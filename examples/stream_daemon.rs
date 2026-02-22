//! ACE-Step streaming music daemon with unix socket control interface.
//!
//! Loads the pipeline once, generates music continuously, and accepts
//! connections on a unix socket for live control and event streaming.
//!
//! Usage:
//!   cargo run --release --features cli --example stream_daemon -- --socket /tmp/ace-step.sock
//!
//! Connect with:
//!   socat - UNIX-CONNECT:/tmp/ace-step.sock
//!   nc -U /tmp/ace-step.sock
//!
//! Protocol (newline-delimited text):
//!
//! Client → Daemon (commands):
//!   caption <text>                — update style/genre description
//!   lyrics <text>                 — update lyrics (\n for newlines)
//!   bpm <n|none>                  — set/unset BPM
//!   key <scale|none>              — set/unset key (e.g. "A minor")
//!   lang <code>                   — language (en, zh, ...)
//!   seed <n>                      — seed for next chunk
//!   step <n>                      — step size in seconds (new audio + virtual position advance)
//!   mode fixed|advancing          — virtual position mode (fixed = always same arc point; advancing = progresses through song)
//!   set window <n>                — window size in seconds (takes effect on restart)
//!   save <name>                   — save window snapshot
//!   load <name>                   — load window snapshot
//!   state                         — request current state snapshot
//!   - . +                         — rate current step (bad/neutral/good)
//!   restart                       — re-exec binary
//!   q / quit / exit               — shut down daemon
//!
//! Daemon → Client (events):
//!   event:ready                   — pipeline loaded, generation starting
//!   event:generating chunk=<n>    — chunk generation started
//!   event:chunk chunk=<n> audio=<s>s gen=<s>s
//!   event:error <message>
//!   event:ok <ack message>        — command acknowledged
//!   event:state <key>=<val> ...   — state snapshot (on connect + 'state' cmd)

use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use ace_step_rs::pipeline::AceStepPipeline;
use ace_step_rs::streaming::{
    ChunkRequest, PositionMode, SlidingWindowConfig, SlidingWindowGenerator, WindowSnapshot,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapProd, HeapRb};

// ── Messages ─────────────────────────────────────────────────────────────

enum GenCmd {
    /// Override fields for the next step (caption, lyrics, seed, step_s, etc.)
    Override(ChunkRequest),
    /// Save current window snapshot under a name
    SaveSnapshot(String),
    /// Load a previously saved snapshot
    LoadSnapshot(String),
    Rate(usize, String),
    Quit,
}

enum GenEvent {
    Ready(SlidingWindowConfig),
    Generating {
        step_index: usize,
    },
    Step {
        step_index: usize,
        audio_samples: usize,
        gen_time_s: f64,
        step_s: f64,
        window_s: f64,
        pos_s: f64,
        caption: String,
        lyrics: String,
        language: String,
        bpm: Option<u32>,
        key_scale: Option<String>,
        config: SlidingWindowConfig,
    },
    SnapshotSaved(String),
    SnapshotLoaded(String),
    Error(String),
}

// ── Broadcast: send events to all connected clients ───────────────────────

type ClientList = Arc<Mutex<Vec<mpsc::Sender<String>>>>;

fn broadcast(clients: &ClientList, msg: &str) {
    let mut list = clients.lock().unwrap();
    list.retain(|tx| tx.send(msg.to_string()).is_ok());
}

// ── Daemon state (shared between socket threads and main loop) ────────────

struct DaemonState {
    caption: String,
    lyrics: String,
    language: String,
    bpm: Option<u32>,
    key_scale: Option<String>,
    step_index: usize,
    pos_s: f64,
    config: SlidingWindowConfig,
    is_generating: bool,
    /// Just the names of saved snapshots (full state lives in generator thread).
    snapshot_names: std::collections::HashSet<String>,
}

impl DaemonState {
    fn state_line(&self) -> String {
        let bpm = self
            .bpm
            .map(|b| b.to_string())
            .unwrap_or_else(|| "none".to_string());
        let key = self.key_scale.clone().unwrap_or_else(|| "none".to_string());
        let caption_flat: String = self
            .caption
            .chars()
            .map(|c| if c == '\n' { '↵' } else { c })
            .take(80)
            .collect();
        let lyrics_flat: String = self
            .lyrics
            .chars()
            .map(|c| if c == '\n' { '↵' } else { c })
            .take(80)
            .collect();
        let snapshot_names: Vec<&String> = self.snapshot_names.iter().collect();
        let mode = match self.config.position_mode {
            ace_step_rs::streaming::PositionMode::Advancing => "advancing",
            ace_step_rs::streaming::PositionMode::Fixed => "fixed",
        };
        format!(
            "event:state step={} pos={:.0}s generating={} bpm={} key={:?} lang={} \
             window={:.0}s step_s={:.0}s mode={} snapshots={:?} \
             caption={:?} lyrics={:?}",
            self.step_index,
            self.pos_s,
            self.is_generating,
            bpm,
            key,
            self.language,
            self.config.window_s,
            self.config.step_s,
            mode,
            snapshot_names,
            caption_flat,
            lyrics_flat,
        )
    }
}

type SharedState = Arc<Mutex<DaemonState>>;

// ── Command parsing ───────────────────────────────────────────────────────

enum ParsedCmd {
    FieldUpdate(String, String),
    Set(String, String),
    Rate(String),
    StateQuery,
    SaveSnapshot(String),
    LoadSnapshot(String),
    Restart,
    Quit,
    Unknown(String),
}

fn parse_cmd(line: &str) -> ParsedCmd {
    let line = line.trim();
    if line.is_empty() {
        return ParsedCmd::Unknown(String::new());
    }
    if line == "-" || line == "." || line == "+" {
        return ParsedCmd::Rate(line.to_string());
    }
    let parts: Vec<&str> = line.splitn(3, ' ').collect();
    let cmd = parts[0].to_lowercase();
    match cmd.as_str() {
        "caption" | "lyrics" | "lang" | "seed" | "bpm" | "key" | "step" | "mode"
            if parts.len() >= 2 =>
        {
            let rest = line[cmd.len()..].trim().to_string();
            ParsedCmd::FieldUpdate(cmd, rest)
        }
        "set" if parts.len() >= 3 => ParsedCmd::Set(parts[1].to_lowercase(), parts[2].to_string()),
        "save" if parts.len() >= 2 => ParsedCmd::SaveSnapshot(parts[1].to_string()),
        "load" if parts.len() >= 2 => ParsedCmd::LoadSnapshot(parts[1].to_string()),
        "state" => ParsedCmd::StateQuery,
        "restart" => ParsedCmd::Restart,
        "quit" | "q" | "exit" => ParsedCmd::Quit,
        _ => {
            // Free-form text → caption update
            ParsedCmd::FieldUpdate("caption".to_string(), line.to_string())
        }
    }
}

// ── Process a command from a client ──────────────────────────────────────
//
// Returns the ack line to send back to the client, or None for quit.
// Also sends GenCmd to the generator thread and updates shared state.

fn process_cmd(line: &str, cmd_tx: &mpsc::Sender<GenCmd>, state: &SharedState) -> Option<String> {
    match parse_cmd(line) {
        ParsedCmd::FieldUpdate(field, value) => {
            let mut st = state.lock().unwrap();
            let ack = match field.as_str() {
                "bpm" => {
                    if value == "none" || value == "off" {
                        st.bpm = None;
                    } else if let Ok(b) = value.parse::<u32>() {
                        st.bpm = Some(b);
                    } else {
                        return Some("event:error invalid bpm value".to_string());
                    }
                    let bpm = st.bpm;
                    cmd_tx
                        .send(GenCmd::Override(ChunkRequest {
                            bpm: Some(bpm),
                            ..Default::default()
                        }))
                        .ok();
                    format!(
                        "event:ok bpm={}",
                        bpm.map(|b| b.to_string())
                            .unwrap_or_else(|| "none".to_string())
                    )
                }
                "key" => {
                    if value == "none" || value == "off" {
                        st.key_scale = None;
                    } else {
                        st.key_scale = Some(value.clone());
                    }
                    let key = st.key_scale.clone();
                    cmd_tx
                        .send(GenCmd::Override(ChunkRequest {
                            key_scale: Some(key.clone()),
                            ..Default::default()
                        }))
                        .ok();
                    format!("event:ok key={:?}", key)
                }
                "caption" => {
                    st.caption = value.clone();
                    cmd_tx
                        .send(GenCmd::Override(ChunkRequest {
                            caption: Some(value.clone()),
                            ..Default::default()
                        }))
                        .ok();
                    format!("event:ok caption: {}", &value[..value.len().min(60)])
                }
                "lyrics" => {
                    let lyrics = value.replace("\\n", "\n");
                    st.lyrics = lyrics.clone();
                    cmd_tx
                        .send(GenCmd::Override(ChunkRequest {
                            lyrics: Some(lyrics),
                            ..Default::default()
                        }))
                        .ok();
                    "event:ok lyrics updated".to_string()
                }
                "lang" => {
                    st.language = value.clone();
                    cmd_tx
                        .send(GenCmd::Override(ChunkRequest {
                            language: Some(value.clone()),
                            ..Default::default()
                        }))
                        .ok();
                    format!("event:ok lang={value}")
                }
                "seed" => {
                    if let Ok(s) = value.parse::<u64>() {
                        cmd_tx
                            .send(GenCmd::Override(ChunkRequest {
                                seed: Some(s),
                                ..Default::default()
                            }))
                            .ok();
                        format!("event:ok seed={s}")
                    } else {
                        "event:error invalid seed".to_string()
                    }
                }
                "step" => {
                    if let Ok(s) = value.parse::<f64>() {
                        let window = st.config.window_s;
                        let clamped = s.clamp(1.0, window - 1.0);
                        st.config.step_s = clamped;
                        cmd_tx
                            .send(GenCmd::Override(ChunkRequest {
                                step_s: Some(clamped),
                                ..Default::default()
                            }))
                            .ok();
                        format!("event:ok step={clamped:.0}s (window={window:.0}s)")
                    } else {
                        "event:error invalid step value".to_string()
                    }
                }
                "mode" => {
                    let mode = match value.to_lowercase().as_str() {
                        "fixed" => Some(PositionMode::Fixed),
                        "advancing" => Some(PositionMode::Advancing),
                        _ => None,
                    };
                    if let Some(m) = mode {
                        st.config.position_mode = m.clone();
                        cmd_tx
                            .send(GenCmd::Override(ChunkRequest {
                                position_mode: Some(m.clone()),
                                ..Default::default()
                            }))
                            .ok();
                        format!("event:ok mode={value}")
                    } else {
                        "event:error mode must be 'fixed' or 'advancing'".to_string()
                    }
                }
                _ => format!("event:error unknown field: {field}"),
            };
            Some(ack)
        }

        ParsedCmd::Set(param, value) => {
            let result = match param.as_str() {
                "window" => value.parse::<f64>().ok().map(|v| {
                    state.lock().unwrap().config.window_s = v;
                    format!("event:ok window={v:.0}s (takes effect on restart)")
                }),
                _ => Some(format!("event:error unknown setting: {param}")),
            };
            Some(result.unwrap_or_else(|| format!("event:error invalid value for {param}")))
        }

        ParsedCmd::Rate(r) => {
            let idx = state.lock().unwrap().step_index.saturating_sub(1);
            cmd_tx.send(GenCmd::Rate(idx, r.clone())).ok();
            Some(format!("event:ok rated step {idx}: {r}"))
        }

        ParsedCmd::SaveSnapshot(name) => {
            cmd_tx.send(GenCmd::SaveSnapshot(name.clone())).ok();
            Some(format!("event:ok saving snapshot {name:?}"))
        }

        ParsedCmd::LoadSnapshot(name) => {
            let has = state.lock().unwrap().snapshot_names.contains(&name);
            if has {
                cmd_tx.send(GenCmd::LoadSnapshot(name.clone())).ok();
                Some(format!("event:ok loading snapshot {name:?}"))
            } else {
                let names: Vec<String> = state
                    .lock()
                    .unwrap()
                    .snapshot_names
                    .iter()
                    .cloned()
                    .collect();
                Some(format!(
                    "event:error snapshot {name:?} not found, available: {names:?}"
                ))
            }
        }

        ParsedCmd::StateQuery => {
            let st = state.lock().unwrap();
            Some(st.state_line())
        }

        ParsedCmd::Restart => Some("event:restart".to_string()),

        ParsedCmd::Quit => None,

        ParsedCmd::Unknown(s) if s.is_empty() => Some(String::new()),
        ParsedCmd::Unknown(s) => Some(format!("event:error unknown command: {s}")),
    }
}

// ── Socket accept loop ────────────────────────────────────────────────────

fn handle_client(
    stream: UnixStream,
    cmd_tx: mpsc::Sender<GenCmd>,
    state: SharedState,
    clients: ClientList,
    quit_flag: Arc<AtomicBool>,
) {
    // Channel for daemon→client events
    let (event_tx, event_rx) = mpsc::channel::<String>();

    // Register this client for broadcasts
    clients.lock().unwrap().push(event_tx.clone());

    // Send current state immediately on connect
    {
        let st = state.lock().unwrap();
        let state_line = st.state_line();
        event_tx.send(state_line).ok();
    }

    let write_stream = stream.try_clone().expect("clone socket");

    // Writer thread: pull from event_rx, write to socket
    let write_handle = thread::spawn(move || {
        let mut w = io::BufWriter::new(write_stream);
        for line in event_rx {
            if line.is_empty() {
                continue;
            }
            if writeln!(w, "{line}").is_err() {
                break;
            }
            let _ = w.flush();
        }
    });

    // Reader: parse commands from socket
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        tracing::info!("socket cmd: {:?}", line);
        match process_cmd(&line, &cmd_tx, &state) {
            Some(ack) if ack == "event:restart" => {
                broadcast(&clients, "event:restarting");
                quit_flag.store(true, Ordering::Relaxed);
                cmd_tx.send(GenCmd::Quit).ok();
                // Use /proc/self/exe so Linux resolves to the current binary
                // on disk (even after a rebuild replaced the inode).
                let exe = std::path::PathBuf::from("/proc/self/exe");
                let args: Vec<String> = std::env::args().skip(1).collect();
                tracing::info!("Restarting: {:?} {:?}", exe, args);
                thread::spawn(move || {
                    thread::sleep(std::time::Duration::from_millis(500));
                    std::process::Command::new(exe)
                        .args(args)
                        .spawn()
                        .expect("re-exec failed");
                    std::process::exit(0);
                });
                break;
            }
            Some(ack) => {
                event_tx.send(ack).ok();
            }
            None => {
                // Quit command
                broadcast(&clients, "event:ok shutting down");
                quit_flag.store(true, Ordering::Relaxed);
                cmd_tx.send(GenCmd::Quit).ok();
                break;
            }
        }
    }

    drop(event_tx);
    write_handle.join().ok();
}

fn socket_accept_loop(
    listener: UnixListener,
    cmd_tx: mpsc::Sender<GenCmd>,
    state: SharedState,
    clients: ClientList,
    quit_flag: Arc<AtomicBool>,
) {
    for stream in listener.incoming() {
        if quit_flag.load(Ordering::Relaxed) {
            break;
        }
        match stream {
            Ok(s) => {
                let cmd_tx = cmd_tx.clone();
                let state = Arc::clone(&state);
                let clients = Arc::clone(&clients);
                let quit_flag = Arc::clone(&quit_flag);
                thread::spawn(move || handle_client(s, cmd_tx, state, clients, quit_flag));
            }
            Err(e) => {
                tracing::warn!("socket accept error: {e}");
            }
        }
    }
}

// ── Generator thread ──────────────────────────────────────────────────────

fn generator_thread(
    cmd_rx: mpsc::Receiver<GenCmd>,
    event_tx: mpsc::Sender<GenEvent>,
    mut ring_prod: HeapProd<f32>,
    initial_caption: String,
    initial_lyrics: String,
    initial_bpm: Option<u32>,
    initial_key_scale: Option<String>,
    initial_lang: String,
    quit_flag: Arc<AtomicBool>,
    // Snapshots are shared with DaemonState for client queries
    snapshots: Arc<
        Mutex<std::collections::HashMap<String, ace_step_rs::streaming::WindowSnapshot>>,
    >,
) {
    let device = match candle_core::Device::cuda_if_available(0) {
        Ok(d) => d,
        Err(e) => {
            event_tx.send(GenEvent::Error(format!("Device: {e}"))).ok();
            return;
        }
    };

    let mut pipeline = match AceStepPipeline::load(&device, candle_core::DType::F32) {
        Ok(p) => p,
        Err(e) => {
            event_tx
                .send(GenEvent::Error(format!("Pipeline: {e}")))
                .ok();
            return;
        }
    };

    let config = SlidingWindowConfig {
        language: initial_lang,
        ..Default::default()
    };

    event_tx.send(GenEvent::Ready(config.clone())).ok();

    let mut generator = SlidingWindowGenerator::new(
        config,
        &initial_caption,
        initial_bpm,
        initial_key_scale.as_deref(),
        "4/4",
        &initial_lyrics,
    );

    loop {
        if quit_flag.load(Ordering::Relaxed) {
            return;
        }

        // Drain all pending commands; merge overrides, last-write-wins per field
        let mut override_req = ChunkRequest::default();
        let mut has_override = false;

        loop {
            match cmd_rx.try_recv() {
                Ok(GenCmd::Quit) => return,
                Ok(GenCmd::Override(req)) => {
                    if req.caption.is_some() {
                        override_req.caption = req.caption;
                    }
                    if req.bpm.is_some() {
                        override_req.bpm = req.bpm;
                    }
                    if req.key_scale.is_some() {
                        override_req.key_scale = req.key_scale;
                    }
                    if req.time_signature.is_some() {
                        override_req.time_signature = req.time_signature;
                    }
                    if req.lyrics.is_some() {
                        override_req.lyrics = req.lyrics;
                    }
                    if req.language.is_some() {
                        override_req.language = req.language;
                    }
                    if req.seed.is_some() {
                        override_req.seed = req.seed;
                    }
                    if req.step_s.is_some() {
                        override_req.step_s = req.step_s;
                    }
                    if req.position_mode.is_some() {
                        override_req.position_mode = req.position_mode;
                    }
                    has_override = true;
                }
                Ok(GenCmd::SaveSnapshot(name)) => {
                    if let Some(snap) = generator.save_snapshot() {
                        snapshots.lock().unwrap().insert(name.clone(), snap);
                        event_tx.send(GenEvent::SnapshotSaved(name)).ok();
                    }
                }
                Ok(GenCmd::LoadSnapshot(name)) => {
                    let snap = snapshots.lock().unwrap().remove(&name);
                    if let Some(snap) = snap {
                        generator.load_snapshot(&snap);
                        // Put it back so it can be loaded again
                        if let Some(new_snap) = generator.save_snapshot() {
                            snapshots.lock().unwrap().insert(name.clone(), new_snap);
                        }
                        event_tx.send(GenEvent::SnapshotLoaded(name)).ok();
                    }
                }
                Ok(GenCmd::Rate(idx, rating)) => {
                    tracing::warn!("RATING step={idx} rating={rating}");
                }
                Err(_) => break,
            }
        }

        let req = if has_override {
            Some(override_req)
        } else {
            None
        };

        event_tx
            .send(GenEvent::Generating {
                step_index: generator.steps_generated(),
            })
            .ok();

        let t0 = std::time::Instant::now();
        tracing::info!(
            "next_step: caption={:?} lyrics_len={} bpm={:?}",
            &generator.caption()[..generator.caption().len().min(60)],
            generator.lyrics().len(),
            generator.bpm(),
        );
        match generator.next_step(&mut pipeline, req) {
            Ok(step) => {
                let gen_time = t0.elapsed().as_secs_f64();
                let audio_samples = step.audio.samples.len();
                // Push directly into ring buffer, spinning when full.
                // This blocks the generator naturally: ring holds ~1.5 steps,
                // so once the ring is full the generator waits for playback
                // to consume samples before generating the next step.
                {
                    let samples = step.audio.samples;
                    let mut offset = 0;
                    while offset < samples.len() {
                        if quit_flag.load(Ordering::Relaxed) {
                            return;
                        }
                        let written = ring_prod.push_slice(&samples[offset..]);
                        offset += written;
                        if written == 0 {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    }
                }
                event_tx
                    .send(GenEvent::Step {
                        step_index: step.step_index,
                        audio_samples,
                        gen_time_s: gen_time,
                        step_s: step.step_s,
                        window_s: step.window_s,
                        pos_s: step.pos_s,
                        caption: generator.caption().to_string(),
                        lyrics: generator.lyrics().to_string(),
                        language: generator.language().to_string(),
                        bpm: generator.bpm(),
                        key_scale: generator.key_scale().map(|s| s.to_string()),
                        config: generator.config.clone(),
                    })
                    .ok();
            }
            Err(e) => {
                event_tx
                    .send(GenEvent::Error(format!("Generation: {e}")))
                    .ok();
                return;
            }
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    // Parse --socket <path>
    let args: Vec<String> = std::env::args().collect();
    let socket_path = args
        .windows(2)
        .find(|w| w[0] == "--socket")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "/tmp/ace-step.sock".to_string());

    // File logging
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("stream_daemon.log")
        .expect("failed to open stream_daemon.log");
    tracing_subscriber::fmt()
        .with_writer(log_file)
        .with_ansi(false)
        .with_target(false)
        .init();

    // Remove stale socket
    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path).expect("failed to bind unix socket");
    eprintln!("Listening on {socket_path}");
    tracing::info!("Listening on {socket_path}");

    let quit_flag = Arc::new(AtomicBool::new(false));
    let clients: ClientList = Arc::new(Mutex::new(Vec::new()));

    // Initial generation parameters
    let initial_caption = "melodic electronic, lush pads, warm bassline, hypnotic groove, \
         subtle percussion, ambient textures, flowing and immersive";
    let initial_lyrics = "la la la la la la la la\nla la la la la la la la\n\
                          oh oh oh oh oh oh oh oh\noh oh oh oh oh oh oh oh";
    let initial_lang = "en";
    let initial_bpm: Option<u32> = Some(120);
    let initial_key: Option<&str> = Some("F major");
    let initial_config = SlidingWindowConfig {
        language: initial_lang.to_string(),
        ..Default::default()
    };

    // Snapshots shared between generator thread and daemon state
    let snapshots: Arc<Mutex<std::collections::HashMap<String, WindowSnapshot>>> =
        Arc::new(Mutex::new(std::collections::HashMap::new()));

    // Shared state (for socket clients to query / update display fields)
    let state = Arc::new(Mutex::new(DaemonState {
        caption: initial_caption.to_string(),
        lyrics: initial_lyrics.to_string(),
        language: initial_lang.to_string(),
        bpm: initial_bpm,
        key_scale: initial_key.map(|s| s.to_string()),
        step_index: 0,
        pos_s: 0.0,
        config: initial_config,
        is_generating: false,
        snapshot_names: std::collections::HashSet::new(),
    }));

    // Channels
    let (cmd_tx, cmd_rx) = mpsc::channel::<GenCmd>();
    let (event_tx, event_rx) = mpsc::channel::<GenEvent>();

    // Ring holds 2 steps so the generator stays exactly 1 step ahead of playback.
    // While cpal plays step N, the generator fills step N+1 into the ring.
    // Once full, generator blocks until cpal drains enough for step N+2 (~13s).
    // At 30s steps / 13s gen, always ~17s surplus per step → solid headroom.
    const STEP_SAMPLES: usize = 48000 * 2 * 30; // 30s stereo @ 48kHz
    const RING_CAPACITY: usize = STEP_SAMPLES * 2; // 2 steps = 1 step ahead
    let ring = HeapRb::<f32>::new(RING_CAPACITY);
    let (ring_prod, mut ring_cons) = ring.split();

    // cpal audio output: callback pulls from ring buffer
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no audio output device");

    // Log supported configs for diagnostics
    if let Ok(mut configs) = device.supported_output_configs() {
        while let Some(c) = configs.next() {
            tracing::info!(
                "cpal supported: ch={} rate={}-{} fmt={:?}",
                c.channels(),
                c.min_sample_rate().0,
                c.max_sample_rate().0,
                c.sample_format(),
            );
        }
    }

    let stream_config = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(48000),
        buffer_size: cpal::BufferSize::Default,
    };
    tracing::info!(
        "cpal opening: device={:?} ch={} rate={}",
        device.name().unwrap_or_default(),
        stream_config.channels,
        stream_config.sample_rate.0,
    );
    // Playback doesn't start until the first chunk is in the ring,
    // eliminating the startup silence gap.
    let playback_started = Arc::new(AtomicBool::new(false));
    let playback_started2 = Arc::clone(&playback_started);
    let _audio_stream = device
        .build_output_stream(
            &stream_config,
            move |data: &mut [f32], _| {
                if !playback_started2.load(Ordering::Relaxed) {
                    data.fill(0.0);
                    return;
                }
                let n = ring_cons.pop_slice(data);
                // Fill remainder with silence if ring is behind
                if n < data.len() {
                    tracing::warn!("ring underrun: needed {} got {}", data.len(), n);
                    data[n..].fill(0.0);
                }
            },
            |e| tracing::error!("cpal error: {e}"),
            None,
        )
        .expect("failed to build audio stream");
    _audio_stream.play().expect("failed to start audio stream");

    // Generator thread: owns ring_prod and pushes audio directly,
    // blocking when the ring is full (natural backpressure).
    let gen_handle = thread::spawn({
        let caption = initial_caption.to_string();
        let lyrics = initial_lyrics.to_string();
        let bpm = initial_bpm;
        let key = initial_key.map(|s| s.to_string());
        let lang = initial_lang.to_string();
        let quit = Arc::clone(&quit_flag);
        let snaps = Arc::clone(&snapshots);
        move || {
            generator_thread(
                cmd_rx, event_tx, ring_prod, caption, lyrics, bpm, key, lang, quit, snaps,
            )
        }
    });

    // Socket accept thread
    {
        let cmd_tx = cmd_tx.clone();
        let state = Arc::clone(&state);
        let clients = Arc::clone(&clients);
        let quit = Arc::clone(&quit_flag);
        thread::spawn(move || socket_accept_loop(listener, cmd_tx, state, clients, quit));
    }

    // Ctrl+C
    {
        let quit = Arc::clone(&quit_flag);
        let cmd_tx = cmd_tx.clone();
        let socket_path = socket_path.clone();
        ctrlc::set_handler(move || {
            eprintln!("\nShutting down...");
            quit.store(true, Ordering::Relaxed);
            cmd_tx.send(GenCmd::Quit).ok();
            let _ = std::fs::remove_file(&socket_path);
            std::process::exit(0);
        })
        .ok();
    }

    // Wait for pipeline ready then kick off first generation
    match event_rx.recv() {
        Ok(GenEvent::Ready(config)) => {
            state.lock().unwrap().config = config;
            eprintln!("Pipeline ready. Generating first chunk...");
            tracing::info!("Pipeline ready");
            broadcast(&clients, "event:ready");
        }
        Ok(GenEvent::Error(e)) => {
            eprintln!("Error: {e}");
            return;
        }
        _ => return,
    }

    // Main event loop: receive generator events, update state, broadcast
    loop {
        if quit_flag.load(Ordering::Relaxed) {
            break;
        }

        match event_rx.recv_timeout(std::time::Duration::from_millis(200)) {
            Ok(GenEvent::Generating { step_index }) => {
                state.lock().unwrap().is_generating = true;
                let msg = format!("event:generating step={step_index}");
                eprintln!("{msg}");
                broadcast(&clients, &msg);
            }
            Ok(GenEvent::Step {
                step_index,
                audio_samples,
                gen_time_s,
                step_s,
                window_s,
                pos_s,
                caption,
                lyrics,
                language,
                bpm,
                key_scale,
                config,
            }) => {
                // Start playback once the first chunk is in the ring.
                if step_index == 0 {
                    playback_started.store(true, Ordering::Relaxed);
                    tracing::info!("Playback started (first chunk in ring)");
                }

                {
                    let mut st = state.lock().unwrap();
                    st.step_index = step_index + 1;
                    st.pos_s = pos_s + step_s; // pos after this step
                    st.caption = caption;
                    st.lyrics = lyrics;
                    st.language = language;
                    st.bpm = bpm;
                    st.key_scale = key_scale;
                    st.config = config;
                    st.is_generating = false;
                }

                let audio_secs = audio_samples as f64 / (2.0 * 48000.0);
                let msg = format!(
                    "event:step step={step_index} pos={pos_s:.0}s audio={audio_secs:.1}s \
                     gen={gen_time_s:.2}s step_s={step_s:.0}s window_s={window_s:.0}s"
                );
                eprintln!("{msg}");
                tracing::info!("{msg}");
                broadcast(&clients, &msg);
            }
            Ok(GenEvent::SnapshotSaved(name)) => {
                state.lock().unwrap().snapshot_names.insert(name.clone());
                let msg = format!("event:snapshot_saved {name}");
                eprintln!("{msg}");
                broadcast(&clients, &msg);
            }
            Ok(GenEvent::SnapshotLoaded(name)) => {
                let msg = format!("event:snapshot_loaded {name}");
                eprintln!("{msg}");
                broadcast(&clients, &msg);
            }
            Ok(GenEvent::Error(e)) => {
                let msg = format!("event:error {e}");
                eprintln!("{msg}");
                tracing::error!("{e}");
                broadcast(&clients, &msg);
            }
            Ok(GenEvent::Ready(_)) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    drop(cmd_tx);
    gen_handle.join().ok();
    let _ = std::fs::remove_file(&socket_path);
    eprintln!("Done.");
}
