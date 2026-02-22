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
//!   toggle overlap|timbre|crossfade
//!   set duration|overlap|crossfade|total <val>
//!   state                         — request current state snapshot
//!   - . +                         — rate current chunk (bad/neutral/good)
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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use ace_step_rs::pipeline::{format_metas, AceStepPipeline};
use ace_step_rs::streaming::{ChunkRequest, StreamConfig, StreamingGenerator};
use rodio::{OutputStream, Sink, Source};

// ── Audio source (copied from stream_cli) ────────────────────────────────

struct ChannelSource {
    rx: mpsc::Receiver<Vec<f32>>,
    current: Vec<f32>,
    pos: usize,
    channels: u16,
    sample_rate: u32,
    consumed: Arc<AtomicUsize>,
}

impl ChannelSource {
    fn new(
        rx: mpsc::Receiver<Vec<f32>>,
        channels: u16,
        sample_rate: u32,
        consumed: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            rx,
            current: Vec::new(),
            pos: 0,
            channels,
            sample_rate,
            consumed,
        }
    }
}

impl Iterator for ChannelSource {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        loop {
            if self.pos < self.current.len() {
                let s = self.current[self.pos];
                self.pos += 1;
                self.consumed.fetch_add(1, Ordering::Relaxed);
                return Some(s);
            }
            match self.rx.recv() {
                Ok(buf) => {
                    self.current = buf;
                    self.pos = 0;
                }
                Err(_) => return None,
            }
        }
    }
}

impl Source for ChannelSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        self.channels
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn total_duration(&self) -> Option<std::time::Duration> {
        None
    }
}

// ── Messages ─────────────────────────────────────────────────────────────

enum GenCmd {
    Generate(Option<ChunkRequest>),
    Config(ConfigUpdate),
    Rate(usize, String),
    Quit,
}

enum ConfigUpdate {
    ToggleOverlap,
    ToggleTimbre,
    ToggleCrossfade,
    SetDuration(f64),
    SetOverlap(f64),
    SetCrossfade(u32),
    SetTotal(f64),
}

enum GenEvent {
    Ready(StreamConfig),
    Generating {
        chunk_index: usize,
    },
    Chunk {
        chunk_index: usize,
        audio_samples: usize,
        gen_time_s: f64,
        caption: String,
        metas: String,
        lyrics: String,
        language: String,
        config: StreamConfig,
    },
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
    metas: String,
    lyrics: String,
    language: String,
    bpm: Option<u32>,
    key_scale: Option<String>,
    chunk_index: usize,
    config: StreamConfig,
    is_generating: bool,
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
        format!(
            "event:state chunk={} generating={} bpm={} key={:?} lang={} \
             dur={:.0} ovlp={:.0} total={:.0} overlap={} timbre={} xfade={} \
             caption={:?} lyrics={:?}",
            self.chunk_index,
            self.is_generating,
            bpm,
            key,
            self.language,
            self.config.chunk_duration_s,
            self.config.overlap_s,
            self.config.total_duration_s,
            self.config.use_overlap,
            self.config.use_timbre_from_prev,
            self.config.use_crossfade,
            caption_flat,
            lyrics_flat,
        )
    }
}

type SharedState = Arc<Mutex<DaemonState>>;

// ── Command parsing ───────────────────────────────────────────────────────

enum ParsedCmd {
    FieldUpdate(String, String),
    Toggle(String),
    Set(String, String),
    Rate(String),
    StateQuery,
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
        "caption" | "lyrics" | "lang" | "seed" | "bpm" | "key" if parts.len() >= 2 => {
            let rest = line[cmd.len()..].trim().to_string();
            ParsedCmd::FieldUpdate(cmd, rest)
        }
        "toggle" if parts.len() >= 2 => ParsedCmd::Toggle(parts[1].to_lowercase()),
        "set" if parts.len() >= 3 => ParsedCmd::Set(parts[1].to_lowercase(), parts[2].to_string()),
        "state" => ParsedCmd::StateQuery,
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
                    rebuild_metas(&mut st);
                    format!(
                        "event:ok bpm={}",
                        st.bpm
                            .map(|b| b.to_string())
                            .unwrap_or_else(|| "none".to_string())
                    )
                }
                "key" => {
                    if value == "none" || value == "off" {
                        st.key_scale = None;
                    } else {
                        st.key_scale = Some(value.clone());
                    }
                    rebuild_metas(&mut st);
                    format!("event:ok key={:?}", st.key_scale)
                }
                "caption" => {
                    st.caption = value.clone();
                    cmd_tx
                        .send(GenCmd::Generate(Some(ChunkRequest {
                            caption: Some(value.clone()),
                            ..Default::default()
                        })))
                        .ok();
                    format!(
                        "event:ok caption updated: {}",
                        &value[..value.len().min(60)]
                    )
                }
                "lyrics" => {
                    let lyrics = value.replace("\\n", "\n");
                    st.lyrics = lyrics.clone();
                    cmd_tx
                        .send(GenCmd::Generate(Some(ChunkRequest {
                            lyrics: Some(lyrics),
                            ..Default::default()
                        })))
                        .ok();
                    "event:ok lyrics updated".to_string()
                }
                "lang" => {
                    st.language = value.clone();
                    cmd_tx
                        .send(GenCmd::Generate(Some(ChunkRequest {
                            language: Some(value.clone()),
                            ..Default::default()
                        })))
                        .ok();
                    format!("event:ok lang={value}")
                }
                "seed" => {
                    if let Ok(s) = value.parse::<u64>() {
                        cmd_tx
                            .send(GenCmd::Generate(Some(ChunkRequest {
                                seed: Some(s),
                                ..Default::default()
                            })))
                            .ok();
                        format!("event:ok seed={s}")
                    } else {
                        "event:error invalid seed".to_string()
                    }
                }
                _ => format!("event:error unknown field: {field}"),
            };
            Some(ack)
        }

        ParsedCmd::Toggle(what) => {
            let upd = match what.as_str() {
                "overlap" => Some(ConfigUpdate::ToggleOverlap),
                "timbre" => Some(ConfigUpdate::ToggleTimbre),
                "crossfade" => Some(ConfigUpdate::ToggleCrossfade),
                _ => None,
            };
            if let Some(upd) = upd {
                cmd_tx.send(GenCmd::Config(upd)).ok();
                Some(format!("event:ok toggle {what}"))
            } else {
                Some(format!("event:error unknown toggle: {what}"))
            }
        }

        ParsedCmd::Set(param, value) => {
            let result = match param.as_str() {
                "duration" => value.parse::<f64>().ok().map(|v| {
                    cmd_tx
                        .send(GenCmd::Config(ConfigUpdate::SetDuration(v)))
                        .ok();
                    state.lock().unwrap().config.chunk_duration_s = v;
                    format!("event:ok duration={v}s")
                }),
                "overlap" => value.parse::<f64>().ok().map(|v| {
                    cmd_tx
                        .send(GenCmd::Config(ConfigUpdate::SetOverlap(v)))
                        .ok();
                    state.lock().unwrap().config.overlap_s = v;
                    format!("event:ok overlap={v}s")
                }),
                "crossfade" => value.parse::<u32>().ok().map(|v| {
                    cmd_tx
                        .send(GenCmd::Config(ConfigUpdate::SetCrossfade(v)))
                        .ok();
                    state.lock().unwrap().config.crossfade_ms = v;
                    format!("event:ok crossfade={v}ms")
                }),
                "total" => value.parse::<f64>().ok().map(|v| {
                    cmd_tx.send(GenCmd::Config(ConfigUpdate::SetTotal(v))).ok();
                    let mut st = state.lock().unwrap();
                    st.config.total_duration_s = v;
                    rebuild_metas(&mut st);
                    format!("event:ok total={v}s")
                }),
                _ => Some(format!("event:error unknown setting: {param}")),
            };
            Some(result.unwrap_or_else(|| format!("event:error invalid value for {param}")))
        }

        ParsedCmd::Rate(r) => {
            let idx = state.lock().unwrap().chunk_index.saturating_sub(1);
            cmd_tx.send(GenCmd::Rate(idx, r.clone())).ok();
            Some(format!("event:ok rated chunk {idx}: {r}"))
        }

        ParsedCmd::StateQuery => {
            let st = state.lock().unwrap();
            Some(st.state_line())
        }

        ParsedCmd::Quit => None,

        ParsedCmd::Unknown(s) if s.is_empty() => Some(String::new()),
        ParsedCmd::Unknown(s) => Some(format!("event:error unknown command: {s}")),
    }
}

fn rebuild_metas(st: &mut DaemonState) {
    let duration = if st.config.total_duration_s > 0.0 {
        st.config.total_duration_s
    } else {
        st.config.chunk_duration_s
    };
    st.metas = format_metas(st.bpm, Some("4/4"), st.key_scale.as_deref(), Some(duration));
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
            Some(ack) => {
                event_tx.send(ack).ok();
            }
            None => {
                // Quit command
                event_tx.send("event:ok shutting down".to_string()).ok();
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

fn apply_config_update(cfg: &mut StreamConfig, upd: &ConfigUpdate) {
    match upd {
        ConfigUpdate::ToggleOverlap => cfg.use_overlap = !cfg.use_overlap,
        ConfigUpdate::ToggleTimbre => cfg.use_timbre_from_prev = !cfg.use_timbre_from_prev,
        ConfigUpdate::ToggleCrossfade => cfg.use_crossfade = !cfg.use_crossfade,
        ConfigUpdate::SetDuration(v) => cfg.chunk_duration_s = *v,
        ConfigUpdate::SetOverlap(v) => cfg.overlap_s = *v,
        ConfigUpdate::SetCrossfade(v) => cfg.crossfade_ms = *v,
        ConfigUpdate::SetTotal(v) => cfg.total_duration_s = *v,
    }
}

fn generator_thread(
    cmd_rx: mpsc::Receiver<GenCmd>,
    event_tx: mpsc::Sender<GenEvent>,
    audio_tx: mpsc::SyncSender<Vec<f32>>,
    initial_caption: String,
    initial_metas: String,
    initial_lyrics: String,
    initial_lang: String,
    quit_flag: Arc<AtomicBool>,
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

    let config = StreamConfig {
        language: initial_lang,
        ..Default::default()
    };

    event_tx.send(GenEvent::Ready(config.clone())).ok();

    let mut streamer =
        StreamingGenerator::new(config, &initial_caption, &initial_metas, &initial_lyrics);

    loop {
        if quit_flag.load(Ordering::Relaxed) {
            return;
        }

        let mut pending_request: Option<Option<ChunkRequest>> = None;
        let mut should_quit = false;

        match cmd_rx.try_recv() {
            Ok(GenCmd::Quit) => return,
            Ok(GenCmd::Config(upd)) => apply_config_update(streamer.config_mut(), &upd),
            Ok(GenCmd::Generate(req)) => pending_request = Some(req),
            Ok(GenCmd::Rate(idx, rating)) => streamer.rate_chunk(idx, &rating),
            Err(_) => {}
        }
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                GenCmd::Quit => {
                    should_quit = true;
                    break;
                }
                GenCmd::Config(upd) => apply_config_update(streamer.config_mut(), &upd),
                GenCmd::Generate(req) => pending_request = Some(req),
                GenCmd::Rate(idx, rating) => streamer.rate_chunk(idx, &rating),
            }
        }
        if should_quit {
            return;
        }

        if pending_request.is_none() {
            thread::sleep(std::time::Duration::from_millis(10));
        }

        if let Some(req) = pending_request {
            let chunk_idx = streamer.chunks_generated();
            event_tx
                .send(GenEvent::Generating {
                    chunk_index: chunk_idx,
                })
                .ok();

            let t0 = std::time::Instant::now();
            match streamer.next_chunk(&mut pipeline, req) {
                Ok(chunk) => {
                    let gen_time = t0.elapsed().as_secs_f64();
                    let audio_samples = chunk.audio.samples.len();
                    if audio_tx.try_send(chunk.audio.samples).is_err() {
                        return;
                    }
                    event_tx
                        .send(GenEvent::Chunk {
                            chunk_index: chunk.chunk_index,
                            audio_samples,
                            gen_time_s: gen_time,
                            caption: streamer.caption().to_string(),
                            metas: streamer.metas().to_string(),
                            lyrics: streamer.lyrics().to_string(),
                            language: streamer.language().to_string(),
                            config: streamer.config().clone(),
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
    let initial_caption =
        "Driving synthwave with retro 80s analog synths, pulsing bass, soaring arpeggios, \
         neon-lit atmospheric pads, energetic beat with heavy kick and hi-hats, upbeat futuristic dance music";
    let initial_lyrics = "la la la la la la la la\nla la la la la la la la\n\
                          oh oh oh oh oh oh oh oh\noh oh oh oh oh oh oh oh";
    let initial_lang = "en";
    let initial_bpm: Option<u32> = Some(128);
    let initial_key: Option<&str> = Some("A minor");
    let initial_metas = format_metas(initial_bpm, Some("4/4"), initial_key, Some(30.0));

    // Shared state (for socket clients to query / update display fields)
    let state = Arc::new(Mutex::new(DaemonState {
        caption: initial_caption.to_string(),
        metas: initial_metas.clone(),
        lyrics: initial_lyrics.to_string(),
        language: initial_lang.to_string(),
        bpm: initial_bpm,
        key_scale: initial_key.map(|s| s.to_string()),
        chunk_index: 0,
        config: StreamConfig::default(),
        is_generating: false,
    }));

    // Channels
    let (cmd_tx, cmd_rx) = mpsc::channel::<GenCmd>();
    let (event_tx, event_rx) = mpsc::channel::<GenEvent>();
    let (audio_tx, audio_rx) = mpsc::sync_channel::<Vec<f32>>(1);

    // Audio output
    let playback_consumed = Arc::new(AtomicUsize::new(0));
    let (_stream, stream_handle) =
        OutputStream::try_default().expect("failed to open audio output");
    let sink = Sink::try_new(&stream_handle).expect("failed to create audio sink");
    let source = ChannelSource::new(audio_rx, 2, 48000, Arc::clone(&playback_consumed));
    sink.append(source);

    // Generator thread
    let gen_handle = thread::spawn({
        let caption = initial_caption.to_string();
        let metas = initial_metas;
        let lyrics = initial_lyrics.to_string();
        let lang = initial_lang.to_string();
        let quit = Arc::clone(&quit_flag);
        move || {
            generator_thread(
                cmd_rx, event_tx, audio_tx, caption, metas, lyrics, lang, quit,
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
            cmd_tx.send(GenCmd::Generate(None)).ok();
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
            Ok(GenEvent::Generating { chunk_index }) => {
                state.lock().unwrap().is_generating = true;
                let msg = format!("event:generating chunk={chunk_index}");
                eprintln!("{msg}");
                broadcast(&clients, &msg);
            }
            Ok(GenEvent::Chunk {
                chunk_index,
                audio_samples,
                gen_time_s,
                caption,
                metas,
                lyrics,
                language,
                config,
            }) => {
                {
                    let mut st = state.lock().unwrap();
                    st.chunk_index = chunk_index + 1;
                    st.caption = caption;
                    st.metas = metas;
                    st.lyrics = lyrics;
                    st.language = language;
                    st.config = config;
                    st.is_generating = false;
                }

                let audio_secs = audio_samples as f64 / (2.0 * 48000.0);
                let msg = format!(
                    "event:chunk chunk={chunk_index} audio={audio_secs:.1}s gen={gen_time_s:.2}s"
                );
                eprintln!("{msg}");
                tracing::info!("{msg}");
                broadcast(&clients, &msg);

                // Schedule next chunk
                cmd_tx.send(GenCmd::Generate(None)).ok();
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
