//! ACE-Step radio station daemon.
//!
//! Generates whole songs from a request queue, always keeping one song
//! buffered ahead of playback. Accepts connections on a unix socket
//! for live control (queue songs, skip, change style).
//!
//! Uses the GenerationManager for VRAM monitoring and OOM retry.
//!
//! Usage:
//!   cargo run --release --features cli --example radio_daemon -- --socket /tmp/ace-radio.sock
//!
//! Connect with:
//!   nc -U /tmp/ace-radio.sock
//!
//! Protocol (newline-delimited text):
//!
//! Client → Daemon:
//!   <free text>                    — queue a song (LLM translates, or used as caption directly)
//!   caption <text>                 — queue with explicit caption
//!   request <json>                 — queue a structured SongRequest JSON
//!   skip                           — skip current song, play next buffered
//!   queue                          — show queued requests
//!   history                        — show recently played songs
//!   clear                          — clear the request queue
//!   state                          — current playback state
//!   q / quit / exit                — shut down
//!
//! Daemon → Client:
//!   event:ready                    — pipeline loaded
//!   event:generating song=<n> caption=<text> duration=<s>s
//!   event:playing song=<n> caption=<text> duration=<s>s
//!   event:song_done song=<n>
//!   event:skip
//!   event:ok <ack>
//!   event:error <message>
//!   event:state <key>=<val> ...

use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use ace_step_rs::manager::{GenerationManager, ManagerConfig};
use ace_step_rs::radio::{RadioStation, SongRequest};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

// ── Messages between threads ─────────────────────────────────────────────

enum RadioCmd {
    /// Queue a structured song request.
    Enqueue(SongRequest),
    /// Skip the currently playing song.
    Skip,
    /// Clear the queue.
    ClearQueue,
    /// Shut down.
    Quit,
}

enum RadioEvent {
    /// Pipeline loaded, generation starting.
    Ready,
    /// Song generation started.
    Generating {
        song_index: usize,
        caption: String,
        duration_s: f64,
    },
    /// Song generated and queued for playback.
    SongReady {
        song_index: usize,
        caption: String,
        duration_s: f64,
        gen_time_s: f64,
        seed: u64,
    },
    /// Song started playing.
    Playing {
        song_index: usize,
        caption: String,
        duration_s: f64,
    },
    /// Song finished playing.
    SongDone { song_index: usize },
    /// Skip acknowledged.
    Skipped,
    /// Error.
    Error(String),
}

// ── Broadcast ────────────────────────────────────────────────────────────

type ClientList = Arc<Mutex<Vec<mpsc::Sender<String>>>>;

fn broadcast(clients: &ClientList, msg: &str) {
    let mut list = clients.lock().unwrap();
    list.retain(|tx| tx.send(msg.to_string()).is_ok());
}

// ── Shared state for socket queries ──────────────────────────────────────

struct DaemonState {
    /// Currently playing song info.
    now_playing: Option<NowPlaying>,
    /// Queue snapshot (captions only, for display).
    queue_captions: Vec<String>,
    /// History snapshot (captions only, for display).
    history_captions: Vec<String>,
    /// Total songs generated.
    song_counter: usize,
    /// Is the generator currently working?
    is_generating: bool,
}

#[derive(Clone)]
struct NowPlaying {
    song_index: usize,
    caption: String,
    duration_s: f64,
    samples_total: usize,
}

impl DaemonState {
    fn state_line(&self) -> String {
        let playing = match &self.now_playing {
            Some(np) => format!(
                "song={} caption={:?} duration={:.0}s",
                np.song_index,
                &np.caption[..np.caption.len().min(60)],
                np.duration_s
            ),
            None => "idle".to_string(),
        };
        format!(
            "event:state playing={} queued={} history={} total={}",
            playing,
            self.queue_captions.len(),
            self.history_captions.len(),
            self.song_counter,
        )
    }
}

type SharedState = Arc<Mutex<DaemonState>>;

// ── Command parsing ──────────────────────────────────────────────────────

enum ParsedCmd {
    Enqueue(SongRequest),
    Skip,
    Queue,
    History,
    Clear,
    State,
    Quit,
    Unknown(String),
}

fn parse_cmd(line: &str) -> ParsedCmd {
    let line = line.trim();
    if line.is_empty() {
        return ParsedCmd::Unknown(String::new());
    }

    let parts: Vec<&str> = line.splitn(2, ' ').collect();
    let cmd = parts[0].to_lowercase();

    match cmd.as_str() {
        "skip" => ParsedCmd::Skip,
        "queue" => ParsedCmd::Queue,
        "history" => ParsedCmd::History,
        "clear" => ParsedCmd::Clear,
        "state" => ParsedCmd::State,
        "quit" | "q" | "exit" => ParsedCmd::Quit,
        "caption" if parts.len() >= 2 => ParsedCmd::Enqueue(SongRequest {
            caption: parts[1].to_string(),
            ..Default::default()
        }),
        "request" if parts.len() >= 2 => match serde_json::from_str::<SongRequestJson>(parts[1]) {
            Ok(json) => ParsedCmd::Enqueue(json.into()),
            Err(e) => ParsedCmd::Unknown(format!("invalid JSON: {e}")),
        },
        _ => {
            // Free-form text → use as caption directly
            // (LLM translator would go here later)
            ParsedCmd::Enqueue(SongRequest {
                caption: line.to_string(),
                ..Default::default()
            })
        }
    }
}

/// JSON-friendly version of SongRequest for the `request` command.
#[derive(serde::Deserialize)]
struct SongRequestJson {
    #[serde(default)]
    caption: String,
    #[serde(default)]
    lyrics: String,
    #[serde(default)]
    bpm: Option<u32>,
    #[serde(default)]
    key_scale: Option<String>,
    #[serde(default)]
    time_signature: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration_s: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    shift: Option<f64>,
}

impl From<SongRequestJson> for SongRequest {
    fn from(j: SongRequestJson) -> Self {
        SongRequest {
            caption: j.caption,
            lyrics: j.lyrics,
            bpm: j.bpm,
            key_scale: j.key_scale,
            time_signature: j.time_signature.unwrap_or_else(|| "4/4".to_string()),
            language: j.language.unwrap_or_else(|| "en".to_string()),
            duration_s: j.duration_s,
            seed: j.seed,
            shift: j.shift.unwrap_or(3.0),
            auto_generated: false,
        }
    }
}

// ── Process a command ────────────────────────────────────────────────────

fn process_cmd(
    line: &str,
    cmd_tx: &mpsc::Sender<RadioCmd>,
    state: &SharedState,
    skip_flag: &AtomicBool,
) -> Option<String> {
    match parse_cmd(line) {
        ParsedCmd::Enqueue(req) => {
            let caption = req.caption.clone();
            cmd_tx.send(RadioCmd::Enqueue(req)).ok();
            Some(format!(
                "event:ok queued: {:?}",
                &caption[..caption.len().min(60)]
            ))
        }
        ParsedCmd::Skip => {
            skip_flag.store(true, Ordering::Relaxed);
            Some("event:ok skipping".to_string())
        }
        ParsedCmd::Queue => {
            let st = state.lock().unwrap();
            if st.queue_captions.is_empty() {
                Some("event:ok queue is empty".to_string())
            } else {
                let lines: Vec<String> = st
                    .queue_captions
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("  {}. {}", i + 1, c))
                    .collect();
                Some(format!("event:ok queue:\n{}", lines.join("\n")))
            }
        }
        ParsedCmd::History => {
            let st = state.lock().unwrap();
            if st.history_captions.is_empty() {
                Some("event:ok no history yet".to_string())
            } else {
                let lines: Vec<String> = st
                    .history_captions
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("  {}. {}", i + 1, c))
                    .collect();
                Some(format!("event:ok history:\n{}", lines.join("\n")))
            }
        }
        ParsedCmd::Clear => {
            cmd_tx.send(RadioCmd::ClearQueue).ok();
            Some("event:ok queue cleared".to_string())
        }
        ParsedCmd::State => {
            let st = state.lock().unwrap();
            Some(st.state_line())
        }
        ParsedCmd::Quit => None,
        ParsedCmd::Unknown(s) if s.is_empty() => Some(String::new()),
        ParsedCmd::Unknown(s) => Some(format!("event:error {s}")),
    }
}

// ── Socket handling ──────────────────────────────────────────────────────

fn handle_client(
    stream: UnixStream,
    cmd_tx: mpsc::Sender<RadioCmd>,
    state: SharedState,
    clients: ClientList,
    quit_flag: Arc<AtomicBool>,
    skip_flag: Arc<AtomicBool>,
) {
    let (event_tx, event_rx) = mpsc::channel::<String>();
    clients.lock().unwrap().push(event_tx.clone());

    // Send current state on connect
    {
        let st = state.lock().unwrap();
        event_tx.send(st.state_line()).ok();
    }

    let write_stream = stream.try_clone().expect("clone socket");
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

    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        tracing::info!("socket cmd: {:?}", line);
        match process_cmd(&line, &cmd_tx, &state, &skip_flag) {
            Some(ack) => {
                event_tx.send(ack).ok();
            }
            None => {
                // Quit
                broadcast(&clients, "event:ok shutting down");
                quit_flag.store(true, Ordering::Relaxed);
                cmd_tx.send(RadioCmd::Quit).ok();
                break;
            }
        }
    }

    drop(event_tx);
    write_handle.join().ok();
}

fn socket_accept_loop(
    listener: UnixListener,
    cmd_tx: mpsc::Sender<RadioCmd>,
    state: SharedState,
    clients: ClientList,
    quit_flag: Arc<AtomicBool>,
    skip_flag: Arc<AtomicBool>,
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
                let skip_flag = Arc::clone(&skip_flag);
                thread::spawn(move || {
                    handle_client(s, cmd_tx, state, clients, quit_flag, skip_flag)
                });
            }
            Err(e) => {
                tracing::warn!("socket accept error: {e}");
            }
        }
    }
}

// ── Generator thread ─────────────────────────────────────────────────────

/// A song ready for playback.
struct PlayableSong {
    song_index: usize,
    caption: String,
    duration_s: f64,
    samples: Vec<f32>,
}

fn generator_thread(
    rt: Arc<tokio::runtime::Runtime>,
    manager: GenerationManager,
    cmd_rx: mpsc::Receiver<RadioCmd>,
    event_tx: mpsc::Sender<RadioEvent>,
    station: Arc<Mutex<RadioStation>>,
    song_tx: mpsc::SyncSender<PlayableSong>,
    songs_playing: Arc<AtomicUsize>,
    quit_flag: Arc<AtomicBool>,
) {
    // Fallback captions when queue is empty — rotated to avoid repetition.
    let fallback_captions = [
        "aggressive rap, hard-hitting 808s, dark trap beat, raw vocals, heavy bass drops",
        "indie rock, jangly guitars, driving drums, raw vocals, garage energy, lo-fi warmth",
        "punk rock, distorted power chords, fast tempo, shouted vocals, rebellious energy",
        "hip hop boom bap, dusty vinyl samples, hard snare, deep sub bass, lyrical flow",
        "grunge rock, heavy distortion, angst-filled vocals, slow crushing riffs, raw emotion",
        "drum and bass, rolling breakbeats, massive sub bass, dark atmosphere, high energy",
        "metal, down-tuned guitars, double kick drums, aggressive vocals, crushing heaviness",
        "industrial electronic, harsh textures, pounding rhythm, distorted synths, relentless",
        "post-punk, angular guitar riffs, driving bass, cold atmosphere, urgent vocals",
        "garage rock, fuzzy guitars, stomping beat, raw energy, no-frills rock and roll",
        "techno, relentless kick drum, acid bassline, hypnotic loop, warehouse rave energy",
        "ska punk, offbeat guitar, horn section, fast tempo, bouncy energy, sing-along chorus",
        "psychedelic rock, wah guitar, swirling organ, heavy drums, trippy atmosphere",
        "hardcore punk, blast beats, screamed vocals, breakneck speed, chaotic energy",
        "trip hop, dark beats, haunting samples, heavy bass, nocturnal atmosphere",
        "surf rock, reverb-drenched guitar, driving drums, energetic twang, retro vibes",
    ];
    let mut fallback_idx = 0usize;
    let fallback = |idx: &mut usize| {
        let caption = fallback_captions[*idx % fallback_captions.len()].to_string();
        *idx += 1;
        SongRequest {
            caption,
            duration_s: Some(180.0),
            auto_generated: true,
            ..Default::default()
        }
    };

    /// Drain all pending commands. Returns `true` if Quit was received.
    fn drain_commands(cmd_rx: &mpsc::Receiver<RadioCmd>, station: &Mutex<RadioStation>) -> bool {
        loop {
            match cmd_rx.try_recv() {
                Ok(RadioCmd::Quit) => return true,
                Ok(RadioCmd::Enqueue(req)) => {
                    station.lock().unwrap().enqueue(req);
                }
                Ok(RadioCmd::ClearQueue) => {
                    station.lock().unwrap().clear_queue();
                }
                Ok(RadioCmd::Skip) => {
                    // Skip is handled by the playback side; we just keep generating.
                }
                Err(_) => return false,
            }
        }
    }

    loop {
        if quit_flag.load(Ordering::Relaxed) {
            return;
        }

        // Drain commands right before deciding what to generate.
        // This is the critical spot: after song_tx.send() blocks waiting
        // for playback to finish, we want to pick up any commands that
        // arrived during that wait (e.g., user enqueued songs while we
        // were blocked).
        if drain_commands(&cmd_rx, &station) {
            return;
        }

        // Get next request: pop from queue, or use fallback
        let request = {
            let mut st = station.lock().unwrap();
            st.pop_request().unwrap_or_else(|| {
                // TODO: LLM translator generates something based on history
                let f = fallback(&mut fallback_idx);
                tracing::info!(
                    "Queue empty, using fallback: {:?}",
                    &f.caption[..f.caption.len().min(60)]
                );
                f
            })
        };

        let is_auto_generated = request.auto_generated;
        let duration_s = RadioStation::resolve_duration(&request);
        let song_index = station.lock().unwrap().song_counter();
        let caption = request.caption.clone();

        event_tx
            .send(RadioEvent::Generating {
                song_index,
                caption: caption.clone(),
                duration_s,
            })
            .ok();

        let params = RadioStation::to_generation_params(&request);
        let seed = params.seed.unwrap_or(0);
        let t0 = std::time::Instant::now();

        // Use the GenerationManager (async) via the tokio runtime
        let gen_result = rt.block_on(manager.generate(params));

        match gen_result {
            Ok(audio) => {
                let gen_time = t0.elapsed().as_secs_f64();
                let samples = audio.samples;

                // Record in history (with the actual seed used)
                let mut request = request;
                request.seed = Some(seed);
                station.lock().unwrap().record_completed(request);

                event_tx
                    .send(RadioEvent::SongReady {
                        song_index,
                        caption: caption.clone(),
                        duration_s,
                        gen_time_s: gen_time,
                        seed,
                    })
                    .ok();

                let is_fallback = is_auto_generated;

                if song_tx
                    .send(PlayableSong {
                        song_index,
                        caption,
                        duration_s,
                        samples,
                    })
                    .is_err()
                {
                    return; // playback shut down
                }

                // Fallback songs: wait until this song actually starts
                // playing before generating the next one. This ensures at
                // most 1 fallback is buffered ahead, so user requests
                // don't get stuck behind a pile of pre-generated filler.
                if is_fallback {
                    let target = song_index + 1; // our song's playing count
                    loop {
                        if quit_flag.load(Ordering::Relaxed) {
                            return;
                        }
                        // Check for user commands while we wait.
                        match cmd_rx.recv_timeout(std::time::Duration::from_millis(200)) {
                            Ok(RadioCmd::Quit) => return,
                            Ok(RadioCmd::Enqueue(req)) => {
                                station.lock().unwrap().enqueue(req);
                                // Got a user request — stop waiting, go
                                // generate it immediately.
                                break;
                            }
                            Ok(RadioCmd::ClearQueue) => {
                                station.lock().unwrap().clear_queue();
                            }
                            Ok(RadioCmd::Skip) => {}
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                // Our fallback started playing — playback
                                // thread is now ready to receive the next
                                // song. Generate one more.
                                if songs_playing.load(Ordering::Relaxed) >= target {
                                    break;
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        }
                    }
                }
            }
            Err(e) => {
                event_tx
                    .send(RadioEvent::Error(format!("Generation failed: {e}")))
                    .ok();
                // Don't crash — continue to next song
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
}

// ── Double-buffer playback ────────────────────────────────────────────────
//
// Two pre-allocated buffers, each sized for the maximum song (600 s).
// cpal reads from the "front" buffer via an atomic cursor. The playback
// thread writes into the "back" buffer and flips the active index when
// cpal finishes draining the front. No mutex in the audio hot path.

/// 600 s × 48 kHz × 2 ch = 57,600,000 samples per buffer.
const MAX_SONG_SAMPLES: usize = 600 * 48000 * 2;

/// One of the two song buffers.
struct SongBuf {
    /// Pre-allocated sample storage. Only `len` samples are valid.
    data: Box<[f32]>,
    /// Number of valid samples in `data` (≤ MAX_SONG_SAMPLES).
    len: AtomicUsize,
    /// Read cursor, advanced by the cpal callback.
    cursor: AtomicUsize,
}

impl SongBuf {
    fn new() -> Self {
        Self {
            data: vec![0.0f32; MAX_SONG_SAMPLES].into_boxed_slice(),
            len: AtomicUsize::new(0),
            cursor: AtomicUsize::new(0),
        }
    }

    /// Load a song's samples into this buffer. Called by the playback thread
    /// while this buffer is the *back* buffer (cpal is not reading it).
    fn load(&self, samples: &[f32]) {
        let n = samples.len().min(MAX_SONG_SAMPLES);
        // Safety: we're the only writer and cpal is reading the other buffer.
        // Using a raw pointer to write into the boxed slice without &mut.
        let ptr = self.data.as_ptr() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(samples.as_ptr(), ptr, n);
        }
        self.len.store(n, Ordering::Release);
        self.cursor.store(0, Ordering::Release);
    }

    /// How many samples remain unplayed.
    fn remaining(&self) -> usize {
        self.len
            .load(Ordering::Relaxed)
            .saturating_sub(self.cursor.load(Ordering::Relaxed))
    }

    /// Mark this buffer as empty (skip / clear).
    fn clear(&self) {
        self.len.store(0, Ordering::Release);
        self.cursor.store(0, Ordering::Release);
    }

    /// cpal callback: copy samples into output, advance cursor.
    /// Returns number of samples written.
    fn fill(&self, data: &mut [f32]) -> usize {
        let pos = self.cursor.load(Ordering::Relaxed);
        let end = self.len.load(Ordering::Acquire);
        let available = end.saturating_sub(pos);
        let n = available.min(data.len());
        if n > 0 {
            data[..n].copy_from_slice(&self.data[pos..pos + n]);
            self.cursor.store(pos + n, Ordering::Relaxed);
        }
        n
    }
}

/// The double buffer shared between cpal and the playback thread.
struct DoubleBuffer {
    bufs: [SongBuf; 2],
    /// Which buffer cpal is currently reading from (0 or 1).
    active: AtomicU8,
}

impl DoubleBuffer {
    fn new() -> Self {
        Self {
            bufs: [SongBuf::new(), SongBuf::new()],
            active: AtomicU8::new(0),
        }
    }

    /// Get the buffer cpal should read from.
    fn front(&self) -> &SongBuf {
        &self.bufs[self.active.load(Ordering::Acquire) as usize]
    }

    /// Get the buffer the playback thread should write into.
    fn back(&self) -> &SongBuf {
        &self.bufs[1 - self.active.load(Ordering::Acquire) as usize]
    }

    /// Swap front and back. Called by the playback thread after loading
    /// a song into the back buffer and the front buffer is drained.
    fn swap(&self) {
        let old = self.active.load(Ordering::Acquire);
        self.active.store(1 - old, Ordering::Release);
    }
}

// Safety: The buffers are accessed from two threads (cpal callback + playback
// thread) but never the same buffer simultaneously — cpal reads front while
// playback writes back, and swap only happens when front is drained.
unsafe impl Sync for DoubleBuffer {}

/// The playback thread: receives songs from the generator, loads them into
/// the back buffer, waits for the front buffer to drain (gapless), then
/// swaps so the new song becomes the front that cpal reads.
///
/// Flow per song:
///   1. recv song from channel
///   2. load samples into back buffer (instant memcpy, cpal reads front)
///   3. wait for front to drain (gapless) or skip
///   4. swap — new song is now front, cpal starts reading it
///   5. emit Playing event
///   6. goto 1 — the NEXT iteration's step 3 is what blocks until
///      this song finishes playing (or gets skipped)
///
/// SongDone fires at step 3 of the NEXT iteration (front drained).
fn playback_thread(
    song_rx: mpsc::Receiver<PlayableSong>,
    dbuf: Arc<DoubleBuffer>,
    event_tx: mpsc::Sender<RadioEvent>,
    songs_playing: Arc<AtomicUsize>,
    skip_flag: Arc<AtomicBool>,
    quit_flag: Arc<AtomicBool>,
) {
    let mut prev_song_index: Option<usize> = None;

    for song in song_rx {
        if quit_flag.load(Ordering::Relaxed) {
            return;
        }

        // Load the new song into the back buffer while cpal drains the front.
        let n_samples = song.samples.len();
        dbuf.back().load(&song.samples);
        tracing::info!(
            "Loaded song #{} into back buffer ({} samples, {:.1}s)",
            song.song_index,
            n_samples,
            n_samples as f64 / (48000.0 * 2.0),
        );

        // Wait for the front buffer (previous song) to finish playing.
        // First song: front is empty (len=0), so this exits immediately.
        let wait_start = std::time::Instant::now();
        let mut logged_wait = false;
        loop {
            if quit_flag.load(Ordering::Relaxed) {
                return;
            }
            if skip_flag.load(Ordering::Relaxed) {
                if let Some(idx) = prev_song_index {
                    tracing::info!("Skipping song #{}", idx);
                }
                dbuf.front().clear();
                break;
            }
            let rem = dbuf.front().remaining();
            if rem == 0 {
                break;
            }
            if !logged_wait {
                tracing::info!(
                    "Waiting for front buffer to drain: {} samples remaining ({:.1}s)",
                    rem,
                    rem as f64 / (48000.0 * 2.0),
                );
                logged_wait = true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if logged_wait {
            tracing::info!(
                "Front buffer drained after {:.1}s",
                wait_start.elapsed().as_secs_f64(),
            );
        }

        // Previous song finished (or was skipped). Emit SongDone for it.
        if let Some(idx) = prev_song_index {
            event_tx.send(RadioEvent::SongDone { song_index: idx }).ok();
        }

        // Swap: back (new song) becomes front. cpal starts reading it.
        dbuf.swap();
        skip_flag.store(false, Ordering::Relaxed);
        prev_song_index = Some(song.song_index);

        // Signal the generator that this song started playing.
        songs_playing.fetch_add(1, Ordering::Relaxed);

        event_tx
            .send(RadioEvent::Playing {
                song_index: song.song_index,
                caption: song.caption.clone(),
                duration_s: song.duration_s,
            })
            .ok();

        // Don't wait here — go back to recv(). The NEXT iteration's
        // "wait for front to drain" is what blocks until this song
        // finishes. This means recv() is called immediately, pulling
        // the next song from the channel so the generator can start
        // working on the one after that. Exactly 1-song-ahead.
    }

    // Channel closed — wait for final song to finish
    if prev_song_index.is_some() {
        loop {
            if quit_flag.load(Ordering::Relaxed) || skip_flag.load(Ordering::Relaxed) {
                break;
            }
            if dbuf.front().remaining() == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if let Some(idx) = prev_song_index {
            event_tx.send(RadioEvent::SongDone { song_index: idx }).ok();
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let socket_path = args
        .windows(2)
        .find(|w| w[0] == "--socket")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "/tmp/ace-radio.sock".to_string());

    // File logging
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("radio_daemon.log")
        .expect("failed to open radio_daemon.log");
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
    let skip_flag = Arc::new(AtomicBool::new(false));
    let clients: ClientList = Arc::new(Mutex::new(Vec::new()));

    // Tokio runtime for GenerationManager
    let rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("failed to build tokio runtime"),
    );

    // Start GenerationManager
    let manager = rt
        .block_on(GenerationManager::start(ManagerConfig::default()))
        .expect("failed to start GenerationManager");

    eprintln!("Pipeline ready.");
    tracing::info!("Pipeline ready");

    // Radio station state
    let station = Arc::new(Mutex::new(RadioStation::new(10)));

    // Shared daemon state for socket queries
    let state: SharedState = Arc::new(Mutex::new(DaemonState {
        now_playing: None,
        queue_captions: Vec::new(),
        history_captions: Vec::new(),
        song_counter: 0,
        is_generating: false,
    }));

    // Double buffer: two pre-allocated 600s buffers. cpal reads the front,
    // playback thread loads the back. Swap when front is drained.
    // ~460 MB total (2 × 230 MB). No mutex in the audio hot path.
    let dbuf = Arc::new(DoubleBuffer::new());

    // Channels
    let (cmd_tx, cmd_rx) = mpsc::channel::<RadioCmd>();
    let (event_tx, event_rx) = mpsc::channel::<RadioEvent>();
    // SyncSender(1): generator can have 1 song buffered ahead.
    let (song_tx, song_rx) = mpsc::sync_channel::<PlayableSong>(1);
    // Counter incremented by playback thread each time a song starts playing.
    // Used by generator to avoid piling up fallback songs.
    let songs_playing = Arc::new(AtomicUsize::new(0));

    // cpal audio output
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no audio output device");
    let stream_config = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(48000),
        buffer_size: cpal::BufferSize::Default,
    };
    let dbuf_for_cpal = Arc::clone(&dbuf);
    let _audio_stream = device
        .build_output_stream(
            &stream_config,
            move |data: &mut [f32], _| {
                let n = dbuf_for_cpal.front().fill(data);
                if n < data.len() {
                    data[n..].fill(0.0);
                }
            },
            |e| tracing::error!("cpal error: {e}"),
            None,
        )
        .expect("failed to build audio stream");
    _audio_stream.play().expect("failed to start audio stream");

    // Playback thread: receives songs, loads into back buffer, swaps
    // when cpal finishes the front.
    {
        let dbuf = Arc::clone(&dbuf);
        let event_tx = event_tx.clone();
        let songs_playing = Arc::clone(&songs_playing);
        let skip_flag = Arc::clone(&skip_flag);
        let quit_flag = Arc::clone(&quit_flag);
        thread::spawn(move || {
            playback_thread(song_rx, dbuf, event_tx, songs_playing, skip_flag, quit_flag)
        });
    }

    // Generator thread
    {
        let station = Arc::clone(&station);
        let songs_playing = Arc::clone(&songs_playing);
        let quit = Arc::clone(&quit_flag);
        let event_tx = event_tx.clone();
        thread::spawn(move || {
            generator_thread(
                rt,
                manager,
                cmd_rx,
                event_tx,
                station,
                song_tx,
                songs_playing,
                quit,
            )
        });
    }

    // Socket accept thread
    {
        let cmd_tx = cmd_tx.clone();
        let state = Arc::clone(&state);
        let clients = Arc::clone(&clients);
        let quit = Arc::clone(&quit_flag);
        let skip = Arc::clone(&skip_flag);
        thread::spawn(move || socket_accept_loop(listener, cmd_tx, state, clients, quit, skip));
    }

    // Ctrl+C
    {
        let quit = Arc::clone(&quit_flag);
        let cmd_tx = cmd_tx.clone();
        let socket_path = socket_path.clone();
        ctrlc::set_handler(move || {
            eprintln!("\nShutting down...");
            quit.store(true, Ordering::Relaxed);
            cmd_tx.send(RadioCmd::Quit).ok();
            let _ = std::fs::remove_file(&socket_path);
            std::process::exit(0);
        })
        .ok();
    }

    broadcast(&clients, "event:ready");

    // Main event loop
    loop {
        if quit_flag.load(Ordering::Relaxed) {
            break;
        }

        match event_rx.recv_timeout(std::time::Duration::from_millis(200)) {
            Ok(RadioEvent::Ready) => {
                broadcast(&clients, "event:ready");
            }
            Ok(RadioEvent::Generating {
                song_index,
                caption,
                duration_s,
            }) => {
                {
                    let mut st = state.lock().unwrap();
                    st.is_generating = true;
                    // Update queue snapshot
                    let station = station.lock().unwrap();
                    st.queue_captions = station
                        .queue()
                        .iter()
                        .map(|r| r.caption[..r.caption.len().min(60)].to_string())
                        .collect();
                }
                let msg = format!(
                    "event:generating song={song_index} duration={duration_s:.0}s caption={:?}",
                    &caption[..caption.len().min(60)]
                );
                eprintln!("{msg}");
                tracing::info!("{msg}");
                broadcast(&clients, &msg);
            }
            Ok(RadioEvent::SongReady {
                song_index,
                caption,
                duration_s,
                gen_time_s,
                seed,
            }) => {
                state.lock().unwrap().is_generating = false;
                let msg = format!(
                    "event:song_ready song={song_index} seed={seed} duration={duration_s:.0}s gen={gen_time_s:.1}s caption={:?}",
                    &caption[..caption.len().min(60)]
                );
                eprintln!("{msg}");
                tracing::info!("{msg}");
                broadcast(&clients, &msg);
            }
            Ok(RadioEvent::Playing {
                song_index,
                caption,
                duration_s,
            }) => {
                {
                    let mut st = state.lock().unwrap();
                    st.now_playing = Some(NowPlaying {
                        song_index,
                        caption: caption.clone(),
                        duration_s,
                        samples_total: (duration_s * 48000.0 * 2.0) as usize,
                    });
                    st.song_counter = song_index + 1;
                    // Update history snapshot
                    let station = station.lock().unwrap();
                    st.history_captions = station
                        .history()
                        .iter()
                        .map(|r| {
                            let seed_str = r.seed.map(|s| format!(" seed={s}")).unwrap_or_default();
                            format!(
                                "{}{} ({:.0}s)",
                                &r.caption[..r.caption.len().min(60)],
                                seed_str,
                                r.duration_s.unwrap_or(180.0),
                            )
                        })
                        .collect();
                }
                let msg = format!(
                    "event:playing song={song_index} duration={duration_s:.0}s caption={:?}",
                    &caption[..caption.len().min(60)]
                );
                eprintln!("{msg}");
                tracing::info!("{msg}");
                broadcast(&clients, &msg);
            }
            Ok(RadioEvent::SongDone { song_index }) => {
                {
                    let mut st = state.lock().unwrap();
                    if st
                        .now_playing
                        .as_ref()
                        .is_some_and(|np| np.song_index == song_index)
                    {
                        st.now_playing = None;
                    }
                }
                let msg = format!("event:song_done song={song_index}");
                eprintln!("{msg}");
                tracing::info!("{msg}");
                broadcast(&clients, &msg);
            }
            Ok(RadioEvent::Skipped) => {
                let msg = "event:skip";
                eprintln!("{msg}");
                broadcast(&clients, msg);
            }
            Ok(RadioEvent::Error(e)) => {
                let msg = format!("event:error {e}");
                eprintln!("{msg}");
                tracing::error!("{e}");
                broadcast(&clients, &msg);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    drop(cmd_tx);
    let _ = std::fs::remove_file(&socket_path);
    eprintln!("Done.");
}
