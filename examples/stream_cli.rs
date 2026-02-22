//! Interactive streaming music CLI with live TUI.
//!
//! Logs go to `stream_cli.log` (append mode). Rate chunks with `-` `.` `+`.
//!
//! Usage:
//!   cargo run --release --features cli --example stream_cli
//!
//! Commands:
//!   caption <text>       — describe the music (natural language)
//!   lyrics <text>        — set lyrics (\n for newlines)
//!   lang <code>          — language (en, zh, etc.)
//!   bpm <n|none>         — set or unset BPM
//!   key <scale|none>     — set or unset key (e.g. "D minor")
//!   seed <number>        — seed for next chunk
//!   toggle overlap|timbre|crossfade
//!   set duration|overlap|crossfade|total <value>
//!   - . +                — rate current chunk (bad / neutral / good)
//!   q / Esc / Ctrl-C     — exit

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

use ace_step_rs::pipeline::{format_metas, AceStepPipeline};
use ace_step_rs::streaming::{ChunkRequest, StreamConfig, StreamingGenerator};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    style::Stylize,
    terminal::{self, ClearType},
};
use rodio::{OutputStream, Sink, Source};
// ── Audio source with sample counter ──────────────────────────────────────

struct ChannelSource {
    rx: mpsc::Receiver<Vec<f32>>,
    current: Vec<f32>,
    pos: usize,
    channels: u16,
    sample_rate: u32,
    /// Counts total interleaved samples consumed (shared with TUI).
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
                let sample = self.current[self.pos];
                self.pos += 1;
                self.consumed.fetch_add(1, Ordering::Relaxed);
                return Some(sample);
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

// ── Messages ──────────────────────────────────────────────────────────────

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

enum GenResult {
    Ready(StreamConfig),
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

// ── TUI state ─────────────────────────────────────────────────────────────

struct TuiState {
    current_caption: String,
    current_metas: String,
    current_lyrics: String,
    current_lang: String,

    bpm: Option<u32>,
    key_scale: Option<String>,

    pending: ChunkRequest,
    has_pending: bool,

    chunk_index: usize,
    last_rating: Option<String>,
    status_line: String,
    config: StreamConfig,

    /// Total interleaved samples sent to playback so far.
    total_generated_samples: usize,
    /// Shared counter: interleaved samples consumed by playback.
    playback_consumed: Arc<AtomicUsize>,

    input_buf: String,

    /// Whether a chunk is currently being generated.
    is_generating: bool,
}

impl TuiState {
    fn new(
        caption: &str,
        lyrics: &str,
        lang: &str,
        bpm: Option<u32>,
        key: Option<&str>,
        playback_consumed: Arc<AtomicUsize>,
    ) -> Self {
        let metas = build_metas(bpm, key, 30.0);
        Self {
            current_caption: caption.to_string(),
            current_metas: metas,
            current_lyrics: lyrics.to_string(),
            current_lang: lang.to_string(),
            bpm,
            key_scale: key.map(|s| s.to_string()),
            pending: ChunkRequest::default(),
            has_pending: false,
            chunk_index: 0,
            last_rating: None,
            status_line: "Loading pipeline...".to_string(),
            config: StreamConfig::default(),
            total_generated_samples: 0,
            playback_consumed,
            input_buf: String::new(),
            is_generating: false,
        }
    }

    fn rebuild_metas(&mut self) {
        // Use total_duration_s for metas when set — tells the model the full artistic duration.
        // When unset, fall back to chunk_duration_s.
        let metas_duration = if self.config.total_duration_s > 0.0 {
            self.config.total_duration_s
        } else {
            self.config.chunk_duration_s
        };
        self.current_metas = build_metas(self.bpm, self.key_scale.as_deref(), metas_duration);
        self.has_pending = true;
        self.pending.metas = Some(self.current_metas.clone());
    }

    fn take_pending(&mut self) -> Option<ChunkRequest> {
        if self.has_pending {
            self.has_pending = false;
            Some(std::mem::take(&mut self.pending))
        } else {
            None
        }
    }

    fn merge_into_pending(&mut self, field: &str, value: String) {
        self.has_pending = true;
        match field {
            "caption" => self.pending.caption = Some(value),
            "lyrics" => self.pending.lyrics = Some(value.replace("\\n", "\n")),
            "lang" => self.pending.language = Some(value),
            "seed" => {
                if let Ok(s) = value.parse::<u64>() {
                    self.pending.seed = Some(s);
                }
            }
            _ => {}
        }
    }

    /// Playback position in seconds.
    fn playback_secs(&self) -> f64 {
        let consumed = self.playback_consumed.load(Ordering::Relaxed);
        // interleaved stereo at 48kHz: 2 samples per frame
        consumed as f64 / (2.0 * 48000.0)
    }

    /// Total generated audio in seconds.
    fn generated_secs(&self) -> f64 {
        self.total_generated_samples as f64 / (2.0 * 48000.0)
    }

    /// Seconds of audio buffered ahead of playback.
    fn buffer_ahead_secs(&self) -> f64 {
        (self.generated_secs() - self.playback_secs()).max(0.0)
    }

    fn display_str(s: &str, max: usize) -> String {
        let flat: String = s.chars().map(|c| if c == '\n' { '↵' } else { c }).collect();
        let char_count = flat.chars().count();
        if char_count > max {
            let truncated: String = flat.chars().take(max - 1).collect();
            format!("{truncated}…")
        } else {
            flat
        }
    }

    fn pending_str(val: &Option<String>, max: usize) -> String {
        match val {
            Some(s) => Self::display_str(s, max),
            None => String::new(),
        }
    }
}

fn build_metas(bpm: Option<u32>, key: Option<&str>, duration_s: f64) -> String {
    format_metas(bpm, Some("4/4"), key, Some(duration_s))
}

// ── Drawing ───────────────────────────────────────────────────────────────

fn on_off(b: bool) -> &'static str {
    if b {
        "ON"
    } else {
        "OFF"
    }
}

/// Word-wrap a string to fit within `width` chars, returning lines.
fn wrap(s: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![s.to_string()];
    }
    let flat: String = s.chars().map(|c| if c == '\n' { '↵' } else { c }).collect();
    let chars: Vec<char> = flat.chars().collect();
    if chars.len() <= width {
        return vec![flat];
    }
    let mut lines = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + width).min(chars.len());
        lines.push(chars[start..end].iter().collect());
        start = end;
    }
    lines
}

fn draw(out: &mut io::Stderr, state: &TuiState) -> io::Result<()> {
    let config = &state.config;
    let (cols, _rows) = terminal::size().unwrap_or((100, 30));
    let w = cols as usize;

    execute!(out, cursor::MoveTo(0, 0), terminal::Clear(ClearType::All))?;

    writeln!(out, "{}\r", " ACE-Step Streaming CLI ".bold().reverse())?;

    // Generating indicator
    if state.is_generating {
        writeln!(out, "  {}\r", "● GENERATING...".bold().green())?;
    } else {
        writeln!(out, "  {}\r", "○ Ready".bold().dark_grey())?;
    }
    writeln!(out, "\r")?;

    // ── Progress bar ──────────────────────────────────────────────────
    let played = state.playback_secs();
    let generated = state.generated_secs();
    let ahead = state.buffer_ahead_secs();

    let bar_w = w.saturating_sub(4).min(80);
    let bar_label = format!(
        "  {:.0}s played | {:.0}s buffered ahead | {:.0}s total generated",
        played, ahead, generated,
    );
    writeln!(out, "{}\r", bar_label.dark_cyan())?;

    // Draw a bar: [=====>    ] where = is played, > is buffer ahead
    if generated > 0.0 && bar_w > 2 {
        let played_frac = (played / generated).clamp(0.0, 1.0);
        let filled = (played_frac * (bar_w - 2) as f64) as usize;
        let buffered_frac = (ahead / generated).clamp(0.0, 1.0);
        let buffered = (buffered_frac * (bar_w - 2) as f64) as usize;
        let empty = (bar_w - 2).saturating_sub(filled + buffered);

        write!(out, "  [")?;
        write!(out, "{}", "=".repeat(filled).green())?;
        write!(out, "{}", "+".repeat(buffered).dark_yellow())?;
        write!(out, "{}", " ".repeat(empty))?;
        writeln!(out, "]\r")?;
    }
    writeln!(out, "\r")?;

    // ── Status ────────────────────────────────────────────────────────
    let rating_str = match &state.last_rating {
        Some(r) => format!("  [{r}]"),
        None => String::new(),
    };
    writeln!(
        out,
        "{}\r",
        format!(
            "  Chunk: {}  |  {}{}",
            state.chunk_index, state.status_line, rating_str
        )
        .dark_cyan()
    )?;
    writeln!(out, "\r")?;

    // ── Config line ───────────────────────────────────────────────────
    let overlap_str = if config.use_overlap {
        format!("{}", on_off(true).green())
    } else {
        format!("{}", on_off(false).red())
    };
    let timbre_str = if config.use_timbre_from_prev {
        format!("{}", on_off(true).green())
    } else {
        format!("{}", on_off(false).red())
    };
    let crossfade_str = if config.use_crossfade {
        format!("{}", on_off(true).green())
    } else {
        format!("{}", on_off(false).red())
    };
    let total_str = if config.total_duration_s > 0.0 {
        format!("{:.0}s", config.total_duration_s)
    } else {
        "off".to_string()
    };
    writeln!(
        out,
        "  dur:{:.0}s  ovlp:{:.0}s  xfade:{}ms  shift:{}  total:{}  |  overlap:{}  timbre:{}  xfade:{}\r",
        config.chunk_duration_s,
        config.overlap_s,
        config.crossfade_ms,
        config.shift,
        total_str,
        overlap_str,
        timbre_str,
        crossfade_str,
    )?;

    let bpm_str = match state.bpm {
        Some(b) => format!("{b}"),
        None => "N/A".to_string(),
    };
    let key_str = match &state.key_scale {
        Some(k) => k.clone(),
        None => "N/A".to_string(),
    };
    writeln!(
        out,
        "  bpm: {}  key: {}  lang: {}\r",
        bpm_str, key_str, state.current_lang
    )?;
    writeln!(out, "\r")?;

    // ── Fields (multi-line, full width) ───────────────────────────────
    let label_w = 10;
    let val_w = w.saturating_sub(label_w + 4);

    // Caption
    let pend_caption = TuiState::pending_str(&state.pending.caption, val_w);
    write!(out, "  {:<label_w$}", "caption:".bold())?;
    let caption_lines = wrap(&state.current_caption, val_w);
    for (i, line) in caption_lines.iter().enumerate() {
        if i > 0 {
            write!(out, "  {:<label_w$}", "")?;
        }
        writeln!(out, "{}\r", line)?;
    }
    if !pend_caption.is_empty() {
        write!(out, "  {:<label_w$}", "")?;
        writeln!(out, "{}\r", format!("-> {pend_caption}").yellow())?;
    }

    // Lyrics
    let pend_lyrics = TuiState::pending_str(&state.pending.lyrics, val_w);
    write!(out, "  {:<label_w$}", "lyrics:".bold())?;
    let lyrics_lines = wrap(&state.current_lyrics, val_w);
    for (i, line) in lyrics_lines.iter().enumerate() {
        if i > 0 {
            write!(out, "  {:<label_w$}", "")?;
        }
        writeln!(out, "{}\r", line)?;
    }
    if !pend_lyrics.is_empty() {
        write!(out, "  {:<label_w$}", "")?;
        writeln!(out, "{}\r", format!("-> {pend_lyrics}").yellow())?;
    }

    writeln!(out, "\r")?;

    // ── Help ──────────────────────────────────────────────────────────
    writeln!(
        out,
        "  {}\r",
        "caption <desc>  lyrics <text>  bpm <n|none>  key <scale|none>  lang <code>  seed <n>"
            .dark_grey()
    )?;
    writeln!(
        out,
        "  {}\r",
        "toggle overlap|timbre|crossfade   set duration|overlap|crossfade|total <val>".dark_grey()
    )?;
    writeln!(
        out,
        "  {}\r",
        "Rate: - (bad) . (neutral) + (good)   q/Esc/Ctrl-C exit".dark_grey()
    )?;
    writeln!(out, "\r")?;

    write!(out, "  > {}", state.input_buf)?;
    out.flush()?;

    Ok(())
}

// ── Command parsing ───────────────────────────────────────────────────────

enum ParsedInput {
    FieldUpdate(String, String),
    Toggle(String),
    Set(String, String),
    Rate(String),
    NaturalLanguage(String), // Free text like "make it happier, faster"
    Quit,
    Unknown(String),
}

fn parse_input(line: &str) -> ParsedInput {
    let line = line.trim();
    if line.is_empty() {
        return ParsedInput::Unknown(String::new());
    }

    if line == "-" || line == "." || line == "+" {
        return ParsedInput::Rate(line.to_string());
    }

    // If it doesn't match known commands, treat as natural language
    let parts: Vec<&str> = line.splitn(3, ' ').collect();
    let cmd = parts[0].to_lowercase();

    match cmd.as_str() {
        "caption" | "lyrics" | "lang" | "seed" | "bpm" | "key" if parts.len() >= 2 => {
            let rest = line[cmd.len()..].trim().to_string();
            ParsedInput::FieldUpdate(cmd, rest)
        }
        "toggle" if parts.len() >= 2 => ParsedInput::Toggle(parts[1].to_lowercase()),
        "set" if parts.len() >= 3 => {
            ParsedInput::Set(parts[1].to_lowercase(), parts[2].to_string())
        }
        "quit" | "q" | "exit" => ParsedInput::Quit,
        // Unknown commands are treated as natural language
        _ => ParsedInput::NaturalLanguage(line.to_string()),
    }
}

// ── Generator thread ──────────────────────────────────────────────────────

fn generator_thread(
    cmd_rx: mpsc::Receiver<GenCmd>,
    result_tx: mpsc::Sender<GenResult>,
    audio_tx: mpsc::SyncSender<Vec<f32>>,
    initial_caption: String,
    initial_metas: String,
    initial_lyrics: String,
    initial_lang: String,
) {
    let device = match candle_core::Device::cuda_if_available(0) {
        Ok(d) => d,
        Err(e) => {
            result_tx
                .send(GenResult::Error(format!("Device: {e}")))
                .ok();
            return;
        }
    };
    let dtype = candle_core::DType::F32;

    let mut pipeline = match AceStepPipeline::load(&device, dtype) {
        Ok(p) => p,
        Err(e) => {
            result_tx
                .send(GenResult::Error(format!("Pipeline load: {e}")))
                .ok();
            return;
        }
    };

    let config = StreamConfig {
        language: initial_lang,
        ..Default::default()
    };

    result_tx.send(GenResult::Ready(config.clone())).ok();

    let mut streamer =
        StreamingGenerator::new(config, &initial_caption, &initial_metas, &initial_lyrics);

    loop {
        // Check quit flag first (for Ctrl+C)
        if QUIT_FLAG.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let mut pending_request: Option<Option<ChunkRequest>> = None;
        let mut should_quit = false;

        // Use try_recv instead of recv to avoid blocking
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

        // If no pending request and not currently generating, wait a bit to avoid spinning
        if pending_request.is_none() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        if let Some(req) = pending_request {
            let t0 = std::time::Instant::now();
            match streamer.next_chunk(&mut pipeline, req) {
                Ok(chunk) => {
                    let gen_time = t0.elapsed().as_secs_f64();
                    let audio_samples = chunk.audio.samples.len();

                    if audio_tx.send(chunk.audio.samples).is_err() {
                        return;
                    }

                    let msg = GenResult::Chunk {
                        chunk_index: chunk.chunk_index,
                        audio_samples,
                        gen_time_s: gen_time,
                        caption: streamer.caption().to_string(),
                        metas: streamer.metas().to_string(),
                        lyrics: streamer.lyrics().to_string(),
                        language: streamer.language().to_string(),
                        config: streamer.config().clone(),
                    };
                    if result_tx.send(msg).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    result_tx
                        .send(GenResult::Error(format!("Generation: {e}")))
                        .ok();
                    return;
                }
            }
        }
    }
}

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

// ── Main ──────────────────────────────────────────────────────────────────

fn restore_terminal() {
    execute!(io::stderr(), terminal::LeaveAlternateScreen, cursor::Show).ok();
    terminal::disable_raw_mode().ok();
}

static CMD_TX: std::sync::OnceLock<mpsc::Sender<GenCmd>> = std::sync::OnceLock::new();
static QUIT_FLAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn main() {
    // File-based logging (TUI owns stderr)
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("stream_cli.log")
        .expect("failed to open stream_cli.log");
    tracing_subscriber::fmt()
        .with_writer(log_file)
        .with_ansi(false)
        .with_target(false)
        .init();

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        default_hook(info);
    }));

    terminal::enable_raw_mode().expect("failed to enable raw mode");
    execute!(io::stderr(), terminal::EnterAlternateScreen, cursor::Hide).ok();

    // Setup Ctrl+C handler. CMD_TX is a static OnceLock — at signal time,
    // CMD_TX.get() will return the sender if it has been initialized by run().
    ctrlc::set_handler(move || {
        restore_terminal();
        if let Some(tx) = CMD_TX.get() {
            tx.send(GenCmd::Quit).ok();
        }
        QUIT_FLAG.store(true, std::sync::atomic::Ordering::Relaxed);
        std::process::exit(0);
    })
    .ok();

    let result = run();

    restore_terminal();

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> ace_step_rs::Result<()> {
    let initial_caption =
        "Driving synthwave with retro 80s analog synths, pulsing bass, soaring arpeggios, neon-lit atmospheric pads, energetic beat with heavy kick and hi-hats, upbeat futuristic dance music";
    let initial_lyrics = "la la la la la la la la\nla la la la la la la la\noh oh oh oh oh oh oh oh\noh oh oh oh oh oh oh oh";
    let initial_lang = "en";
    let initial_bpm: Option<u32> = Some(128);
    let initial_key: Option<&str> = Some("A minor");

    let initial_metas = build_metas(initial_bpm, initial_key, 30.0);

    let playback_counter = Arc::new(AtomicUsize::new(0));

    let mut tui = TuiState::new(
        initial_caption,
        initial_lyrics,
        initial_lang,
        initial_bpm,
        initial_key,
        Arc::clone(&playback_counter),
    );
    let mut stderr = io::stderr();
    draw(&mut stderr, &tui)?;

    // --- Audio output ---
    let (_stream, stream_handle) =
        OutputStream::try_default().expect("failed to open audio output");
    let sink = Sink::try_new(&stream_handle).expect("failed to create audio sink");

    let (audio_tx, audio_rx) = mpsc::sync_channel::<Vec<f32>>(3);
    let source = ChannelSource::new(audio_rx, 2, 48000, playback_counter);
    sink.append(source);

    // --- Channels ---
    let (cmd_tx, cmd_rx) = mpsc::channel::<GenCmd>();
    let (result_tx, result_rx) = mpsc::channel::<GenResult>();

    // Register cmd_tx for signal handler
    CMD_TX.set(cmd_tx.clone()).ok();

    let gen_handle = thread::spawn({
        let caption = initial_caption.to_string();
        let metas = initial_metas;
        let lyrics = initial_lyrics.to_string();
        let lang = initial_lang.to_string();
        move || generator_thread(cmd_rx, result_tx, audio_tx, caption, metas, lyrics, lang)
    });

    tui.status_line = "Loading pipeline...".to_string();
    draw(&mut stderr, &tui)?;

    match result_rx.recv() {
        Ok(GenResult::Ready(config)) => {
            tui.config = config;
            tui.status_line = "Generating first chunk...".to_string();
            draw(&mut stderr, &tui)?;
        }
        Ok(GenResult::Error(e)) => return Err(ace_step_rs::Error::Config(e)),
        _ => {
            return Err(ace_step_rs::Error::Config(
                "Unexpected message from generator".to_string(),
            ))
        }
    }

    let pending = tui.take_pending();
    tui.is_generating = true;
    cmd_tx.send(GenCmd::Generate(pending)).ok();

    // --- Main event loop ---
    loop {
        // Check generator results
        while let Ok(msg) = result_rx.try_recv() {
            match msg {
                GenResult::Chunk {
                    chunk_index,
                    audio_samples,
                    gen_time_s,
                    caption,
                    metas,
                    lyrics,
                    language,
                    config,
                } => {
                    tui.is_generating = false;
                    tui.chunk_index = chunk_index + 1;
                    tui.current_caption = caption;
                    tui.current_metas = metas;
                    tui.current_lyrics = lyrics;
                    tui.current_lang = language;
                    tui.config = config;
                    tui.total_generated_samples += audio_samples;

                    let audio_secs = audio_samples as f64 / (2.0 * 48000.0);
                    tui.status_line = format!(
                        "Chunk {} done: {:.1}s audio in {:.2}s",
                        chunk_index, audio_secs, gen_time_s,
                    );
                    draw(&mut stderr, &tui)?;

                    let pending = tui.take_pending();
                    tui.is_generating = true;
                    cmd_tx.send(GenCmd::Generate(pending)).ok();
                }
                GenResult::Error(e) => {
                    tui.status_line = format!("ERROR: {e}");
                    draw(&mut stderr, &tui)?;
                }
                GenResult::Ready(_) => {}
            }
        }

        // Check if we received Ctrl+C via signal handler
        if QUIT_FLAG.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }

        // Poll input
        if event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
            if let Ok(event) = event::read() {
                match event {
                    Event::Key(KeyEvent {
                        code, modifiers, ..
                    }) => {
                        match code {
                            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                                QUIT_FLAG.store(true, std::sync::atomic::Ordering::Relaxed);
                                cmd_tx.send(GenCmd::Quit).ok();
                                break;
                            }
                            KeyCode::Esc => {
                                QUIT_FLAG.store(true, std::sync::atomic::Ordering::Relaxed);
                                cmd_tx.send(GenCmd::Quit).ok();
                                break;
                            }
                            KeyCode::Char(c) => {
                        tui.input_buf.push(c);
                        draw(&mut stderr, &tui)?;
                    }
                    KeyCode::Backspace => {
                        tui.input_buf.pop();
                        draw(&mut stderr, &tui)?;
                    }
                    KeyCode::Enter => {
                        let line = std::mem::take(&mut tui.input_buf);
                        handle_input(&line, &mut tui, &cmd_tx);
                        draw(&mut stderr, &tui)?;
                        // Check if quit was requested
                        if line.trim() == "q" || line.trim() == "quit" || line.trim() == "exit" {
                            break;
                        }
                    }
                    _ => {}
                }
            }
        } else {
            // No input event — just refresh the progress bar
            draw(&mut stderr, &tui)?;
        }
    }

    drop(cmd_tx);
    gen_handle.join().ok();
    // Give sink a moment to finish current audio, but don't wait forever
    sink.sleep_until_end();

    restore_terminal();
    Ok(())
}

fn handle_input(line: &str, tui: &mut TuiState, cmd_tx: &mpsc::Sender<GenCmd>) {
    match parse_input(line) {
        ParsedInput::FieldUpdate(field, value) => match field.as_str() {
            "bpm" => {
                if value == "none" || value == "off" || value == "n/a" {
                    tui.bpm = None;
                    tui.rebuild_metas();
                    tui.status_line = "BPM: unset".to_string();
                } else if let Ok(b) = value.parse::<u32>() {
                    tui.bpm = Some(b);
                    tui.rebuild_metas();
                    tui.status_line = format!("BPM: {b}");
                } else {
                    tui.status_line = "Invalid BPM (number or 'none')".to_string();
                }
            }
            "key" => {
                if value == "none" || value == "off" || value == "n/a" {
                    tui.key_scale = None;
                    tui.rebuild_metas();
                    tui.status_line = "Key: unset".to_string();
                } else {
                    tui.key_scale = Some(value.clone());
                    tui.rebuild_metas();
                    tui.status_line = format!("Key: {value}");
                }
            }
            _ => tui.merge_into_pending(&field, value),
        },
        ParsedInput::Toggle(what) => match what.as_str() {
            "overlap" => {
                tui.config.use_overlap = !tui.config.use_overlap;
                cmd_tx
                    .send(GenCmd::Config(ConfigUpdate::ToggleOverlap))
                    .ok();
                tui.status_line = format!("Overlap: {}", on_off(tui.config.use_overlap));
            }
            "timbre" => {
                tui.config.use_timbre_from_prev = !tui.config.use_timbre_from_prev;
                cmd_tx.send(GenCmd::Config(ConfigUpdate::ToggleTimbre)).ok();
                tui.status_line = format!("Timbre: {}", on_off(tui.config.use_timbre_from_prev));
            }
            "crossfade" => {
                tui.config.use_crossfade = !tui.config.use_crossfade;
                cmd_tx
                    .send(GenCmd::Config(ConfigUpdate::ToggleCrossfade))
                    .ok();
                tui.status_line = format!("Crossfade: {}", on_off(tui.config.use_crossfade));
            }
            _ => tui.status_line = format!("Unknown toggle: {what}"),
        },
        ParsedInput::Set(param, value) => match param.as_str() {
            "duration" => {
                if let Ok(v) = value.parse::<f64>() {
                    if (5.0..=600.0).contains(&v) {
                        tui.config.chunk_duration_s = v;
                        cmd_tx
                            .send(GenCmd::Config(ConfigUpdate::SetDuration(v)))
                            .ok();
                        tui.rebuild_metas();
                        tui.status_line = format!("Duration: {v:.0}s");
                    } else {
                        tui.status_line = "Duration must be 5-600s".to_string();
                    }
                }
            }
            "overlap" => {
                if let Ok(v) = value.parse::<f64>() {
                    if v >= 0.0 && v < tui.config.chunk_duration_s {
                        tui.config.overlap_s = v;
                        cmd_tx
                            .send(GenCmd::Config(ConfigUpdate::SetOverlap(v)))
                            .ok();
                        tui.status_line = format!("Overlap: {v:.0}s");
                    } else {
                        tui.status_line = "Overlap must be < duration".to_string();
                    }
                }
            }
            "crossfade" => {
                if let Ok(v) = value.parse::<u32>() {
                    if v <= 5000 {
                        tui.config.crossfade_ms = v;
                        cmd_tx
                            .send(GenCmd::Config(ConfigUpdate::SetCrossfade(v)))
                            .ok();
                        tui.status_line = format!("Crossfade: {v}ms");
                    } else {
                        tui.status_line = "Crossfade must be <= 5000ms".to_string();
                    }
                }
            }
            "total" => {
                if let Ok(v) = value.parse::<f64>() {
                    tui.config.total_duration_s = v;
                    cmd_tx.send(GenCmd::Config(ConfigUpdate::SetTotal(v))).ok();
                    tui.rebuild_metas();
                    if v > 0.0 {
                        tui.status_line =
                            format!("Total: {v}s — metas updated, model plans full-track structure");
                    } else {
                        tui.status_line = "Total duration: disabled (metas reverted to chunk duration)".to_string();
                    }
                }
            }
            _ => tui.status_line = format!("Unknown setting: {param}"),
        },
        ParsedInput::Rate(r) => {
            let idx = tui.chunk_index.saturating_sub(1);
            cmd_tx.send(GenCmd::Rate(idx, r.clone())).ok();
            tui.last_rating = Some(format!("chunk {idx}: {r}"));
            tui.status_line = format!("Rated chunk {idx}: {r}");
        }
        ParsedInput::Quit => {
            cmd_tx.send(GenCmd::Quit).ok();
        }
        ParsedInput::NaturalLanguage(s) => {
            // Treat free-form text as a caption update
            tui.current_caption = s.clone();
            tui.has_pending = true;
            tui.pending.caption = Some(s.clone());
            tui.status_line = format!("Caption updated: {}", &s[..s.len().min(60)]);
        }
        ParsedInput::Unknown(s) if !s.is_empty() => {
            tui.status_line = format!("Unknown: {s}");
        }
        _ => {}
    }
}
