//! ACE-Step generation daemon — Unix socket, line-delimited JSON.
//!
//! Keeps the pipeline resident across requests. Each client connection sends
//! one JSON request line and receives one JSON response line, then closes.
//!
//! # Socket path
//!
//! Default: `/tmp/ace-step-gen.sock`. Override with `--socket`.
//!
//! # Protocol
//!
//! **Request** (one JSON line):
//! ```json
//! {
//!   "caption": "upbeat jazz, 120 BPM",
//!   "lyrics":  "[verse]\nSome words",   // optional, "" = instrumental
//!   "metas":   "bpm: 120, key: C",      // optional
//!   "language": "en",                   // optional, default "en"
//!   "duration_s": 30.0,                 // optional, default 30
//!   "shift": 3.0,                       // optional, default 3
//!   "seed": 42,                         // optional, null = random
//!   "output": "/tmp/music.ogg"          // optional, auto-generated if omitted
//! }
//! ```
//!
//! **Response on success** (one JSON line):
//! ```json
//! {"ok": true, "path": "/tmp/music.ogg", "duration_s": 30.0, "sample_rate": 48000, "channels": 2}
//! ```
//!
//! **Response on error** (one JSON line):
//! ```json
//! {"ok": false, "error": "generation failed: ..."}
//! ```
//!
//! **Commands** (one JSON line):
//! ```json
//! {"command": "unload"}
//! ```
//! Drops the pipeline to free VRAM. The next generation request will reload it
//! automatically.
//!
//! # Example (shell)
//!
//! ```sh
//! echo '{"caption":"ambient piano","duration_s":20,"output":"/tmp/piano.ogg"}' \
//!   | socat - UNIX-CONNECT:/tmp/ace-step-gen.sock
//!
//! # Unload pipeline to free VRAM:
//! echo '{"command":"unload"}' | socat - UNIX-CONNECT:/tmp/ace-step-gen.sock
//! ```

use std::path::PathBuf;

use ace_step_rs::{

    manager::{GenerationManager, ManagerConfig},
    pipeline::GenerationParams,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{UnixListener, UnixStream},
};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "generation_daemon",
    about = "ACE-Step generation daemon — resident pipeline, Unix socket JSON interface"
)]
struct Args {
    /// Unix socket path to listen on.
    #[arg(long, default_value = "/home/marenz/.spacebot/sockets/ace-step-gen.sock")]
    socket: PathBuf,

    /// CUDA device ordinal (0 = first GPU).
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Automatically unload the pipeline from VRAM after this many seconds of
    /// inactivity. The next request will reload it automatically (~10-20s).
    /// Set to 0 to disable.
    #[arg(long, default_value_t = 0)]
    idle_unload_secs: u64,
}

// ── Wire types ───────────────────────────────────────────────────────────────

/// A message received over the socket — either a command or a generation request.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Message {
    Command { command: String },
    Generate(GenerateRequest),
}

/// A generation request received over the socket.
#[derive(Debug, Deserialize)]
struct GenerateRequest {
    caption: String,

    #[serde(default)]
    lyrics: String,

    #[serde(default)]
    metas: String,

    #[serde(default = "default_language")]
    language: String,

    #[serde(default = "default_duration")]
    duration_s: f64,

    #[serde(default = "default_shift")]
    shift: f64,

    /// Random seed. `null` or absent = random.
    #[serde(default)]
    seed: Option<u64>,

    /// Output file path. `null` or absent = auto-generated under `/tmp/`.
    #[serde(default)]
    output: Option<String>,
}

fn default_language() -> String {
    "en".into()
}
fn default_duration() -> f64 {
    30.0
}
fn default_shift() -> f64 {
    3.0
}

/// Response sent back to the client.
///
/// For generation responses, `data_length` indicates how many raw audio bytes
/// follow the JSON line. The client reads the JSON header, then reads exactly
/// `data_length` bytes of encoded audio data.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Response {
    Ok {
        ok: bool, // always true
        duration_s: f64,
        sample_rate: u32,
        channels: u16,
        format: String,
        data_length: usize,
    },
    Simple {
        ok: bool,
        message: String,
    },
    Err {
        ok: bool, // always false
        error: String,
    },
}

/// The result of processing a request: a response header and optional audio data.
struct ProcessResult {
    response: Response,
    audio_data: Option<Vec<u8>>,
}

impl Response {
    fn ok_with_data(
        duration_s: f64,
        sample_rate: u32,
        channels: u16,
        format: String,
        data_length: usize,
    ) -> Self {
        Self::Ok {
            ok: true,
            duration_s,
            sample_rate,
            channels,
            format,
            data_length,
        }
    }

    fn ok_simple(msg: impl Into<String>) -> Self {
        Self::Simple {
            ok: true,
            message: msg.into(),
        }
    }

    fn err(msg: impl Into<String>) -> Self {
        Self::Err {
            ok: false,
            error: msg.into(),
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Remove stale socket file if present.
    if args.socket.exists() {
        std::fs::remove_file(&args.socket)?;
    }

    let idle_unload_after = if args.idle_unload_secs > 0 {
        tracing::info!(secs = args.idle_unload_secs, "idle auto-unload enabled");
        Some(std::time::Duration::from_secs(args.idle_unload_secs))
    } else {
        None
    };

    tracing::info!("Loading ACE-Step pipeline (this may take a minute on first run)...");
    let config = ManagerConfig {
        cuda_device: args.device,
        idle_unload_after,
        ..ManagerConfig::default()
    };
    let manager = GenerationManager::start(config).await?;
    tracing::info!("Pipeline ready. Listening on {:?}", args.socket);

    let listener = UnixListener::bind(&args.socket)?;

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let manager = manager.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream, manager).await {
                        tracing::warn!("connection error: {e}");
                    }
                });
            }
            Err(e) => {
                tracing::error!("accept error: {e}");
            }
        }
    }
}

// ── Connection handler ────────────────────────────────────────────────────────

async fn handle_connection(stream: UnixStream, manager: GenerationManager) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    // Read exactly one line (the JSON request).
    let line = match lines.next_line().await? {
        Some(l) if !l.trim().is_empty() => l,
        _ => {
            send_response(&mut writer, Response::err("empty request"), None).await?;
            return Ok(());
        }
    };

    let result = process_request(&line, &manager).await;
    send_response(&mut writer, result.response, result.audio_data.as_deref()).await?;
    Ok(())
}

async fn process_request(line: &str, manager: &GenerationManager) -> ProcessResult {
    // Parse message.
    let message: Message = match serde_json::from_str(line) {
        Ok(m) => m,
        Err(e) => return ProcessResult { response: Response::err(format!("invalid JSON: {e}")), audio_data: None },
    };

    match message {
        Message::Command { ref command } => process_command(command, manager).await,
        Message::Generate(req) => process_generate(req, manager).await,
    }
}

async fn process_command(command: &str, manager: &GenerationManager) -> ProcessResult {
    let response = match command {
        "unload" => match manager.unload().await {
            Ok(()) => Response::ok_simple("pipeline unloaded"),
            Err(e) => Response::err(format!("unload failed: {e}")),
        },
        _ => Response::err(format!("unknown command: {command}")),
    };
    ProcessResult { response, audio_data: None }
}

async fn process_generate(req: GenerateRequest, manager: &GenerationManager) -> ProcessResult {
    use ace_step_rs::audio::{AudioFormat, encode_audio};

    let err = |msg: String| ProcessResult { response: Response::err(msg), audio_data: None };

    // Validate.
    if req.caption.trim().is_empty() {
        return err("'caption' field is required and must not be empty".into());
    }
    if req.duration_s < 1.0 || req.duration_s > 600.0 {
        return err(format!("duration_s must be between 1 and 600, got {}", req.duration_s));
    }

    // Determine output format from the output path extension, default to mp3.
    let format = req.output.as_ref()
        .and_then(|p| std::path::Path::new(p).extension())
        .and_then(|e| e.to_str())
        .and_then(AudioFormat::parse)
        .unwrap_or(AudioFormat::Mp3);

    let params = GenerationParams {
        caption: req.caption,
        metas: req.metas,
        lyrics: req.lyrics,
        language: req.language,
        duration_s: req.duration_s,
        shift: req.shift,
        seed: req.seed,
        src_latents: None,
        chunk_masks: None,
        refer_audio: None,
        refer_order: None,
    };

    tracing::info!(
        caption = %params.caption,
        duration_s = params.duration_s,
        format = %format,
        "generating"
    );

    let audio = match manager.generate(params).await {
        Ok(a) => a,
        Err(ref e) if e.to_string().contains("manager has shut down") => {
            tracing::error!("generation manager has shut down — exiting for restart");
            std::process::exit(1);
        }
        Err(e) => return err(format!("generation failed: {e}")),
    };

    let encoded = match encode_audio(format, &audio.samples, audio.sample_rate, audio.channels) {
        Ok(data) => data,
        Err(e) => return err(format!("failed to encode audio: {e}")),
    };

    tracing::info!(
        format = %format,
        bytes = encoded.len(),
        "done"
    );

    let data_length = encoded.len();
    ProcessResult {
        response: Response::ok_with_data(
            req.duration_s,
            audio.sample_rate,
            audio.channels,
            format.extension().to_string(),
            data_length,
        ),
        audio_data: Some(encoded),
    }
}

async fn send_response(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    response: Response,
    audio_data: Option<&[u8]>,
) -> anyhow::Result<()> {
    let mut json = serde_json::to_string(&response)?;
    json.push('\n');
    writer.write_all(json.as_bytes()).await?;

    // Send raw audio bytes after the JSON header.
    if let Some(data) = audio_data {
        writer.write_all(data).await?;
    }

    Ok(())
}
