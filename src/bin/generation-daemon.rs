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
//! # Example (shell)
//!
//! ```sh
//! echo '{"caption":"ambient piano","duration_s":20,"output":"/tmp/piano.ogg"}' \
//!   | socat - UNIX-CONNECT:/tmp/ace-step-gen.sock
//! ```

use std::path::PathBuf;

use ace_step_rs::{
    audio::write_audio,
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
    #[arg(long, default_value = "/tmp/ace-step-gen.sock")]
    socket: PathBuf,

    /// CUDA device ordinal (0 = first GPU).
    #[arg(long, default_value_t = 0)]
    device: usize,
}

// ── Wire types ───────────────────────────────────────────────────────────────

/// A generation request received over the socket.
#[derive(Debug, Deserialize)]
struct Request {
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
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Response {
    Ok {
        ok: bool, // always true
        path: String,
        duration_s: f64,
        sample_rate: u32,
        channels: u16,
    },
    Err {
        ok: bool, // always false
        error: String,
    },
}

impl Response {
    fn ok(path: String, duration_s: f64, sample_rate: u32, channels: u16) -> Self {
        Self::Ok {
            ok: true,
            path,
            duration_s,
            sample_rate,
            channels,
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

    tracing::info!("Loading ACE-Step pipeline (this may take a minute on first run)...");
    let config = ManagerConfig {
        cuda_device: args.device,
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
            send_response(&mut writer, Response::err("empty request")).await?;
            return Ok(());
        }
    };

    let response = process_request(&line, &manager).await;
    send_response(&mut writer, response).await?;
    Ok(())
}

async fn process_request(line: &str, manager: &GenerationManager) -> Response {
    // Parse request.
    let req: Request = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => return Response::err(format!("invalid JSON request: {e}")),
    };

    // Validate.
    if req.caption.trim().is_empty() {
        return Response::err("'caption' field is required and must not be empty");
    }
    if req.duration_s < 1.0 || req.duration_s > 600.0 {
        return Response::err(format!(
            "duration_s must be between 1 and 600, got {}",
            req.duration_s
        ));
    }

    // Resolve output path.
    let output_path = req.output.unwrap_or_else(|| {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let spool = dirs::data_local_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
            .join("ace-step/spool");
        // Ensure spool dir exists; fall back to /tmp on error.
        if std::fs::create_dir_all(&spool).is_ok() {
            spool
                .join(format!("{ts}.ogg"))
                .to_string_lossy()
                .into_owned()
        } else {
            format!("/tmp/ace-step-{ts}.ogg")
        }
    });

    // Validate extension.
    let ext = std::path::Path::new(&output_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("wav");
    if ace_step_rs::audio::AudioFormat::from_str(ext).is_none() {
        return Response::err(format!(
            "unsupported output format '{ext}'. Use .wav or .ogg"
        ));
    }

    // Ensure output directory exists.
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return Response::err(format!("could not create output directory: {e}"));
            }
        }
    }

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
        output = %output_path,
        "generating"
    );

    let audio = match manager.generate(params).await {
        Ok(a) => a,
        Err(ref e) if e.to_string().contains("manager has shut down") => {
            // The manager background thread has died (e.g. double OOM failure).
            // Exit so systemd can restart the process and reload the pipeline.
            tracing::error!("generation manager has shut down — exiting for restart");
            std::process::exit(1);
        }
        Err(e) => return Response::err(format!("generation failed: {e}")),
    };

    if let Err(e) = write_audio(
        &output_path,
        &audio.samples,
        audio.sample_rate,
        audio.channels,
    ) {
        return Response::err(format!("failed to write audio file: {e}"));
    }

    tracing::info!(output = %output_path, "done");

    Response::ok(
        output_path,
        req.duration_s,
        audio.sample_rate,
        audio.channels,
    )
}

async fn send_response(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    response: Response,
) -> anyhow::Result<()> {
    let mut json = serde_json::to_string(&response)?;
    json.push('\n');
    writer.write_all(json.as_bytes()).await?;
    Ok(())
}
