//! Simple command-line client for the ACE-Step generation daemon.
//!
//! Connects to the Unix socket, sends a JSON generation request, waits for the
//! response, and exits 0 on success or 1 on error.
//!
//! # Usage
//!
//! ```sh
//! ace-step-client \
//!   --caption "upbeat jazz, 120 BPM" \
//!   --output /tmp/music.mp3 \
//!   --duration 30
//!
//! # With lyrics:
//! ace-step-client \
//!   --caption "silly novelty pop, bouncy" \
//!   --lyrics "[verse]\nCat cat cat cat\n[chorus]\nMeow meow meow" \
//!   --output /tmp/cat.mp3 \
//!   --duration 30
//!
//! # Unload pipeline to free VRAM:
//! ace-step-client --unload
//! ```

use std::{path::PathBuf, process::Stdio, time::Duration};

use anyhow::{Context, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
    time::timeout,
};

const DEFAULT_SOCKET: &str = "/home/marenz/.spacebot/sockets/ace-step-gen.sock";
const SYSTEMD_SERVICE: &str = "ace-step-gen.service";

#[derive(Parser)]
#[command(name = "ace-step-client", about = "Send a generation request to the ACE-Step daemon")]
struct Args {
    /// Style description: genre, mood, tempo, instruments
    #[arg(long)]
    caption: Option<String>,

    /// Output file path (.mp3, .ogg, or .wav)
    #[arg(long)]
    output: Option<PathBuf>,

    /// Duration in seconds (default: 30)
    #[arg(long, default_value = "30.0")]
    duration: f64,

    /// Lyrics with [verse]/[chorus]/[bridge] tags; omit for instrumental
    #[arg(long)]
    lyrics: Option<String>,

    /// Metadata string, e.g. "bpm: 120, key: C major"
    #[arg(long)]
    metas: Option<String>,

    /// Lyrics language code (default: en)
    #[arg(long, default_value = "en")]
    language: String,

    /// ODE schedule shift 1–3 (default: 3.0)
    #[arg(long, default_value = "3.0")]
    shift: f64,

    /// Fixed seed for reproducibility (omit for random)
    #[arg(long)]
    seed: Option<u64>,

    /// Socket path
    #[arg(long, default_value = DEFAULT_SOCKET)]
    socket: PathBuf,

    /// Don't try to auto-start the daemon via systemd if the socket is missing
    #[arg(long)]
    no_autostart: bool,

    /// Timeout in seconds to wait for generation (default: 300)
    #[arg(long, default_value = "300")]
    timeout_secs: u64,

    /// Unload the pipeline from VRAM instead of generating
    #[arg(long)]
    unload: bool,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Request {
    Generate(GenerateRequest),
    Command(CommandRequest),
}

#[derive(Serialize)]
struct GenerateRequest {
    caption: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
    duration_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    lyrics: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metas: Option<String>,
    language: String,
    shift: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

#[derive(Serialize)]
struct CommandRequest {
    command: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Response {
    /// Generation success — audio data follows the JSON line.
    GenerateOk(GenerateOkResponse),
    /// Simple success (e.g. unload command).
    SimpleOk(SimpleOkResponse),
    /// Error.
    Error(ErrorResponse),
}

#[derive(Deserialize)]
struct GenerateOkResponse {
    ok: bool,
    duration_s: f64,
    #[allow(dead_code)]
    sample_rate: u32,
    #[allow(dead_code)]
    channels: u16,
    format: String,
    data_length: usize,
}

#[derive(Deserialize)]
struct SimpleOkResponse {
    ok: bool,
    message: String,
}

#[derive(Deserialize)]
struct ErrorResponse {
    #[allow(dead_code)]
    ok: bool,
    error: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let output_path = args.output.clone();

    let request = if args.unload {
        Request::Command(CommandRequest { command: "unload".into() })
    } else {
        let caption = args.caption.context("--caption is required for generation")?;
        Request::Generate(GenerateRequest {
            caption,
            output: args.output.map(|p| p.to_string_lossy().into_owned()),
            duration_s: args.duration,
            lyrics: args.lyrics,
            metas: args.metas,
            language: args.language,
            shift: args.shift,
            seed: args.seed,
        })
    };

    let request_line = serde_json::to_string(&request)? + "\n";

    let stream = connect_or_start(&args.socket, args.no_autostart).await?;

    let (reader, mut writer) = stream.into_split();

    writer
        .write_all(request_line.as_bytes())
        .await
        .context("failed to send request")?;
    writer.flush().await?;
    // Signal EOF so the daemon knows we're done writing.
    drop(writer);

    let mut reader = BufReader::new(reader);
    let mut response_line = String::new();

    timeout(Duration::from_secs(args.timeout_secs), reader.read_line(&mut response_line))
        .await
        .context("timed out waiting for daemon response")?
        .context("failed to read response")?;

    if response_line.is_empty() {
        bail!("daemon closed connection without sending a response");
    }

    let response: Response =
        serde_json::from_str(response_line.trim()).context("failed to parse daemon response")?;

    match response {
        Response::GenerateOk(r) if r.ok => {
            // Read the audio data that follows the JSON header.
            let mut audio_data = vec![0u8; r.data_length];
            timeout(
                Duration::from_secs(30),
                tokio::io::AsyncReadExt::read_exact(&mut reader, &mut audio_data),
            )
            .await
            .context("timed out reading audio data")?
            .context("failed to read audio data")?;

            // Determine output path — use --output if given, else auto-generate.
            let output_path = output_path.unwrap_or_else(|| {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                std::path::PathBuf::from(format!("/tmp/ace-step-{ts}.{}", r.format))
            });

            // Ensure parent directory exists.
            if let Some(parent) = output_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("failed to create dir {}", parent.display()))?;
                }
            }

            std::fs::write(&output_path, &audio_data)
                .with_context(|| format!("failed to write {}", output_path.display()))?;

            eprintln!("generated {:.1}s of audio → {}", r.duration_s, output_path.display());
            println!("{}", output_path.display());
            Ok(())
        }
        Response::GenerateOk(_) => {
            bail!("daemon returned ok=false in generate response");
        }
        Response::SimpleOk(r) if r.ok => {
            eprintln!("{}", r.message);
            Ok(())
        }
        Response::SimpleOk(_) => {
            bail!("daemon returned ok=false in simple response");
        }
        Response::Error(r) => {
            bail!("generation failed: {}", r.error);
        }
    }
}

/// Try to connect to the daemon socket. If the socket doesn't exist and
/// auto-start is allowed, start the systemd user service and poll until the
/// socket appears (up to ~60s for pipeline loading).
async fn connect_or_start(socket: &PathBuf, no_autostart: bool) -> anyhow::Result<UnixStream> {
    // First attempt — fast path when daemon is already running.
    match timeout(Duration::from_secs(5), UnixStream::connect(socket)).await {
        Ok(Ok(stream)) => return Ok(stream),
        Ok(Err(_)) | Err(_) => {}
    }

    if no_autostart {
        bail!(
            "daemon not reachable at {} (use --no-autostart=false or start it manually)",
            socket.display()
        );
    }

    // Auto-start requires XDG_RUNTIME_DIR to talk to the user's systemd.
    // Inside a sandbox this is typically unset, so we skip autostart gracefully.
    if std::env::var_os("XDG_RUNTIME_DIR").is_none() {
        bail!(
            "daemon not reachable at {} and auto-start unavailable \
             (no XDG_RUNTIME_DIR — likely running inside sandbox). \
             Start the daemon manually: systemctl --user start {SYSTEMD_SERVICE}",
            socket.display()
        );
    }

    eprintln!("daemon not running, starting {SYSTEMD_SERVICE}...");

    let status = std::process::Command::new("systemctl")
        .args(["--user", "start", SYSTEMD_SERVICE])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to run systemctl")?;

    if !status.success() {
        bail!(
            "systemctl --user start {SYSTEMD_SERVICE} failed (exit {})",
            status.code().unwrap_or(-1)
        );
    }

    // Poll for the socket to appear — the daemon needs time to load the
    // pipeline into VRAM (~10-20s on RTX 3090).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    let mut interval = tokio::time::interval(Duration::from_secs(2));
    interval.tick().await; // consume immediate first tick

    while tokio::time::Instant::now() < deadline {
        interval.tick().await;
        match timeout(Duration::from_secs(5), UnixStream::connect(socket)).await {
            Ok(Ok(stream)) => {
                eprintln!("daemon ready");
                return Ok(stream);
            }
            Ok(Err(_)) | Err(_) => continue,
        }
    }

    bail!(
        "daemon did not become reachable at {} within 60s after starting",
        socket.display()
    );
}
