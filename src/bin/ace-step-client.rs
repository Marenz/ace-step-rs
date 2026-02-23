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

use std::{path::PathBuf, time::Duration};

use anyhow::{Context, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
    time::timeout,
};

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

    /// Socket path (default: /tmp/ace-step-gen.sock)
    #[arg(long, default_value = "/tmp/ace-step-gen.sock")]
    socket: PathBuf,

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
    Success(SuccessResponse),
    Error(ErrorResponse),
}

#[derive(Deserialize)]
struct SuccessResponse {
    ok: bool,
    path: Option<String>,
    duration_s: Option<f64>,
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

    let stream = timeout(Duration::from_secs(10), UnixStream::connect(&args.socket))
        .await
        .context("timed out connecting to daemon socket")?
        .with_context(|| format!("failed to connect to {}", args.socket.display()))?;

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
        Response::Success(r) if r.ok => {
            if let Some(path) = r.path {
                if let Some(duration) = r.duration_s {
                    eprintln!("generated {:.1}s of audio → {path}", duration);
                } else {
                    eprintln!("done → {path}");
                }
                println!("{path}");
            } else {
                eprintln!("ok");
            }
            Ok(())
        }
        Response::Success(r) => {
            bail!("daemon returned ok=false without error field (raw: {:?})", r.path);
        }
        Response::Error(r) => {
            bail!("generation failed: {}", r.error);
        }
    }
}
