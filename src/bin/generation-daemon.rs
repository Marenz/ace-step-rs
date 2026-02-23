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
use std::sync::{Arc, Mutex};

use ace_step_rs::{
    audio::write_audio,
    manager::{GenerationManager, ManagerConfig},
    model::lm_planner::LmPlanner,
    pipeline::GenerationParams,
};
use clap::Parser;
use hf_hub::api::sync::Api;
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

    /// Load the 5Hz LM planner and use it to expand captions before generation.
    ///
    /// Adds ~3.5GB VRAM usage. When enabled, the LM rewrites the caption and
    /// fills in BPM, key/scale, time signature, and duration from the text.
    #[arg(long, default_value_t = false)]
    use_lm: bool,
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

    /// Duration in seconds. When absent the LM planner may suggest one; otherwise defaults to 30.
    #[serde(default)]
    duration_s: Option<f64>,

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

    // Bind the socket immediately so callers can connect right away.
    // Connections that arrive before loading completes will wait in the channel.
    let listener = UnixListener::bind(&args.socket)?;
    tracing::info!("Listening on {:?} (loading pipeline...)", args.socket);

    // When the LM planner is resident it consumes ~3.5GB, so the pipeline
    // itself only needs ~6.3GB — leave 512MB headroom instead of the default 2GB.
    let min_free_vram_bytes = if args.use_lm {
        512 * 1024 * 1024
    } else {
        ManagerConfig::default().min_free_vram_bytes
    };
    let config = ManagerConfig {
        cuda_device: args.device,
        min_free_vram_bytes,
        ..ManagerConfig::default()
    };
    let manager = GenerationManager::start(config).await?;

    // Optionally load the LM planner (blocking, on the current thread).
    let lm_planner: Option<Arc<Mutex<LmPlanner>>> = if args.use_lm {
        tracing::info!("Loading 5Hz LM planner...");
        let device = ace_step_rs::manager::preferred_device(args.device);
        let planner = tokio::task::spawn_blocking(move || -> anyhow::Result<LmPlanner> {
            let api = Api::new()?;
            let repo = api.model("ACE-Step/Ace-Step1.5".to_string());
            let weights = repo.get("acestep-5Hz-lm-1.7B/model.safetensors")?;
            let tokenizer = repo.get("acestep-5Hz-lm-1.7B/tokenizer.json")?;
            let planner = LmPlanner::load(&weights, &tokenizer, &device, candle_core::DType::BF16)?;
            Ok(planner)
        })
        .await??;
        tracing::info!("LM planner ready");
        Some(Arc::new(Mutex::new(planner)))
    } else {
        None
    };

    tracing::info!("Pipeline ready");

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let manager = manager.clone();
                let lm = lm_planner.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream, manager, lm).await {
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

async fn handle_connection(
    stream: UnixStream,
    manager: GenerationManager,
    lm: Option<Arc<Mutex<LmPlanner>>>,
) -> anyhow::Result<()> {
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

    let response = process_request(&line, &manager, lm).await;
    send_response(&mut writer, response).await?;
    Ok(())
}

async fn process_request(
    line: &str,
    manager: &GenerationManager,
    lm: Option<Arc<Mutex<LmPlanner>>>,
) -> Response {
    // Parse request.
    let req: Request = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => return Response::err(format!("invalid JSON request: {e}")),
    };

    // Validate.
    if req.caption.trim().is_empty() {
        return Response::err("'caption' field is required and must not be empty");
    }
    if let Some(d) = req.duration_s {
        if d < 1.0 || d > 600.0 {
            return Response::err(format!(
                "duration_s must be between 1 and 600, got {d}"
            ));
        }
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
    if ace_step_rs::audio::AudioFormat::parse(ext).is_none() {
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

    // Optionally run the LM planner to expand the caption into structured metadata.
    // Resolve duration: user value takes priority; LM suggestion used only when omitted.
    const DEFAULT_DURATION: f64 = 30.0;
    let user_duration = req.duration_s; // None = user did not specify

    let (caption, metas, language, duration_s) =
        if let Some(lm_arc) = lm {
            let caption = req.caption.clone();
            let lyrics = req.lyrics.clone();
            let lm_fallback_duration = user_duration.unwrap_or(DEFAULT_DURATION);
            let result = tokio::task::spawn_blocking(move || {
                let mut planner = lm_arc.lock().unwrap();
                planner.plan(&caption, &lyrics, 512, 0.0)
            })
            .await;

            match result {
                Ok(Ok(plan)) => {
                    tracing::info!(
                        bpm = ?plan.bpm,
                        keyscale = ?plan.keyscale,
                        language = ?plan.language,
                        lm_duration_s = ?plan.duration_s,
                        "LM planner output"
                    );
                    let metas = plan.to_metas_string(lm_fallback_duration);
                    let caption = plan.caption.unwrap_or(req.caption);
                    let language = plan.language.unwrap_or(req.language);
                    // User-specified duration always wins; LM suggestion only if user omitted it.
                    let duration_s = user_duration
                        .or_else(|| plan.duration_s.map(|d| d as f64))
                        .unwrap_or(DEFAULT_DURATION);
                    (caption, metas, language, duration_s)
                }
                Ok(Err(e)) => {
                    tracing::warn!("LM planner failed, falling back to raw caption: {e}");
                    (req.caption, req.metas, req.language, user_duration.unwrap_or(DEFAULT_DURATION))
                }
                Err(e) => {
                    tracing::warn!("LM planner task panicked, falling back: {e}");
                    (req.caption, req.metas, req.language, user_duration.unwrap_or(DEFAULT_DURATION))
                }
            }
        } else {
            (req.caption, req.metas, req.language, user_duration.unwrap_or(DEFAULT_DURATION))
        };

    let params = GenerationParams {
        caption,
        metas,
        lyrics: req.lyrics,
        language,
        duration_s,
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

    Response::ok(output_path, duration_s, audio.sample_rate, audio.channels)
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
