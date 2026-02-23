//! ACE-Step v1.5 CLI — text-to-music generation.
//!
//! Generates audio from a text caption and optional lyrics.
//! Downloads model weights from HuggingFace on first run (~2 GB total).
//!
//! # Output
//!
//! Writes a single audio file to the path given by --output.
//! Also prints a one-line JSON summary to stdout on success:
//!
//! ```json
//! {"path":"/tmp/music.ogg","duration_s":30.0,"sample_rate":48000,"channels":2}
//! ```
//!
//! Exit code 0 on success, non-zero on error.

use ace_step_rs::{
    audio::write_audio,
    pipeline::{AceStepPipeline, GenerationParams},
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "ace-step",
    about = "ACE-Step v1.5 text-to-music generation",
    long_about = "Generate music from a text caption and optional lyrics.\n\
                  Downloads ~2 GB of model weights from HuggingFace on first run.\n\
                  Output is written to --output; a JSON summary line is printed to stdout."
)]
struct Args {
    /// Text description of the music style, genre, and mood.
    /// Include BPM, key, genre, instruments for best results.
    #[arg(long, short = 'c')]
    caption: String,

    /// Lyrics text. Use [verse], [chorus], [bridge] tags.
    /// Leave empty for instrumental.
    #[arg(long, short = 'l', default_value = "")]
    lyrics: String,

    /// Language of the lyrics ("en", "zh", etc.)
    #[arg(long, default_value = "en")]
    language: String,

    /// Metadata string: "bpm: 120, key: C major, genre: jazz"
    #[arg(long, default_value = "")]
    metas: String,

    /// Duration in seconds (1–600).
    #[arg(long, short = 'd', default_value_t = 30.0)]
    duration: f64,

    /// Output file path. Format determined by extension (.wav or .ogg).
    #[arg(long, short = 'o')]
    output: String,

    /// Random seed. Omit for a random seed each run.
    #[arg(long, short = 's')]
    seed: Option<u64>,

    /// Shift parameter for the turbo ODE schedule (1, 2, or 3).
    /// Higher values produce more variation at the cost of consistency.
    #[arg(long, default_value_t = 3.0)]
    shift: f64,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Validate duration
    if args.duration < 1.0 || args.duration > 600.0 {
        anyhow::bail!(
            "duration must be between 1 and 600 seconds, got {}",
            args.duration
        );
    }

    // Validate output path has a supported extension
    let output_path = std::path::Path::new(&args.output);
    let ext = output_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("wav");

    if ace_step_rs::audio::AudioFormat::parse(ext).is_none() {
        anyhow::bail!("unsupported output format '{}'. Use .wav or .ogg", ext);
    }

    // Ensure output directory exists
    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let device = candle_core::Device::cuda_if_available(0)?;
    let dtype = candle_core::DType::F32;

    tracing::info!("Using device: {:?}", device);
    tracing::info!("Loading ACE-Step v1.5 pipeline...");

    let mut pipeline = AceStepPipeline::load(&device, dtype)
        .map_err(|e| anyhow::anyhow!("failed to load pipeline: {e}"))?;

    let params = GenerationParams {
        caption: args.caption,
        metas: args.metas,
        lyrics: args.lyrics,
        language: args.language,
        duration_s: args.duration,
        shift: args.shift,
        seed: args.seed,
        src_latents: None,
        chunk_masks: None,
        refer_audio: None,
        refer_order: None,
    };

    tracing::info!("Generating {:.1}s of audio...", params.duration_s);

    let audio = pipeline
        .generate(&params)
        .map_err(|e| anyhow::anyhow!("generation failed: {e}"))?;

    write_audio(
        &args.output,
        &audio.samples,
        audio.sample_rate,
        audio.channels,
    )
    .map_err(|e| anyhow::anyhow!("failed to write audio: {e}"))?;

    // Print machine-readable summary to stdout for the caller
    println!(
        r#"{{"path":"{path}","duration_s":{duration},"sample_rate":{sr},"channels":{ch}}}"#,
        path = args.output,
        duration = args.duration,
        sr = audio.sample_rate,
        ch = audio.channels,
    );

    Ok(())
}
