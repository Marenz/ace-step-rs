//! Example: generate a short audio clip with ACE-Step v1.5.
//!
//! Usage: cargo run --release --example generate [-- <duration_seconds> [output_format]]
//!
//! output_format: wav (default), ogg

use ace_step_rs::audio::{write_audio, AudioFormat};
use ace_step_rs::pipeline::{AceStepPipeline, GenerationParams};

fn main() -> ace_step_rs::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let duration_s: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30.0);

    let format = args
        .get(2)
        .and_then(|s| AudioFormat::from_str(s))
        .unwrap_or(AudioFormat::Wav);

    let output_path = format!("output.{}", format.extension());

    let device = candle_core::Device::cuda_if_available(0)?;
    let dtype = candle_core::DType::F32;
    println!("Using device: {:?}", device);

    println!("Loading ACE-Step v1.5 pipeline...");
    let mut pipeline = AceStepPipeline::load(&device, dtype)?;

    println!("Generating {duration_s}s of audio...");
    let params = GenerationParams {
        caption: "German-style philosophical rap with heavy bass, lo-fi beats, spoken word flow, quirky and intellectual, Käptn Peng inspired, celebratory energy, triumphant horns".to_string(),
        metas: "bpm: 95, key: D minor, genre: hip-hop rap, instruments: bass, drums, horns, synth, turntable".to_string(),
        lyrics: "[verse]\nWe wrote the code in Rust they said it couldn't be done\nTwo billion parameters underneath the sun\n[chorus]\nAce Step in Rust, we made the machine sing\nThree point four seconds, hear the future ring\n".to_string(),
        language: "en".to_string(),
        duration_s,
        shift: 3.0,
        seed: Some(1337),
        ..Default::default()
    };

    let audio = pipeline.generate(&params)?;

    println!(
        "Generated {} samples at {}Hz ({} channels)",
        audio.samples.len(),
        audio.sample_rate,
        audio.channels
    );

    // Save to audio file (format detected from extension)
    write_audio(
        &output_path,
        &audio.samples,
        audio.sample_rate,
        audio.channels,
    )?;
    println!("Saved to {output_path}");

    Ok(())
}
