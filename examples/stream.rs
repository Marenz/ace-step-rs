//! Example: infinite streaming music generation with lyric/style injection.
//!
//! Generates 3 chunks of music, changing lyrics and style between chunks.
//! Outputs a single WAV file with all chunks concatenated.
//!
//! Usage: cargo run --release --example stream

use ace_step_rs::pipeline::AceStepPipeline;
use ace_step_rs::streaming::{ChunkRequest, StreamConfig, StreamingGenerator};

fn main() -> ace_step_rs::Result<()> {
    tracing_subscriber::fmt::init();

    let device = candle_core::Device::cuda_if_available(0)?;
    let dtype = candle_core::DType::F32;
    println!("Using device: {device:?}");

    println!("Loading ACE-Step v1.5 pipeline...");
    let mut pipeline = AceStepPipeline::load(&device, dtype)?;

    let config = StreamConfig {
        chunk_duration_s: 30.0,
        overlap_s: 8.0,
        crossfade_ms: 100,
        shift: 3.0,
        language: "en".to_string(),
        ..Default::default()
    };

    let mut streamer = StreamingGenerator::new(
        config,
        "smooth jazz piano, relaxed late-night vibe, warm bass",
        "bpm: 90, key: Bb major, genre: jazz",
        "[verse]\nMidnight keys on ivory white\nThe city sleeps but we ignite\n",
    );

    let mut all_samples: Vec<f32> = Vec::new();

    // Chunk 1: Initial jazz
    println!("\n--- Chunk 1: Jazz piano ---");
    let chunk1 = streamer.next_chunk(&mut pipeline, None)?;
    println!(
        "Chunk 1: {} samples ({:.1}s)",
        chunk1.audio.samples.len() / 2,
        chunk1.audio.samples.len() as f64 / (2.0 * 48000.0)
    );
    all_samples.extend_from_slice(&chunk1.audio.samples);

    // Chunk 2: Shift to funk, new lyrics
    println!("\n--- Chunk 2: Transition to funk ---");
    let chunk2 = streamer.next_chunk(
        &mut pipeline,
        Some(ChunkRequest {
            caption: Some(
                "funky groove, slap bass, wah guitar, tight drums, energetic".to_string(),
            ),
            bpm: Some(Some(110)),
            key_scale: Some(Some("E minor".to_string())),
            lyrics: Some(
                "[chorus]\nGet up get down the groove is in the sound\nFeel the rhythm moving round and round\n"
                    .to_string(),
            ),
            ..Default::default()
        }),
    )?;
    println!(
        "Chunk 2: {} samples ({:.1}s), overlap: {} frames",
        chunk2.audio.samples.len() / 2,
        chunk2.audio.samples.len() as f64 / (2.0 * 48000.0),
        chunk2.overlap_frames,
    );
    all_samples.extend_from_slice(&chunk2.audio.samples);

    // Chunk 3: Keep funk style, new verse
    println!("\n--- Chunk 3: Funk continues, new verse ---");
    let chunk3 = streamer.next_chunk(
        &mut pipeline,
        Some(ChunkRequest {
            lyrics: Some(
                "[verse]\nBass line walking down the street\nEvery step a funky beat\n".to_string(),
            ),
            ..Default::default()
        }),
    )?;
    println!(
        "Chunk 3: {} samples ({:.1}s), overlap: {} frames",
        chunk3.audio.samples.len() / 2,
        chunk3.audio.samples.len() as f64 / (2.0 * 48000.0),
        chunk3.overlap_frames,
    );
    all_samples.extend_from_slice(&chunk3.audio.samples);

    // Save combined output
    let total_duration = all_samples.len() as f64 / (2.0 * 48000.0);
    println!("\nTotal: {:.1}s across {} chunks", total_duration, 3);

    let output_path = "output-stream.wav";
    ace_step_rs::audio::write_wav(output_path, &all_samples, 48000, 2)?;
    println!("Saved to {output_path}");

    Ok(())
}
