//! Benchmark: text encoder on CPU vs GPU.
//!
//! Loads Qwen3TextEncoder twice — once on CPU, once on CUDA — and times
//! `encode_text()` with a realistic caption input. Runs several warmup
//! iterations then reports mean and min over N timed iterations.
//!
//! Usage: cargo run --release --example bench_text_encoder

use ace_step_rs::model::encoder::text::{self, Qwen3TextEncoder};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 10;

/// A typical caption token sequence length. The actual tokenized caption
/// "German-style philosophical rap... " is about 60 tokens.
const SEQ_LEN: usize = 64;

fn load_encoder(device: &Device) -> anyhow::Result<Qwen3TextEncoder> {
    let api = Api::new()?;
    let repo = api.model("ACE-Step/Ace-Step1.5".to_string());
    let safetensors = repo.get("Qwen3-Embedding-0.6B/model.safetensors")?;

    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&safetensors], dtype, device)? };
    let vb = vb.rename_f(|name: &str| name.strip_prefix("model.").unwrap_or(name).to_string());

    let cfg = text::default_qwen3_config();
    Ok(Qwen3TextEncoder::new(&cfg, vb)?)
}

fn bench(label: &str, encoder: &mut Qwen3TextEncoder, device: &Device) -> anyhow::Result<()> {
    // Realistic token IDs — values don't matter for timing.
    let input = Tensor::zeros((1, SEQ_LEN), DType::U32, device)?;

    // Warmup
    for _ in 0..WARMUP {
        encoder.clear_kv_cache();
        let _ = encoder.encode_text(&input)?;
    }

    // Timed iterations
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        encoder.clear_kv_cache();
        let t = Instant::now();
        let _ = encoder.encode_text(&input)?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "{label:6}  mean={mean:7.1}ms  min={min:7.1}ms  max={max:7.1}ms  (seq_len={SEQ_LEN}, {ITERS} iters)"
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    println!("Loading text encoder on CPU...");
    let cpu = Device::Cpu;
    let mut cpu_encoder = load_encoder(&cpu)?;

    println!("Loading text encoder on CUDA...");
    let cuda = Device::cuda_if_available(0)?;
    let mut cuda_encoder = load_encoder(&cuda)?;

    println!();
    println!("device  timing (encode_text, seq_len={SEQ_LEN})");
    println!("{}", "-".repeat(60));

    bench("CPU", &mut cpu_encoder, &cpu)?;
    bench("CUDA", &mut cuda_encoder, &cuda)?;

    Ok(())
}
