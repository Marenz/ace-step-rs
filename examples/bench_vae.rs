//! Benchmark VAE decode at different latent lengths.
//!
//! Usage: cargo run --release --example bench_vae

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

fn main() -> ace_step_rs::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::F32;
    println!("Device: {:?}", device);

    // Load VAE
    let api = hf_hub::api::sync::Api::new().unwrap();
    let repo = api.model("ACE-Step/Ace-Step1.5".to_string());
    let vae_path = repo.get("vae/diffusion_pytorch_model.safetensors").unwrap();

    let vae_cfg = ace_step_rs::config::VaeConfig::default();
    let vae_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&vae_path], dtype, &device)? };
    let vae = ace_step_rs::vae::OobleckDecoder::new(&vae_cfg, vae_vb.pp("decoder"))?;
    println!("VAE loaded\n");

    // Warmup
    let warmup = Tensor::randn(0f32, 1.0, (1, 64, 10), &device)?;
    let _ = vae.decode(&warmup)?;

    // Test Snake1d vs Conv speed at large tensor sizes
    // The VAE decoder has 5 blocks, each upsampling by strides [10, 6, 4, 4, 2]
    // After each block the temporal dim grows. The last blocks operate on huge tensors.
    // Block 0: 750 -> 7500 (stride 10), channels 2048->1024
    // Block 1: 7500 -> 45000 (stride 6), channels 1024->512
    // Block 2: 45000 -> 180000 (stride 4), channels 512->256
    // Block 3: 180000 -> 720000 (stride 4), channels 256->128
    // Block 4: 720000 -> 1440000 (stride 2), channels 128->128

    println!("\n--- Snake1d benchmark (the activation runs many times per block) ---");
    for (ch, len) in [
        (2048, 750),
        (1024, 7500),
        (512, 45000),
        (256, 180000),
        (128, 720000),
    ] {
        let x = Tensor::randn(0f32, 1.0, (1, ch, len), &device)?;
        let alpha = Tensor::randn(0f32, 0.1, (1, ch, 1), &device)?;
        let beta = Tensor::randn(0f32, 0.1, (1, ch, 1), &device)?;

        // Snake: x + (1/exp(beta)) * sin(exp(alpha) * x)^2
        let t0 = Instant::now();
        for _ in 0..10 {
            let alpha_exp = alpha.exp()?;
            let beta_exp = beta.exp()?;
            let sin_term = alpha_exp.broadcast_mul(&x)?.sin()?;
            let sin_sq = (&sin_term * &sin_term)?;
            let recip_beta = beta_exp.recip()?;
            let _result = (&x + recip_beta.broadcast_mul(&sin_sq)?)?;
        }
        let elapsed = t0.elapsed().as_secs_f64() / 10.0;
        let mels = (ch * len) as f64 / 1e6;
        println!("  ch={ch:4} len={len:7} ({mels:.1}M elements): {elapsed:.4}s/call");
    }

    // Benchmark decode with warmup, multiple runs
    println!("\n--- Direct decode (3 runs each, after warmup) ---");
    for frames in [250, 500, 750, 1500] {
        let latents = Tensor::randn(0f32, 1.0, (1, 64, frames), &device)?;
        // warmup
        let _ = vae.decode(&latents);
        let mut times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let _w = vae.decode(&latents)?;
            times.push(t0.elapsed().as_secs_f64());
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        println!("  frames={frames:5}: avg={avg:.3}s min={min:.3}s  runs={times:?}");
    }

    Ok(())
}
