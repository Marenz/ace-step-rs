//! Benchmark Conv1d and ConvTranspose1d at Oobleck VAE shapes.
//!
//! Isolates each convolution type to find where the non-cuDNN path is slow.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};
use std::time::Instant;

fn sync(t: &Tensor) {
    // Force CUDA sync by reading from the result tensor
    let _ = t.sum_all().unwrap().to_scalar::<f32>();
}

fn main() -> candle_core::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::F32;
    let vb = VarBuilder::zeros(dtype, &device);

    println!("Device: {:?}\n", device);

    // ---- ConvTranspose1d (upsampling) ----
    println!("=== ConvTranspose1d (5 blocks, 3 runs each) ===");
    let ct_configs: Vec<(usize, usize, usize, usize, usize, &str)> = vec![
        (2048, 1024, 16, 8, 750, "Block 0: stride=8"),
        (1024, 512, 16, 8, 6000, "Block 1: stride=8"),
        (512, 256, 8, 4, 48000, "Block 2: stride=4"),
        (256, 128, 8, 4, 192000, "Block 3: stride=4"),
        (128, 128, 4, 2, 768000, "Block 4: stride=2"),
    ];

    for (in_ch, out_ch, k, s, length, desc) in &ct_configs {
        let padding = s.div_ceil(2);
        let cfg = ConvTranspose1dConfig {
            stride: *s,
            padding,
            ..Default::default()
        };
        let conv = ConvTranspose1d::new(
            vb.get((*in_ch, *out_ch, 2 * s), "weight")?,
            Some(vb.get(*out_ch, "bias")?),
            cfg,
        );
        let x = Tensor::randn(0f32, 1.0, (1, *in_ch, *length), &device)?;

        // warmup
        let w = conv.forward(&x)?;
        sync(&w);
        drop(w);

        let mut times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let y = conv.forward(&x)?;
            sync(&y);
            times.push(t0.elapsed().as_secs_f64());
            drop(y);
        }
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        println!("  {desc} ({in_ch}->{out_ch}, len={length}): min={min:.4}s  runs={times:.4?}");
        // Drop conv and input to free GPU memory before next block
        drop(conv);
        drop(x);
    }

    // ---- Conv1d (residual blocks) ----
    println!("\n=== Conv1d (residual block shapes, 3 runs each) ===");
    // Each decoder block has 3 residual units, each with 2 Conv1d ops (dilated + 1x1)
    let c1_configs: Vec<(usize, usize, usize, usize, usize, &str)> = vec![
        // After Block 0 upsample: ch=1024, len~6000
        (1024, 1024, 7, 1, 6000, "ResUnit ch=1024 dil=1 k=7"),
        (1024, 1024, 7, 3, 6000, "ResUnit ch=1024 dil=3 k=7"),
        (1024, 1024, 7, 9, 6000, "ResUnit ch=1024 dil=9 k=7"),
        // After Block 2 upsample: ch=256, len~192000
        (256, 256, 7, 1, 192000, "ResUnit ch=256 dil=1 k=7"),
        (256, 256, 7, 3, 192000, "ResUnit ch=256 dil=3 k=7"),
        (256, 256, 7, 9, 192000, "ResUnit ch=256 dil=9 k=7"),
        // After Block 4 upsample: ch=128, len~1536000
        (128, 128, 7, 1, 1536000, "ResUnit ch=128 dil=1 k=7"),
        (128, 128, 7, 3, 1536000, "ResUnit ch=128 dil=3 k=7"),
        (128, 128, 7, 9, 1536000, "ResUnit ch=128 dil=9 k=7"),
        // 1x1 convs (in residual units)
        (1024, 1024, 1, 1, 6000, "1x1 ch=1024"),
        (256, 256, 1, 1, 192000, "1x1 ch=256"),
        (128, 128, 1, 1, 1536000, "1x1 ch=128"),
    ];

    for (in_ch, out_ch, k, dil, length, desc) in &c1_configs {
        let pad = ((k - 1) * dil) / 2;
        let cfg = Conv1dConfig {
            dilation: *dil,
            padding: pad,
            ..Default::default()
        };
        let conv = Conv1d::new(
            vb.get((*out_ch, *in_ch, *k), "weight")?,
            Some(vb.get(*out_ch, "bias")?),
            cfg,
        );
        let x = Tensor::randn(0f32, 1.0, (1, *in_ch, *length), &device)?;

        // warmup
        let w = conv.forward(&x)?;
        sync(&w);
        drop(w);

        let mut times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let y = conv.forward(&x)?;
            sync(&y);
            times.push(t0.elapsed().as_secs_f64());
            drop(y);
        }
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        println!("  {desc} (len={length}): min={min:.4}s  runs={times:.4?}");
        // Drop conv and input to free GPU memory before next block
        drop(conv);
        drop(x);
    }

    Ok(())
}
