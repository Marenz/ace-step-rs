//! Benchmark ConvTranspose1d at Oobleck decoder sizes.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};
use std::time::Instant;

fn main() -> candle_core::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::F32;
    let vb = VarBuilder::zeros(dtype, &device);

    // (in_ch, out_ch, kernel, stride, input_len, description)
    let configs: Vec<(usize, usize, usize, usize, usize, &str)> = vec![
        (2048, 1024, 20, 10, 750, "Block 0: 750->7500"),
        (1024, 512, 12, 6, 7500, "Block 1: 7500->45000"),
        (512, 256, 8, 4, 45000, "Block 2: 45000->180000"),
        (256, 128, 8, 4, 180000, "Block 3: 180k->720k"),
        (128, 128, 4, 2, 720000, "Block 4: 720k->1440k"),
    ];

    for (in_ch, out_ch, k, s, length, desc) in &configs {
        let padding = k / 2 - s / 2;
        let cfg = ConvTranspose1dConfig {
            stride: *s,
            padding,
            ..Default::default()
        };
        let conv = ConvTranspose1d::new(
            vb.get((*in_ch, *out_ch, *k), "weight")?,
            Some(vb.get(*out_ch, "bias")?),
            cfg,
        );
        let x = Tensor::randn(0f32, 1.0, (1, *in_ch, *length), &device)?;

        // warmup
        let _ = conv.forward(&x)?;
        // Force sync by reading a value
        let _ = x.sum_all()?.to_scalar::<f32>()?;

        let t0 = Instant::now();
        let y = conv.forward(&x)?;
        // Force CUDA sync
        let _ = y.sum_all()?.to_scalar::<f32>()?;
        let elapsed = t0.elapsed().as_secs_f64();
        println!("{desc}: {elapsed:.4}s  out={:?}", y.dims());
    }

    Ok(())
}
