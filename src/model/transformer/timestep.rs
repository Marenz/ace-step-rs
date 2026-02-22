//! Sinusoidal timestep embedding for diffusion conditioning.
//!
//! Converts scalar timestep values to high-dimensional embeddings:
//! sinusoidal positional encoding → Linear → SiLU → Linear → SiLU → Linear(6*D)
//!
//! Returns (temb, timestep_proj) where timestep_proj has shape [B, 6, D]
//! for the 6 AdaLN modulation parameters per DiT layer.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

/// Timestep embedding module matching the Python `TimestepEmbedding`.
#[derive(Debug, Clone)]
pub struct TimestepEmbedding {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
    time_proj: nn::Linear,
    in_channels: usize,
    scale: f64,
}

impl TimestepEmbedding {
    /// Create a new timestep embedding.
    ///
    /// - `in_channels`: dimension of sinusoidal embedding (256 in ACE-Step)
    /// - `time_embed_dim`: output dimension (= hidden_size, 2048)
    pub fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = nn::linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        let time_proj = nn::linear(time_embed_dim, time_embed_dim * 6, vb.pp("time_proj"))?;
        Ok(Self {
            linear_1,
            linear_2,
            time_proj,
            in_channels,
            scale: 1000.0,
        })
    }

    /// Create sinusoidal timestep embeddings.
    ///
    /// `t`: [B] scalar timestep values → returns [B, in_channels] embedding
    fn timestep_embedding(&self, t: &Tensor, dev: &Device) -> Result<Tensor> {
        let t = (t * self.scale)?;
        let half = self.in_channels / 2;
        let max_period: f64 = 10000.0;

        // freqs = exp(-log(max_period) * arange(0, half) / half)
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(max_period.ln()) * i as f64 / half as f64).exp() as f32)
            .collect();
        let freqs = Tensor::new(freqs.as_slice(), dev)?;

        // args = t[:, None] * freqs[None, :]
        let t_f32 = t.to_dtype(DType::F32)?;
        let args = t_f32.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;

        let cos = args.cos()?;
        let sin = args.sin()?;
        Tensor::cat(&[&cos, &sin], 1)
    }

    /// Forward pass: t [B] → (temb [B, D], timestep_proj [B, 6, D]).
    pub fn forward(&self, t: &Tensor) -> Result<(Tensor, Tensor)> {
        let dev = t.device();
        let dtype = t.dtype();

        let t_freq = self.timestep_embedding(t, dev)?.to_dtype(dtype)?;
        let temb = t_freq
            .apply(&self.linear_1)?
            .silu()?
            .apply(&self.linear_2)?;
        let proj = temb.silu()?.apply(&self.time_proj)?;

        // Reshape proj from [B, 6*D] → [B, 6, D]
        let b = temb.dim(0)?;
        let d = temb.dim(1)?;
        let timestep_proj = proj.reshape((b, 6, d))?;

        Ok((temb, timestep_proj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_timestep_embedding_shape() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let te = TimestepEmbedding::new(256, 32, vb.pp("te")).unwrap();
        let t = Tensor::new(&[0.5f32, 0.8], &dev).unwrap();
        let (temb, proj) = te.forward(&t).unwrap();
        assert_eq!(temb.dims(), &[2, 32]);
        assert_eq!(proj.dims(), &[2, 6, 32]);
    }

    #[test]
    fn test_sinusoidal_varies_with_timestep() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let te = TimestepEmbedding::new(256, 32, vb.pp("te")).unwrap();

        let t1 = Tensor::new(&[0.1f32], &dev).unwrap();
        let t2 = Tensor::new(&[0.9f32], &dev).unwrap();
        let emb1 = te.timestep_embedding(&t1, &dev).unwrap();
        let emb2 = te.timestep_embedding(&t2, &dev).unwrap();

        // Different timesteps should produce different embeddings
        let diff = (emb1 - emb2).unwrap().abs().unwrap().sum_all().unwrap();
        let diff_val: f32 = diff.to_scalar().unwrap();
        assert!(diff_val > 0.1);
    }
}
