//! Rotary position embedding (RoPE).
//!
//! Qwen2/LLaMA-style RoPE with `rope_theta = 1_000_000.0`.
//!
//! Applied separately to:
//! - Self-attention: query and key both use latent position frequencies
//! - Cross-attention: query uses latent freqs, key uses encoder freqs

use candle_core::{DType, Device, Tensor};

use crate::Result;

/// Pre-computed rotary embedding tables.
pub struct RotaryEmbedding {
    head_dim: usize,
    theta: f64,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, theta: f64) -> Self {
        Self { head_dim, theta }
    }

    /// Compute cos and sin tables for positions `0..seq_len`.
    ///
    /// Returns `(cos, sin)` each of shape `[seq_len, head_dim]`.
    pub fn compute_freqs(
        &self,
        seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let half_dim = self.head_dim / 2;

        // inv_freq = 1 / (theta ^ (2i / head_dim)) for i in 0..half_dim
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / self.theta.powf(2.0 * i as f64 / self.head_dim as f64))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?.to_dtype(DType::F64)?;

        // positions = [0, 1, 2, ..., seq_len-1]
        let positions: Vec<f64> = (0..seq_len).map(|i| i as f64).collect();
        let positions = Tensor::from_vec(positions, (seq_len, 1), device)?;

        // freqs = outer(positions, inv_freq) → [seq_len, half_dim]
        let freqs = positions.matmul(&inv_freq)?;

        // Duplicate: [freqs, freqs] → [seq_len, head_dim]
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok((cos, sin))
    }

    /// Apply rotary embedding to a tensor `x` of shape `[B, H, S, D]`.
    ///
    /// `cos` and `sin` have shape `[S, D]`.
    pub fn apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();

        // Cast to f32 for precision (matching Python impl).
        let x = x.to_dtype(DType::F32)?;
        let cos = cos.to_dtype(DType::F32)?;
        let sin = sin.to_dtype(DType::F32)?;

        // rotate_half: [-x[..., D/2:], x[..., :D/2]]
        let half = x.dim(candle_core::D::Minus1)? / 2;
        let x_first = x.narrow(candle_core::D::Minus1, 0, half)?;
        let x_second = x.narrow(candle_core::D::Minus1, half, half)?;
        let x_rotated = Tensor::cat(&[&x_second.neg()?, &x_first], candle_core::D::Minus1)?;

        // x * cos + rotate_half(x) * sin
        // cos/sin are [S, D], need broadcast to [1, 1, S, D]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let result = (x.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?)?;
        result.to_dtype(x_dtype).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn freqs_shape() {
        let rope = RotaryEmbedding::new(128, 1_000_000.0);
        let (cos, sin) = rope.compute_freqs(512, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[512, 128]);
        assert_eq!(sin.dims(), &[512, 128]);
    }

    #[test]
    fn cos_sin_bounded() {
        let rope = RotaryEmbedding::new(128, 1_000_000.0);
        let (cos, sin) = rope.compute_freqs(100, DType::F32, &Device::Cpu).unwrap();

        let cos_max: f32 = cos
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        let sin_max: f32 = sin
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();

        assert!(cos_max <= 1.0 + 1e-6, "cos max = {cos_max}");
        assert!(sin_max <= 1.0 + 1e-6, "sin max = {sin_max}");
    }

    #[test]
    fn apply_preserves_shape() {
        let rope = RotaryEmbedding::new(128, 1_000_000.0);
        let device = Device::Cpu;

        let x = Tensor::randn(0.0_f32, 1.0, (1, 20, 512, 128), &device).unwrap();
        let (cos, sin) = rope.compute_freqs(512, DType::F32, &device).unwrap();

        let result = RotaryEmbedding::apply(&x, &cos, &sin).unwrap();
        assert_eq!(result.dims(), x.dims());
    }

    #[test]
    fn apply_preserves_norm_approximately() {
        let rope = RotaryEmbedding::new(128, 1_000_000.0);
        let device = Device::Cpu;

        let x = Tensor::randn(0.0_f32, 1.0, (1, 4, 32, 128), &device).unwrap();
        let (cos, sin) = rope.compute_freqs(32, DType::F32, &device).unwrap();

        let result = RotaryEmbedding::apply(&x, &cos, &sin).unwrap();

        // RoPE is a rotation — should preserve L2 norm.
        let x_norm: f32 = x.sqr().unwrap().mean_all().unwrap().to_scalar().unwrap();
        let r_norm: f32 = result
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();

        assert!(
            (x_norm - r_norm).abs() < 0.01,
            "RoPE should preserve norm: {x_norm} vs {r_norm}"
        );
    }
}
