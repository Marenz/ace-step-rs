//! GLUMBConv: Gated depthwise-separable 1D conv feed-forward.
//!
//! This replaces the standard MLP in the transformer blocks:
//! ```text
//! x [B, T, dim]
//!   → transpose → [B, dim, T]
//!   → 1×1 conv (dim → 2*hidden)
//!   → depthwise conv (groups=2*hidden, k=3)
//!   → chunk into (gate, x)
//!   → x * SiLU(gate)
//!   → 1×1 conv (hidden → dim)
//!   → transpose → [B, T, dim]
//! ```

use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;

use crate::Result;

/// GLU Mobilenet-Block Convolution.
pub struct GluMbConv {
    /// 1×1 pointwise expansion: dim → 2 * hidden_features
    inverted_conv: candle_nn::Conv1d,
    /// Depthwise conv: groups = 2 * hidden_features, kernel_size = 3
    depth_conv: candle_nn::Conv1d,
    /// 1×1 pointwise projection: hidden_features → dim
    point_conv: candle_nn::Conv1d,
}

impl GluMbConv {
    pub fn load(vb: VarBuilder, in_features: usize, hidden_features: usize) -> Result<Self> {
        let double_hidden = 2 * hidden_features;

        // inverted_conv: Conv1d(in_features, 2*hidden, kernel=1, bias=True)
        let inv_cfg = candle_nn::Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let inverted_conv = candle_nn::conv1d(
            in_features,
            double_hidden,
            1,
            inv_cfg,
            vb.pp("inverted_conv"),
        )?;

        // depth_conv: Conv1d(2*hidden, 2*hidden, kernel=3, padding=1, groups=2*hidden, bias=True)
        let depth_cfg = candle_nn::Conv1dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: double_hidden,
            ..Default::default()
        };
        let depth_conv = candle_nn::conv1d(
            double_hidden,
            double_hidden,
            3,
            depth_cfg,
            vb.pp("depth_conv"),
        )?;

        // point_conv: Conv1d(hidden, in_features, kernel=1, bias=False)
        // Note: the Python code uses use_bias=(True, True, False) — no bias on point_conv.
        let point_cfg = candle_nn::Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let point_conv = candle_nn::conv1d_no_bias(
            hidden_features,
            in_features,
            1,
            point_cfg,
            vb.pp("point_conv"),
        )?;

        Ok(Self {
            inverted_conv,
            depth_conv,
            point_conv,
        })
    }

    /// Forward pass.
    ///
    /// Input: `[B, T, dim]` → Output: `[B, T, dim]`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // [B, T, dim] → [B, dim, T] for Conv1d
        let x = x.transpose(1, 2)?;

        // 1×1 expansion: [B, dim, T] → [B, 2*hidden, T]
        let x = self.inverted_conv.forward(&x)?;

        // Depthwise conv: [B, 2*hidden, T] → [B, 2*hidden, T]
        let x = self.depth_conv.forward(&x)?;

        // GLU: chunk into gate and value, apply SiLU gating
        let chunks = x.chunk(2, 1)?;
        let gate = &chunks[0];
        let value = &chunks[1];
        let x = (value * candle_nn::Activation::Silu.forward(gate)?)?;

        // 1×1 projection: [B, hidden, T] → [B, dim, T]
        let x = self.point_conv.forward(&x)?;

        // [B, dim, T] → [B, T, dim]
        x.transpose(1, 2).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn glumbconv_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let dim = 256;
        let hidden = (dim as f64 * 2.5) as usize; // 640
        let conv = GluMbConv::load(vb, dim, hidden).unwrap();

        let x = Tensor::randn(0.0_f32, 1.0, (1, 64, dim), &device).unwrap();
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 64, dim]);
    }

    #[test]
    fn glumbconv_actual_config() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Deployed model: dim=2560, mlp_ratio=2.5, hidden=6400
        let conv = GluMbConv::load(vb, 2560, 6400).unwrap();

        let x = Tensor::randn(0.0_f32, 1.0, (1, 32, 2560), &device).unwrap();
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 32, 2560]);
    }
}
