//! Patch embedding for ACE-Step.
//!
//! Converts latent `[B, 8, 16, T]` to token sequence `[B, T, 2560]`.
//!
//! The patch size `[16, 1]` collapses the full height dimension (16)
//! into a single token per time frame. The result is a purely 1D
//! temporal sequence — no height dimension in the transformer.
//!
//! Architecture:
//! ```text
//! Conv2d(8, 2048, kernel=[16,1], stride=[16,1])  → [B, 2048, 1, T]
//! GroupNorm(32, 2048)
//! Conv2d(2048, 2560, kernel=1)                    → [B, 2560, 1, T]
//! flatten + transpose                              → [B, T, 2560]
//! ```
//!
//! Note: candle's `conv2d` uses square kernels, so we manually construct
//! the Conv2d with a non-square `[16, 1]` weight tensor.

use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;

use crate::Result;

/// Patch embedding layer.
pub struct PatchEmbed {
    /// Conv2d weight: [out_ch, in_ch, 16, 1]
    conv1_weight: Tensor,
    /// Conv2d bias: [out_ch]
    conv1_bias: Tensor,
    group_norm: candle_nn::GroupNorm,
    conv2: candle_nn::Conv2d,
}

impl PatchEmbed {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        inner_dim: usize,
        patch_size: [usize; 2],
    ) -> Result<Self> {
        // hidden = 8 * 16 * 1 * 16 = 2048
        let hidden = in_channels * patch_size[0] * patch_size[1] * patch_size[0];

        // Load conv1 weight/bias manually for non-square [16, 1] kernel.
        let vb_conv1 = vb.pp("early_conv_layers.0");
        let conv1_weight = vb_conv1.get(
            (hidden, in_channels, patch_size[0], patch_size[1]),
            "weight",
        )?;
        let conv1_bias = vb_conv1.get(hidden, "bias")?;

        let group_norm = candle_nn::group_norm(32, hidden, 1e-5, vb.pp("early_conv_layers.1"))?;

        let conv2_cfg = candle_nn::Conv2dConfig::default();
        let conv2 = candle_nn::conv2d(
            hidden,
            inner_dim,
            1,
            conv2_cfg,
            vb.pp("early_conv_layers.2"),
        )?;

        Ok(Self {
            conv1_weight,
            conv1_bias,
            group_norm,
            conv2,
        })
    }

    /// Forward pass: `[B, C, H, T]` → `[B, T, inner_dim]`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Manual Conv2d with [16, 1] kernel and stride [16, 1]:
        // Input [B, 8, 16, T] → Output [B, 2048, 1, T]
        let x = x.conv2d(
            &self.conv1_weight,
            /*padding=*/ 0,
            /*stride=*/ 16,
            /*dilation=*/ 1,
            /*groups=*/ 1,
        )?;
        // conv2d with stride=16 on height (16→1) and stride=16 on width.
        // But width kernel is 1, so with stride 16 we'd get T/16.
        // We need stride [16, 1] but candle only supports uniform stride.
        //
        // Workaround: reshape to use conv1d on the time dimension only.
        // Since kernel height = input height = 16, we can reshape:
        // [B, 8, 16, T] → [B, 8*16, T] → conv1d with kernel=1 → [B, 2048, T]
        // Actually, let's do the correct thing: reshape input.

        // Redo: reshape [B, 8, 16, T] to [B, 128, T] (flatten channels × height)
        // Then use conv1d with kernel=1 to project to 2048.
        // This is equivalent to Conv2d([16,1], stride=[16,1]) since kernel covers full height.
        // ... but we already loaded weights as [2048, 8, 16, 1].
        // Reshape weight to [2048, 128, 1] for conv1d.
        todo!("implement non-square conv2d patch embedding — see AGENTS.md pitfall")
    }
}

/// Alternative patch embedding using reshape + Conv1d.
///
/// Since patch_size=[16,1] covers the full height, we can flatten
/// height into channels and use Conv1d:
/// ```text
/// [B, 8, 16, T] → reshape → [B, 128, T] → Conv1d(128, 2048, k=1) → Conv1d(2048, 2560, k=1)
/// ```
pub struct PatchEmbed1d {
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    group_norm: candle_nn::GroupNorm,
    conv2: candle_nn::Conv1d,
}

impl PatchEmbed1d {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        inner_dim: usize,
        patch_size: [usize; 2],
    ) -> Result<Self> {
        let hidden = in_channels * patch_size[0] * patch_size[1] * patch_size[0];
        let flat_in = in_channels * patch_size[0]; // 8 * 16 = 128

        // Load original [2048, 8, 16, 1] weight, reshape to [2048, 128, 1]
        let vb_conv1 = vb.pp("early_conv_layers.0");
        let conv1_weight = vb_conv1.get(
            (hidden, in_channels, patch_size[0], patch_size[1]),
            "weight",
        )?;
        let conv1_weight = conv1_weight.reshape((hidden, flat_in, 1))?;
        let conv1_bias = vb_conv1.get(hidden, "bias")?;

        let group_norm = candle_nn::group_norm(32, hidden, 1e-5, vb.pp("early_conv_layers.1"))?;

        let conv2_cfg = candle_nn::Conv1dConfig::default();
        let conv2 = candle_nn::conv1d(
            hidden,
            inner_dim,
            1,
            conv2_cfg,
            vb.pp("early_conv_layers.2"),
        )?;

        Ok(Self {
            conv1_weight,
            conv1_bias,
            group_norm,
            conv2,
        })
    }

    /// Forward pass: `[B, C, H, T]` → `[B, T, inner_dim]`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _channels, _height, time) = x.dims4()?;

        // Flatten height into channels: [B, 8, 16, T] → [B, 128, T]
        let x = x.reshape((batch, _channels * _height, time))?;

        // Conv1d: [B, 128, T] → [B, 2048, T]
        let x = x.conv1d(
            &self.conv1_weight,
            /*padding=*/ 0,
            /*stride=*/ 1,
            /*dilation=*/ 1,
            /*groups=*/ 1,
        )?;
        let x = x.broadcast_add(&self.conv1_bias.reshape((1, (), 1))?)?;

        // GroupNorm expects [B, C, ...] — we have [B, 2048, T], which works.
        let x = self.group_norm.forward(&x)?;

        // Conv1d: [B, 2048, T] → [B, 2560, T]
        let x = self.conv2.forward(&x)?;

        // Transpose: [B, 2560, T] → [B, T, 2560]
        x.transpose(1, 2).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn patch_embed_1d_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let embed = PatchEmbed1d::load(vb, 8, 2560, [16, 1]).unwrap();

        // Input: [1, 8, 16, 644] (60s of audio)
        let x = Tensor::randn(0.0_f32, 1.0, (1, 8, 16, 644), &device).unwrap();
        let out = embed.forward(&x).unwrap();

        // Should be [1, 644, 2560] — one token per time frame
        assert_eq!(out.dims(), &[1, 644, 2560]);
    }

    #[test]
    fn patch_embed_1d_short_sequence() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let embed = PatchEmbed1d::load(vb, 8, 2560, [16, 1]).unwrap();

        // 10s of audio → ~108 frames
        let x = Tensor::randn(0.0_f32, 1.0, (1, 8, 16, 108), &device).unwrap();
        let out = embed.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 108, 2560]);
    }
}
