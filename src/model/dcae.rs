//! DCAE decoder (Deep Compression AutoEncoder).
//!
//! Converts diffusion latents to mel spectrograms using a `diffusers::AutoencoderDC`
//! style decoder.
//!
//! ## Architecture
//!
//! ```text
//! [B, 8, 16, W] ─→ Conv2d(8, 1024, 3) + in_shortcut
//!   ─→ 3× EfficientViTBlock(1024)                         [B, 1024, 16, W]
//!   ─→ DCUpBlock2d(1024→512) + 3× ResBlock(512)           [B, 512, 32, 2W]
//!   ─→ DCUpBlock2d(512→256)  + 3× ResBlock(256)           [B, 256, 64, 4W]
//!   ─→ DCUpBlock2d(256→128)  + 3× ResBlock(128)           [B, 128, 128, 8W]
//!   ─→ RMSNorm + ReLU + Conv2d(128, 2, 3)                 [B, 2, 128, 8W]
//! ```
//!
//! ## Normalization
//!
//! ```text
//! Decode latent denorm: latents = latents / scale_factor + shift_factor
//! Mel denorm: mel = (x * 0.5 + 0.5) * (max_mel - min_mel) + min_mel
//! ```

use candle_core::{DType, Module, Tensor};
use candle_nn::VarBuilder;

use crate::Result;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// DCAE latent normalization constants.
pub const SHIFT_FACTOR: f64 = -1.9091;
pub const SCALE_FACTOR: f64 = 0.1786;

/// Mel spectrogram normalization constants.
pub const MIN_MEL_VALUE: f64 = -11.0;
pub const MAX_MEL_VALUE: f64 = 3.0;

/// Denormalize latents from model space to DCAE input space.
pub fn denormalize_latent(latent: &Tensor) -> Result<Tensor> {
    Ok(((latent / SCALE_FACTOR)? + SHIFT_FACTOR)?)
}

/// Denormalize mel spectrogram from DCAE output `[-1, 1]` to log-mel values.
pub fn denormalize_mel(mel: &Tensor) -> Result<Tensor> {
    let range = MAX_MEL_VALUE - MIN_MEL_VALUE;
    let scaled = ((mel * 0.5)? + 0.5)?;
    let stretched = (scaled * range)?;
    Ok((stretched + MIN_MEL_VALUE)?)
}

/// Calculate the number of latent time frames for a given duration.
pub fn calculate_frame_length(duration_secs: f64) -> usize {
    (duration_secs * 44100.0 / 512.0 / 8.0).ceil() as usize
}

// ---------------------------------------------------------------------------
// Channel-last RMSNorm (with weight + bias)
// ---------------------------------------------------------------------------

/// RMSNorm applied along the last dimension, with elementwise affine and bias.
///
/// Used in channel-last mode throughout the DCAE:
/// `x.movedim(1, -1)` → RMSNorm → `movedim(-1, 1)`
struct ChannelRmsNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl ChannelRmsNorm {
    fn load(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    /// Apply RMSNorm to the last dimension of x, then affine transform.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [..., C]
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rsqrt = (variance + self.eps)?.recip()?.sqrt()?;
        let normed = x_f32.broadcast_mul(&rsqrt)?;
        let normed = normed.to_dtype(x.dtype())?;
        // weight * x + bias
        let out = normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?;
        Ok(out)
    }

    /// Apply to a `[B, C, H, W]` tensor in channel-last mode.
    fn forward_channels_first(&self, x: &Tensor) -> Result<Tensor> {
        // [B, C, H, W] → [B, H, W, C]
        let x = x.permute([0, 2, 3, 1])?;
        let x = self.forward(&x)?;
        // [B, H, W, C] → [B, C, H, W]
        Ok(x.permute([0, 3, 1, 2])?)
    }
}

// ---------------------------------------------------------------------------
// ResBlock
// ---------------------------------------------------------------------------

/// Residual block: Conv2d → SiLU → Conv2d(no bias) → RMSNorm → +residual.
struct ResBlock {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    norm: ChannelRmsNorm,
}

impl ResBlock {
    fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let cfg3x3 = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = candle_nn::conv2d(channels, channels, 3, cfg3x3, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv2d_no_bias(channels, channels, 3, cfg3x3, vb.pp("conv2"))?;
        let norm = ChannelRmsNorm::load(vb.pp("norm"), channels, 1e-5)?;
        Ok(Self { conv1, conv2, norm })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.conv1.forward(x)?;
        let x = x.apply(&candle_nn::Activation::Silu)?;
        let x = self.conv2.forward(&x)?;
        let x = self.norm.forward_channels_first(&x)?;
        Ok((x + residual)?)
    }
}

// ---------------------------------------------------------------------------
// DCUpBlock2d (interpolate + conv + pixel_shuffle shortcut)
// ---------------------------------------------------------------------------

/// Upsample block: nearest-neighbor interpolation 2× + Conv2d, with
/// repeat_interleave + pixel_shuffle skip connection.
struct DcUpBlock2d {
    conv: candle_nn::Conv2d,
    repeats: usize,
}

impl DcUpBlock2d {
    fn load(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let cfg3x3 = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = candle_nn::conv2d(in_channels, out_channels, 3, cfg3x3, vb.pp("conv"))?;
        let repeats = out_channels * 4 / in_channels;
        Ok(Self { conv, repeats })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = x.dims4()?;

        // Main path: nearest-neighbor interpolate 2x then conv
        let upsampled = x.upsample_nearest2d(h * 2, w * 2)?;
        let main_path = self.conv.forward(&upsampled)?;

        // Shortcut: repeat_interleave on channels then pixel_shuffle(2)
        let shortcut = repeat_interleave_channels(x, self.repeats)?;
        let shortcut = pixel_shuffle(&shortcut, 2)?;

        Ok((main_path + shortcut)?)
    }
}

/// Repeat each channel `repeats` times: `[B, C, H, W]` → `[B, C*repeats, H, W]`.
fn repeat_interleave_channels(x: &Tensor, repeats: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    // [B, C, H, W] → [B, C, 1, H, W] → expand → [B, C, repeats, H, W] → [B, C*repeats, H, W]
    let x = x.unsqueeze(2)?; // [B, C, 1, H, W]
    let x = x.expand((b, c, repeats, h, w))?; // [B, C, repeats, H, W]
    Ok(x.reshape((b, c * repeats, h, w))?)
}

/// Pixel shuffle: `[B, C*r², H, W]` → `[B, C, H*r, W*r]`.
fn pixel_shuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let oc = c / (r * r);
    // Reshape to [B, oc, r, r, H, W]
    let x = x.reshape((b, oc, r, r, h, w))?;
    // Permute to [B, oc, H, r, W, r]
    let x = x.permute([0, 1, 4, 2, 5, 3])?;
    // Reshape to [B, oc, H*r, W*r]
    Ok(x.reshape((b, oc, h * r, w * r))?)
}

// ---------------------------------------------------------------------------
// SanaMultiscaleLinearAttention
// ---------------------------------------------------------------------------

/// Multi-scale linear attention used in EfficientViTBlock.
///
/// Uses ReLU-kernel linear attention with multi-scale QKV projections.
struct SanaMultiscaleLinearAttention {
    to_q: candle_nn::Linear,
    to_k: candle_nn::Linear,
    to_v: candle_nn::Linear,
    /// Multi-scale depthwise conv projection.
    ms_proj_in: candle_nn::Conv2d,
    ms_proj_out: candle_nn::Conv2d,
    to_out: candle_nn::Linear,
    norm_out: ChannelRmsNorm,
    num_heads: usize,
    head_dim: usize,
}

impl SanaMultiscaleLinearAttention {
    fn load(
        vb: VarBuilder,
        dim: usize,
        head_dim: usize,
        qkv_multiscale_kernel: usize,
    ) -> Result<Self> {
        let num_heads = dim / head_dim;
        let inner_dim = num_heads * head_dim; // = dim

        let to_q = candle_nn::linear_no_bias(dim, inner_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear_no_bias(dim, inner_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear_no_bias(dim, inner_dim, vb.pp("to_v"))?;

        // Multi-scale projection: depthwise conv + grouped pointwise
        let qkv_dim = 3 * inner_dim; // 3072 for dim=1024
        let pad = qkv_multiscale_kernel / 2;
        let ms_cfg_in = candle_nn::Conv2dConfig {
            padding: pad,
            groups: qkv_dim,
            ..Default::default()
        };
        let ms_proj_in = candle_nn::conv2d_no_bias(
            qkv_dim,
            qkv_dim,
            qkv_multiscale_kernel,
            ms_cfg_in,
            vb.pp("to_qkv_multiscale.0.proj_in"),
        )?;
        // Grouped pointwise: groups = 3 * head_dim
        let ms_cfg_out = candle_nn::Conv2dConfig {
            groups: 3 * num_heads,
            ..Default::default()
        };
        let ms_proj_out = candle_nn::conv2d_no_bias(
            qkv_dim,
            qkv_dim,
            1,
            ms_cfg_out,
            vb.pp("to_qkv_multiscale.0.proj_out"),
        )?;

        // Output: Linear(2 * inner_dim, inner_dim) — 2× because of multi-scale concat
        let to_out = candle_nn::linear_no_bias(2 * inner_dim, inner_dim, vb.pp("to_out"))?;
        let norm_out = ChannelRmsNorm::load(vb.pp("norm_out"), inner_dim, 1e-5)?;

        Ok(Self {
            to_q,
            to_k,
            to_v,
            ms_proj_in,
            ms_proj_out,
            to_out,
            norm_out,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let (b, _c, h, w) = x.dims4()?;
        let n = h * w;
        let nh = self.num_heads;
        let dk = self.head_dim;

        // To channels-last: [B, C, H, W] → [B, H, W, C]
        let x_cl = x.permute([0, 2, 3, 1])?.contiguous()?;
        let x_flat = x_cl.reshape((b, n, ()))?;

        // Q, K, V projections: [B, N, dim]
        let q = self.to_q.forward(&x_flat)?;
        let k = self.to_k.forward(&x_flat)?;
        let v = self.to_v.forward(&x_flat)?;

        // Concatenate for multi-scale: [B, N, 3*dim] → [B, 3*dim, H, W]
        let qkv = Tensor::cat(&[&q, &k, &v], 2)?;
        let qkv_2d = qkv
            .reshape((b, h, w, 3 * nh * dk))?
            .permute([0, 3, 1, 2])?
            .contiguous()?;

        // Multi-scale projection
        let ms_qkv = self.ms_proj_in.forward(&qkv_2d)?;
        let ms_qkv = self.ms_proj_out.forward(&ms_qkv)?;

        // Concatenate identity + multi-scale: [B, 6*dim, H, W]
        let combined = Tensor::cat(&[&qkv_2d, &ms_qkv], 1)?;

        // Reshape to [B, num_heads*(1+num_scales), 3*head_dim, N]
        let total_heads = nh * 2; // identity + 1 scale
        let combined = combined
            .reshape((b, total_heads, 3 * dk, h * w))?
            .contiguous()?;

        // Split into Q, K, V each [B, total_heads, head_dim, N]
        let q = combined.narrow(2, 0, dk)?.contiguous()?;
        let k = combined.narrow(2, dk, dk)?.contiguous()?;
        let v = combined.narrow(2, 2 * dk, dk)?.contiguous()?;

        // ReLU activation on Q and K
        let q = q.relu()?;
        let k = k.relu()?;

        // Linear attention: O(N·d²)
        // scores = V · K^T → [B, total_heads, dk, dk]
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = v.matmul(&k_t)?;

        // out = scores · Q → [B, total_heads, dk, N]
        let out = scores.matmul(&q)?;

        // Normalization: sum(K, dim=-1) → [B, total_heads, dk, 1]
        let k_sum = k.sum(candle_core::D::Minus1)?.unsqueeze(3)?;
        let normalizer = k_sum.transpose(2, 3)?.contiguous()?.matmul(&q)?; // [B, th, 1, N]
        let normalizer = (normalizer + 1e-6)?;
        let out = out.broadcast_div(&normalizer)?;

        // Reshape: [B, total_heads, dk, N] → [B, N, total_heads*dk]
        let out = out
            .permute([0, 3, 1, 2])? // [B, N, total_heads, dk]
            .reshape((b, n, total_heads * dk))?
            .contiguous()?;

        // Output projection + norm + residual
        let out = self.to_out.forward(&out)?;
        // Back to channels-first: [B, N, dim] → [B, dim, H, W]
        let out = out.reshape((b, h, w, nh * dk))?.permute([0, 3, 1, 2])?;
        let out = self.norm_out.forward_channels_first(&out)?;
        Ok((out + residual)?)
    }
}

// ---------------------------------------------------------------------------
// GLUMBConv (2D variant for DCAE)
// ---------------------------------------------------------------------------

/// Gated depthwise-separable 2D conv FFN used in EfficientViTBlock.
struct DcaeGluMbConv {
    conv_inverted: candle_nn::Conv2d,
    conv_depth: candle_nn::Conv2d,
    conv_point: candle_nn::Conv2d,
    norm: ChannelRmsNorm,
}

impl DcaeGluMbConv {
    fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let hidden = 4 * dim;
        let cfg1x1 = candle_nn::Conv2dConfig::default();
        let cfg3x3 = candle_nn::Conv2dConfig {
            padding: 1,
            groups: 2 * hidden, // depthwise
            ..Default::default()
        };

        let conv_inverted = candle_nn::conv2d(dim, 2 * hidden, 1, cfg1x1, vb.pp("conv_inverted"))?;
        let conv_depth = candle_nn::conv2d(2 * hidden, 2 * hidden, 3, cfg3x3, vb.pp("conv_depth"))?;
        let conv_point = candle_nn::conv2d_no_bias(hidden, dim, 1, cfg1x1, vb.pp("conv_point"))?;
        let norm = ChannelRmsNorm::load(vb.pp("norm"), dim, 1e-5)?;

        Ok(Self {
            conv_inverted,
            conv_depth,
            conv_point,
            norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.conv_inverted.forward(x)?;
        let x = x.apply(&candle_nn::Activation::Silu)?;
        let x = self.conv_depth.forward(&x)?;

        // Split + gated activation
        let chunks = x.chunk(2, 1)?;
        let x = &chunks[0];
        let gate = chunks[1].apply(&candle_nn::Activation::Silu)?;
        let x = (x * gate)?;

        let x = self.conv_point.forward(&x)?;
        let x = self.norm.forward_channels_first(&x)?;
        Ok((x + residual)?)
    }
}

// ---------------------------------------------------------------------------
// EfficientViTBlock
// ---------------------------------------------------------------------------

/// EfficientViTBlock = SanaMultiscaleLinearAttention + GLUMBConv.
struct EfficientViTBlock {
    attn: SanaMultiscaleLinearAttention,
    conv_out: DcaeGluMbConv,
}

impl EfficientViTBlock {
    fn load(vb: VarBuilder, dim: usize, head_dim: usize, qkv_kernel: usize) -> Result<Self> {
        let attn = SanaMultiscaleLinearAttention::load(vb.pp("attn"), dim, head_dim, qkv_kernel)?;
        let conv_out = DcaeGluMbConv::load(vb.pp("conv_out"), dim)?;
        Ok(Self { attn, conv_out })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.attn.forward(x)?;
        self.conv_out.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// DCAE Decoder (top-level)
// ---------------------------------------------------------------------------

/// DCAE decoder configuration.
#[derive(Debug, Clone)]
pub struct DcaeDecoderConfig {
    pub in_channels: usize,
    pub latent_channels: usize,
    pub attention_head_dim: usize,
    pub block_out_channels: Vec<usize>,
    pub block_types: Vec<String>,
    pub layers_per_block: Vec<usize>,
    pub qkv_multiscales: Vec<Vec<usize>>,
}

impl Default for DcaeDecoderConfig {
    fn default() -> Self {
        Self {
            in_channels: 2,
            latent_channels: 8,
            attention_head_dim: 32,
            block_out_channels: vec![128, 256, 512, 1024],
            block_types: vec![
                "ResBlock".into(),
                "ResBlock".into(),
                "ResBlock".into(),
                "EfficientViTBlock".into(),
            ],
            layers_per_block: vec![3, 3, 3, 3],
            qkv_multiscales: vec![vec![], vec![], vec![5], vec![5]],
        }
    }
}

/// Up-block: optional upsample + N layers of ResBlock or EfficientViTBlock.
enum UpBlock {
    ResBlocks {
        upsample: Option<DcUpBlock2d>,
        blocks: Vec<ResBlock>,
    },
    EfficientViTBlocks {
        upsample: Option<DcUpBlock2d>,
        blocks: Vec<EfficientViTBlock>,
    },
}

impl UpBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            UpBlock::ResBlocks { upsample, blocks } => {
                let mut x = if let Some(up) = upsample {
                    up.forward(x)?
                } else {
                    x.clone()
                };
                for block in blocks {
                    x = block.forward(&x)?;
                }
                Ok(x)
            }
            UpBlock::EfficientViTBlocks { upsample, blocks } => {
                let mut x = if let Some(up) = upsample {
                    up.forward(x)?
                } else {
                    x.clone()
                };
                for block in blocks {
                    x = block.forward(&x)?;
                }
                Ok(x)
            }
        }
    }
}

/// Complete DCAE decoder.
///
/// Loads from the `decoder.` prefix of an AutoencoderDC checkpoint.
pub struct DcaeDecoder {
    conv_in: candle_nn::Conv2d,
    in_shortcut_repeats: usize,
    /// Up blocks in storage order [0..3]. Forward iterates in reverse.
    up_blocks: Vec<UpBlock>,
    norm_out: ChannelRmsNorm,
    conv_out: candle_nn::Conv2d,
    #[allow(dead_code)]
    config: DcaeDecoderConfig,
}

impl DcaeDecoder {
    /// Load decoder weights.
    ///
    /// The `vb` should be scoped to `decoder.` (e.g., `vb.pp("decoder")`).
    pub fn load(vb: VarBuilder, config: &DcaeDecoderConfig) -> Result<Self> {
        let num_blocks = config.block_out_channels.len();
        let deepest_ch = config.block_out_channels[num_blocks - 1];

        // conv_in: latent_channels → deepest_channels
        let cfg3x3 = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = candle_nn::conv2d(
            config.latent_channels,
            deepest_ch,
            3,
            cfg3x3,
            vb.pp("conv_in"),
        )?;
        let in_shortcut_repeats = deepest_ch / config.latent_channels;

        // Build up_blocks in storage order [0..num_blocks-1]
        // Block i has out_channels[i] channels.
        // Blocks 0..num_blocks-2 have upsample prepended, block num_blocks-1 does not.
        let mut up_blocks = Vec::with_capacity(num_blocks);

        for i in 0..num_blocks {
            let out_ch = config.block_out_channels[i];
            let n_layers = config.layers_per_block[i];
            let block_type = &config.block_types[i];

            // Determine input channels for upsample (from the next-higher block)
            let needs_upsample = i < num_blocks - 1;
            let upsample = if needs_upsample {
                let in_ch = config.block_out_channels[i + 1];
                Some(DcUpBlock2d::load(
                    vb.pp(format!("up_blocks.{i}.0")),
                    in_ch,
                    out_ch,
                )?)
            } else {
                None
            };

            // Layer offset: if upsample exists, layers start at index 1
            let layer_offset = if needs_upsample { 1 } else { 0 };

            match block_type.as_str() {
                "ResBlock" => {
                    let mut blocks = Vec::with_capacity(n_layers);
                    for j in 0..n_layers {
                        let block = ResBlock::load(
                            vb.pp(format!("up_blocks.{i}.{}", j + layer_offset)),
                            out_ch,
                        )?;
                        blocks.push(block);
                    }
                    up_blocks.push(UpBlock::ResBlocks { upsample, blocks });
                }
                "EfficientViTBlock" => {
                    let qkv_kernel = config.qkv_multiscales[i].first().copied().unwrap_or(5);
                    let mut blocks = Vec::with_capacity(n_layers);
                    for j in 0..n_layers {
                        let block = EfficientViTBlock::load(
                            vb.pp(format!("up_blocks.{i}.{}", j + layer_offset)),
                            out_ch,
                            config.attention_head_dim,
                            qkv_kernel,
                        )?;
                        blocks.push(block);
                    }
                    up_blocks.push(UpBlock::EfficientViTBlocks { upsample, blocks });
                }
                other => {
                    return Err(crate::Error::Config(format!(
                        "Unknown DCAE block type: {other}"
                    )));
                }
            }
        }

        let norm_out = ChannelRmsNorm::load(vb.pp("norm_out"), config.block_out_channels[0], 1e-5)?;
        let conv_out = candle_nn::conv2d(
            config.block_out_channels[0],
            config.in_channels,
            3,
            cfg3x3,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            in_shortcut_repeats,
            up_blocks,
            norm_out,
            conv_out,
            config: config.clone(),
        })
    }

    /// Decode latents to mel spectrogram.
    ///
    /// - Input: `[B, 8, 16, W]` (latent)
    /// - Output: `[B, 2, 128, 8W]` (stereo mel spectrogram in [-1, 1])
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        // conv_in + in_shortcut
        let shortcut = repeat_interleave_channels(latent, self.in_shortcut_repeats)?;
        let x = (self.conv_in.forward(latent)? + shortcut)?;

        // Process up_blocks in reverse order (deepest first)
        let mut x = x;
        for up_block in self.up_blocks.iter().rev() {
            x = up_block.forward(&x)?;
        }

        // Output: RMSNorm → ReLU → Conv2d
        let x = self.norm_out.forward_channels_first(&x)?;
        let x = x.relu()?;
        self.conv_out.forward(&x).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarMap;

    fn make_vb(device: &Device) -> (VarMap, VarBuilder<'static>) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        (varmap, vb)
    }

    #[test]
    fn frame_length_60s() {
        assert_eq!(calculate_frame_length(60.0), 646);
    }

    #[test]
    fn frame_length_30s() {
        assert_eq!(calculate_frame_length(30.0), 323);
    }

    #[test]
    fn frame_length_10s() {
        assert_eq!(calculate_frame_length(10.0), 108);
    }

    #[test]
    fn denormalize_roundtrip() {
        let device = Device::Cpu;
        let original = Tensor::randn(0.0_f32, 1.0, (1, 8, 16, 32), &device).unwrap();
        let normalized = ((&original - SHIFT_FACTOR).unwrap() * SCALE_FACTOR).unwrap();
        let recovered = denormalize_latent(&normalized).unwrap();
        let diff: f32 = (&recovered - &original)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-4, "roundtrip error = {diff}");
    }

    #[test]
    fn mel_denormalize_range() {
        let device = Device::Cpu;
        let minus_one = Tensor::full(-1.0_f32, (1, 128, 32), &device).unwrap();
        let plus_one = Tensor::full(1.0_f32, (1, 128, 32), &device).unwrap();
        let mel_low: f32 = denormalize_mel(&minus_one)
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        let mel_high: f32 = denormalize_mel(&plus_one)
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!((mel_low - MIN_MEL_VALUE as f32).abs() < 1e-4);
        assert!((mel_high - MAX_MEL_VALUE as f32).abs() < 1e-4);
    }

    #[test]
    fn channel_rms_norm_shape() {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);
        let norm = ChannelRmsNorm::load(vb, 32, 1e-5).unwrap();
        let x = Tensor::randn(0.0_f32, 1.0, (2, 32, 4, 4), &device).unwrap();
        let out = norm.forward_channels_first(&x).unwrap();
        assert_eq!(out.dims(), &[2, 32, 4, 4]);
    }

    #[test]
    fn pixel_shuffle_shape() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0_f32, 1.0, (1, 16, 4, 4), &device).unwrap();
        let out = pixel_shuffle(&x, 2).unwrap();
        assert_eq!(out.dims(), &[1, 4, 8, 8]);
    }

    #[test]
    fn repeat_interleave_channels_shape() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0_f32, 1.0, (1, 8, 4, 4), &device).unwrap();
        let out = repeat_interleave_channels(&x, 3).unwrap();
        assert_eq!(out.dims(), &[1, 24, 4, 4]);
    }

    #[test]
    fn resblock_preserves_shape() {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);
        let block = ResBlock::load(vb, 32).unwrap();
        let x = Tensor::randn(0.0_f32, 1.0, (1, 32, 8, 8), &device).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 32, 8, 8]);
    }

    #[test]
    fn dc_up_block_doubles_spatial() {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);
        let up = DcUpBlock2d::load(vb, 64, 32).unwrap();
        let x = Tensor::randn(0.0_f32, 1.0, (1, 64, 4, 4), &device).unwrap();
        let out = up.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 32, 8, 8]);
    }

    #[test]
    fn efficient_vit_block_preserves_shape() {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);
        let block = EfficientViTBlock::load(vb, 64, 16, 5).unwrap();
        let x = Tensor::randn(0.0_f32, 1.0, (1, 64, 8, 8), &device).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 64, 8, 8]);
    }

    #[test]
    fn dcae_decoder_small() {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);

        let config = DcaeDecoderConfig {
            in_channels: 2,
            latent_channels: 4,
            attention_head_dim: 8,
            block_out_channels: vec![16, 32, 64],
            block_types: vec![
                "ResBlock".into(),
                "ResBlock".into(),
                "EfficientViTBlock".into(),
            ],
            layers_per_block: vec![1, 1, 1],
            qkv_multiscales: vec![vec![], vec![], vec![3]],
        };

        let decoder = DcaeDecoder::load(vb, &config).unwrap();
        let latent = Tensor::randn(0.0_f32, 1.0, (1, 4, 4, 8), &device).unwrap();
        let out = decoder.forward(&latent).unwrap();
        // 3 blocks = 2 upsamples (blocks 0 and 1 get upsamples, block 2 = deepest, no upsample)
        // 4 * 2^2 = 16 in height, 8 * 2^2 = 32 in width
        assert_eq!(out.dims(), &[1, 2, 16, 32]);
    }

    #[test]
    fn dcae_decoder_default_config() {
        let config = DcaeDecoderConfig::default();
        assert_eq!(config.block_out_channels, vec![128, 256, 512, 1024]);
        assert_eq!(config.latent_channels, 8);
        assert_eq!(config.in_channels, 2);
        // 3 upsamples = 8× spatial upscaling
        // Input [B, 8, 16, W] → Output [B, 2, 128, 8W]
    }
}
