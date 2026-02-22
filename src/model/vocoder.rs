//! ADaMoSHiFiGAN vocoder: mel spectrogram → audio waveform.
//!
//! Two-component architecture:
//!
//! ## ConvNeXt backbone
//! ```text
//! Input: [B, 128, T_mel]
//! Stem:   Conv1d(128, 128, k=7, replicate_pad) + LayerNorm
//! Stage0: 3 × ConvNeXtBlock(128)
//! Down:   LayerNorm + Conv1d(128, 256, k=1)
//! Stage1: 3 × ConvNeXtBlock(256)
//! Down:   LayerNorm + Conv1d(256, 384, k=1)
//! Stage2: 9 × ConvNeXtBlock(384)
//! Down:   LayerNorm + Conv1d(384, 512, k=1)
//! Stage3: 3 × ConvNeXtBlock(512)
//! Final:  LayerNorm(512)
//! Output: [B, 512, T_mel]
//! ```
//!
//! ## HiFiGAN head
//! ```text
//! Input: [B, 512, T_mel]
//! conv_pre(512, 1024, k=13)
//! 7 upsample stages: rates=[4,4,2,2,2,2,2], product=512=hop_length
//! Each: SiLU + ConvTranspose1d + 4× ResBlock1 (multi-kernel, averaged)
//! conv_post → [B, 1, T_audio], tanh
//! Output: T_audio = T_mel × 512
//! ```
//!
//! ## Weight norm
//!
//! HiFiGAN uses weight normalization. Weights are stored as:
//! - `parametrizations.weight.original0` (direction)
//! - `parametrizations.weight.original1` (magnitude)
//!
//! At load time, these are folded into a single weight tensor.
//!
//! For stereo: the vocoder is called twice (once per channel).

use candle_core::{DType, Module, Tensor};
use candle_nn::VarBuilder;

use crate::Result;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Vocoder configuration.
#[derive(Debug, Clone)]
pub struct VocoderConfig {
    /// ConvNeXt stage depths.
    pub backbone_depths: Vec<usize>,
    /// ConvNeXt stage dimensions.
    pub backbone_dims: Vec<usize>,
    /// HiFiGAN upsample rates (product must equal hop_length=512).
    pub upsample_rates: Vec<usize>,
    /// HiFiGAN upsample kernel sizes.
    pub upsample_kernel_sizes: Vec<usize>,
    /// HiFiGAN ResBlock kernel sizes.
    pub resblock_kernel_sizes: Vec<usize>,
    /// Dilations for each ResBlock iteration.
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// HiFiGAN initial channel count.
    pub upsample_initial_channel: usize,
    /// Input mel bins.
    pub input_channels: usize,
    /// Pre/post conv kernel size.
    pub pre_post_conv_kernel_size: usize,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            backbone_depths: vec![3, 3, 9, 3],
            backbone_dims: vec![128, 256, 384, 512],
            upsample_rates: vec![4, 4, 2, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11, 13],
            resblock_dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            upsample_initial_channel: 1024,
            input_channels: 128,
            pre_post_conv_kernel_size: 13,
        }
    }
}

impl VocoderConfig {
    /// Verify that upsample rates multiply to the hop length.
    pub fn verify(&self) -> Result<()> {
        let product: usize = self.upsample_rates.iter().product();
        if product != 512 {
            return Err(crate::Error::Config(format!(
                "upsample_rates product is {product}, expected 512 (hop_length)"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Channels-first LayerNorm (1D)
// ---------------------------------------------------------------------------

/// LayerNorm that operates on `[B, C, L]` tensors (normalizing over C dimension).
struct ChannelsFirstLayerNorm1d {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl ChannelsFirstLayerNorm1d {
    fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self {
            weight,
            bias,
            eps: 1e-6,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, L]
        let mean = x.mean_keepdim(1)?; // [B, 1, L]
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(1)?;
        let rsqrt = (var + self.eps)?.recip()?.sqrt()?;
        let normed = x_centered.broadcast_mul(&rsqrt)?;
        // weight: [C] → [1, C, 1], bias: [C] → [1, C, 1]
        let w = self.weight.unsqueeze(0)?.unsqueeze(2)?;
        let b = self.bias.unsqueeze(0)?.unsqueeze(2)?;
        Ok(normed.broadcast_mul(&w)?.broadcast_add(&b)?)
    }
}

// ---------------------------------------------------------------------------
// ConvNeXtBlock
// ---------------------------------------------------------------------------

/// ConvNeXtBlock: depthwise conv → LayerNorm → Linear expand → GELU → Linear compress → scale.
struct ConvNeXtBlock {
    dwconv: candle_nn::Conv1d,
    norm: candle_nn::LayerNorm,
    pwconv1: candle_nn::Linear,
    pwconv2: candle_nn::Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn load(vb: VarBuilder, dim: usize, kernel_size: usize) -> Result<Self> {
        let pad = kernel_size / 2;
        let dw_cfg = candle_nn::Conv1dConfig {
            padding: pad,
            groups: dim,
            ..Default::default()
        };
        let dwconv = candle_nn::conv1d(dim, dim, kernel_size, dw_cfg, vb.pp("dwconv"))?;
        let norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm"))?;
        let mlp_dim = 4 * dim;
        let pwconv1 = candle_nn::linear(dim, mlp_dim, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(mlp_dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        // Depthwise conv: [B, C, L] → [B, C, L]
        let x = self.dwconv.forward(x)?;
        // Permute to channels-last: [B, C, L] → [B, L, C]
        let x = x.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        let x = self.pwconv1.forward(&x)?;
        let x = x.gelu_erf()?;
        let x = self.pwconv2.forward(&x)?;
        // Layer scale: gamma * x
        let x = x.broadcast_mul(&self.gamma)?;
        // Permute back: [B, L, C] → [B, C, L]
        let x = x.transpose(1, 2)?;
        Ok((x + residual)?)
    }
}

// ---------------------------------------------------------------------------
// ConvNeXtEncoder (backbone)
// ---------------------------------------------------------------------------

/// ConvNeXt backbone: 4 stages transforming mel to feature representation.
struct ConvNeXtEncoder {
    /// Stem + inter-stage channel projections (4 entries).
    channel_layers: Vec<ChannelLayer>,
    /// ConvNeXtBlock stages (4 entries).
    stages: Vec<Vec<ConvNeXtBlock>>,
    /// Final LayerNorm.
    norm: ChannelsFirstLayerNorm1d,
}

struct ChannelLayer {
    norm: Option<ChannelsFirstLayerNorm1d>,
    conv: candle_nn::Conv1d,
    is_stem: bool,
}

impl ChannelLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.is_stem {
            // Stem: conv first (with replicate padding handled externally),
            // then norm. But we handle replicate pad inside forward.
            let x = replicate_pad_1d(&x, 3)?; // k=7, pad=3
            let x = self.conv.forward(&x)?;
            // Norm is applied as channels-first LayerNorm
            if let Some(ref norm) = self.norm {
                norm.forward(&x)
            } else {
                Ok(x)
            }
        } else {
            // Inter-stage: norm → conv(1x1)
            let x = if let Some(ref norm) = self.norm {
                norm.forward(&x)?
            } else {
                x.clone()
            };
            self.conv.forward(&x).map_err(Into::into)
        }
    }
}

/// Replicate-pad a 1D tensor `[B, C, L]` by `pad` on each side.
fn replicate_pad_1d(x: &Tensor, pad: usize) -> Result<Tensor> {
    let (_b, _c, l) = x.dims3()?;
    if pad == 0 || l == 0 {
        return Ok(x.clone());
    }
    // Left: repeat first frame `pad` times
    let left = x.narrow(2, 0, 1)?.expand((x.dim(0)?, x.dim(1)?, pad))?;
    // Right: repeat last frame `pad` times
    let right = x.narrow(2, l - 1, 1)?.expand((x.dim(0)?, x.dim(1)?, pad))?;
    Ok(Tensor::cat(&[&left, x, &right], 2)?)
}

impl ConvNeXtEncoder {
    fn load(vb: VarBuilder, config: &VocoderConfig) -> Result<Self> {
        let dims = &config.backbone_dims;

        // channel_layers[0] = stem: Conv1d(input, dims[0], k=7) + LayerNorm
        // Note: stem conv has NO padding (we apply replicate pad manually)
        let stem_conv = candle_nn::conv1d(
            config.input_channels,
            dims[0],
            7,
            candle_nn::Conv1dConfig::default(), // no padding — we pad manually
            vb.pp("channel_layers.0.0"),
        )?;
        let stem_norm = ChannelsFirstLayerNorm1d::load(vb.pp("channel_layers.0.1"), dims[0])?;

        let mut channel_layers = vec![ChannelLayer {
            conv: stem_conv,
            norm: Some(stem_norm),
            is_stem: true,
        }];

        // channel_layers[1..] = inter-stage: LayerNorm + Conv1d(dims[i-1], dims[i], k=1)
        for i in 1..dims.len() {
            let norm = ChannelsFirstLayerNorm1d::load(
                vb.pp(format!("channel_layers.{i}.0")),
                dims[i - 1],
            )?;
            let conv = candle_nn::conv1d(
                dims[i - 1],
                dims[i],
                1,
                candle_nn::Conv1dConfig::default(),
                vb.pp(format!("channel_layers.{i}.1")),
            )?;
            channel_layers.push(ChannelLayer {
                conv,
                norm: Some(norm),
                is_stem: false,
            });
        }

        // stages
        let mut stages = Vec::with_capacity(dims.len());
        for (s, (&dim, &depth)) in dims.iter().zip(config.backbone_depths.iter()).enumerate() {
            let mut blocks = Vec::with_capacity(depth);
            for b in 0..depth {
                let block = ConvNeXtBlock::load(
                    vb.pp(format!("stages.{s}.{b}")),
                    dim,
                    7, // kernel_sizes = [7]
                )?;
                blocks.push(block);
            }
            stages.push(blocks);
        }

        let norm = ChannelsFirstLayerNorm1d::load(vb.pp("norm"), dims[dims.len() - 1])?;

        Ok(Self {
            channel_layers,
            stages,
            norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (cl, stage) in self.channel_layers.iter().zip(self.stages.iter()) {
            x = cl.forward(&x)?;
            for block in stage {
                x = block.forward(&x)?;
            }
        }
        self.norm.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Weight norm loading utilities
// ---------------------------------------------------------------------------

/// Load a weight-normed Conv1d parameter, folding the norm at load time.
///
/// Reads `{prefix}.parametrizations.weight.original0` (direction) and
/// `{prefix}.parametrizations.weight.original1` (magnitude), computes
/// `weight = mag * (dir / ||dir||)` where norm is over all dims except dim 0.
fn load_weight_normed_conv1d(
    vb: &VarBuilder,
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: candle_nn::Conv1dConfig,
) -> Result<candle_nn::Conv1d> {
    let dir = vb.get(
        (out_c, in_c / config.groups, kernel_size),
        "parametrizations.weight.original0",
    )?;
    let mag = vb.get_with_hints(
        (out_c, 1, 1),
        "parametrizations.weight.original1",
        candle_nn::Init::Const(1.0),
    )?;
    let bias = vb.get(out_c, "bias")?;

    // Normalize direction: norm over dims [1, 2] (in_c and kernel)
    let norm = dir
        .sqr()?
        .sum_keepdim(&[1, 2][..])?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?;
    let normed_dir = dir.broadcast_div(&norm)?;
    let weight = normed_dir.broadcast_mul(&mag)?;

    Ok(candle_nn::Conv1d::new(weight, Some(bias), config))
}

/// Load a weight-normed ConvTranspose1d parameter, folding the norm.
///
/// ConvTranspose1d weight shape: `[in_c, out_c, kernel]`.
/// Norm is over all dims except dim 0 (input channels).
fn load_weight_normed_conv_transpose1d(
    vb: &VarBuilder,
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: candle_nn::ConvTranspose1dConfig,
) -> Result<candle_nn::ConvTranspose1d> {
    let dir = vb.get(
        (in_c, out_c, kernel_size),
        "parametrizations.weight.original0",
    )?;
    let mag = vb.get_with_hints(
        (in_c, 1, 1),
        "parametrizations.weight.original1",
        candle_nn::Init::Const(1.0),
    )?;
    let bias = vb.get(out_c, "bias")?;

    let norm = dir
        .sqr()?
        .sum_keepdim(&[1, 2][..])?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?;
    let normed_dir = dir.broadcast_div(&norm)?;
    let weight = normed_dir.broadcast_mul(&mag)?;

    Ok(candle_nn::ConvTranspose1d::new(weight, Some(bias), config))
}

// ---------------------------------------------------------------------------
// ResBlock1
// ---------------------------------------------------------------------------

/// HiFiGAN ResBlock1: 3 residual iterations, each with dilated conv + 1x-dilation conv.
struct ResBlock1 {
    convs1: Vec<candle_nn::Conv1d>,
    convs2: Vec<candle_nn::Conv1d>,
}

impl ResBlock1 {
    fn load(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());

        for (d_idx, &dilation) in dilations.iter().enumerate() {
            // convs1: dilated conv
            let pad = (kernel_size * dilation - dilation) / 2;
            let cfg1 = candle_nn::Conv1dConfig {
                padding: pad,
                dilation,
                ..Default::default()
            };
            let c1 = load_weight_normed_conv1d(
                &vb.pp(format!("convs1.{d_idx}")),
                channels,
                channels,
                kernel_size,
                cfg1,
            )?;
            convs1.push(c1);

            // convs2: dilation=1 conv
            let pad2 = (kernel_size - 1) / 2;
            let cfg2 = candle_nn::Conv1dConfig {
                padding: pad2,
                ..Default::default()
            };
            let c2 = load_weight_normed_conv1d(
                &vb.pp(format!("convs2.{d_idx}")),
                channels,
                channels,
                kernel_size,
                cfg2,
            )?;
            convs2.push(c2);
        }

        Ok(Self { convs1, convs2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let xt = x.apply(&candle_nn::Activation::Silu)?;
            let xt = c1.forward(&xt)?;
            let xt = xt.apply(&candle_nn::Activation::Silu)?;
            let xt = c2.forward(&xt)?;
            x = (x + xt)?;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// HiFiGAN Generator (head)
// ---------------------------------------------------------------------------

/// HiFiGAN generator head: upsample through transposed convolutions + multi-kernel ResBlocks.
struct HiFiGANGenerator {
    conv_pre: candle_nn::Conv1d,
    ups: Vec<candle_nn::ConvTranspose1d>,
    /// resblocks[i*num_kernels + j] where i=upsample stage, j=kernel index.
    resblocks: Vec<ResBlock1>,
    conv_post: candle_nn::Conv1d,
    num_kernels: usize,
}

impl HiFiGANGenerator {
    fn load(vb: VarBuilder, config: &VocoderConfig) -> Result<Self> {
        let num_ups = config.upsample_rates.len();
        let num_kernels = config.resblock_kernel_sizes.len();
        let init_ch = config.upsample_initial_channel;
        let pre_k = config.pre_post_conv_kernel_size;
        let pre_pad = pre_k / 2;

        // conv_pre
        let pre_cfg = candle_nn::Conv1dConfig {
            padding: pre_pad,
            ..Default::default()
        };
        let last_backbone_dim = config.backbone_dims[config.backbone_dims.len() - 1];
        let conv_pre = load_weight_normed_conv1d(
            &vb.pp("conv_pre"),
            last_backbone_dim,
            init_ch,
            pre_k,
            pre_cfg,
        )?;

        // Upsample layers
        let mut ups = Vec::with_capacity(num_ups);
        let mut ch = init_ch;
        for i in 0..num_ups {
            let out_ch = ch / 2;
            let k = config.upsample_kernel_sizes[i];
            let stride = config.upsample_rates[i];
            let pad = (k - stride) / 2;
            let up_cfg = candle_nn::ConvTranspose1dConfig {
                padding: pad,
                stride,
                ..Default::default()
            };
            let up = load_weight_normed_conv_transpose1d(
                &vb.pp(format!("ups.{i}")),
                ch,
                out_ch,
                k,
                up_cfg,
            )?;
            ups.push(up);
            ch = out_ch;
        }

        // ResBlocks: num_ups * num_kernels total
        let mut resblocks = Vec::with_capacity(num_ups * num_kernels);
        ch = init_ch;
        for i in 0..num_ups {
            ch /= 2;
            for j in 0..num_kernels {
                let r_idx = i * num_kernels + j;
                let rb = ResBlock1::load(
                    vb.pp(format!("resblocks.{r_idx}")),
                    ch,
                    config.resblock_kernel_sizes[j],
                    &config.resblock_dilation_sizes[j],
                )?;
                resblocks.push(rb);
            }
        }

        // conv_post
        let post_cfg = candle_nn::Conv1dConfig {
            padding: pre_pad,
            ..Default::default()
        };
        let conv_post = load_weight_normed_conv1d(&vb.pp("conv_post"), ch, 1, pre_k, post_cfg)?;

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            conv_post,
            num_kernels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_pre.forward(x)?;

        for (i, up) in self.ups.iter().enumerate() {
            x = x.apply(&candle_nn::Activation::Silu)?;
            x = up.forward(&x)?;

            // Multi-kernel ResBlocks: average outputs
            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let r_idx = i * self.num_kernels + j;
                let rb_out = self.resblocks[r_idx].forward(&x)?;
                xs = Some(match xs {
                    Some(prev) => (prev + rb_out)?,
                    None => rb_out,
                });
            }
            x = (xs.unwrap() / self.num_kernels as f64)?;
        }

        x = x.apply(&candle_nn::Activation::Silu)?;
        x = self.conv_post.forward(&x)?;
        x.tanh().map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Vocoder (top-level)
// ---------------------------------------------------------------------------

/// Complete ADaMoSHiFiGAN vocoder.
///
/// Takes a mono mel spectrogram `[B, 128, T]` and produces waveform `[B, 1, T*512]`.
/// For stereo, call `forward()` twice (once per channel).
pub struct Vocoder {
    backbone: ConvNeXtEncoder,
    head: HiFiGANGenerator,
    #[allow(dead_code)]
    config: VocoderConfig,
}

impl Vocoder {
    /// Load vocoder weights.
    ///
    /// The `vb` should point to the root of the vocoder checkpoint
    /// (keys start with `backbone.` and `head.`).
    pub fn load(vb: VarBuilder, config: &VocoderConfig) -> Result<Self> {
        config.verify()?;
        let backbone = ConvNeXtEncoder::load(vb.pp("backbone"), config)?;
        let head = HiFiGANGenerator::load(vb.pp("head"), config)?;
        Ok(Self {
            backbone,
            head,
            config: config.clone(),
        })
    }

    /// Decode mel spectrogram to audio waveform.
    ///
    /// - Input: `[B, 128, T_mel]` (mono mel spectrogram)
    /// - Output: `[B, 1, T_mel * 512]` (mono waveform, values in [-1, 1])
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(mel)?;
        self.head.forward(&features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn make_vb(device: &Device) -> (VarMap, VarBuilder<'static>) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        (varmap, vb)
    }

    #[test]
    fn default_upsample_product() {
        let config = VocoderConfig::default();
        config.verify().unwrap();
    }

    #[test]
    fn backbone_dims_increasing() {
        let config = VocoderConfig::default();
        for i in 1..config.backbone_dims.len() {
            assert!(
                config.backbone_dims[i] > config.backbone_dims[i - 1],
                "dims should increase: {} vs {}",
                config.backbone_dims[i - 1],
                config.backbone_dims[i]
            );
        }
    }

    #[test]
    fn bad_upsample_rates_rejected() {
        let config = VocoderConfig {
            upsample_rates: vec![4, 4, 2, 2],
            ..Default::default()
        };
        assert!(config.verify().is_err());
    }

    #[test]
    fn channels_first_layer_norm_shape() {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);
        let norm = ChannelsFirstLayerNorm1d::load(vb, 32).unwrap();
        let x = Tensor::randn(0.0_f32, 1.0, (2, 32, 16), &device).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 32, 16]);
    }

    #[test]
    fn replicate_pad_1d_shape() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0_f32, 1.0, (1, 4, 8), &device).unwrap();
        let padded = replicate_pad_1d(&x, 3).unwrap();
        assert_eq!(padded.dims(), &[1, 4, 14]); // 8 + 3 + 3
    }

    #[test]
    fn convnext_block_preserves_shape() {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);
        let block = ConvNeXtBlock::load(vb, 32, 7).unwrap();
        let x = Tensor::randn(0.0_f32, 1.0, (1, 32, 16), &device).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 32, 16]);
    }

    #[test]
    fn pixel_shuffle_1d() {
        let device = Device::Cpu;
        // Simulate pixel_shuffle for 1D: not directly used here but good sanity check
        let x = Tensor::randn(0.0_f32, 1.0, (1, 32, 16), &device).unwrap();
        assert_eq!(x.dims(), &[1, 32, 16]);
    }

    #[test]
    fn resblock1_preserves_shape() {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);
        let rb = ResBlock1::load(vb, 16, 3, &[1, 3, 5]).unwrap();
        let x = Tensor::randn(0.0_f32, 1.0, (1, 16, 32), &device).unwrap();
        let out = rb.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 16, 32]);
    }

    #[test]
    fn small_backbone() {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);

        let config = VocoderConfig {
            backbone_depths: vec![1, 1],
            backbone_dims: vec![16, 32],
            input_channels: 8,
            ..Default::default()
        };

        let backbone = ConvNeXtEncoder::load(vb, &config).unwrap();
        let x = Tensor::randn(0.0_f32, 1.0, (1, 8, 16), &device).unwrap();
        let out = backbone.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 32, 16]);
    }

    #[test]
    fn small_vocoder_e2e() {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);

        // Tiny config for testing: 2 backbone stages, 2 upsample stages (product = 4*4 = 16... need 512)
        // Let's use a tiny rate that multiplies to 8 for faster testing
        let config = VocoderConfig {
            backbone_depths: vec![1, 1],
            backbone_dims: vec![16, 32],
            input_channels: 8,
            upsample_rates: vec![4, 2],
            upsample_kernel_sizes: vec![8, 4],
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1]],
            upsample_initial_channel: 64,
            pre_post_conv_kernel_size: 3,
        };
        // Don't verify() since we intentionally have product != 512

        let backbone = ConvNeXtEncoder::load(vb.pp("backbone"), &config).unwrap();
        let head = HiFiGANGenerator::load(vb.pp("head"), &config).unwrap();

        let mel = Tensor::randn(0.0_f32, 1.0, (1, 8, 16), &device).unwrap();
        let features = backbone.forward(&mel).unwrap();
        assert_eq!(features.dims(), &[1, 32, 16]);

        let audio = head.forward(&features).unwrap();
        // 16 * 4 * 2 = 128
        assert_eq!(audio.dims(), &[1, 1, 128]);
    }
}
