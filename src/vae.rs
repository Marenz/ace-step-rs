//! AutoencoderOobleck VAE decoder.
//!
//! Converts latents [B, 64, T] to stereo waveform [B, 2, T*2048] at 48kHz.
//! Architecture is structurally identical to DAC (Descript Audio Codec) decoder
//! with different config and Snake1d having both alpha and beta parameters.
//!
//! Uses weight-norm loading via `candle_transformers::models::encodec`.

use candle_core::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};
use candle_transformers::models::encodec;

use crate::config::VaeConfig;

// ---------------------------------------------------------------------------
// Snake1d activation: x + (1/exp(beta)) * sin(exp(alpha) * x)^2
// ---------------------------------------------------------------------------

/// Snake activation with learnable alpha and beta (Oobleck variant).
///
/// Unlike DAC's Snake1d which only has alpha, Oobleck also has a beta parameter.
#[derive(Debug, Clone)]
pub struct Snake1d {
    alpha: Tensor, // [1, channels, 1]
    beta: Tensor,  // [1, channels, 1]
}

impl Snake1d {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        let beta = vb.get((1, channels, 1), "beta")?;
        Ok(Self { alpha, beta })
    }
}

impl Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // x + (1/exp(beta)) * sin(exp(alpha) * x)^2
        let alpha_exp = self.alpha.exp()?;
        let beta_exp = self.beta.exp()?;
        let sin_term = alpha_exp.broadcast_mul(xs)?.sin()?;
        let sin_sq = (&sin_term * &sin_term)?;
        let recip_beta = beta_exp.recip()?;
        xs + recip_beta.broadcast_mul(&sin_sq)?
    }
}

// ---------------------------------------------------------------------------
// Residual unit: Snake → Conv1d(dil) → Snake → Conv1d(1) + residual
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OobleckResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl OobleckResidualUnit {
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        let snake1 = Snake1d::new(dim, vb.pp("snake1"))?;
        let cfg1 = Conv1dConfig {
            dilation,
            padding: pad,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(dim, dim, 7, cfg1, vb.pp("conv1"))?;
        let snake2 = Snake1d::new(dim, vb.pp("snake2"))?;
        let conv2 = encodec::conv1d_weight_norm(dim, dim, 1, Default::default(), vb.pp("conv2"))?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }
}

impl Module for OobleckResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs
            .apply(&self.snake1)?
            .apply(&self.conv1)?
            .apply(&self.snake2)?
            .apply(&self.conv2)?;
        // Residual connection (handle potential length mismatch from conv)
        let pad = (xs.dim(D::Minus1)? - ys.dim(D::Minus1)?) / 2;
        if pad > 0 {
            &ys + xs.narrow(D::Minus1, pad, ys.dim(D::Minus1)?)
        } else {
            ys + xs
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder block: Snake → ConvTranspose1d(upsample) → 3x ResidualUnit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OobleckDecoderBlock {
    snake1: Snake1d,
    conv_t1: ConvTranspose1d,
    res_unit1: OobleckResidualUnit,
    res_unit2: OobleckResidualUnit,
    res_unit3: OobleckResidualUnit,
}

impl OobleckDecoderBlock {
    pub fn new(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let snake1 = Snake1d::new(in_dim, vb.pp("snake1"))?;
        let cfg = ConvTranspose1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv_t1 = encodec::conv_transpose1d_weight_norm(
            in_dim,
            out_dim,
            2 * stride,
            true,
            cfg,
            vb.pp("conv_t1"),
        )?;
        let res_unit1 = OobleckResidualUnit::new(out_dim, 1, vb.pp("res_unit1"))?;
        let res_unit2 = OobleckResidualUnit::new(out_dim, 3, vb.pp("res_unit2"))?;
        let res_unit3 = OobleckResidualUnit::new(out_dim, 9, vb.pp("res_unit3"))?;
        Ok(Self {
            snake1,
            conv_t1,
            res_unit1,
            res_unit2,
            res_unit3,
        })
    }
}

impl Module for OobleckDecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.snake1)?
            .apply(&self.conv_t1)?
            .apply(&self.res_unit1)?
            .apply(&self.res_unit2)?
            .apply(&self.res_unit3)
    }
}

// ---------------------------------------------------------------------------
// Full Oobleck decoder
// ---------------------------------------------------------------------------

/// AutoencoderOobleck decoder: [B, 64, T] → [B, 2, T*2048].
#[derive(Debug, Clone)]
pub struct OobleckDecoder {
    conv1: Conv1d,
    blocks: Vec<OobleckDecoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl OobleckDecoder {
    pub fn new(vae_cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let channels = vae_cfg.decoder_channels;
        let in_channels = vae_cfg.decoder_input_channels;
        let out_channels = vae_cfg.audio_channels;

        // Channel multiples: [1, 1, 2, 4, 8, 16] (prepend 1 to channel_multiples)
        let mut cm = vec![1usize];
        cm.extend_from_slice(&vae_cfg.channel_multiples);
        // cm = [1, 1, 2, 4, 8, 16] for default config

        let strides: Vec<usize> = vae_cfg.downsampling_ratios.iter().rev().cloned().collect();
        // strides = [8, 8, 4, 4, 2] for default config

        let n_blocks = strides.len();

        // conv1: in_channels → channels * cm[last]
        let first_dim = channels * cm[n_blocks];
        let conv1_cfg = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 =
            encodec::conv1d_weight_norm(in_channels, first_dim, 7, conv1_cfg, vb.pp("conv1"))?;

        // Decoder blocks
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let in_d = channels * cm[n_blocks - i];
            let out_d = channels * cm[n_blocks - i - 1];
            let block =
                OobleckDecoderBlock::new(in_d, out_d, strides[i], vb.pp(format!("block.{i}")))?;
            blocks.push(block);
        }

        // Final activation + conv
        let final_dim = channels * cm[0]; // channels * 1 = 128
        let snake1 = Snake1d::new(final_dim, vb.pp("snake1"))?;
        // No bias on final conv
        let conv2 = encodec::conv1d_weight_norm_no_bias(
            final_dim,
            out_channels,
            7,
            conv1_cfg,
            vb.pp("conv2"),
        )?;

        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }

    /// Decode latents to waveform.
    ///
    /// Input: [B, 64, T] latents
    /// Output: [B, 2, T*hop] stereo waveform at 48kHz
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut h = latents.apply(&self.conv1)?;
        for block in &self.blocks {
            h = h.apply(block)?;
        }
        h.apply(&self.snake1)?.apply(&self.conv2)
    }

    /// Chunked/tiled decode for long sequences that don't fit in VRAM.
    ///
    /// Splits the latent sequence into overlapping chunks, decodes each independently,
    /// trims the overlap regions, and concatenates. This matches the Python reference's
    /// `_tiled_decode_gpu` / `_tiled_decode_offload_cpu` approach.
    ///
    /// - `chunk_size`: number of latent frames per chunk (default: 128)
    /// - `overlap`: overlap in latent frames between adjacent windows (default: 16)
    pub fn tiled_decode(
        &self,
        latents: &Tensor,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Tensor> {
        let (_bsz, _channels, latent_frames) = latents.dims3()?;

        // If it fits in one chunk, just decode directly
        if latent_frames <= chunk_size {
            return self.decode(latents);
        }

        // Ensure overlap is workable
        let overlap = if chunk_size <= 2 * overlap {
            let mut ov = overlap;
            while chunk_size <= 2 * ov && ov > 0 {
                ov /= 2;
            }
            ov
        } else {
            overlap
        };

        let stride = chunk_size - 2 * overlap;
        assert!(stride > 0, "chunk_size must be > 2 * overlap");

        let num_steps = latent_frames.div_ceil(stride);
        let mut decoded_chunks: Vec<Tensor> = Vec::with_capacity(num_steps);
        let mut upsample_factor: Option<f64> = None;

        for i in 0..num_steps {
            let core_start = i * stride;
            let core_end = (core_start + stride).min(latent_frames);
            let win_start = core_start.saturating_sub(overlap);
            let win_end = (core_end + overlap).min(latent_frames);

            let latent_chunk = latents.i((.., .., win_start..win_end))?.contiguous()?;
            let audio_chunk = self.decode(&latent_chunk)?;

            let uf = match upsample_factor {
                Some(f) => f,
                None => {
                    let f = audio_chunk.dim(2)? as f64 / latent_chunk.dim(2)? as f64;
                    upsample_factor = Some(f);
                    f
                }
            };

            let added_start = core_start - win_start;
            let trim_start = (added_start as f64 * uf).round() as usize;
            let added_end = win_end - core_end;
            let trim_end = (added_end as f64 * uf).round() as usize;

            let audio_len = audio_chunk.dim(2)?;
            let end_idx = if trim_end > 0 {
                audio_len - trim_end
            } else {
                audio_len
            };
            let audio_core = audio_chunk.i((.., .., trim_start..end_idx))?.contiguous()?;
            decoded_chunks.push(audio_core);
        }

        Tensor::cat(&decoded_chunks, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_snake1d() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let snake = Snake1d::new(4, vb.pp("snake")).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, 4, 10), &dev).unwrap();
        let y = snake.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 4, 10]);
    }

    #[test]
    fn test_snake1d_identity_at_zero_alpha() {
        // With alpha=0, beta=0: sin(exp(0)*x)^2 / exp(0) = sin(x)^2
        // so output = x + sin(x)^2 (not identity, but deterministic)
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let snake = Snake1d::new(2, vb.pp("snake")).unwrap();
        let x = Tensor::zeros((1, 2, 5), DType::F32, &dev).unwrap();
        let y = snake.forward(&x).unwrap();
        // sin(0)^2 = 0, so output should be 0
        let sum: f32 = y.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(sum < 1e-6);
    }
}
