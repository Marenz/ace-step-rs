//! AceStepDiTModel — the full diffusion transformer.
//!
//! Patchifies input latents, runs through N DiT layers with timestep conditioning
//! and cross-attention to encoder states, then unpatchifies back.
//!
//! Input: [B, T, 64] latents (after context concatenation → [B, T, 192])
//! Output: [B, T, 64] predicted velocity

use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{self as nn, Conv1d, Conv1dConfig, VarBuilder};

use super::attention::{RmsNorm, RotaryEmbedding};
use super::layers::AceStepDiTLayer;
use super::mask::create_4d_mask;
use super::timestep::TimestepEmbedding;
use crate::config::AceStepConfig;

/// The full DiT model.
#[derive(Debug, Clone)]
pub struct AceStepDiTModel {
    rotary_emb: RotaryEmbedding,
    layers: Vec<AceStepDiTLayer>,
    proj_in: Conv1d,
    proj_out: nn::ConvTranspose1d,
    time_embed: TimestepEmbedding,
    time_embed_r: TimestepEmbedding,
    condition_embedder: nn::Linear,
    norm_out: RmsNorm,
    scale_shift_table: Tensor, // [1, 2, hidden_size]
    patch_size: usize,
    cfg: AceStepConfig,
}

impl AceStepDiTModel {
    pub fn new(
        cfg: &AceStepConfig,
        dtype: DType,
        dev: &candle_core::Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = cfg.hidden_size;
        let in_channels = cfg.in_channels;
        let patch_size = cfg.patch_size;

        let rotary_emb = RotaryEmbedding::new(cfg, dtype, dev)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(AceStepDiTLayer::new(cfg, i, vb.pp(format!("layers.{i}")))?);
        }

        // Patch embedding: Conv1d(in_channels, inner_dim, kernel=patch_size, stride=patch_size)
        let proj_in_cfg = Conv1dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj_in = nn::conv1d(
            in_channels,
            inner_dim,
            patch_size,
            proj_in_cfg,
            vb.pp("proj_in.1"),
        )?;

        // Unpatchify: ConvTranspose1d(inner_dim, acoustic_dim, kernel=patch_size, stride=patch_size)
        let proj_out_cfg = candle_nn::ConvTranspose1dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj_out = nn::conv_transpose1d(
            inner_dim,
            cfg.audio_acoustic_hidden_dim,
            patch_size,
            proj_out_cfg,
            vb.pp("proj_out.1"),
        )?;

        let time_embed = TimestepEmbedding::new(256, inner_dim, vb.pp("time_embed"))?;
        let time_embed_r = TimestepEmbedding::new(256, inner_dim, vb.pp("time_embed_r"))?;
        let condition_embedder = nn::linear(inner_dim, inner_dim, vb.pp("condition_embedder"))?;
        let norm_out = RmsNorm::new(inner_dim, cfg.rms_norm_eps, vb.pp("norm_out"))?;
        let scale_shift_table = vb.get((1, 2, inner_dim), "scale_shift_table")?;

        Ok(Self {
            rotary_emb,
            layers,
            proj_in,
            proj_out,
            time_embed,
            time_embed_r,
            condition_embedder,
            norm_out,
            scale_shift_table,
            patch_size,
            cfg: cfg.clone(),
        })
    }

    /// Forward pass of the DiT.
    ///
    /// - `hidden_states`: [B, T, acoustic_dim] — the noisy latents
    /// - `timestep`: [B] — current timestep t
    /// - `timestep_r`: [B] — reference timestep r (= t for standard inference)
    /// - `encoder_hidden_states`: [B, E, D] — packed condition sequence
    /// - `context_latents`: [B, T, in_channels - acoustic_dim] — src_latents + chunk_masks
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        timestep_r: &Tensor,
        encoder_hidden_states: &Tensor,
        context_latents: &Tensor,
    ) -> Result<Tensor> {
        // 1. Timestep embeddings
        let (temb_t, proj_t) = self.time_embed.forward(timestep)?;
        let t_minus_r = (timestep - timestep_r)?;
        let (temb_r, proj_r) = self.time_embed_r.forward(&t_minus_r)?;
        let temb = (&temb_t + &temb_r)?;
        let timestep_proj = (&proj_t + &proj_r)?; // [B, 6, D]

        // 2. Concatenate context with hidden states: [B, T, in_channels]
        let h = Tensor::cat(&[context_latents, hidden_states], 2)?;

        let original_seq_len = h.dim(1)?;

        // Pad sequence to be divisible by patch_size
        let h = if original_seq_len % self.patch_size != 0 {
            let pad_len = self.patch_size - (original_seq_len % self.patch_size);
            h.pad_with_zeros(1, 0, pad_len)?
        } else {
            h
        };

        // 3. Patchify: [B, T, C] → transpose → Conv1d → transpose → [B, T/P, D]
        let h = h.transpose(1, 2)?.contiguous()?; // [B, C, T]
        let h = h.apply(&self.proj_in)?; // [B, D, T/P]
        let h = h.transpose(1, 2)?.contiguous()?; // [B, T/P, D]

        // 4. Project encoder states
        let enc = encoder_hidden_states.apply(&self.condition_embedder)?;

        // 5. Position embeddings
        let seq_len = h.dim(1)?;
        let pos_ids = Tensor::arange(0u32, seq_len as u32, h.device())?.unsqueeze(0)?;
        let pos_emb = self.rotary_emb.forward(&pos_ids)?;

        // 6. Build attention masks (SDPA mode — 4D masks)
        let dtype = h.dtype();
        let device = h.device();
        let enc_seq_len = enc.dim(1)?;

        let full_attn_mask = create_4d_mask(seq_len, dtype, device, false, None)?;

        let sliding_attn_mask = if self.cfg.use_sliding_window {
            Some(create_4d_mask(
                seq_len,
                dtype,
                device,
                false,
                Some(self.cfg.sliding_window),
            )?)
        } else {
            None
        };

        // Cross-attention mask: [1, 1, seq_len, enc_seq_len]
        let max_len = seq_len.max(enc_seq_len);
        let enc_mask_full = create_4d_mask(max_len, dtype, device, false, None)?;
        let enc_attn_mask = enc_mask_full.i((.., .., ..seq_len, ..enc_seq_len))?;

        // 7. Run DiT layers
        let mut h = h;
        for layer in &self.layers {
            // Select mask based on layer type
            let self_mask = if layer.self_attn_has_sliding_window() {
                sliding_attn_mask.as_ref()
            } else {
                Some(&full_attn_mask)
            };
            h = layer.forward(
                &h,
                &timestep_proj,
                self_mask,
                &enc,
                Some(&enc_attn_mask),
                &pos_emb,
            )?;
        }

        // 8. Output: AdaLN + unpatchify
        let temb_unsqueezed = temb.unsqueeze(1)?; // [B, 1, D]
        let modulation = self.scale_shift_table.broadcast_add(&temb_unsqueezed)?; // [1,2,D]+[B,1,D] → [B,2,D]
        let shift = modulation.i((.., 0..1, ..))?; // [B, 1, D]
        let scale = modulation.i((.., 1..2, ..))?; // [B, 1, D]

        let h = self.norm_out.forward(&h)?;
        let scale_plus_one = (scale + 1.0)?;
        let h = h.broadcast_mul(&scale_plus_one)?.broadcast_add(&shift)?;

        // Unpatchify: [B, T/P, D] → transpose → ConvTranspose1d → transpose → [B, T, acoustic_dim]
        let h = h.transpose(1, 2)?.contiguous()?;
        let h = h.apply(&self.proj_out)?;
        let h = h.transpose(1, 2)?.contiguous()?;

        // Crop to original length
        if h.dim(1)? > original_seq_len {
            h.narrow(1, 0, original_seq_len)
        } else {
            Ok(h)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_dit_model_shape() {
        let dev = Device::Cpu;
        let cfg = AceStepConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            num_hidden_layers: 2,
            in_channels: 192,
            audio_acoustic_hidden_dim: 64,
            patch_size: 2,
            layer_types: vec![
                crate::config::LayerType::SlidingAttention,
                crate::config::LayerType::FullAttention,
            ],
            ..AceStepConfig::default()
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let dit = AceStepDiTModel::new(&cfg, DType::F32, &dev, vb.pp("dit")).unwrap();

        let b = 1;
        let t = 10;
        let hidden = Tensor::randn(0f32, 1.0, (b, t, 64), &dev).unwrap();
        let context = Tensor::randn(0f32, 1.0, (b, t, 128), &dev).unwrap(); // 192 - 64 = 128
        let timestep = Tensor::new(&[0.5f32], &dev).unwrap();
        let enc = Tensor::randn(0f32, 1.0, (b, 20, 32), &dev).unwrap();

        let out = dit
            .forward(&hidden, &timestep, &timestep, &enc, &context)
            .unwrap();
        assert_eq!(out.dims(), &[b, t, 64]);
    }
}
