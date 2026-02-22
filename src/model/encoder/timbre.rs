//! Timbre encoder — extracts timbre from reference audio.
//!
//! Processes packed reference audio acoustic features [N, 750, 64] through
//! a 4-layer transformer, extracts CLS token per sample, and unpacks
//! to batch format.

use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::config::AceStepConfig;
use crate::model::transformer::attention::{RmsNorm, RotaryEmbedding};
use crate::model::transformer::layers::AceStepEncoderLayer;
use crate::model::transformer::mask::create_4d_mask;

/// Timbre encoder matching Python `AceStepTimbreEncoder`.
#[derive(Debug, Clone)]
pub struct AceStepTimbreEncoder {
    embed_tokens: nn::Linear,
    norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    special_token: Tensor,
    layers: Vec<AceStepEncoderLayer>,
    cfg: AceStepConfig,
}

impl AceStepTimbreEncoder {
    pub fn new(
        cfg: &AceStepConfig,
        dtype: DType,
        dev: &candle_core::Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embed_tokens = nn::linear(
            cfg.timbre_hidden_dim,
            cfg.hidden_size,
            vb.pp("embed_tokens"),
        )?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = RotaryEmbedding::new(cfg, dtype, dev)?;
        let special_token = vb.get((1, 1, cfg.hidden_size), "special_token")?;

        let mut layers = Vec::with_capacity(cfg.num_timbre_encoder_hidden_layers);
        for i in 0..cfg.num_timbre_encoder_hidden_layers {
            layers.push(AceStepEncoderLayer::new(
                cfg,
                i,
                vb.pp(format!("layers.{i}")),
            )?);
        }

        Ok(Self {
            embed_tokens,
            norm,
            rotary_emb,
            special_token,
            layers,
            cfg: cfg.clone(),
        })
    }

    /// Forward pass.
    ///
    /// - `refer_audio`: [N, T, timbre_hidden_dim] — packed reference audio features
    /// - `order_mask`: [N] — batch assignment for each packed sample
    ///
    /// Returns (unpacked_embeds [B, max_refs, D], mask [B, max_refs]).
    pub fn forward(&self, refer_audio: &Tensor, order_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = refer_audio.apply(&self.embed_tokens)?;
        let (_n, seq_len, _d) = h.dims3()?;

        // Position embeddings
        let pos_ids = Tensor::arange(0u32, seq_len as u32, h.device())?.unsqueeze(0)?;
        let pos_emb = self.rotary_emb.forward(&pos_ids)?;

        // Build mask
        let full_mask = create_4d_mask(seq_len, h.dtype(), h.device(), false, None)?;

        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(&h, Some(&full_mask), &pos_emb)?;
        }
        let h = self.norm.forward(&h)?;

        // Extract first position as timbre embedding: [N, seq_len, D] → [N, D]
        let timbre_embs = h.i((.., 0, ..))?;

        // Unpack to batch format
        self.unpack_timbre_embeddings(&timbre_embs, order_mask)
    }

    /// Unpack packed timbre embeddings [N, D] → [B, max_count, D] using order_mask.
    fn unpack_timbre_embeddings(
        &self,
        timbre_embs: &Tensor,
        order_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_n, d) = timbre_embs.dims2()?;
        let dev = timbre_embs.device();

        // Get batch assignments as Vec
        let order: Vec<i64> = order_mask.to_vec1()?;
        let b = *order.iter().max().unwrap_or(&0) as usize + 1;

        // Count per batch
        let mut counts = vec![0usize; b];
        for &o in &order {
            counts[o as usize] += 1;
        }
        let max_count = *counts.iter().max().unwrap_or(&1);

        // Build unpacked tensor by gathering
        let mut positions = vec![0usize; b];
        let _indices: Vec<i64> = Vec::with_capacity(b * max_count);
        let mut mask_vals = Vec::with_capacity(b * max_count);

        // We'll gather specific rows from timbre_embs
        // For padding positions, we use index 0 (will be masked out)
        let mut gather_indices = vec![0i64; b * max_count];

        for (idx, &batch_id) in order.iter().enumerate() {
            let bid = batch_id as usize;
            let pos = positions[bid];
            gather_indices[bid * max_count + pos] = idx as i64;
            positions[bid] = pos + 1;
        }

        // Build mask
        for bid in 0..b {
            for pos in 0..max_count {
                mask_vals.push(if pos < counts[bid] { 1i64 } else { 0i64 });
            }
        }

        let gather_idx = Tensor::new(gather_indices.as_slice(), dev)?;
        let unpacked = timbre_embs.contiguous()?.index_select(&gather_idx, 0)?;
        let unpacked = unpacked.reshape((b, max_count, d))?;

        let mask = Tensor::new(mask_vals.as_slice(), dev)?;
        let mask = mask.reshape((b, max_count))?;

        Ok((unpacked, mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_timbre_encoder_shape() {
        let dev = Device::Cpu;
        let cfg = AceStepConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            num_hidden_layers: 2,
            num_timbre_encoder_hidden_layers: 2,
            timbre_hidden_dim: 8,
            ..AceStepConfig::default()
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let enc = AceStepTimbreEncoder::new(&cfg, DType::F32, &dev, vb.pp("timbre")).unwrap();

        // 3 reference audios: 2 for batch 0, 1 for batch 1
        let audio = Tensor::randn(0f32, 1.0, (3, 10, 8), &dev).unwrap();
        let order = Tensor::new(&[0i64, 0, 1], &dev).unwrap();
        let (embs, mask) = enc.forward(&audio, &order).unwrap();
        assert_eq!(embs.dims(), &[2, 2, 32]); // B=2, max_count=2
        assert_eq!(mask.dims(), &[2, 2]);
    }
}
