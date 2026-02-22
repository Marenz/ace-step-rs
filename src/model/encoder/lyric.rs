//! Lyric encoder — 8-layer transformer on Qwen3 token embeddings.
//!
//! Takes lyric embeddings [B, L, 1024] from Qwen3-Embedding's embed_tokens,
//! projects to hidden_size, processes through transformer layers with RoPE,
//! and returns encoded lyric features [B, L, hidden_size].

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::config::AceStepConfig;
use crate::model::transformer::attention::{RmsNorm, RotaryEmbedding};
use crate::model::transformer::layers::AceStepEncoderLayer;
use crate::model::transformer::mask::create_4d_mask;

/// Lyric encoder matching Python `AceStepLyricEncoder`.
#[derive(Debug, Clone)]
pub struct AceStepLyricEncoder {
    embed_tokens: nn::Linear,
    norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    layers: Vec<AceStepEncoderLayer>,
    cfg: AceStepConfig,
}

impl AceStepLyricEncoder {
    pub fn new(
        cfg: &AceStepConfig,
        dtype: DType,
        dev: &candle_core::Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embed_tokens = nn::linear(cfg.text_hidden_dim, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = RotaryEmbedding::new(cfg, dtype, dev)?;

        let mut layers = Vec::with_capacity(cfg.num_lyric_encoder_hidden_layers);
        for i in 0..cfg.num_lyric_encoder_hidden_layers {
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
            layers,
            cfg: cfg.clone(),
        })
    }

    /// Forward pass.
    ///
    /// - `inputs_embeds`: [B, L, text_hidden_dim] — raw token embeddings from Qwen3
    /// - `attention_mask`: [B, L] — 1 for valid, 0 for padding
    pub fn forward(&self, inputs_embeds: &Tensor, _attention_mask: &Tensor) -> Result<Tensor> {
        let h = inputs_embeds.apply(&self.embed_tokens)?;
        let seq_len = h.dim(1)?;
        let pos_ids = Tensor::arange(0u32, seq_len as u32, h.device())?.unsqueeze(0)?;
        let pos_emb = self.rotary_emb.forward(&pos_ids)?;

        // Build attention masks
        let dtype = h.dtype();
        let device = h.device();
        let full_mask = create_4d_mask(seq_len, dtype, device, false, None)?;
        let sliding_mask = if self.cfg.use_sliding_window {
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

        let mut h = h;
        for layer in &self.layers {
            let mask = if layer.self_attn_has_sliding_window() {
                sliding_mask.as_ref()
            } else {
                Some(&full_mask)
            };
            h = layer.forward(&h, mask, &pos_emb)?;
        }

        self.norm.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_lyric_encoder_shape() {
        let dev = Device::Cpu;
        let cfg = AceStepConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            num_hidden_layers: 2,
            num_lyric_encoder_hidden_layers: 2,
            text_hidden_dim: 16,
            ..AceStepConfig::default()
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let enc = AceStepLyricEncoder::new(&cfg, DType::F32, &dev, vb.pp("lyric")).unwrap();
        let embeds = Tensor::randn(0f32, 1.0, (2, 10, 16), &dev).unwrap();
        let mask = Tensor::ones((2, 10), DType::F32, &dev).unwrap();
        let out = enc.forward(&embeds, &mask).unwrap();
        assert_eq!(out.dims(), &[2, 10, 32]);
    }
}
