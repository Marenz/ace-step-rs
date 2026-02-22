//! Audio token detokenizer — 5Hz → 25Hz expansion.
//!
//! Expands each quantized token into `pool_window_size=5` patches using
//! learnable special tokens and self-attention.

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::AceStepConfig;
use crate::model::transformer::attention::{RmsNorm, RotaryEmbedding};
use crate::model::transformer::layers::AceStepEncoderLayer;

/// Audio token detokenizer matching Python `AudioTokenDetokenizer`.
#[derive(Debug, Clone)]
pub struct AudioTokenDetokenizer {
    embed_tokens: candle_nn::Linear,
    norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    special_tokens: Tensor,
    layers: Vec<AceStepEncoderLayer>,
    proj_out: candle_nn::Linear,
    pool_window_size: usize,
}

impl AudioTokenDetokenizer {
    pub fn new(
        cfg: &AceStepConfig,
        dtype: DType,
        dev: &candle_core::Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embed_tokens =
            candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = RotaryEmbedding::new(cfg, dtype, dev)?;
        let special_tokens =
            vb.get((1, cfg.pool_window_size, cfg.hidden_size), "special_tokens")?;

        let mut layers = Vec::with_capacity(cfg.num_attention_pooler_hidden_layers);
        for i in 0..cfg.num_attention_pooler_hidden_layers {
            layers.push(AceStepEncoderLayer::new(
                cfg,
                i,
                vb.pp(format!("layers.{i}")),
            )?);
        }

        let proj_out = candle_nn::linear(
            cfg.hidden_size,
            cfg.audio_acoustic_hidden_dim,
            vb.pp("proj_out"),
        )?;

        Ok(Self {
            embed_tokens,
            norm,
            rotary_emb,
            special_tokens,
            layers,
            proj_out,
            pool_window_size: cfg.pool_window_size,
        })
    }

    /// Forward pass.
    ///
    /// Input: [B, T, D] — quantized token embeddings at 5Hz
    /// Output: [B, T*pool_window_size, acoustic_dim] — expanded to 25Hz
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let d = x.dim(2)?;
        let x = x.apply(&self.embed_tokens)?;

        // Expand: [B, T, D] → [B, T, P, D] by repeating + adding special tokens
        let x = x.unsqueeze(2)?; // [B, T, 1, D]
        let x = x.repeat(&[1, 1, self.pool_window_size, 1])?; // [B, T, P, D]
        let special = self
            .special_tokens
            .broadcast_as((b, t, self.pool_window_size, d))?;
        let x = (x + special)?; // [B, T, P, D]

        // Reshape: [B*T, P, D]
        let bt = b * t;
        let x = x.reshape((bt, self.pool_window_size, d))?;

        let pos_ids =
            Tensor::arange(0u32, self.pool_window_size as u32, x.device())?.unsqueeze(0)?;
        let pos_emb = self.rotary_emb.forward(&pos_ids)?;

        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(&h, None, &pos_emb)?;
        }
        let h = self.norm.forward(&h)?;
        let h = h.apply(&self.proj_out)?;

        // Reshape back: [B*T, P, acoustic_dim] → [B, T*P, acoustic_dim]
        let acoustic_dim = h.dim(2)?;
        h.reshape((b, t * self.pool_window_size, acoustic_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_detokenizer_shape() {
        let dev = Device::Cpu;
        let cfg = AceStepConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            num_hidden_layers: 2,
            num_attention_pooler_hidden_layers: 1,
            pool_window_size: 5,
            audio_acoustic_hidden_dim: 8,
            ..AceStepConfig::default()
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let detok = AudioTokenDetokenizer::new(&cfg, DType::F32, &dev, vb.pp("detok")).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 4, 16), &dev).unwrap();
        let out = detok.forward(&x).unwrap();
        // 4 tokens * 5 pool_window = 20 frames
        assert_eq!(out.dims(), &[2, 20, 8]);
    }
}
