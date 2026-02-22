//! Attention-based pooler for 25Hz → 5Hz downsampling.
//!
//! Pools groups of `pool_window_size=5` frames using a CLS-like special token
//! and self-attention. Used by the audio tokenizer.

use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::AceStepConfig;
use crate::model::transformer::attention::{RmsNorm, RotaryEmbedding};
use crate::model::transformer::layers::AceStepEncoderLayer;

/// Attention pooler matching Python `AttentionPooler`.
#[derive(Debug, Clone)]
pub struct AttentionPooler {
    embed_tokens: candle_nn::Linear,
    norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    special_token: Tensor,
    layers: Vec<AceStepEncoderLayer>,
    cfg: AceStepConfig,
}

impl AttentionPooler {
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
        let special_token = vb.get((1, 1, cfg.hidden_size), "special_token")?;

        let mut layers = Vec::with_capacity(cfg.num_attention_pooler_hidden_layers);
        for i in 0..cfg.num_attention_pooler_hidden_layers {
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
    /// Input: [B, T, pool_window_size, D] — grouped patches
    /// Output: [B, T, D] — pooled representations (CLS token output)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, p, _d) = x.dims4()?;
        let x = x.apply(&self.embed_tokens)?;

        // Prepend special token: [B, T, 1, D]
        let special = self
            .special_token
            .broadcast_as((b, t, 1, self.cfg.hidden_size))?;
        let x = Tensor::cat(&[&special, &x], 2)?; // [B, T, P+1, D]

        // Reshape for processing: [B*T, P+1, D]
        let bt = b * t;
        let p_plus_1 = p + 1;
        let d = self.cfg.hidden_size;
        let x = x.reshape((bt, p_plus_1, d))?;

        let pos_ids = Tensor::arange(0u32, p_plus_1 as u32, x.device())?.unsqueeze(0)?;
        let pos_emb = self.rotary_emb.forward(&pos_ids)?;

        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(&h, None, &pos_emb)?;
        }
        let h = self.norm.forward(&h)?;

        // Extract CLS token (position 0): [B*T, D]
        let cls = h.i((.., 0, ..))?;
        cls.reshape((b, t, d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_pooler_shape() {
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
            ..AceStepConfig::default()
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let pooler = AttentionPooler::new(&cfg, DType::F32, &dev, vb.pp("pooler")).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 4, 5, 16), &dev).unwrap();
        let out = pooler.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 4, 16]);
    }
}
