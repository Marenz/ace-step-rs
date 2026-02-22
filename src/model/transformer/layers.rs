//! Transformer layer types for ACE-Step v1.5.
//!
//! - [`AceStepEncoderLayer`] — pre-norm self-attention + MLP (used by lyric/timbre/pooler encoders)
//! - [`AceStepDiTLayer`] — AdaLN self-attention + cross-attention + MLP (used by DiT)

use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::attention::{AceStepAttention, RmsNorm, SiluMlp};
use crate::config::AceStepConfig;

// ---------------------------------------------------------------------------
// Encoder Layer (pre-norm, no cross-attention, no timestep modulation)
// ---------------------------------------------------------------------------

/// Standard transformer encoder layer: pre-norm self-attention + pre-norm MLP.
///
/// Used by lyric encoder, timbre encoder, attention pooler, and detokenizer.
#[derive(Debug, Clone)]
pub struct AceStepEncoderLayer {
    self_attn: AceStepAttention,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: SiluMlp,
    has_sliding_window: bool,
}

impl AceStepEncoderLayer {
    pub fn new(cfg: &AceStepConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: AceStepAttention::new(cfg, layer_idx, false, vb.pp("self_attn"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: SiluMlp::new(cfg, vb.pp("mlp"))?,
            has_sliding_window: cfg.layer_types[layer_idx]
                == crate::config::LayerType::SlidingAttention,
        })
    }

    /// Whether this layer uses sliding window attention.
    pub fn self_attn_has_sliding_window(&self) -> bool {
        self.has_sliding_window
    }

    /// Forward pass.
    ///
    /// - `hidden_states`: [B, S, D]
    /// - `attention_mask`: optional [B, 1, S, S]
    /// - `position_embeddings`: (cos, sin) from RoPE
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_embeddings: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        // Self-attention with residual
        let residual = hidden_states;
        let h = self.input_layernorm.forward(hidden_states)?;
        let h = self
            .self_attn
            .forward(&h, attention_mask, None, Some(position_embeddings))?;
        let h = (residual + h)?;

        // MLP with residual
        let residual = &h;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }
}

// ---------------------------------------------------------------------------
// DiT Layer (AdaLN self-attention + cross-attention + AdaLN MLP)
// ---------------------------------------------------------------------------

/// DiT transformer layer with adaptive layer normalization from timestep embeddings.
///
/// Three sub-layers:
/// 1. AdaLN self-attention (scale/shift/gate from timestep)
/// 2. Cross-attention to encoder hidden states
/// 3. AdaLN MLP (scale/shift/gate from timestep)
#[derive(Debug, Clone)]
pub struct AceStepDiTLayer {
    // Self-attention
    self_attn_norm: RmsNorm,
    self_attn: AceStepAttention,
    // Cross-attention
    cross_attn_norm: RmsNorm,
    cross_attn: AceStepAttention,
    // MLP
    mlp_norm: RmsNorm,
    mlp: SiluMlp,
    // AdaLN scale-shift-gate table: [1, 6, hidden_size]
    scale_shift_table: Tensor,
    // Whether this layer uses sliding window attention
    has_sliding_window: bool,
}

impl AceStepDiTLayer {
    pub fn new(cfg: &AceStepConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("self_attn_norm"),
            )?,
            self_attn: AceStepAttention::new(cfg, layer_idx, false, vb.pp("self_attn"))?,
            cross_attn_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("cross_attn_norm"),
            )?,
            cross_attn: AceStepAttention::new(cfg, layer_idx, true, vb.pp("cross_attn"))?,
            mlp_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("mlp_norm"))?,
            mlp: SiluMlp::new(cfg, vb.pp("mlp"))?,
            scale_shift_table: vb.get((1, 6, cfg.hidden_size), "scale_shift_table")?,
            has_sliding_window: cfg.layer_types[layer_idx]
                == crate::config::LayerType::SlidingAttention,
        })
    }

    /// Whether this layer uses sliding window attention.
    pub fn self_attn_has_sliding_window(&self) -> bool {
        self.has_sliding_window
    }

    /// Forward pass.
    ///
    /// - `hidden_states`: [B, S, D]
    /// - `temb`: timestep projection [B, 6, D] (chunked into 6 modulation params)
    /// - `self_attn_mask`: optional [B, 1, S, S]
    /// - `encoder_hidden_states`: [B, E, D]
    /// - `encoder_attn_mask`: optional [B, 1, S, E]
    /// - `position_embeddings`: (cos, sin) for RoPE
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        temb: &Tensor,
        self_attn_mask: Option<&Tensor>,
        encoder_hidden_states: &Tensor,
        encoder_attn_mask: Option<&Tensor>,
        position_embeddings: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        // Extract 6 modulation parameters from scale_shift_table + temb
        // temb: [B, 6, D], scale_shift_table: [1, 6, D]
        let modulation = self.scale_shift_table.broadcast_add(temb)?;

        let shift_msa = modulation.i((.., 0, ..))?; // [B, D]
        let scale_msa = modulation.i((.., 1, ..))?;
        let gate_msa = modulation.i((.., 2, ..))?;
        let c_shift_msa = modulation.i((.., 3, ..))?;
        let c_scale_msa = modulation.i((.., 4, ..))?;
        let c_gate_msa = modulation.i((.., 5, ..))?;

        // Unsqueeze for broadcasting: [B, D] → [B, 1, D]
        let shift_msa = shift_msa.unsqueeze(1)?;
        let scale_msa = scale_msa.unsqueeze(1)?;
        let gate_msa = gate_msa.unsqueeze(1)?;
        let c_shift_msa = c_shift_msa.unsqueeze(1)?;
        let c_scale_msa = c_scale_msa.unsqueeze(1)?;
        let c_gate_msa = c_gate_msa.unsqueeze(1)?;

        // 1. Self-attention with AdaLN: norm(x) * (1 + scale) + shift
        let norm_h = self.self_attn_norm.forward(hidden_states)?;
        let scale_plus_one = (scale_msa + 1.0)?;
        let norm_h = norm_h
            .broadcast_mul(&scale_plus_one)?
            .broadcast_add(&shift_msa)?;
        let attn_out =
            self.self_attn
                .forward(&norm_h, self_attn_mask, None, Some(position_embeddings))?;
        // Gated residual: x = x + attn * gate
        let h = (hidden_states + attn_out.broadcast_mul(&gate_msa)?)?;

        // 2. Cross-attention (no AdaLN, just pre-norm + residual)
        let norm_h = self.cross_attn_norm.forward(&h)?;
        let cross_out = self.cross_attn.forward(
            &norm_h,
            encoder_attn_mask,
            Some(encoder_hidden_states),
            None,
        )?;
        let h = (&h + cross_out)?;

        // 3. MLP with AdaLN: norm(x) * (1 + scale) + shift
        let norm_h = self.mlp_norm.forward(&h)?;
        let c_scale_plus_one = (c_scale_msa + 1.0)?;
        let norm_h = norm_h
            .broadcast_mul(&c_scale_plus_one)?
            .broadcast_add(&c_shift_msa)?;
        let ff_out = self.mlp.forward(&norm_h)?;
        // Gated residual: x = x + mlp * gate
        &h + ff_out.broadcast_mul(&c_gate_msa)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn small_cfg() -> AceStepConfig {
        AceStepConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            num_hidden_layers: 2,
            ..AceStepConfig::default()
        }
    }

    #[test]
    fn test_encoder_layer_shape() {
        let dev = Device::Cpu;
        let cfg = small_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let layer = AceStepEncoderLayer::new(&cfg, 0, vb.pp("layer")).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 8, 16), &dev).unwrap();
        let rope = super::super::attention::RotaryEmbedding::new(&cfg, DType::F32, &dev).unwrap();
        let pos_ids = Tensor::arange(0u32, 8, &dev).unwrap().unsqueeze(0).unwrap();
        let pos_emb = rope.forward(&pos_ids).unwrap();
        let y = layer.forward(&x, None, &pos_emb).unwrap();
        assert_eq!(y.dims(), &[2, 8, 16]);
    }

    #[test]
    fn test_dit_layer_shape() {
        let dev = Device::Cpu;
        let cfg = small_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let layer = AceStepDiTLayer::new(&cfg, 0, vb.pp("layer")).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 8, 16), &dev).unwrap();
        let temb = Tensor::randn(0f32, 1.0, (2, 6, 16), &dev).unwrap();
        let enc = Tensor::randn(0f32, 1.0, (2, 12, 16), &dev).unwrap();
        let rope = super::super::attention::RotaryEmbedding::new(&cfg, DType::F32, &dev).unwrap();
        let pos_ids = Tensor::arange(0u32, 8, &dev).unwrap().unsqueeze(0).unwrap();
        let pos_emb = rope.forward(&pos_ids).unwrap();
        let y = layer
            .forward(&x, &temb, None, &enc, None, &pos_emb)
            .unwrap();
        assert_eq!(y.dims(), &[2, 8, 16]);
    }
}
