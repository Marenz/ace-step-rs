//! GQA attention with RoPE for ACE-Step v1.5.
//!
//! Implements Qwen3-style grouped query attention with:
//! - Q/K RMSNorm (per-head normalization)
//! - Rotary position embeddings
//! - Sliding window support (bidirectional)
//! - Cross-attention mode (no RoPE, attends to encoder states)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::config::{AceStepConfig, LayerType};

// ---------------------------------------------------------------------------
// RoPE (Qwen3-style rotary position embedding)
// ---------------------------------------------------------------------------

/// Precomputed rotary embedding cos/sin tables.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor, // [max_seq, head_dim/2]
    sin: Tensor, // [max_seq, head_dim/2]
}

impl RotaryEmbedding {
    pub fn new(cfg: &AceStepConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let max_seq = cfg.max_position_embeddings;
        let theta = cfg.rope_theta;
        let half = head_dim / 2;

        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / theta.powf(i as f64 / half as f64))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), dev)?; // [half]

        let positions: Vec<f64> = (0..max_seq).map(|p| p as f64).collect();
        let positions = Tensor::new(positions.as_slice(), dev)?; // [max_seq]

        // [max_seq, half]
        let freqs = positions
            .unsqueeze(1)?
            .broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    /// Get (cos, sin) for given position_ids: [1, seq_len] → ([1, seq_len, 1, head_dim/2], ...)
    pub fn forward(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.index_select(&position_ids.flatten_all()?, 0)?;
        let sin = self.sin.index_select(&position_ids.flatten_all()?, 0)?;
        // Reshape to [1, seq_len, 1, head_dim/2] for broadcasting with [B, seq, heads, head_dim]
        let seq_len = position_ids.dim(1)?;
        let half = self.cos.dim(1)?;
        let cos = cos.reshape((1, seq_len, 1, half))?;
        let sin = sin.reshape((1, seq_len, 1, half))?;
        Ok((cos, sin))
    }
}

/// Apply rotary position embedding to query and key tensors.
///
/// q, k: [B, seq_len, num_heads, head_dim]
/// cos, sin: [1, seq_len, 1, head_dim/2]
pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_rot = apply_rope_single(q, cos, sin)?;
    let k_rot = apply_rope_single(k, cos, sin)?;
    Ok((q_rot, k_rot))
}

fn apply_rope_single(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _s, _h, d) = x.dims4()?;
    let half = d / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    // [x1*cos - x2*sin, x1*sin + x2*cos]
    let out1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let out2 = (x1.broadcast_mul(sin)? + x2.broadcast_mul(cos)?)?;
    Tensor::cat(&[&out1, &out2], 3)
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// RMSNorm matching Qwen3RMSNorm — normalizes last dimension.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(xs, &self.weight, self.eps as f32)
    }
}

// ---------------------------------------------------------------------------
// GQA Attention
// ---------------------------------------------------------------------------

/// Grouped Query Attention with optional cross-attention and sliding window.
#[derive(Debug, Clone)]
pub struct AceStepAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
    is_cross_attention: bool,
    #[allow(dead_code)]
    sliding_window: Option<usize>,
}

impl AceStepAttention {
    pub fn new(
        cfg: &AceStepConfig,
        layer_idx: usize,
        is_cross_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let hidden = cfg.hidden_size;

        let q_proj = nn::linear_no_bias(hidden, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = nn::linear_no_bias(hidden, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = nn::linear_no_bias(hidden, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = nn::linear_no_bias(num_heads * head_dim, hidden, vb.pp("o_proj"))?;

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let sliding_window =
            if !is_cross_attention && cfg.layer_types[layer_idx] == LayerType::SlidingAttention {
                Some(cfg.sliding_window)
            } else {
                None
            };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            is_cross_attention,
            sliding_window,
        })
    }

    /// Forward pass.
    ///
    /// - `hidden_states`: [B, S, D]
    /// - `attention_mask`: optional [B, 1, S, S] additive mask
    /// - `encoder_hidden_states`: optional [B, E, D] for cross-attention
    /// - `position_embeddings`: (cos, sin) for RoPE (skipped for cross-attention)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        position_embeddings: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, s, _d) = hidden_states.dims3()?;

        // Project Q
        let q = hidden_states.apply(&self.q_proj)?;
        let q = q.reshape((b, s, self.num_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;

        // Determine K/V source
        let kv_input = if self.is_cross_attention {
            encoder_hidden_states.unwrap_or(hidden_states)
        } else {
            hidden_states
        };
        let kv_len = kv_input.dim(1)?;

        let k = kv_input.apply(&self.k_proj)?;
        let k = k.reshape((b, kv_len, self.num_kv_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?;

        let v = kv_input.apply(&self.v_proj)?;
        let v = v.reshape((b, kv_len, self.num_kv_heads, self.head_dim))?;

        // Apply RoPE (self-attention only)
        let (q, k) = if !self.is_cross_attention {
            if let Some((cos, sin)) = position_embeddings {
                apply_rotary_pos_emb(&q, &k, cos, sin)?
            } else {
                (q, k)
            }
        } else {
            (q, k)
        };

        // Expand KV heads for GQA: [B, S, num_kv_heads, D] → [B, S, num_heads, D]
        let num_groups = self.num_heads / self.num_kv_heads;
        let k = if num_groups > 1 {
            let k = k.unsqueeze(3)?; // [B, S, num_kv, 1, D]
            let k = k.expand((b, kv_len, self.num_kv_heads, num_groups, self.head_dim))?;
            k.reshape((b, kv_len, self.num_heads, self.head_dim))?
        } else {
            k
        };
        let v = if num_groups > 1 {
            let v = v.unsqueeze(3)?;
            let v = v.expand((b, kv_len, self.num_kv_heads, num_groups, self.head_dim))?;
            v.reshape((b, kv_len, self.num_heads, self.head_dim))?
        } else {
            v
        };

        // Transpose to [B, heads, S, D]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scaling)?;

        // Apply attention mask
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Transpose back: [B, heads, S, D] → [B, S, heads*D]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b,
            s,
            self.num_heads * self.head_dim,
        ))?;

        attn_output.apply(&self.o_proj)
    }
}

// ---------------------------------------------------------------------------
// SiLU MLP (Qwen3MLP)
// ---------------------------------------------------------------------------

/// Gated SiLU MLP matching Qwen3MLP: gate_proj * silu * up_proj → down_proj.
#[derive(Debug, Clone)]
pub struct SiluMlp {
    gate_proj: nn::Linear,
    up_proj: nn::Linear,
    down_proj: nn::Linear,
}

impl SiluMlp {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: nn::linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: nn::linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: nn::linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }
}

impl Module for SiluMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?.silu()?;
        let up = xs.apply(&self.up_proj)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_rope_shapes() {
        let cfg = AceStepConfig::default();
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(&cfg, DType::F32, &dev).unwrap();
        let pos_ids = Tensor::arange(0u32, 10, &dev)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let (cos, sin) = rope.forward(&pos_ids).unwrap();
        // [1, 10, 1, 64] (head_dim/2 = 128/2 = 64)
        assert_eq!(cos.dims(), &[1, 10, 1, 64]);
        assert_eq!(sin.dims(), &[1, 10, 1, 64]);
    }

    #[test]
    fn test_apply_rope() {
        let dev = Device::Cpu;
        let (b, s, h, d) = (1, 4, 2, 8);
        let q = Tensor::randn(0f32, 1.0, (b, s, h, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1.0, (b, s, h, d), &dev).unwrap();
        let cos = Tensor::ones((1, s, 1, d / 2), DType::F32, &dev).unwrap();
        let sin = Tensor::zeros((1, s, 1, d / 2), DType::F32, &dev).unwrap();
        let (q_rot, k_rot) = apply_rotary_pos_emb(&q, &k, &cos, &sin).unwrap();
        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());
    }

    #[test]
    fn test_rms_norm() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let norm = RmsNorm::new(4, 1e-6, vb).unwrap();
        let x = Tensor::ones((2, 3, 4), DType::F32, &dev).unwrap();
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_silu_mlp_shape() {
        let dev = Device::Cpu;
        let cfg = AceStepConfig {
            hidden_size: 16,
            intermediate_size: 32,
            ..AceStepConfig::default()
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let mlp = SiluMlp::new(&cfg, vb.pp("mlp")).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 5, 16), &dev).unwrap();
        let y = mlp.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }
}
