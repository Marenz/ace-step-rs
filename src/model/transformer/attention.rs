//! Attention implementations for ACE-Step.
//!
//! Two types:
//! - [`LinearAttention`] — ReLU-kernel linear attention for self-attention (O(n·d²))
//! - [`CrossAttention`] — standard scaled dot-product attention for cross-attention

use candle_core::{DType, Module, Tensor};
use candle_nn::VarBuilder;

use crate::Result;

/// Linear attention with ReLU kernel.
///
/// Instead of softmax attention O(n²·d), uses:
/// ```text
/// φ(q) = ReLU(q),  φ(k) = ReLU(k)
/// O = (V^T · φ(K)) · φ(Q)  normalized
/// ```
/// giving O(n·d²) complexity — critical for long audio sequences.
pub struct LinearAttention {
    to_q: candle_nn::Linear,
    to_k: candle_nn::Linear,
    to_v: candle_nn::Linear,
    to_out: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl LinearAttention {
    pub fn load(vb: VarBuilder, dim: usize, num_heads: usize, head_dim: usize) -> Result<Self> {
        let inner_dim = num_heads * head_dim;
        let to_q = candle_nn::linear(dim, inner_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear(dim, inner_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear(dim, inner_dim, vb.pp("to_v"))?;
        let to_out = candle_nn::linear(inner_dim, dim, vb.pp("to_out.0"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass.
    ///
    /// - `hidden_states`: `[B, S, dim]`
    /// - `attention_mask`: `[B, S]` (1=valid, 0=padding)
    /// - `rope_cos`, `rope_sin`: `[S, head_dim]`
    ///
    /// Returns `[B, S, dim]`.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _dim) = hidden_states.dims3()?;

        // Project to Q, K, V: [B, S, inner_dim]
        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(hidden_states)?;
        let v = self.to_v.forward(hidden_states)?;

        // Reshape to [B, H, S, D]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        let q = super::rope::RotaryEmbedding::apply(&q, rope_cos, rope_sin)?;
        let k = super::rope::RotaryEmbedding::apply(&k, rope_cos, rope_sin)?;

        // Transpose to [B, H, D, S] for linear attention kernel
        let q = q.transpose(2, 3)?; // [B, H, D, S]
        let k = k.transpose(2, 3)?; // [B, H, D, S]
        let v = v.transpose(2, 3)?; // [B, H, D, S]

        // Apply mask by zeroing padded positions
        let (q, k, v) = if let Some(mask) = attention_mask {
            // mask: [B, S] → [B, 1, 1, S]
            let mask = mask.unsqueeze(1)?.unsqueeze(1)?.to_dtype(q.dtype())?;
            (
                q.broadcast_mul(&mask)?,
                k.broadcast_mul(&mask)?,
                v.broadcast_mul(&mask)?,
            )
        } else {
            (q, k, v)
        };

        // ReLU kernel: φ(x) = max(0, x)
        let q = q.relu()?;
        let k = k.relu()?;

        // Cast to f32 for numerical stability
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // Pad V with a normalization row: [B, H, D+1, S]
        let ones = Tensor::ones((batch, self.num_heads, 1, seq_len), DType::F32, v.device())?;
        let v_padded = Tensor::cat(&[&v, &ones], 2)?; // [B, H, D+1, S]

        // Linear attention: VK = V^T · K, then VK · Q
        let vk = v_padded.matmul(&k.transpose(2, 3)?)?; // [B, H, D+1, D]
        let out = vk.matmul(&q)?; // [B, H, D+1, S]

        // Normalize: out[:-1] / out[-1:]
        let out_main = out.narrow(2, 0, self.head_dim)?; // [B, H, D, S]
        let out_norm = out.narrow(2, self.head_dim, 1)?; // [B, H, 1, S]
        let out = out_main.broadcast_div(&(out_norm + 1e-15)?)?;

        // Reshape back: [B, H, D, S] → [B, S, H*D]
        let out = out
            .transpose(2, 3)? // [B, H, S, D]
            .transpose(1, 2)? // [B, S, H, D]
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        let out = out.to_dtype(hidden_states.dtype())?;
        self.to_out.forward(&out).map_err(Into::into)
    }
}

/// Standard scaled dot-product cross-attention.
///
/// Query from latent tokens, key/value from encoder context.
pub struct CrossAttention {
    to_q: candle_nn::Linear,
    to_k: candle_nn::Linear,
    to_v: candle_nn::Linear,
    to_out: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    pub fn load(
        vb: VarBuilder,
        query_dim: usize,
        cross_dim: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let inner_dim = num_heads * head_dim;
        let to_q = candle_nn::linear(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear(cross_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear(cross_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = candle_nn::linear(inner_dim, query_dim, vb.pp("to_out.0"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass.
    ///
    /// - `hidden_states`: `[B, S_lat, dim]` — latent tokens
    /// - `encoder_hidden_states`: `[B, S_enc, cross_dim]` — encoder context
    /// - `attention_mask`: `[B, S_lat]` — latent mask
    /// - `encoder_attention_mask`: `[B, S_enc]` — encoder mask
    /// - `rope_cos_q`, `rope_sin_q`: `[S_lat, head_dim]` — RoPE for query
    /// - `rope_cos_k`, `rope_sin_k`: `[S_enc, head_dim]` — RoPE for key
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        rope_cos_q: &Tensor,
        rope_sin_q: &Tensor,
        rope_cos_k: &Tensor,
        rope_sin_k: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_lat, _) = hidden_states.dims3()?;
        let (_, seq_enc, _) = encoder_hidden_states.dims3()?;

        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(encoder_hidden_states)?;
        let v = self.to_v.forward(encoder_hidden_states)?;

        // Reshape to [B, H, S, D]
        let q = q
            .reshape((batch, seq_lat, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_enc, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_enc, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // RoPE: query uses latent freqs, key uses encoder freqs.
        let q = super::rope::RotaryEmbedding::apply(&q, rope_cos_q, rope_sin_q)?;
        let k = super::rope::RotaryEmbedding::apply(&k, rope_cos_k, rope_sin_k)?;

        let scale = (self.head_dim as f64).sqrt();

        // Attention scores: Q · K^T / sqrt(d)
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;

        // Build combined mask: attn_mask[:,:,None] * enc_mask[:,None,:]
        let attn_weights = match (attention_mask, encoder_attention_mask) {
            (Some(q_mask), Some(k_mask)) => {
                // q_mask: [B, S_lat] → [B, 1, S_lat, 1]
                // k_mask: [B, S_enc] → [B, 1, 1, S_enc]
                let q_mask = q_mask.unsqueeze(1)?.unsqueeze(3)?;
                let k_mask = k_mask.unsqueeze(1)?.unsqueeze(2)?;
                let combined = q_mask.broadcast_mul(&k_mask)?;
                // Where combined == 0, set to -inf.
                // Build additive bias: 0 where mask=1, -inf where mask=0.
                // (mask - 1) * inf gives -inf for 0s and 0 for 1s.
                let bias = ((combined - 1.0)? * f64::INFINITY)?;
                let bias = bias.to_dtype(attn_weights.dtype())?;
                attn_weights.broadcast_add(&bias)?
            }
            _ => attn_weights,
        };

        // Softmax + V
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let out = attn_weights.matmul(&v)?; // [B, H, S_lat, D]

        // Reshape: [B, S_lat, H*D]
        let out = out
            .transpose(1, 2)? // [B, S_lat, H, D]
            .reshape((batch, seq_lat, self.num_heads * self.head_dim))?;

        self.to_out.forward(&out).map_err(Into::into)
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
    fn linear_attention_output_shape() {
        let device = Device::Cpu;
        let (_varmap, vb) = make_vb(&device);

        let attn = LinearAttention::load(vb, 256, 4, 64).unwrap();

        let x = Tensor::randn(0.0_f32, 1.0, (1, 32, 256), &device).unwrap();
        let rope = super::super::rope::RotaryEmbedding::new(64, 1_000_000.0);
        let (cos, sin) = rope.compute_freqs(32, DType::F32, &device).unwrap();

        let out = attn.forward(&x, None, &cos, &sin).unwrap();
        assert_eq!(out.dims(), &[1, 32, 256]);
    }

    #[test]
    fn cross_attention_output_shape() {
        let device = Device::Cpu;
        let (_varmap, vb) = make_vb(&device);

        let attn = CrossAttention::load(vb, 256, 256, 4, 64).unwrap();

        let q_input = Tensor::randn(0.0_f32, 1.0, (1, 32, 256), &device).unwrap();
        let kv_input = Tensor::randn(0.0_f32, 1.0, (1, 16, 256), &device).unwrap();
        let rope = super::super::rope::RotaryEmbedding::new(64, 1_000_000.0);
        let (cos_q, sin_q) = rope.compute_freqs(32, DType::F32, &device).unwrap();
        let (cos_k, sin_k) = rope.compute_freqs(16, DType::F32, &device).unwrap();

        let out = attn
            .forward(
                &q_input, &kv_input, None, None, &cos_q, &sin_q, &cos_k, &sin_k,
            )
            .unwrap();
        assert_eq!(out.dims(), &[1, 32, 256]);
    }
}
