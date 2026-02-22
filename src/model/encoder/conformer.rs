//! Conformer encoder for lyric conditioning.
//!
//! Despite the name, the actual ACE-Step instantiation uses `macaron_style=False`
//! and `use_cnn_module=False`, making this effectively a **plain Transformer encoder
//! with Espnet-style relative position attention**. No convolution modules or
//! macaron-style dual FFN are active.
//!
//! ## Architecture
//!
//! ```text
//! LinearEmbed:
//!   Linear(1024, 1024) → LayerNorm(1024) → Dropout(0.1)
//!   → scale by √1024 → EspnetRelPositionalEncoding
//!   → (xs: [B, T, 1024], pos_emb: [1, 2T-1, 1024])
//!
//! 6 × ConformerEncoderLayer (pre-norm):
//!   1. norm_mha → RelPositionMHA(16 heads, 64 d_k) → dropout → residual
//!   2. norm_ff  → FFN(1024→4096→1024, SiLU) → dropout → residual
//!
//! after_norm: LayerNorm(1024)
//! ```
//!
//! ## Relative Position Attention (Espnet-style)
//!
//! From Transformer-XL (Dai et al. 2019). Attention scores:
//! ```text
//! matrix_ac = (Q + pos_bias_u) · K^T     (content-to-content)
//! matrix_bd = (Q + pos_bias_v) · P^T     (content-to-position)
//! scores = (matrix_ac + rel_shift(matrix_bd)) / √d_k
//! ```
//!
//! ## Weight key paths (under `lyric_encoder.` prefix)
//!
//! ```text
//! embed.out.0.{weight,bias}         — Linear(1024, 1024)
//! embed.out.1.{weight,bias}         — LayerNorm(1024)
//! after_norm.{weight,bias}          — LayerNorm(1024)
//! encoders.{i}.self_attn.linear_q.{weight,bias}
//! encoders.{i}.self_attn.linear_k.{weight,bias}
//! encoders.{i}.self_attn.linear_v.{weight,bias}
//! encoders.{i}.self_attn.linear_out.{weight,bias}
//! encoders.{i}.self_attn.linear_pos.weight      (no bias)
//! encoders.{i}.self_attn.pos_bias_u             (16, 64)
//! encoders.{i}.self_attn.pos_bias_v             (16, 64)
//! encoders.{i}.norm_mha.{weight,bias}
//! encoders.{i}.feed_forward.w_1.{weight,bias}   — Linear(1024, 4096)
//! encoders.{i}.feed_forward.w_2.{weight,bias}   — Linear(4096, 1024)
//! encoders.{i}.norm_ff.{weight,bias}
//! ```

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

use crate::Result;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Conformer lyric encoder.
#[derive(Debug, Clone)]
pub struct ConformerConfig {
    /// Input/output feature size.
    pub output_size: usize,
    /// Number of attention heads.
    pub attention_heads: usize,
    /// FFN inner size.
    pub linear_units: usize,
    /// Number of encoder blocks.
    pub num_blocks: usize,
    /// Dropout rate (applied in training; ignored during inference).
    pub dropout_rate: f64,
    /// Attention dropout rate.
    pub attention_dropout_rate: f64,
}

impl Default for ConformerConfig {
    fn default() -> Self {
        Self {
            output_size: 1024,
            attention_heads: 16,
            linear_units: 4096,
            num_blocks: 6,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Espnet Relative Positional Encoding
// ---------------------------------------------------------------------------

/// Espnet-style relative positional encoding.
///
/// Produces a `[1, 2T-1, d_model]` tensor covering both positive (left)
/// and negative (right) relative positions. The input is scaled by `√d_model`.
///
/// No learnable parameters — the PE is computed on-the-fly.
pub struct EspnetRelPositionalEncoding {
    d_model: usize,
    xscale: f64,
}

impl EspnetRelPositionalEncoding {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            xscale: (d_model as f64).sqrt(),
        }
    }

    /// Compute relative positional encoding for sequence length `size`.
    ///
    /// Returns `pos_emb: [1, 2*size - 1, d_model]`.
    pub fn compute(&self, size: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        let d = self.d_model;
        let half_d = d / 2;

        // div_term = exp(arange(0, d, 2) * -(ln(10000) / d))
        let div_term: Vec<f32> = (0..half_d)
            .map(|i| (-(i as f64) * (10000.0_f64).ln() / d as f64).exp() as f32)
            .collect();
        let div_term = Tensor::from_vec(div_term, (1, half_d), device)?;

        // position = arange(0, size)
        let position: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let position = Tensor::from_vec(position, (size, 1), device)?;

        // args = position * div_term → [size, half_d]
        let args = position.broadcast_mul(&div_term)?;

        // pe_positive: sin/cos interleaved → [size, d_model]
        let sin_pos = args.sin()?;
        let cos_pos = args.cos()?;
        // Interleave: [sin0, cos0, sin1, cos1, ...] by stacking and reshaping
        let sin_pos = sin_pos.unsqueeze(2)?; // [size, half_d, 1]
        let cos_pos = cos_pos.unsqueeze(2)?; // [size, half_d, 1]
        let interleaved = Tensor::cat(&[&sin_pos, &cos_pos], 2)?; // [size, half_d, 2]
        let pe_positive = interleaved.reshape((size, d))?; // [size, d_model]

        // pe_negative: sin(-pos)/cos(-pos) = [-sin(pos), cos(pos)]
        let neg_sin = sin_pos.neg()?;
        let neg_interleaved = Tensor::cat(&[&neg_sin, &cos_pos], 2)?;
        let pe_negative = neg_interleaved.reshape((size, d))?;

        // Flip positive (so position 0 is last, position size-1 is first)
        // Then concat: [flipped_positive, pe_negative[1:]]
        // Result shape: [2*size - 1, d_model]
        let pe_positive_flipped = pe_positive.flip(&[0])?;
        let pe_negative_tail = pe_negative.narrow(0, 1, size - 1)?;
        let pe = Tensor::cat(&[&pe_positive_flipped, &pe_negative_tail], 0)?;
        let pe = pe.unsqueeze(0)?.to_dtype(dtype)?; // [1, 2*size - 1, d_model]

        Ok(pe)
    }

    /// Extract the relevant slice for sequence length `size`.
    ///
    /// From a precomputed PE of shape `[1, L, d]`, extracts `[1, 2*size-1, d]`.
    pub fn extract_slice(pe: &Tensor, size: usize) -> Result<Tensor> {
        let l = pe.dim(1)?;
        let center = l / 2;
        let start = center + 1 - size;
        let len = 2 * size - 1;
        Ok(pe.narrow(1, start, len)?)
    }

    /// Forward: scale input and compute positional encoding.
    ///
    /// - `x`: `[B, T, d_model]`
    ///
    /// Returns `(scaled_x, pos_emb)` where:
    /// - `scaled_x`: `[B, T, d_model]` (x * √d_model)
    /// - `pos_emb`: `[1, 2T-1, d_model]`
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let size = x.dim(1)?;
        let scaled_x = (x * self.xscale)?;
        let pos_emb = self.compute(size, x.dtype(), x.device())?;
        Ok((scaled_x, pos_emb))
    }
}

// ---------------------------------------------------------------------------
// Relative Position Multi-Headed Attention
// ---------------------------------------------------------------------------

/// Multi-headed attention with Transformer-XL style relative position encoding.
///
/// Uses learnable biases `pos_bias_u` and `pos_bias_v` and a `linear_pos`
/// projection for the positional encoding.
pub struct RelPositionMultiHeadedAttention {
    num_heads: usize,
    head_dim: usize,
    linear_q: candle_nn::Linear,
    linear_k: candle_nn::Linear,
    linear_v: candle_nn::Linear,
    linear_out: candle_nn::Linear,
    linear_pos: candle_nn::Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

impl RelPositionMultiHeadedAttention {
    pub fn load(vb: VarBuilder, n_feat: usize, n_head: usize) -> Result<Self> {
        let d_k = n_feat / n_head;

        let linear_q = candle_nn::linear(n_feat, n_feat, vb.pp("linear_q"))?;
        let linear_k = candle_nn::linear(n_feat, n_feat, vb.pp("linear_k"))?;
        let linear_v = candle_nn::linear(n_feat, n_feat, vb.pp("linear_v"))?;
        let linear_out = candle_nn::linear(n_feat, n_feat, vb.pp("linear_out"))?;
        let linear_pos = candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_pos"))?;
        let pos_bias_u = vb.get((n_head, d_k), "pos_bias_u")?;
        let pos_bias_v = vb.get((n_head, d_k), "pos_bias_v")?;

        Ok(Self {
            num_heads: n_head,
            head_dim: d_k,
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
        })
    }

    /// Relative shift (skew trick) for matrix_bd.
    ///
    /// Input: `[B, H, T1, 2*T1-1]` → Output: `[B, H, T1, T1]`
    ///
    /// Pads a zero column, reshapes, slices to extract proper relative positions.
    fn rel_shift(x: &Tensor) -> Result<Tensor> {
        let (b, h, t1, t2) = x.dims4()?; // t2 = 2*t1-1

        // Pad with zero column on the left: [B, H, T1, 1]
        let zero_pad = Tensor::zeros((b, h, t1, 1), x.dtype(), x.device())?;
        let x_padded = Tensor::cat(&[&zero_pad, x], 3)?; // [B, H, T1, t2+1]

        // Reshape: [B, H, t2+1, T1]
        let x_padded = x_padded.reshape((b, h, t2 + 1, t1))?;

        // Skip first row, view as original shape: [B, H, T1, t2]
        let x = x_padded.narrow(2, 1, t2)?.reshape((b, h, t1, t2))?;

        // Keep only positions 0..T1 (= t2 // 2 + 1)
        let keep = t2 / 2 + 1; // = T1
        let x = x.narrow(3, 0, keep)?;

        Ok(x)
    }

    /// Forward pass.
    ///
    /// - `query`, `key`, `value`: `[B, T, n_feat]` (self-attention: all same)
    /// - `pos_emb`: `[1, 2T-1, n_feat]` from EspnetRelPositionalEncoding
    /// - `mask`: `[B, 1, T]` boolean-like mask (1=attend, 0=mask)
    ///
    /// Returns `[B, T, n_feat]`.
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        pos_emb: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, time1, _) = query.dims3()?;
        let h = self.num_heads;
        let d_k = self.head_dim;

        // Q, K, V projections: [B, T, n_feat] → [B, H, T, d_k]
        let q = self
            .linear_q
            .forward(query)?
            .reshape((batch, time1, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .linear_k
            .forward(key)?
            .reshape((batch, time1, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(value)?
            .reshape((batch, time1, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        // In the Python code, q is transposed to (batch, time1, head, d_k)
        // then pos_bias_u/v are added, then transposed back to (batch, head, time1, d_k).
        // q shape here: [B, H, T, d_k], pos_bias_u: [H, d_k]

        // (q + pos_bias_u) → [B, H, T, d_k]
        let bias_u = self.pos_bias_u.unsqueeze(0)?.unsqueeze(2)?; // [1, H, 1, d_k]
        let q_with_bias_u = q.broadcast_add(&bias_u)?;

        // (q + pos_bias_v) → [B, H, T, d_k]
        let bias_v = self.pos_bias_v.unsqueeze(0)?.unsqueeze(2)?;
        let q_with_bias_v = q.broadcast_add(&bias_v)?;

        // Positional encoding projection: [1, 2T-1, n_feat] → [1, H, 2T-1, d_k]
        let p_len = pos_emb.dim(1)?;
        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((1, p_len, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        // matrix_ac = (Q + bias_u) · K^T → [B, H, T, T]
        let matrix_ac = q_with_bias_u.broadcast_matmul(&k.transpose(2, 3)?.contiguous()?)?;

        // matrix_bd = (Q + bias_v) · P^T → [B, H, T, 2T-1]
        // P has batch=1, needs broadcast to match Q's batch size
        let matrix_bd = q_with_bias_v.broadcast_matmul(&p.transpose(2, 3)?.contiguous()?)?;

        // Apply rel_shift if shapes differ (they always do with espnet PE)
        let matrix_bd = if matrix_ac.dims() != matrix_bd.dims() {
            Self::rel_shift(&matrix_bd)?
        } else {
            matrix_bd
        };

        // scores = (ac + bd) / √d_k
        let scale = (d_k as f64).sqrt();
        let scores = ((matrix_ac + matrix_bd)? / scale)?;

        // Apply mask: where mask==0, set scores to -inf
        let scores = match mask {
            Some(mask) => {
                // mask: [B, 1, T] → additive bias
                // Python: mask.unsqueeze(1).eq(0) → masked_fill(-inf)
                // We use the additive approach: (mask - 1) * inf
                let mask_unsqueezed = mask.unsqueeze(1)?; // [B, 1, 1, T]
                let bias = ((mask_unsqueezed - 1.0)? * f64::INFINITY)?;
                scores.broadcast_add(&bias)?
            }
            None => scores,
        };

        // Softmax + matmul with V
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?; // [B, H, T, d_k]

        // Reshape back: [B, T, n_feat]
        let out = out
            .transpose(1, 2)? // [B, T, H, d_k]
            .reshape((batch, time1, h * d_k))?;

        self.linear_out.forward(&out).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Positionwise Feed-Forward
// ---------------------------------------------------------------------------

/// Standard feed-forward: Linear(dim, hidden) → SiLU → Linear(hidden, dim).
pub struct PositionwiseFeedForward {
    w_1: candle_nn::Linear,
    w_2: candle_nn::Linear,
}

impl PositionwiseFeedForward {
    pub fn load(vb: VarBuilder, dim: usize, hidden: usize) -> Result<Self> {
        let w_1 = candle_nn::linear(dim, hidden, vb.pp("w_1"))?;
        let w_2 = candle_nn::linear(hidden, dim, vb.pp("w_2"))?;
        Ok(Self { w_1, w_2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.w_1.forward(x)?;
        let x = x.apply(&candle_nn::Activation::Silu)?;
        self.w_2.forward(&x).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Conformer Encoder Layer
// ---------------------------------------------------------------------------

/// Single encoder layer: pre-norm MHA → pre-norm FFN.
///
/// Since macaron_style=False and use_cnn_module=False, this is a standard
/// Transformer encoder layer with relative position attention.
pub struct ConformerEncoderLayer {
    norm_mha: candle_nn::LayerNorm,
    self_attn: RelPositionMultiHeadedAttention,
    norm_ff: candle_nn::LayerNorm,
    feed_forward: PositionwiseFeedForward,
}

impl ConformerEncoderLayer {
    pub fn load(vb: VarBuilder, config: &ConformerConfig) -> Result<Self> {
        let dim = config.output_size;
        let norm_mha = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm_mha"))?;
        let self_attn =
            RelPositionMultiHeadedAttention::load(vb.pp("self_attn"), dim, config.attention_heads)?;
        let norm_ff = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm_ff"))?;
        let feed_forward =
            PositionwiseFeedForward::load(vb.pp("feed_forward"), dim, config.linear_units)?;

        Ok(Self {
            norm_mha,
            self_attn,
            norm_ff,
            feed_forward,
        })
    }

    /// Forward pass (pre-norm style).
    ///
    /// - `x`: `[B, T, dim]`
    /// - `mask`: `[B, 1, T]` (1=attend, 0=mask)
    /// - `pos_emb`: `[1, 2T-1, dim]`
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, pos_emb: &Tensor) -> Result<Tensor> {
        // 1. MHA sub-block (pre-norm)
        let residual = x.clone();
        let x_normed = self.norm_mha.forward(x)?;
        let x_att = self
            .self_attn
            .forward(&x_normed, &x_normed, &x_normed, pos_emb, mask)?;
        let x = (residual + x_att)?;

        // 2. FFN sub-block (pre-norm)
        let residual = x.clone();
        let x_normed = self.norm_ff.forward(&x)?;
        let ff_out = self.feed_forward.forward(&x_normed)?;
        let x = (residual + ff_out)?;

        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Conformer Encoder (top-level)
// ---------------------------------------------------------------------------

/// Conformer encoder: LinearEmbed + N layers + after_norm.
///
/// Takes lyric embeddings `[B, T, 1024]` and returns encoded features `[B, T, 1024]`.
pub struct ConformerEncoder {
    /// LinearEmbed: Linear(input, output) + LayerNorm + Dropout(implicit).
    embed_linear: candle_nn::Linear,
    embed_norm: candle_nn::LayerNorm,
    /// Relative positional encoding.
    pos_enc: EspnetRelPositionalEncoding,
    /// Encoder layers.
    layers: Vec<ConformerEncoderLayer>,
    /// Final LayerNorm.
    after_norm: candle_nn::LayerNorm,
    #[allow(dead_code)]
    config: ConformerConfig,
}

impl ConformerEncoder {
    /// Load weights from safetensors.
    ///
    /// The `vb` should be scoped to the `lyric_encoder` namespace.
    pub fn load(vb: VarBuilder, config: &ConformerConfig) -> Result<Self> {
        let dim = config.output_size;

        // LinearEmbed: embed.out.0 = Linear, embed.out.1 = LayerNorm
        let embed_linear = candle_nn::linear(dim, dim, vb.pp("embed.out.0"))?;
        let embed_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("embed.out.1"))?;

        let pos_enc = EspnetRelPositionalEncoding::new(dim);

        let mut layers = Vec::with_capacity(config.num_blocks);
        for i in 0..config.num_blocks {
            let layer = ConformerEncoderLayer::load(vb.pp(format!("encoders.{i}")), config)?;
            layers.push(layer);
        }

        let after_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("after_norm"))?;

        Ok(Self {
            embed_linear,
            embed_norm,
            pos_enc,
            layers,
            after_norm,
            config: config.clone(),
        })
    }

    /// Forward pass.
    ///
    /// - `xs`: `[B, T, 1024]` lyric embeddings (from `lyric_embs` Embedding)
    /// - `mask`: `[B, T]` padding mask (1=valid, 0=padding)
    ///
    /// Returns `[B, T, 1024]` encoded features.
    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // LinearEmbed: Linear → LayerNorm (dropout omitted at inference)
        let xs = self.embed_linear.forward(xs)?;
        let xs = self.embed_norm.forward(&xs)?;

        // Positional encoding: scale + compute PE
        let (xs, pos_emb) = self.pos_enc.forward(&xs)?;

        // Mask: [B, T] → [B, 1, T] for attention masking
        let mask_3d = mask.unsqueeze(1)?; // [B, 1, T]

        // Encoder layers (full bidirectional attention: chunk_masks = masks)
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs, Some(&mask_3d), &pos_emb)?;
        }

        // Final LayerNorm
        self.after_norm.forward(&xs).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn make_vb(device: &Device) -> (VarMap, VarBuilder<'static>) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        (varmap, vb)
    }

    #[test]
    fn conformer_config_defaults() {
        let config = ConformerConfig::default();
        assert_eq!(config.output_size, 1024);
        assert_eq!(config.attention_heads, 16);
        assert_eq!(config.num_blocks, 6);
        assert_eq!(config.output_size / config.attention_heads, 64); // head_dim
    }

    #[test]
    fn espnet_rel_pos_encoding_shape() {
        let enc = EspnetRelPositionalEncoding::new(64);
        let pe = enc.compute(32, DType::F32, &Device::Cpu).unwrap();
        // 2*32 - 1 = 63
        assert_eq!(pe.dims(), &[1, 63, 64]);
    }

    #[test]
    fn espnet_rel_pos_encoding_symmetry() {
        // Position 0 should be the center of the PE
        let enc = EspnetRelPositionalEncoding::new(16);
        let pe = enc.compute(5, DType::F32, &Device::Cpu).unwrap();
        // Shape: [1, 9, 16]
        assert_eq!(pe.dim(1).unwrap(), 9);

        // The center (index 4) corresponds to relative position 0
        // pe_positive[0] = sin(0) = 0, cos(0) = 1 pattern
        let center: Vec<f32> = pe
            .squeeze(0)
            .unwrap()
            .narrow(0, 4, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Even indices (sin(0)) should be ~0, odd indices (cos(0)) should be ~1
        assert!(center[0].abs() < 1e-6, "sin(0) should be 0");
        assert!((center[1] - 1.0).abs() < 1e-6, "cos(0) should be 1");
    }

    #[test]
    fn espnet_forward_scales_input() {
        let device = Device::Cpu;
        let d_model = 64;
        let enc = EspnetRelPositionalEncoding::new(d_model);

        let x = Tensor::ones((1, 10, d_model), DType::F32, &device).unwrap();
        let (scaled_x, pos_emb) = enc.forward(&x).unwrap();

        // scaled_x should be x * √d_model = √64 = 8.0
        let val: f32 = scaled_x
            .flatten_all()
            .unwrap()
            .get(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!((val - 8.0).abs() < 1e-5);
        assert_eq!(pos_emb.dims(), &[1, 19, 64]); // 2*10 - 1 = 19
    }

    #[test]
    fn rel_shift_shape() {
        let device = Device::Cpu;
        let t1 = 8;
        let t2 = 2 * t1 - 1; // 15
        let x = Tensor::randn(0.0_f32, 1.0, (1, 2, t1, t2), &device).unwrap();
        let shifted = RelPositionMultiHeadedAttention::rel_shift(&x).unwrap();
        assert_eq!(shifted.dims(), &[1, 2, t1, t1]);
    }

    #[test]
    fn rel_position_mha_output_shape() {
        let device = Device::Cpu;
        let (_varmap, vb) = make_vb(&device);
        let n_feat = 64;
        let n_head = 4;

        let attn = RelPositionMultiHeadedAttention::load(vb, n_feat, n_head).unwrap();

        let x = Tensor::randn(0.0_f32, 1.0, (2, 16, n_feat), &device).unwrap();
        let enc = EspnetRelPositionalEncoding::new(n_feat);
        let pe = enc.compute(16, DType::F32, &device).unwrap();

        let out = attn.forward(&x, &x, &x, &pe, None).unwrap();
        assert_eq!(out.dims(), &[2, 16, n_feat]);
    }

    #[test]
    fn rel_position_mha_with_mask() {
        let device = Device::Cpu;
        let (_varmap, vb) = make_vb(&device);
        let n_feat = 32;
        let n_head = 2;

        let attn = RelPositionMultiHeadedAttention::load(vb, n_feat, n_head).unwrap();

        let x = Tensor::randn(0.0_f32, 1.0, (1, 8, n_feat), &device).unwrap();
        let enc = EspnetRelPositionalEncoding::new(n_feat);
        let pe = enc.compute(8, DType::F32, &device).unwrap();

        // Mask: first 6 positions valid, last 2 masked
        let mask_data: Vec<f32> = vec![1., 1., 1., 1., 1., 1., 0., 0.];
        let mask = Tensor::from_vec(mask_data, (1, 1, 8), &device).unwrap();

        let out = attn.forward(&x, &x, &x, &pe, Some(&mask)).unwrap();
        assert_eq!(out.dims(), &[1, 8, n_feat]);
    }

    #[test]
    fn ffn_output_shape() {
        let device = Device::Cpu;
        let (_varmap, vb) = make_vb(&device);

        let ffn = PositionwiseFeedForward::load(vb, 64, 256).unwrap();
        let x = Tensor::randn(0.0_f32, 1.0, (2, 16, 64), &device).unwrap();
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 16, 64]);
    }

    #[test]
    fn encoder_layer_output_shape() {
        let device = Device::Cpu;
        let (_varmap, vb) = make_vb(&device);

        let config = ConformerConfig {
            output_size: 64,
            attention_heads: 4,
            linear_units: 128,
            num_blocks: 1,
            ..Default::default()
        };

        let layer = ConformerEncoderLayer::load(vb, &config).unwrap();

        let x = Tensor::randn(0.0_f32, 1.0, (1, 16, 64), &device).unwrap();
        let enc = EspnetRelPositionalEncoding::new(64);
        let pe = enc.compute(16, DType::F32, &device).unwrap();

        let out = layer.forward(&x, None, &pe).unwrap();
        assert_eq!(out.dims(), &[1, 16, 64]);
    }

    #[test]
    fn full_conformer_encoder() {
        let device = Device::Cpu;
        let (_varmap, vb) = make_vb(&device);

        let config = ConformerConfig {
            output_size: 64,
            attention_heads: 4,
            linear_units: 128,
            num_blocks: 2,
            ..Default::default()
        };

        let encoder = ConformerEncoder::load(vb, &config).unwrap();

        let xs = Tensor::randn(0.0_f32, 1.0, (2, 20, 64), &device).unwrap();
        let mask = Tensor::ones((2, 20), DType::F32, &device).unwrap();

        let out = encoder.forward(&xs, &mask).unwrap();
        assert_eq!(out.dims(), &[2, 20, 64]);
    }

    #[test]
    fn full_conformer_encoder_with_padding() {
        let device = Device::Cpu;
        let (_varmap, vb) = make_vb(&device);

        let config = ConformerConfig {
            output_size: 32,
            attention_heads: 2,
            linear_units: 64,
            num_blocks: 2,
            ..Default::default()
        };

        let encoder = ConformerEncoder::load(vb, &config).unwrap();

        let xs = Tensor::randn(0.0_f32, 1.0, (2, 12, 32), &device).unwrap();
        // Second batch element only has 8 valid tokens
        let mask_data: Vec<f32> = [
            vec![1.0; 12],                                          // batch 0: all valid
            vec![1.0; 8].into_iter().chain(vec![0.0; 4]).collect(), // batch 1: 8 valid
        ]
        .concat();
        let mask = Tensor::from_vec(mask_data, (2, 12), &device).unwrap();

        let out = encoder.forward(&xs, &mask).unwrap();
        assert_eq!(out.dims(), &[2, 12, 32]);
    }

    #[test]
    fn default_config_matches_deployed_model() {
        let config = ConformerConfig::default();
        // Verify all defaults match the deployed ACE-Step model
        assert_eq!(config.output_size, 1024);
        assert_eq!(config.attention_heads, 16);
        assert_eq!(config.linear_units, 4096);
        assert_eq!(config.num_blocks, 6);
        let head_dim = config.output_size / config.attention_heads;
        assert_eq!(head_dim, 64);
    }
}
