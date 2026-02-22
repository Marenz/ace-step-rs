//! ACE-Step diffusion transformer.
//!
//! A 24-layer single-stream DiT with:
//! - Linear self-attention (ReLU kernel, O(n·d²))
//! - Standard cross-attention (SDPA) to encoder context
//! - GLUMBConv feed-forward (gated depthwise-separable 1D conv)
//! - AdaLN-Single conditioning from timestep
//!
//! ## Config (deployed model)
//!
//! ```text
//! num_layers:           24
//! inner_dim:            2560
//! num_attention_heads:  20
//! attention_head_dim:   128
//! mlp_ratio:            2.5
//! in_channels:          8
//! out_channels:         8
//! patch_size:           [16, 1]
//! rope_theta:           1_000_000.0
//! text_embedding_dim:   768
//! speaker_embedding_dim: 512
//! lyric_hidden_size:    1024
//! ```

pub mod attention;
pub mod config;
pub mod glumbconv;
pub mod patch_embed;
pub mod rope;

use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::Result;
use config::TransformerConfig;

/// Complete ACE-Step transformer (encode + decode).
pub struct AceStepTransformer {
    config: TransformerConfig,
    /// Patch embedding: Conv2d(8, 2560, kernel=[16,1], stride=[16,1])
    patch_embed: patch_embed::PatchEmbed,
    /// Timestep projection: sinusoidal → MLP → [B, 2560]
    time_proj: TimestepProjection,
    /// t_block: SiLU + Linear(2560, 6*2560) → per-block temb
    t_block: candle_nn::Linear,
    /// Speaker embedding: Linear(512, 2560)
    speaker_embedder: candle_nn::Linear,
    /// Text embedding: Linear(768, 2560)
    genre_embedder: candle_nn::Linear,
    /// Lyric token embeddings: Embedding(6693, 1024)
    lyric_embs: candle_nn::Embedding,
    /// Lyric Conformer encoder
    // TODO: lyric_encoder: ConformerEncoder,
    /// Lyric projection: Linear(1024, 2560)
    lyric_proj: candle_nn::Linear,
    /// RoPE frequencies
    rope: rope::RotaryEmbedding,
    /// Transformer blocks
    blocks: Vec<LinearTransformerBlock>,
    /// Final layer
    final_layer: FinalLayer,
}

/// Single transformer block.
pub struct LinearTransformerBlock {
    /// RMSNorm before self-attention.
    norm1: candle_nn::RmsNorm,
    /// Linear self-attention (ReLU kernel).
    self_attn: attention::LinearAttention,
    /// Standard cross-attention (SDPA).
    cross_attn: attention::CrossAttention,
    /// RMSNorm before FFN.
    norm2: candle_nn::RmsNorm,
    /// Gated depthwise conv FFN.
    ffn: glumbconv::GluMbConv,
    /// AdaLN-Single: [6, inner_dim] learned table.
    scale_shift_table: Tensor,
}

/// Timestep sinusoidal projection + MLP.
pub struct TimestepProjection {
    /// Linear(256, inner_dim)
    linear1: candle_nn::Linear,
    /// Linear(inner_dim, inner_dim)
    linear2: candle_nn::Linear,
}

/// Final output layer: RMSNorm + modulate + linear unpatchify.
pub struct FinalLayer {
    norm: candle_nn::RmsNorm,
    /// Linear(inner_dim, patch_h * patch_w * out_channels) = Linear(2560, 128)
    linear: candle_nn::Linear,
    /// [2, inner_dim] scale/shift table.
    scale_shift_table: Tensor,
}

impl AceStepTransformer {
    /// Load from safetensors via VarBuilder.
    pub fn load(vb: VarBuilder, config: &TransformerConfig) -> Result<Self> {
        let _ = (vb, config);
        todo!("implement weight loading")
    }

    /// Encode conditioning context (text + speaker + lyrics).
    ///
    /// Returns `(encoder_hidden_states, encoder_mask)`:
    /// - `encoder_hidden_states`: `[B, 1 + S_text + S_lyric, 2560]`
    /// - `encoder_mask`: `[B, 1 + S_text + S_lyric]`
    pub fn encode(
        &self,
        _text_hidden_states: &Tensor, // [B, S_text, 768]
        _text_mask: &Tensor,          // [B, S_text]
        _speaker_embeds: &Tensor,     // [B, 512]
        _lyric_token_ids: &Tensor,    // [B, S_lyric]
        _lyric_mask: &Tensor,         // [B, S_lyric]
    ) -> Result<(Tensor, Tensor)> {
        todo!("implement encode")
    }

    /// Run the denoising forward pass.
    ///
    /// - `latent`: `[B, 8, 16, T]` current noisy latent
    /// - `attention_mask`: `[B, T]`
    /// - `encoder_hidden_states`: `[B, S_enc, 2560]` from encode()
    /// - `encoder_mask`: `[B, S_enc]`
    /// - `timestep`: scalar timestep value
    /// - `output_length`: target temporal length T
    ///
    /// Returns predicted velocity `[B, 8, 16, T]`.
    pub fn decode(
        &self,
        _latent: &Tensor,
        _attention_mask: &Tensor,
        _encoder_hidden_states: &Tensor,
        _encoder_mask: &Tensor,
        _timestep: f64,
        _output_length: usize,
    ) -> Result<Tensor> {
        todo!("implement decode")
    }
}

#[cfg(test)]
mod tests {
    use super::config::TransformerConfig;

    #[test]
    fn default_config_matches_deployed() {
        let config = TransformerConfig::default();
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.inner_dim, 2560);
        assert_eq!(config.num_attention_heads, 20);
        assert_eq!(config.attention_head_dim, 128);
        assert_eq!(config.in_channels, 8);
        assert_eq!(config.out_channels, 8);
        assert_eq!(config.patch_size, [16, 1]);
    }
}
