//! Conformer encoder for lyric conditioning.
//!
//! A 6-layer Conformer with relative-position multi-head attention.
//!
//! ## Architecture per layer
//!
//! ```text
//! Input [B, T, 1024]
//!   → LayerNorm + RelPositionMultiHeadedAttention(16 heads, 1024 dim)
//!   → residual
//!   → LayerNorm + FFN(1024 → 4096 → 1024)
//!   → residual
//! ```
//!
//! ## Relative Position Attention
//!
//! Uses Espnet-style relative positional encoding with learnable biases
//! `pos_bias_u` and `pos_bias_v`. Attention scores combine:
//! - Content-content: `(Q + pos_bias_u) · K^T`
//! - Content-position: `(Q + pos_bias_v) · R^T` (where R = positional encoding)

// TODO: implement ConformerEncoder
// TODO: implement RelPositionMultiHeadedAttention
// TODO: implement EspnetRelPositionalEncoding

/// Configuration for the Conformer lyric encoder.
#[derive(Debug, Clone)]
pub struct ConformerConfig {
    /// Input/output size.
    pub output_size: usize,
    /// Number of attention heads.
    pub attention_heads: usize,
    /// FFN inner size.
    pub linear_units: usize,
    /// Number of Conformer blocks.
    pub num_blocks: usize,
    /// Dropout rate.
    pub dropout_rate: f64,
}

impl Default for ConformerConfig {
    fn default() -> Self {
        Self {
            output_size: 1024,
            attention_heads: 16,
            linear_units: 4096,
            num_blocks: 6,
            dropout_rate: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformer_config_defaults() {
        let config = ConformerConfig::default();
        assert_eq!(config.output_size, 1024);
        assert_eq!(config.attention_heads, 16);
        assert_eq!(config.num_blocks, 6);
        assert_eq!(config.output_size / config.attention_heads, 64); // head_dim
    }
}
