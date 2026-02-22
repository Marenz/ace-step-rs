//! Transformer configuration.

/// Configuration for the ACE-Step transformer.
///
/// Defaults match the deployed `ace_step_transformer/config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TransformerConfig {
    /// Number of transformer blocks.
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,

    /// Hidden dimension (num_attention_heads × attention_head_dim).
    #[serde(default = "default_inner_dim")]
    pub inner_dim: usize,

    /// Number of attention heads.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    /// Dimension per attention head.
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: usize,

    /// MLP expansion ratio for GLUMBConv.
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,

    /// Input latent channels.
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,

    /// Output latent channels.
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,

    /// Patch size [height, width] for patch embedding.
    #[serde(default = "default_patch_size")]
    pub patch_size: [usize; 2],

    /// Maximum position for RoPE.
    #[serde(default = "default_max_position")]
    pub max_position: usize,

    /// RoPE theta base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// UMT5 text embedding dimension.
    #[serde(default = "default_text_embedding_dim")]
    pub text_embedding_dim: usize,

    /// Speaker embedding input dimension.
    #[serde(default = "default_speaker_embedding_dim")]
    pub speaker_embedding_dim: usize,

    /// Lyric BPE vocabulary size.
    #[serde(default = "default_lyric_encoder_vocab_size")]
    pub lyric_encoder_vocab_size: usize,

    /// Lyric encoder hidden size (Conformer).
    #[serde(default = "default_lyric_hidden_size")]
    pub lyric_hidden_size: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            num_layers: default_num_layers(),
            inner_dim: default_inner_dim(),
            num_attention_heads: default_num_attention_heads(),
            attention_head_dim: default_attention_head_dim(),
            mlp_ratio: default_mlp_ratio(),
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            patch_size: default_patch_size(),
            max_position: default_max_position(),
            rope_theta: default_rope_theta(),
            text_embedding_dim: default_text_embedding_dim(),
            speaker_embedding_dim: default_speaker_embedding_dim(),
            lyric_encoder_vocab_size: default_lyric_encoder_vocab_size(),
            lyric_hidden_size: default_lyric_hidden_size(),
        }
    }
}

fn default_num_layers() -> usize {
    24
}
fn default_inner_dim() -> usize {
    2560
}
fn default_num_attention_heads() -> usize {
    20
}
fn default_attention_head_dim() -> usize {
    128
}
fn default_mlp_ratio() -> f64 {
    2.5
}
fn default_in_channels() -> usize {
    8
}
fn default_out_channels() -> usize {
    8
}
fn default_patch_size() -> [usize; 2] {
    [16, 1]
}
fn default_max_position() -> usize {
    32768
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_text_embedding_dim() -> usize {
    768
}
fn default_speaker_embedding_dim() -> usize {
    512
}
fn default_lyric_encoder_vocab_size() -> usize {
    6693
}
fn default_lyric_hidden_size() -> usize {
    1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_dim_equals_heads_times_head_dim() {
        let config = TransformerConfig::default();
        assert_eq!(
            config.inner_dim,
            config.num_attention_heads * config.attention_head_dim,
            "inner_dim should equal num_heads × head_dim"
        );
    }

    #[test]
    fn patch_collapses_full_height() {
        let config = TransformerConfig::default();
        // Latent height is 16 (128 mel bins / 8 DCAE compression).
        // patch_size[0] = 16, so the full height is collapsed.
        assert_eq!(config.patch_size[0], 16);
        assert_eq!(config.patch_size[1], 1);
    }

    #[test]
    fn deserialize_from_json() {
        let json = r#"{"num_layers": 12, "inner_dim": 1536}"#;
        let config: TransformerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.inner_dim, 1536);
        // Unspecified fields should use defaults.
        assert_eq!(config.num_attention_heads, 20);
    }
}
