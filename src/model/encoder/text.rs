//! Text encoder wrapper around Qwen3-Embedding-0.6B.
//!
//! Two encoding paths:
//! - **Caption encoding**: full Qwen3 forward → `last_hidden_state` [B, T, 1024]
//! - **Lyric embedding**: raw `embed_tokens` lookup only [B, T, 1024]
//!
//! The Qwen3 model weights are loaded from the `Qwen3-Embedding-0.6B/` checkpoint
//! directory, separate from the main ACE-Step model weights.

use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};
use candle_transformers::models::qwen3;

/// Qwen3-Embedding-0.6B configuration for ACE-Step v1.5.
///
/// The embedding model has hidden_size=1024, 28 layers, 16 heads, 8 KV heads, head_dim=128.
/// Note: q_proj size = num_heads * head_dim = 16 * 128 = 2048 (larger than hidden_size).
pub fn default_qwen3_config() -> qwen3::Config {
    qwen3::Config {
        vocab_size: 151669,
        hidden_size: 1024,
        intermediate_size: 3072,
        num_hidden_layers: 28,
        num_attention_heads: 16,
        head_dim: 128,
        attention_bias: false,
        num_key_value_heads: 8,
        max_position_embeddings: 32768,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
    }
}

/// Text encoder wrapping Qwen3-Embedding-0.6B.
///
/// Provides two encoding paths:
/// - `encode_text()` — full Qwen3 forward for captions → [B, T, 1024]
/// - `embed_lyrics()` — raw embedding lookup for lyrics → [B, T, 1024]
pub struct Qwen3TextEncoder {
    model: qwen3::Model,
    embed_tokens: nn::Embedding,
}

impl Qwen3TextEncoder {
    /// Load from a VarBuilder pointing at Qwen3 weights.
    ///
    /// The VarBuilder should be constructed from the Qwen3-Embedding-0.6B
    /// safetensors file. Weight paths: `model.embed_tokens`, `model.layers.*`,
    /// `model.norm`.
    pub fn new(cfg: &qwen3::Config, vb: VarBuilder) -> Result<Self> {
        let model = qwen3::Model::new(cfg, vb.clone())?;
        // Load embed_tokens separately since Model doesn't expose it
        let embed_tokens =
            nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        Ok(Self {
            model,
            embed_tokens,
        })
    }

    /// Encode text captions through the full Qwen3 model.
    ///
    /// Input: `token_ids` [B, T] — tokenized caption text
    /// Output: `hidden_states` [B, T, 1024] — last hidden states
    ///
    /// Uses causal attention (matching Python's Qwen3Model behavior).
    /// Clears the KV cache before each call so repeated invocations
    /// don't accumulate stale state.
    pub fn encode_text(&mut self, token_ids: &Tensor) -> Result<Tensor> {
        self.model.clear_kv_cache();
        self.model.forward(token_ids, 0)
    }

    /// Extract raw token embeddings for lyrics.
    ///
    /// Input: `token_ids` [B, T] — tokenized lyric text
    /// Output: `embeddings` [B, T, 1024] — pre-transformer token embeddings
    ///
    /// These embeddings are passed to [`AceStepLyricEncoder`] for further processing.
    pub fn embed_lyrics(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(token_ids)
    }

    /// Clear the KV cache. Must be called between independent forward passes.
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    /// Get the Qwen3 hidden dimension (1024 for Qwen3-Embedding-0.6B).
    pub fn hidden_size(&self) -> usize {
        self.embed_tokens.embeddings().dim(1).unwrap_or(1024)
    }
}

/// Format a caption into the ACE-Step v1.5 prompt template.
///
/// Matches Python's `SFT_GEN_PROMPT` triple-quoted string which ends with a newline after
/// `<|endoftext|>`. The instruction is always the default DiT instruction.
///
/// ```text
/// # Instruction
/// Fill the audio semantic mask based on the given conditions:
///
/// # Caption
/// {caption}
///
/// # Metas
/// {metas}<|endoftext|>
/// ```
///
/// The metas string must be in the training format. Use [`format_metas`] to build it.
pub fn format_caption_prompt(caption: &str, metas: &str) -> String {
    format!(
        "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"
    )
}

/// Build a metas string in the format the model was trained on.
///
/// ```text
/// - bpm: 120
/// - timesignature: 4/4
/// - keyscale: C major
/// - duration: 30 seconds
/// ```
///
/// Pass `None` or empty string for unknown fields — they'll be set to "N/A".
pub fn format_metas(
    bpm: Option<u32>,
    time_signature: Option<&str>,
    key_scale: Option<&str>,
    duration_s: Option<f64>,
) -> String {
    let bpm_str = match bpm {
        Some(b) => b.to_string(),
        None => "N/A".to_string(),
    };
    let ts_str = match time_signature {
        Some(ts) if !ts.is_empty() => ts.to_string(),
        _ => "N/A".to_string(),
    };
    let ks_str = match key_scale {
        Some(ks) if !ks.is_empty() => ks.to_string(),
        _ => "N/A".to_string(),
    };
    let dur_str = match duration_s {
        Some(d) if d > 0.0 => format!("{} seconds", d as u32),
        _ => "N/A".to_string(),
    };
    format!("- bpm: {bpm_str}\n- timesignature: {ts_str}\n- keyscale: {ks_str}\n- duration: {dur_str}\n")
}

/// Format lyrics into the ACE-Step v1.5 lyric template.
///
/// ```text
/// # Languages
/// {language}
///
/// # Lyric
/// {lyrics}<|endoftext|>
/// ```
pub fn format_lyric_prompt(lyrics: &str, language: &str) -> String {
    format!("# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_default_qwen3_config() {
        let cfg = default_qwen3_config();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 151669);
        assert_eq!(cfg.intermediate_size, 3072);
        assert!(!cfg.use_sliding_window);
    }

    #[test]
    fn test_format_caption_prompt() {
        let metas = format_metas(Some(120), Some("4/4"), Some("C major"), Some(30.0));
        let prompt = format_caption_prompt("jazz piano", &metas);
        assert!(prompt.contains("# Caption\njazz piano"));
        assert!(prompt.contains("- bpm: 120\n"));
        assert!(prompt.contains("- keyscale: C major\n"));
        assert!(prompt.starts_with("# Instruction\n"));
    }

    #[test]
    fn test_format_metas() {
        let m = format_metas(Some(120), Some("4/4"), Some("C major"), Some(30.0));
        assert_eq!(
            m,
            "- bpm: 120\n- timesignature: 4/4\n- keyscale: C major\n- duration: 30 seconds\n"
        );

        let m2 = format_metas(None, None, None, None);
        assert_eq!(
            m2,
            "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: N/A\n"
        );

        let m3 = format_metas(Some(90), None, Some("D minor"), Some(60.0));
        assert!(m3.contains("- bpm: 90\n"));
        assert!(m3.contains("- timesignature: N/A\n"));
        assert!(m3.contains("- keyscale: D minor\n"));
        assert!(m3.contains("- duration: 60 seconds\n"));
    }

    #[test]
    fn test_format_lyric_prompt() {
        let prompt = format_lyric_prompt("Hello world", "en");
        assert!(prompt.contains("# Languages\nen"));
        assert!(prompt.contains("# Lyric\nHello world<|endoftext|>"));
    }

    #[test]
    fn test_qwen3_text_encoder_creation() {
        // Test with tiny config (1 layer) and zero weights
        let dev = Device::Cpu;
        let cfg = qwen3::Config {
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            head_dim: 16,
            attention_bias: false,
            num_key_value_heads: 1,
            max_position_embeddings: 128,
            sliding_window: None,
            max_window_layers: 0,
            tie_word_embeddings: true,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let mut enc = Qwen3TextEncoder::new(&cfg, vb).unwrap();

        // Test encode_text
        let input = Tensor::zeros((1, 5), DType::U32, &dev).unwrap();
        let out = enc.encode_text(&input).unwrap();
        assert_eq!(out.dims(), &[1, 5, 32]);

        // Calling encode_text again (different length) must not fail.
        // Regression test: without KV cache clearing, the second call
        // would hit a shape mismatch (stale cache from the first call).
        let input2 = Tensor::zeros((1, 8), DType::U32, &dev).unwrap();
        let out2 = enc.encode_text(&input2).unwrap();
        assert_eq!(out2.dims(), &[1, 8, 32]);

        // Test embed_lyrics
        let lyric_out = enc.embed_lyrics(&input).unwrap();
        assert_eq!(lyric_out.dims(), &[1, 5, 32]);
    }
}
