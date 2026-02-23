//! 5Hz LM planner (CoT-only mode).
//!
//! Runs the `acestep-5Hz-lm-1.7B` Qwen3 causal LM before the DiT to expand
//! a raw user caption into structured metadata (BPM, key/scale, time signature,
//! duration, language) plus a rewritten caption.
//!
//! # CoT-only mode
//!
//! We only run Phase 1: generate until `</think>` is emitted, then parse the
//! YAML metadata block. The audio-code tokens (Phase 2) are not generated —
//! they're only needed for the `llm_dit` infer path which we don't support.
//!
//! # Chat prompt format
//!
//! ```text
//! <|im_start|>system
//! # Instruction
//! Generate audio semantic tokens based on the given conditions:
//!
//! <|im_end|>
//! <|im_start|>user
//! # Caption
//! {caption}
//!
//! # Lyric
//! {lyrics}
//! <|im_end|>
//! <|im_start|>assistant
//! ```
//!
//! The model then generates:
//! ```text
//! <think>
//! bpm: 120
//! caption: A rewritten caption
//! duration: 30
//! genres: pop
//! keyscale: C major
//! language: en
//! timesignature: 4
//! </think>
//! ```
//!
//! We stop sampling as soon as `</think>` (token 151668) is produced.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;
use tokenizers::Tokenizer;

use crate::{Error, Result};

// ── Token IDs (Qwen2/Qwen3 special tokens) ────────────────────────────────────

/// `<|im_start|>` — role-tag open
const TOK_IM_START: u32 = 151644;
/// `<|im_end|>` — role-tag close (also EOS for generation)
const TOK_IM_END: u32 = 151645;
/// `<think>` — start of chain-of-thought
const TOK_THINK_OPEN: u32 = 151667;
/// `</think>` — end of chain-of-thought; we stop here
const TOK_THINK_CLOSE: u32 = 151668;

/// Qwen3 causal LM configuration for `acestep-5Hz-lm-1.7B`.
pub fn lm_planner_config() -> qwen3::Config {
    qwen3::Config {
        vocab_size: 217204,
        hidden_size: 2048,
        intermediate_size: 6144,
        num_hidden_layers: 28,
        num_attention_heads: 16,
        head_dim: 128,
        attention_bias: false,
        num_key_value_heads: 8,
        max_position_embeddings: 40960,
        sliding_window: None,
        max_window_layers: 28,
        tie_word_embeddings: true,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
    }
}

/// Structured metadata produced by the LM planner.
///
/// All fields are optional — the LM may omit some in its CoT output.
#[derive(Debug, Clone, Default)]
pub struct PlannerOutput {
    /// Rewritten caption (more detailed than the user input).
    pub caption: Option<String>,
    /// BPM (beats per minute).
    pub bpm: Option<u32>,
    /// Key and scale, e.g. "C major", "F# minor".
    pub keyscale: Option<String>,
    /// Time signature numerator (e.g. 4 for 4/4).
    pub time_signature: Option<u32>,
    /// Duration in seconds.
    pub duration_s: Option<u32>,
    /// BCP-47 language code, e.g. "en", "zh".
    pub language: Option<String>,
    /// Genre tags, e.g. "pop, rock".
    pub genres: Option<String>,
}

impl PlannerOutput {
    /// Format the planner metadata as the ACE-Step metas string.
    ///
    /// Produces the `- bpm: ...\n- timesignature: ...\n...` format that the
    /// DiT text encoder was trained on.  Falls back to "N/A" for absent fields.
    ///
    /// The `user_duration_s` is used when the LM did not emit a duration.
    pub fn to_metas_string(&self, user_duration_s: f64) -> String {
        let ts = self
            .time_signature
            .map(|n| format!("{n}/4"))
            .unwrap_or_else(|| "N/A".to_string());
        let bpm_str = self
            .bpm
            .map(|b| b.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let ks_str = self.keyscale.as_deref().unwrap_or("N/A");
        let dur_s = self.duration_s.map(|d| d as f64).unwrap_or(user_duration_s);
        let dur_str = if dur_s > 0.0 {
            format!("{} seconds", dur_s as u32)
        } else {
            "N/A".to_string()
        };
        format!("- bpm: {bpm_str}\n- timesignature: {ts}\n- keyscale: {ks_str}\n- duration: {dur_str}\n")
    }
}

/// 5Hz LM planner (CoT-only).
///
/// Holds the Qwen3 causal LM and its tokenizer. Call [`LmPlanner::plan`] to
/// expand a caption + lyrics into [`PlannerOutput`].
pub struct LmPlanner {
    model: qwen3::ModelForCausalLM,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl LmPlanner {
    /// Load from pre-downloaded model files.
    ///
    /// - `weights_path` — path to `acestep-5Hz-lm-1.7B/model.safetensors`
    /// - `tokenizer_path` — path to `acestep-5Hz-lm-1.7B/tokenizer.json`
    pub fn load(
        weights_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let cfg = lm_planner_config();

        tracing::info!("Loading LM planner weights from {:?}", weights_path);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
                .map_err(|e| Error::WeightLoad(format!("LM planner weights: {e}")))?
        };
        // Keys are stored without "model." prefix (e.g. "embed_tokens.weight")
        // but candle's Qwen3 requests them with it (e.g. "model.embed_tokens.weight").
        let vb = vb.rename_f(|name: &str| name.strip_prefix("model.").unwrap_or(name).to_string());

        let model = qwen3::ModelForCausalLM::new(&cfg, vb)
            .map_err(|e| Error::WeightLoad(format!("LM planner model init: {e}")))?;

        tracing::info!("Loading LM planner tokenizer from {:?}", tokenizer_path);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::WeightLoad(format!("LM tokenizer load: {e}")))?;

        tracing::info!("LM planner loaded successfully");
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
        })
    }

    /// Expand a caption + lyrics into structured metadata.
    ///
    /// Runs the Qwen3 LM until `</think>` is emitted (or `max_new_tokens`).
    /// Parses the YAML CoT block and returns the extracted fields.
    ///
    /// - `caption` — raw user description (e.g. "calm piano ballad")
    /// - `lyrics` — lyrics text; use `"[instrumental]"` for no vocals
    /// - `max_new_tokens` — safety cap (default: 512 is enough for CoT)
    ///
    /// `temperature=0.0` → greedy decoding (deterministic, good for CoT).
    pub fn plan(
        &mut self,
        caption: &str,
        lyrics: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<PlannerOutput> {
        // 1. Build prompt tokens
        let prompt_ids = self.build_prompt_tokens(caption, lyrics)?;
        let prompt_len = prompt_ids.len();
        tracing::debug!("LM planner prompt: {} tokens", prompt_len);

        // 2. Encode prompt as tensor [1, T]
        let mut input =
            Tensor::from_vec(prompt_ids, (1, prompt_len), &self.device)?.to_dtype(DType::U32)?;

        // 3. Autoregressive generation loop
        self.model.clear_kv_cache();
        let mut generated: Vec<u32> = Vec::with_capacity(256);
        let mut offset = 0usize;

        for step in 0..max_new_tokens {
            // Forward pass — returns [1, 1, vocab_size] logits for last token
            let logits = self.model.forward(&input, offset)?;

            // logits: [1, 1, V] → [V]
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            // Sample next token
            let next_token = if temperature <= 0.0 {
                logits.argmax(0)?.to_scalar::<u32>()?
            } else {
                let scaled = (logits / temperature as f64)?;
                let probs = candle_nn::ops::softmax_last_dim(&scaled)?;
                sample_multinomial(&probs)?
            };

            generated.push(next_token);
            offset += if step == 0 { prompt_len } else { 1 };

            // Advance input to the single new token for next step
            input =
                Tensor::from_vec(vec![next_token], (1, 1), &self.device)?.to_dtype(DType::U32)?;

            // Stop at </think> or <|im_end|>
            if next_token == TOK_THINK_CLOSE || next_token == TOK_IM_END {
                tracing::debug!(
                    "LM planner stopped at token {} after {} steps",
                    next_token,
                    step + 1
                );
                break;
            }
        }

        // 4. Decode generated tokens to text
        let output_text = self
            .tokenizer
            .decode(&generated, /*skip_special_tokens=*/ false)
            .map_err(|e| {
                Error::Tokenizer(crate::error::TokenizerError(format!("LM decode: {e}")))
            })?;

        tracing::debug!("LM planner output: {:?}", output_text);

        // 5. Parse YAML metadata from <think>...</think>
        Ok(parse_cot_output(&output_text))
    }

    /// Build the chat-template prompt token IDs.
    ///
    /// Encodes the prompt manually (the tokenizers crate doesn't run Jinja
    /// templates), matching Python's `apply_chat_template` output:
    ///
    /// ```text
    /// <|im_start|>system\n# Instruction\n...\n\n<|im_end|>\n
    /// <|im_start|>user\n# Caption\n{caption}\n\n# Lyric\n{lyrics}\n<|im_end|>\n
    /// <|im_start|>assistant\n
    /// ```
    fn build_prompt_tokens(&self, caption: &str, lyrics: &str) -> Result<Vec<u32>> {
        const INSTRUCTION: &str = "Generate audio semantic tokens based on the given conditions:";

        let prompt_str = format!(
            "<|im_start|>system\n# Instruction\n{INSTRUCTION}\n\n<|im_end|>\n\
             <|im_start|>user\n# Caption\n{caption}\n\n# Lyric\n{lyrics}\n<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        let encoding = self
            .tokenizer
            .encode(prompt_str.as_str(), false)
            .map_err(|e| {
                Error::Tokenizer(crate::error::TokenizerError(format!("LM encode: {e}")))
            })?;

        Ok(encoding.get_ids().to_vec())
    }
}

// ── Multinomial sampling helper ───────────────────────────────────────────────

/// Sample one token index from a probability distribution.
///
/// Draws a uniform sample in [0,1) and walks the CDF of `probs`.
fn sample_multinomial(probs: &Tensor) -> Result<u32> {
    use rand::Rng;

    // Move to CPU for sampling
    let probs_cpu = probs.to_device(&Device::Cpu)?;
    let probs_vec: Vec<f32> = probs_cpu.to_vec1()?;

    // Weighted random selection via CDF walk
    let sample: f64 = rand::rng().random();
    let mut cumulative = 0.0f64;
    for (i, &p) in probs_vec.iter().enumerate() {
        cumulative += p as f64;
        if sample < cumulative {
            return Ok(i as u32);
        }
    }
    // Fallback: last token (handles floating-point rounding)
    Ok((probs_vec.len().saturating_sub(1)) as u32)
}

// ── CoT output parser ─────────────────────────────────────────────────────────

/// Parse the LM's chain-of-thought YAML output into [`PlannerOutput`].
///
/// Looks for `<think>...</think>` and parses simple `key: value` lines.
/// Fields outside the `<think>` block are ignored.
fn parse_cot_output(text: &str) -> PlannerOutput {
    let mut out = PlannerOutput::default();

    // Extract inner text from <think>...</think>
    let inner = if let (Some(open), Some(close)) = (text.find("<think>"), text.find("</think>")) {
        let start = open + "<think>".len();
        &text[start..close]
    } else {
        // No tags — try to parse the whole output
        text
    };

    // Parse simple YAML-style `key: value` lines
    for line in inner.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('<') {
            continue;
        }
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim().to_lowercase();
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            match key.as_str() {
                "bpm" => out.bpm = value.parse().ok(),
                "caption" => out.caption = Some(value.to_string()),
                "keyscale" => out.keyscale = Some(value.to_string()),
                "timesignature" => out.time_signature = value.parse().ok(),
                "duration" => out.duration_s = value.parse().ok(),
                "language" => out.language = Some(value.to_string()),
                "genres" => out.genres = Some(value.to_string()),
                _ => {}
            }
        }
    }

    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lm_planner_config() {
        let cfg = lm_planner_config();
        assert_eq!(cfg.vocab_size, 217204);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert!(!cfg.use_sliding_window);
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn test_parse_cot_output_full() {
        let text = "<think>\nbpm: 120\ncaption: A calm piano melody\nduration: 30\n\
                    genres: pop\nkeyscale: C major\nlanguage: en\ntimesignature: 4\n</think>";
        let out = parse_cot_output(text);
        assert_eq!(out.bpm, Some(120));
        assert_eq!(out.caption.as_deref(), Some("A calm piano melody"));
        assert_eq!(out.duration_s, Some(30));
        assert_eq!(out.genres.as_deref(), Some("pop"));
        assert_eq!(out.keyscale.as_deref(), Some("C major"));
        assert_eq!(out.language.as_deref(), Some("en"));
        assert_eq!(out.time_signature, Some(4));
    }

    #[test]
    fn test_parse_cot_output_partial() {
        // Only some fields present
        let text = "<think>\nbpm: 85\nkeyscale: F# minor\n</think>";
        let out = parse_cot_output(text);
        assert_eq!(out.bpm, Some(85));
        assert_eq!(out.keyscale.as_deref(), Some("F# minor"));
        assert!(out.caption.is_none());
        assert!(out.duration_s.is_none());
    }

    #[test]
    fn test_parse_cot_output_no_tags() {
        // Fallback: no <think> tags, parse directly
        let text = "bpm: 100\nlanguage: zh\n";
        let out = parse_cot_output(text);
        assert_eq!(out.bpm, Some(100));
        assert_eq!(out.language.as_deref(), Some("zh"));
    }

    #[test]
    fn test_parse_cot_ignores_audio_codes() {
        // Audio code tokens should not crash the parser
        let text = "<think>\nbpm: 140\n</think>\n<|audio_code_12345|><|audio_code_67890|>";
        let out = parse_cot_output(text);
        assert_eq!(out.bpm, Some(140));
        assert!(out.caption.is_none());
    }
}
