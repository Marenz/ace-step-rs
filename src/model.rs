//! ACE-Step v1.5 model components.
//!
//! The model consists of:
//! - [`transformer`] — DiT layers (GQA self-attention, cross-attention, SiLU MLP)
//! - [`encoder`] — condition encoder (lyric encoder, timbre encoder, text projector)
//! - [`tokenizer`] — audio tokenizer (FSQ) and detokenizer
//! - [`generation`] — top-level generation model combining all components
//! - [`lm_planner`] — 5Hz LM planner (CoT-only), expands raw caption → structured metadata

pub mod encoder;
pub mod generation;
pub mod lm_planner;
pub mod tokenizer;
pub mod transformer;
