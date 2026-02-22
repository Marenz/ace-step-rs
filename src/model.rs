//! ACE-Step v1.5 model components.
//!
//! The model consists of:
//! - [`transformer`] — DiT layers (GQA self-attention, cross-attention, SiLU MLP)
//! - [`encoder`] — condition encoder (lyric encoder, timbre encoder, text projector)
//! - [`tokenizer`] — audio tokenizer (FSQ) and detokenizer
//! - [`generation`] — top-level generation model combining all components

pub mod transformer;
pub mod encoder;
pub mod tokenizer;
pub mod generation;
