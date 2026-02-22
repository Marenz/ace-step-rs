//! Audio tokenizer and detokenizer.
//!
//! - [`fsq`] — Finite Scalar Quantization
//! - [`pooler`] — Attention-based pooling (25Hz → 5Hz)
//! - [`detokenizer`] — Token expansion (5Hz → 25Hz)

pub mod detokenizer;
pub mod fsq;
pub mod pooler;
