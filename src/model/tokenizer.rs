//! Audio tokenizer and detokenizer.
//!
//! - [`fsq`] — Finite Scalar Quantization
//! - [`pooler`] — Attention-based pooling (25Hz → 5Hz)
//! - [`detokenizer`] — Token expansion (5Hz → 25Hz)

pub mod fsq;
pub mod pooler;
pub mod detokenizer;
