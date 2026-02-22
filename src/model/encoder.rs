//! Condition encoders for ACE-Step v1.5.
//!
//! - [`text`] — Qwen3-Embedding-0.6B wrapper (caption encoding + lyric embedding)
//! - [`lyric`] — lyric encoder (8-layer transformer on Qwen3 token embeddings)
//! - [`timbre`] — timbre encoder (4-layer transformer on reference audio features)
//! - [`condition`] — combines text, lyrics, and timbre into packed condition sequence

pub mod condition;
pub mod lyric;
pub mod text;
pub mod timbre;
