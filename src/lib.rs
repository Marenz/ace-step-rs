//! ACE-Step v1.5 music generation in pure Rust.
//!
//! A candle-based implementation of the ACE-Step v1.5 flow-matching diffusion
//! transformer for text-to-music generation. Loads original safetensors
//! weights directly — no ONNX conversion needed.
//!
//! ## Architecture
//!
//! ```text
//! caption → Qwen3-Embedding (full encoder) ──┐
//!                                             ├→ packed condition sequence
//! lyrics → Qwen3-Embedding (embed only) ─────┤
//!   → lyric encoder (8-layer transformer)    │
//!                                             │
//! ref audio → timbre encoder (4-layer) ───────┘
//!                                             ↓
//!              DiT (24 layers, GQA, sliding window + full attn)
//!              flow matching, 8-step turbo ODE
//!                                             ↓
//!              AutoencoderOobleck VAE (latent → 48kHz stereo waveform)
//! ```

pub mod audio;
pub mod config;
pub mod model;
pub mod pipeline;
pub mod streaming;
pub mod vae;

mod error;

pub use error::{Error, Result};
