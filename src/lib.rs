//! ACE-Step music generation in pure Rust.
//!
//! A candle-based implementation of the ACE-Step flow-matching diffusion
//! transformer for text-to-music generation. Loads original safetensors
//! weights directly — no ONNX conversion needed.
//!
//! ## Architecture
//!
//! The pipeline transforms text (tags + lyrics) into stereo audio:
//!
//! ```text
//! tags → UMT5 encoder ──┐
//!                        ├→ cross-attention context
//! lyrics → Conformer ───┘
//!                        ↓
//!              DiT (24 blocks, flow matching)
//!                        ↓
//!              DCAE decoder (latent → mel)
//!                        ↓
//!              ADaMoSHiFiGAN vocoder (mel → audio)
//! ```
//!
//! ## Modules
//!
//! - [`audio`] — mel spectrogram (STFT + filterbank), WAV I/O, resampling
//! - [`model`] — transformer, encoder, DCAE decoder, vocoder
//! - [`scheduler`] — flow-matching schedulers (Euler, Heun, PingPong)
//! - [`pipeline`] — end-to-end inference pipeline

pub mod audio;
pub mod model;
pub mod pipeline;
pub mod scheduler;

mod error;

pub use error::{Error, Result};
