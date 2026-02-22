//! Model components for ACE-Step.
//!
//! ## Components
//!
//! - [`transformer`] — the 24-layer DiT with linear self-attention and cross-attention
//! - [`encoder`] — UMT5 text encoder, Conformer lyric encoder, speaker embedding
//! - [`dcae`] — DCAE latent decoder (latent → mel spectrogram)
//! - [`vocoder`] — ADaMoSHiFiGAN (mel spectrogram → audio waveform)

pub mod dcae;
pub mod encoder;
pub mod transformer;
pub mod vocoder;
