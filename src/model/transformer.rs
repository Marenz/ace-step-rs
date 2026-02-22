//! DiT transformer components.
//!
//! Building blocks for the ACE-Step v1.5 diffusion transformer:
//! - [`attention`] — GQA self-attention and cross-attention with RoPE
//! - [`layers`] — encoder layers, DiT layers with AdaLN
//! - [`timestep`] — sinusoidal timestep embedding
//! - [`mask`] — 4D attention mask creation
//! - [`dit`] — the full DiT model (patch in → transformer → patch out)

pub mod attention;
pub mod dit;
pub mod layers;
pub mod mask;
pub mod timestep;
