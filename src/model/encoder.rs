//! Encoder components: UMT5 text encoder, Conformer lyric encoder.
//!
//! The encoding pipeline produces a unified context for the transformer:
//! ```text
//! speaker_embeds [B, 512]     → Linear(512, 2560) → [B, 1, 2560]
//! text [B, S_text, 768]       → Linear(768, 2560) → [B, S_text, 2560]
//! lyrics [B, S_lyric] (int64) → Embedding(6693, 1024)
//!                              → Conformer(6 layers)
//!                              → Linear(1024, 2560) → [B, S_lyric, 2560]
//!
//! context = cat([speaker, text, lyrics], dim=1) → [B, 1+S_text+S_lyric, 2560]
//! ```

pub mod conformer;
pub mod lyric_tokenizer;

// TODO: implement UMT5 wrapper (reuse candle_transformers::models::t5)
// TODO: implement ConformerEncoder (6-layer relative-position self-attention)
// TODO: implement lyric BPE tokenizer
