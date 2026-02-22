//! End-to-end inference pipeline.
//!
//! Orchestrates the full text-to-music generation:
//! 1. Tokenize tags (UMT5) and lyrics (BPE)
//! 2. Encode text and lyrics into conditioning context
//! 3. Initialize random latent noise
//! 4. Run the denoising loop (scheduler + DiT)
//! 5. Decode latent → mel spectrogram (DCAE)
//! 6. Synthesize mel → audio waveform (vocoder)
//! 7. Resample and write WAV

// TODO: implement pipeline
