//! DCAE decoder (Deep Compression AutoEncoder).
//!
//! Converts diffusion latents to mel spectrograms.
//!
//! The DCAE is a `diffusers::AutoencoderDC` with:
//! - Latent shape: `[B, 8, 16, T_lat]`
//! - Output shape: `[B, 2, 128, T_mel]` (stereo mel spectrogram)
//! - Compression factor: 8× in both height and time (`f8c8`)
//! - `T_mel = T_lat × 8`
//!
//! ## Normalization
//!
//! ```text
//! Encode: latents = (latents - shift_factor) * scale_factor
//! Decode: latents = latents / scale_factor + shift_factor
//! ```
//! Where `shift_factor = -1.9091`, `scale_factor = 0.1786`.
//!
//! ## Mel normalization
//!
//! ```text
//! To model: mel_norm = (mel - min_mel) / (max_mel - min_mel) * 2 - 1  → [-1, 1]
//! From model: mel = (mel_norm * 0.5 + 0.5) * (max_mel - min_mel) + min_mel
//! ```
//! Where `min_mel = -11.0`, `max_mel = 3.0`.

// TODO: implement AutoencoderDC decoder
// TODO: implement chunked decoding (128-frame chunk limit)

/// DCAE latent normalization constants.
pub const SHIFT_FACTOR: f64 = -1.9091;
pub const SCALE_FACTOR: f64 = 0.1786;

/// Mel spectrogram normalization constants.
pub const MIN_MEL_VALUE: f64 = -11.0;
pub const MAX_MEL_VALUE: f64 = 3.0;

/// Denormalize latents from model space to DCAE input space.
///
/// `raw = latent / scale_factor + shift_factor`
pub fn denormalize_latent(latent: &candle_core::Tensor) -> crate::Result<candle_core::Tensor> {
    Ok(((latent / SCALE_FACTOR)? + SHIFT_FACTOR)?)
}

/// Denormalize mel spectrogram from DCAE output space to log-mel values.
///
/// `mel = (x * 0.5 + 0.5) * (max - min) + min`
pub fn denormalize_mel(mel: &candle_core::Tensor) -> crate::Result<candle_core::Tensor> {
    let range = MAX_MEL_VALUE - MIN_MEL_VALUE;
    let scaled = ((mel * 0.5)? + 0.5)?;
    let stretched = (scaled * range)?;
    Ok((stretched + MIN_MEL_VALUE)?)
}

/// Calculate the number of latent time frames for a given duration.
///
/// `frame_length = ceil(duration_secs * 44100 / 512 / 8)`
pub fn calculate_frame_length(duration_secs: f64) -> usize {
    (duration_secs * 44100.0 / 512.0 / 8.0).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn frame_length_60s() {
        let frames = calculate_frame_length(60.0);
        // 60 * 44100 / 512 / 8 = 645.703 → 646
        assert_eq!(frames, 646);
    }

    #[test]
    fn frame_length_30s() {
        let frames = calculate_frame_length(30.0);
        // 30 * 44100 / 512 / 8 = 322.27 → 323
        assert_eq!(frames, 323);
    }

    #[test]
    fn frame_length_10s() {
        let frames = calculate_frame_length(10.0);
        // 10 * 44100 / 512 / 8 = 107.42 → 108
        assert_eq!(frames, 108);
    }

    #[test]
    fn denormalize_roundtrip() {
        let device = Device::Cpu;
        let original = Tensor::randn(0.0_f32, 1.0, (1, 8, 16, 32), &device).unwrap();

        // Normalize (encode direction): (x - shift) * scale
        let normalized = ((&original - SHIFT_FACTOR).unwrap() * SCALE_FACTOR).unwrap();

        // Denormalize (decode direction): x / scale + shift
        let recovered = denormalize_latent(&normalized).unwrap();

        let diff: f32 = (&recovered - &original)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-4, "roundtrip error = {diff}");
    }

    #[test]
    fn mel_denormalize_range() {
        let device = Device::Cpu;

        // Model output in [-1, 1]
        let minus_one = Tensor::full(-1.0_f32, (1, 128, 32), &device).unwrap();
        let plus_one = Tensor::full(1.0_f32, (1, 128, 32), &device).unwrap();

        let mel_low: f32 = denormalize_mel(&minus_one)
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        let mel_high: f32 = denormalize_mel(&plus_one)
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();

        assert!(
            (mel_low - MIN_MEL_VALUE as f32).abs() < 1e-4,
            "mel_low = {mel_low}"
        );
        assert!(
            (mel_high - MAX_MEL_VALUE as f32).abs() < 1e-4,
            "mel_high = {mel_high}"
        );
    }
}
