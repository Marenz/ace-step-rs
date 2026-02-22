//! Mel spectrogram computation via STFT + mel filterbank.
//!
//! Matches the original ACE-Step configuration exactly:
//! - Sample rate: 44100 Hz
//! - FFT size: 2048 (giving 1025 frequency bins)
//! - Window: Hann, length 2048
//! - Hop length: 512
//! - Mel bins: 128, range 40–16000 Hz
//! - Mel scale/norm: Slaney
//! - Padding: reflect, manually applied (not via STFT center=True)
//! - Log compression: `ln(clamp(mel, min=1e-5))`

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Configuration for the mel spectrogram. Defaults match ACE-Step.
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub f_min: f64,
    pub f_max: f64,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            n_fft: 2048,
            win_length: 2048,
            hop_length: 512,
            n_mels: 128,
            f_min: 40.0,
            f_max: 16000.0,
        }
    }
}

/// Mel spectrogram processor.
///
/// Pre-computes the Hann window, FFT plan, and mel filterbank on construction.
/// Then call [`MelSpectrogram::process`] to convert audio samples to a log-mel
/// spectrogram.
pub struct MelSpectrogram {
    config: MelConfig,
    window: Vec<f64>,
    filterbank: Vec<Vec<f64>>,
    fft: std::sync::Arc<dyn rustfft::Fft<f64>>,
}

impl MelSpectrogram {
    /// Create a new mel spectrogram processor with the given config.
    pub fn new(config: MelConfig) -> Self {
        let window = hann_window(config.win_length);
        let filterbank = mel_filterbank(
            config.n_fft,
            config.n_mels,
            config.sample_rate,
            config.f_min,
            config.f_max,
        );
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.n_fft);

        Self {
            config,
            window,
            filterbank,
            fft,
        }
    }

    /// Compute a log-mel spectrogram from raw audio samples.
    ///
    /// Input: mono audio at the configured sample rate.
    /// Output: `[n_mels, num_frames]` log-mel spectrogram.
    pub fn process(&self, samples: &[f32]) -> Vec<Vec<f64>> {
        let samples_f64: Vec<f64> = samples.iter().map(|&s| s as f64).collect();

        // Reflect-pad to match PyTorch's manual padding (center=False after padding).
        let pad_left = (self.config.win_length - self.config.hop_length) / 2;
        let pad_right = (self.config.win_length - self.config.hop_length + 1) / 2;
        let padded = reflect_pad(&samples_f64, pad_left, pad_right);

        // STFT: compute magnitude spectrum for each frame.
        let magnitudes = self.stft(&padded);

        // Apply mel filterbank + log compression.
        let num_frames = magnitudes.len();
        let mut mel_spec = vec![vec![0.0; num_frames]; self.config.n_mels];

        for (frame_idx, frame_magnitudes) in magnitudes.iter().enumerate() {
            for (mel_idx, filter) in self.filterbank.iter().enumerate() {
                let mut sum = 0.0;
                for (bin_idx, &weight) in filter.iter().enumerate() {
                    if weight > 0.0 {
                        sum += weight * frame_magnitudes[bin_idx];
                    }
                }
                // Log compression: ln(clamp(x, min=1e-5))
                mel_spec[mel_idx][frame_idx] = sum.max(1e-5).ln();
            }
        }

        mel_spec
    }

    /// Short-time Fourier transform. Returns magnitude spectra per frame.
    ///
    /// Each inner vec has `n_fft/2 + 1` elements (one-sided).
    fn stft(&self, padded: &[f64]) -> Vec<Vec<f64>> {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let num_bins = n_fft / 2 + 1;

        let num_frames = (padded.len().saturating_sub(n_fft)) / hop + 1;
        let mut frames = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;
            let end = start + n_fft;
            if end > padded.len() {
                break;
            }

            // Window the frame and prepare complex input.
            let mut buffer: Vec<Complex<f64>> = (0..n_fft)
                .map(|i| Complex::new(padded[start + i] * self.window[i], 0.0))
                .collect();

            // In-place FFT.
            self.fft.process(&mut buffer);

            // Magnitude: sqrt(re² + im² + 1e-6) for the one-sided spectrum.
            let magnitudes: Vec<f64> = buffer[..num_bins]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im + 1e-6).sqrt())
                .collect();

            frames.push(magnitudes);
        }

        frames
    }
}

/// Generate a Hann window of the given length.
fn hann_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / length as f64;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// Reflect-pad a signal on both sides.
fn reflect_pad(signal: &[f64], pad_left: usize, pad_right: usize) -> Vec<f64> {
    let len = signal.len();
    let total = pad_left + len + pad_right;
    let mut padded = Vec::with_capacity(total);

    // Left reflection: signal[pad_left], signal[pad_left-1], ..., signal[1]
    for i in (1..=pad_left).rev() {
        padded.push(signal[i.min(len - 1)]);
    }

    padded.extend_from_slice(signal);

    // Right reflection: signal[len-2], signal[len-3], ...
    for i in 0..pad_right {
        let idx = len.saturating_sub(2 + i);
        padded.push(signal[idx]);
    }

    padded
}

/// Build a Slaney-normalized mel filterbank.
///
/// Returns `n_mels` filters, each with `n_fft/2 + 1` weights.
fn mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    sample_rate: u32,
    f_min: f64,
    f_max: f64,
) -> Vec<Vec<f64>> {
    let num_bins = n_fft / 2 + 1;
    let sr = sample_rate as f64;

    // Mel scale conversion points (n_mels + 2 edges).
    let mel_min = hz_to_mel_slaney(f_min);
    let mel_max = hz_to_mel_slaney(f_max);

    let mel_points: Vec<f64> = (0..=(n_mels + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

    // FFT bin frequencies.
    let bin_freqs: Vec<f64> = (0..num_bins)
        .map(|i| sr * i as f64 / n_fft as f64)
        .collect();

    // Build triangular filters with Slaney normalization.
    let mut filters = Vec::with_capacity(n_mels);

    for i in 0..n_mels {
        let f_left = hz_points[i];
        let f_center = hz_points[i + 1];
        let f_right = hz_points[i + 2];

        // Slaney normalization: 2 / (f_right - f_left)
        let norm = 2.0 / (f_right - f_left);

        let filter: Vec<f64> = bin_freqs
            .iter()
            .map(|&f| {
                if f < f_left || f > f_right {
                    0.0
                } else if f <= f_center {
                    norm * (f - f_left) / (f_center - f_left)
                } else {
                    norm * (f_right - f) / (f_right - f_center)
                }
            })
            .collect();

        filters.push(filter);
    }

    filters
}

/// Convert frequency in Hz to Slaney mel scale.
///
/// Below 1000 Hz: linear mapping (mel = 3 * f / 200).
/// Above 1000 Hz: logarithmic (mel = 15 + 27 * ln(f / 1000) / ln(6.4)).
fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < 1000.0 {
        3.0 * hz / 200.0
    } else {
        15.0 + 27.0 * (hz / 1000.0).ln() / (6.4_f64).ln()
    }
}

/// Convert Slaney mel scale to frequency in Hz.
fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < 15.0 {
        200.0 * mel / 3.0
    } else {
        1000.0 * ((mel - 15.0) * (6.4_f64).ln() / 27.0).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_conversion_roundtrip() {
        let test_freqs = [40.0, 100.0, 440.0, 1000.0, 4000.0, 16000.0];
        for &freq in &test_freqs {
            let mel = hz_to_mel_slaney(freq);
            let back = mel_to_hz_slaney(mel);
            assert!(
                (freq - back).abs() < 0.01,
                "roundtrip failed for {freq} Hz: got {back}"
            );
        }
    }

    #[test]
    fn mel_1000hz_is_boundary() {
        // At exactly 1000 Hz, both formulas should agree: mel = 15.0
        let mel = hz_to_mel_slaney(1000.0);
        assert!(
            (mel - 15.0).abs() < 1e-10,
            "mel(1000 Hz) should be 15.0, got {mel}"
        );
    }

    #[test]
    fn hann_window_properties() {
        let w = hann_window(2048);
        assert_eq!(w.len(), 2048);
        // Endpoints should be ~0
        assert!(w[0].abs() < 1e-10);
        // Midpoint should be ~1
        assert!((w[1024] - 1.0).abs() < 1e-10);
        // Symmetric
        assert!((w[100] - w[2048 - 100]).abs() < 1e-10);
    }

    #[test]
    fn filterbank_shape() {
        let fb = mel_filterbank(2048, 128, 44100, 40.0, 16000.0);
        assert_eq!(fb.len(), 128);
        assert_eq!(fb[0].len(), 1025); // n_fft/2 + 1
    }

    #[test]
    fn filterbank_non_negative() {
        let fb = mel_filterbank(2048, 128, 44100, 40.0, 16000.0);
        for (i, filter) in fb.iter().enumerate() {
            for (j, &w) in filter.iter().enumerate() {
                assert!(w >= 0.0, "negative weight at mel={i}, bin={j}: {w}");
            }
        }
    }

    #[test]
    fn filterbank_each_filter_has_nonzero() {
        let fb = mel_filterbank(2048, 128, 44100, 40.0, 16000.0);
        for (i, filter) in fb.iter().enumerate() {
            let sum: f64 = filter.iter().sum();
            assert!(sum > 0.0, "filter {i} is all zeros");
        }
    }

    #[test]
    fn reflect_pad_basic() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad(&signal, 2, 2);
        // Left: signal[2], signal[1] = 3.0, 2.0
        // Right: signal[3], signal[2] = 4.0, 3.0
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn stft_output_shape() {
        let config = MelConfig::default();
        let mel = MelSpectrogram::new(config);

        // 1 second of silence at 44100 Hz
        let samples = vec![0.0_f32; 44100];
        let result = mel.process(&samples);

        assert_eq!(result.len(), 128, "should have 128 mel bins");

        // Expected frames: after padding, ~(44100 + 768 + 769 - 2048) / 512 + 1
        let num_frames = result[0].len();
        assert!(
            num_frames > 80 && num_frames < 90,
            "expected ~86 frames for 1s, got {num_frames}"
        );
    }

    #[test]
    fn mel_spectrogram_sine_wave() {
        let config = MelConfig::default();
        let mel = MelSpectrogram::new(config);

        // 440 Hz sine wave, 0.1 seconds
        let num_samples = 4410;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin() as f32)
            .collect();

        let result = mel.process(&samples);

        // Should have energy (not all the same value).
        let min_val = result
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_val = result
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            max_val > min_val,
            "mel spectrogram should have variation for a sine wave"
        );
    }
}
