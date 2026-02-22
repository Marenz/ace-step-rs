//! WAV file I/O at 48kHz stereo.

use crate::Result;
use std::path::Path;

/// Read a WAV file, return (samples, sample_rate, num_channels).
///
/// Samples are interleaved f32 in [-1, 1].
pub fn read_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32, u16)> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };

    Ok((samples, sample_rate, channels))
}

/// Write interleaved f32 samples as a WAV file.
pub fn write_wav(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
    num_channels: u16,
) -> Result<()> {
    let spec = hound::WavSpec {
        channels: num_channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Peak-normalize audio samples to [-1, 1].
pub fn peak_normalize(samples: &mut [f32]) {
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max_abs > 1e-8 {
        let scale = 1.0 / max_abs;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    }
}

/// Equal-power crossfade between the tail of `prev` and the head of `next`.
///
/// `crossfade_samples` is the number of *per-channel* samples to crossfade.
/// For stereo interleaved audio, the actual number of f32 values affected is
/// `crossfade_samples * channels`.
///
/// Returns the merged audio: `prev[..overlap_start] ++ blended ++ next[overlap_end..]`.
/// Both inputs must be interleaved with the same channel count.
pub fn crossfade(prev: &[f32], next: &[f32], crossfade_samples: usize, channels: u16) -> Vec<f32> {
    let ch = channels as usize;
    let fade_frames = crossfade_samples; // per-channel frames to crossfade
    let fade_values = fade_frames * ch; // total interleaved values in crossfade zone

    // If either buffer is too short for the crossfade, just concatenate.
    if prev.len() < fade_values || next.len() < fade_values || fade_frames == 0 {
        let mut out = Vec::with_capacity(prev.len() + next.len());
        out.extend_from_slice(prev);
        out.extend_from_slice(next);
        return out;
    }

    let prev_keep = prev.len() - fade_values;
    let next_skip = fade_values;

    let mut out = Vec::with_capacity(prev_keep + fade_values + (next.len() - next_skip));

    // Copy non-overlapping prefix from prev
    out.extend_from_slice(&prev[..prev_keep]);

    // Equal-power crossfade in the overlap zone
    let prev_tail = &prev[prev_keep..];
    let next_head = &next[..fade_values];

    for i in 0..fade_frames {
        // Linear position in [0, 1]
        let t = (i as f64 + 0.5) / fade_frames as f64;
        // Equal-power gains: at t=0, prev=1/next=0; at t=1, prev=0/next=1
        let gain_prev = (t * std::f64::consts::FRAC_PI_2).cos() as f32;
        let gain_next = (t * std::f64::consts::FRAC_PI_2).sin() as f32;
        for c in 0..ch {
            let idx = i * ch + c;
            out.push(prev_tail[idx] * gain_prev + next_head[idx] * gain_next);
        }
    }

    // Copy non-overlapping suffix from next
    out.extend_from_slice(&next[next_skip..]);

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_normalize() {
        let mut samples = vec![0.5, -0.25, 0.1];
        peak_normalize(&mut samples);
        assert!((samples[0] - 1.0).abs() < 1e-6);
        assert!((samples[1] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_equal_power() {
        // Mono: prev = [1.0; 200], next = [0.0; 200], crossfade 100 samples
        let prev = vec![1.0f32; 200];
        let next = vec![0.0f32; 200];
        let result = crossfade(&prev, &next, 100, 1);
        // Output length: (200-100) + 100 + (200-100) = 300
        assert_eq!(result.len(), 300);
        // First 100 samples should be untouched prev
        for &s in &result[..100] {
            assert_eq!(s, 1.0);
        }
        // Last 100 samples should be untouched next
        for &s in &result[200..] {
            assert_eq!(s, 0.0);
        }
        // First crossfade sample (t≈0) should be very close to 1.0 (prev)
        assert!(
            result[100] > 0.95,
            "first fade sample {} not near 1.0",
            result[100]
        );
        // Last crossfade sample (t≈1) should be very close to 0.0 (next)
        assert!(
            result[199] < 0.05,
            "last fade sample {} not near 0.0",
            result[199]
        );
        // Middle of crossfade: cos(π/4) ≈ 0.707 (equal-power mid-point)
        let mid = result[150];
        assert!(
            (mid - 0.707).abs() < 0.05,
            "mid-fade sample {} not near 0.707",
            mid
        );
        // Crossfade should be monotonically decreasing
        for i in 100..199 {
            assert!(
                result[i] >= result[i + 1] - 0.001,
                "non-monotonic at {i}: {} < {}",
                result[i],
                result[i + 1]
            );
        }
    }

    #[test]
    fn test_crossfade_stereo() {
        // Stereo interleaved: 8 frames = 16 values per buffer
        let prev = vec![1.0f32; 16]; // 8 stereo frames
        let next = vec![0.0f32; 16];
        let result = crossfade(&prev, &next, 4, 2); // 4-frame crossfade
                                                    // (16-8) + 8 + (16-8) = 24
        assert_eq!(result.len(), 24);
    }

    #[test]
    fn test_crossfade_too_short() {
        // If buffers are shorter than crossfade, should just concatenate
        let prev = vec![1.0f32; 3];
        let next = vec![0.5f32; 3];
        let result = crossfade(&prev, &next, 10, 1);
        assert_eq!(result.len(), 6);
        assert_eq!(&result[..3], &[1.0; 3]);
        assert_eq!(&result[3..], &[0.5; 3]);
    }

    #[test]
    fn test_roundtrip_wav() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wav");
        let original = vec![0.0f32, 0.5, -0.5, 1.0, -1.0, 0.25];
        write_wav(&path, &original, 48000, 2).unwrap();
        let (loaded, sr, ch) = read_wav(&path).unwrap();
        assert_eq!(sr, 48000);
        assert_eq!(ch, 2);
        assert_eq!(loaded.len(), original.len());
        for (a, b) in loaded.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
