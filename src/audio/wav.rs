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
