//! WAV file I/O.
//!
//! Read and write 32-bit float WAV files. ACE-Step generates stereo
//! audio at 44100 Hz internally, optionally resampled to 48000 Hz.

use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};

use crate::{Error, Result};

/// Read a WAV file into channel-separated f32 sample buffers.
///
/// Returns `(sample_rate, channels)` where each channel is a `Vec<f32>`.
pub fn read_wav(path: &Path) -> Result<(u32, Vec<Vec<f32>>)> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let num_channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let mut channels: Vec<Vec<f32>> = vec![Vec::new(); num_channels];

    match spec.sample_format {
        SampleFormat::Float => {
            for (i, sample) in reader.into_samples::<f32>().enumerate() {
                let sample = sample.map_err(|e| Error::Audio(e.to_string()))?;
                channels[i % num_channels].push(sample);
            }
        }
        SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            for (i, sample) in reader.into_samples::<i32>().enumerate() {
                let sample = sample.map_err(|e| Error::Audio(e.to_string()))?;
                channels[i % num_channels].push(sample as f32 / max_val);
            }
        }
    }

    Ok((sample_rate, channels))
}

/// Write stereo f32 audio to a WAV file.
///
/// `channels` should contain exactly 2 channels of equal length.
pub fn write_wav(path: &Path, sample_rate: u32, channels: &[Vec<f32>]) -> Result<()> {
    if channels.is_empty() {
        return Err(Error::Audio("no channels to write".into()));
    }

    let num_channels = channels.len();
    let num_samples = channels[0].len();

    for (i, ch) in channels.iter().enumerate() {
        if ch.len() != num_samples {
            return Err(Error::Audio(format!(
                "channel {i} has {} samples, expected {num_samples}",
                ch.len()
            )));
        }
    }

    let spec = WavSpec {
        channels: num_channels as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)?;

    for sample_idx in 0..num_samples {
        for channel in channels {
            writer
                .write_sample(channel[sample_idx])
                .map_err(|e| Error::Audio(e.to_string()))?;
        }
    }

    writer.finalize().map_err(|e| Error::Audio(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn write_and_read_roundtrip() {
        let tmp = NamedTempFile::with_suffix(".wav").unwrap();
        let path = tmp.path();

        let sample_rate = 44100;
        let num_samples = 1000;

        // Stereo sine wave.
        let left: Vec<f32> = (0..num_samples)
            .map(|i| {
                (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate as f64).sin() as f32
            })
            .collect();
        let right: Vec<f32> = (0..num_samples)
            .map(|i| {
                (2.0 * std::f64::consts::PI * 880.0 * i as f64 / sample_rate as f64).sin() as f32
            })
            .collect();

        write_wav(path, sample_rate, &[left.clone(), right.clone()]).unwrap();

        let (read_sr, read_channels) = read_wav(path).unwrap();
        assert_eq!(read_sr, sample_rate);
        assert_eq!(read_channels.len(), 2);
        assert_eq!(read_channels[0].len(), num_samples);
        assert_eq!(read_channels[1].len(), num_samples);

        // Values should match closely (f32 precision).
        for i in 0..num_samples {
            assert!(
                (read_channels[0][i] - left[i]).abs() < 1e-6,
                "left mismatch at {i}"
            );
            assert!(
                (read_channels[1][i] - right[i]).abs() < 1e-6,
                "right mismatch at {i}"
            );
        }
    }

    #[test]
    fn write_mono() {
        let tmp = NamedTempFile::with_suffix(".wav").unwrap();
        let path = tmp.path();

        let samples = vec![0.0_f32; 100];
        write_wav(path, 44100, &[samples]).unwrap();

        let (sr, channels) = read_wav(path).unwrap();
        assert_eq!(sr, 44100);
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].len(), 100);
    }

    #[test]
    fn empty_channels_error() {
        let tmp = NamedTempFile::with_suffix(".wav").unwrap();
        let result = write_wav(tmp.path(), 44100, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn mismatched_channel_lengths_error() {
        let tmp = NamedTempFile::with_suffix(".wav").unwrap();
        let result = write_wav(tmp.path(), 44100, &[vec![0.0; 100], vec![0.0; 200]]);
        assert!(result.is_err());
    }
}
