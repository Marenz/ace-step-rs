//! MP3 encoding via libmp3lame.
//!
//! Requires the `audio-mp3` feature and `libmp3lame` to be installed on the system.

use std::io::Write;
use std::path::Path;

use mp3lame_encoder::{Builder, FlushNoGap, InterleavedPcm, Quality};

use crate::{Error, Result};

/// Write interleaved f32 samples to a file as MP3 at 192 kbps.
///
/// `samples` must be interleaved stereo: `[L, R, L, R, ...]`.
/// `sample_rate` should be 48000 (ACE-Step's native rate).
pub fn write_mp3(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
    num_channels: u16,
) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())
        .map_err(|e| Error::Audio(format!("failed to create MP3 file: {e}")))?;
    write_mp3_to(file, samples, sample_rate, num_channels)
}

/// Write interleaved f32 samples to a writer as MP3 at 192 kbps.
pub fn write_mp3_to<W: Write>(
    mut writer: W,
    samples: &[f32],
    sample_rate: u32,
    num_channels: u16,
) -> Result<()> {
    if num_channels != 2 {
        return Err(Error::Audio(format!(
            "MP3 encoder only supports stereo (2 channels), got {num_channels}"
        )));
    }

    let mut encoder = Builder::new()
        .ok_or_else(|| Error::Audio("failed to create LAME encoder".into()))?
        .with_num_channels(num_channels as u8)
        .map_err(|e| Error::Audio(format!("LAME set_num_channels failed: {e:?}")))?
        .with_sample_rate(sample_rate)
        .map_err(|e| Error::Audio(format!("LAME set_sample_rate failed: {e:?}")))?
        .with_brate(mp3lame_encoder::Bitrate::Kbps192)
        .map_err(|e| Error::Audio(format!("LAME set_brate failed: {e:?}")))?
        .with_quality(Quality::Best)
        .map_err(|e| Error::Audio(format!("LAME set_quality failed: {e:?}")))?
        .build()
        .map_err(|e| Error::Audio(format!("LAME build failed: {e:?}")))?;

    let num_samples_per_channel = samples.len() / num_channels as usize;
    let mut buf = Vec::new();
    buf.reserve(mp3lame_encoder::max_required_buffer_size(
        num_samples_per_channel,
    ));

    let encoded_size = encoder
        .encode(InterleavedPcm(samples), buf.spare_capacity_mut())
        .map_err(|e| Error::Audio(format!("LAME encode failed: {e:?}")))?;
    // SAFETY: encode filled exactly `encoded_size` bytes into spare capacity.
    unsafe { buf.set_len(encoded_size) };

    let flush_size = encoder
        .flush::<FlushNoGap>(buf.spare_capacity_mut())
        .map_err(|e| Error::Audio(format!("LAME flush failed: {e:?}")))?;
    // SAFETY: flush filled exactly `flush_size` bytes into spare capacity.
    unsafe { buf.set_len(buf.len() + flush_size) };

    writer
        .write_all(&buf)
        .map_err(|e| Error::Audio(format!("failed to write MP3 data: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_mp3_produces_nonempty_output() {
        // 1 second of silence at 48kHz stereo
        let samples = vec![0f32; 48000 * 2];
        let mut buf = std::io::Cursor::new(Vec::new());
        write_mp3_to(&mut buf, &samples, 48000, 2).expect("MP3 encode should succeed");
        let out = buf.into_inner();
        assert!(!out.is_empty(), "MP3 output should not be empty");
        // MP3 files start with either an ID3 header (0x49 0x44 0x33) or a sync word (0xFF 0xFB/0xFA/0xF3...)
        // With LAME and no ID3 tag, expect a sync frame
        assert!(
            out[0] == 0xFF || out[0] == 0x49,
            "expected MP3 sync or ID3 header, got 0x{:02X}",
            out[0]
        );
    }

    #[test]
    fn test_write_mp3_rejects_mono() {
        let samples = vec![0f32; 48000];
        let mut buf = std::io::Cursor::new(Vec::new());
        let result = write_mp3_to(&mut buf, &samples, 48000, 1);
        assert!(result.is_err());
    }
}
