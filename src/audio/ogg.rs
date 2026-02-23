//! OGG/Vorbis audio encoding.

use crate::Result;
use std::io::Write;
use std::num::{NonZeroU8, NonZeroU32};

pub fn write_ogg_to<W: Write>(
    writer: W,
    samples: &[f32],
    sample_rate: u32,
    num_channels: u16,
) -> Result<()> {
    let mut encoder = vorbis_rs::VorbisEncoderBuilder::new(
        NonZeroU32::new(sample_rate).unwrap(),
        NonZeroU8::new(num_channels as u8).unwrap(),
        writer,
    )
    .map_err(|e| crate::Error::Audio(format!("vorbis init: {}", e)))?
    .build()
    .map_err(|e| crate::Error::Audio(format!("vorbis build: {}", e)))?;

    let channels: Vec<Vec<f32>> = (0..num_channels as usize)
        .map(|ch| {
            samples
                .iter()
                .skip(ch)
                .step_by(num_channels as usize)
                .copied()
                .collect()
        })
        .collect();

    encoder
        .encode_audio_block(&channels)
        .map_err(|e| crate::Error::Audio(format!("vorbis encode: {}", e)))?;

    Ok(())
}

pub fn write_ogg(
    path: impl AsRef<std::path::Path>,
    samples: &[f32],
    sample_rate: u32,
    num_channels: u16,
) -> Result<()> {
    let file = std::fs::File::create(path)?;
    write_ogg_to(file, samples, sample_rate, num_channels)
}
