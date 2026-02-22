//! OGG/Vorbis audio encoding.

use crate::Result;
use std::num::{NonZeroU32, NonZeroU8};
use std::path::Path;

pub fn write_ogg(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
    num_channels: u16,
) -> Result<()> {
    let path = path.as_ref();

    let file = std::fs::File::create(path)?;

    let mut encoder = vorbis_rs::VorbisEncoderBuilder::new(
        NonZeroU32::new(sample_rate).unwrap(),
        NonZeroU8::new(num_channels as u8).unwrap(),
        file,
    )
    .map_err(|e| crate::Error::Audio(format!("vorbis init: {}", e)))?
    .build()
    .map_err(|e| crate::Error::Audio(format!("vorbis build: {}", e)))?;

    // Convert interleaved to channel-separated
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
