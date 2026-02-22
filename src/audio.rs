//! Audio I/O utilities.
//!
//! WAV read/write at 48kHz stereo for ACE-Step v1.5.
//! Optional OGG encoding via feature flag.

mod wav;

#[cfg(feature = "audio-ogg")]
mod ogg;

pub use wav::{read_wav, write_wav};

/// Output audio format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AudioFormat {
    #[default]
    Wav,
    Ogg,
}

impl AudioFormat {
    /// Parse from string (wav, ogg).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "wav" => Some(Self::Wav),
            "ogg" | "vorbis" => Some(Self::Ogg),
            _ => None,
        }
    }

    /// Get file extension.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Ogg => "ogg",
        }
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.extension())
    }
}

/// Write interleaved f32 samples to the specified format.
pub fn write_audio(
    path: impl AsRef<std::path::Path>,
    samples: &[f32],
    sample_rate: u32,
    num_channels: u16,
) -> crate::Result<()> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("wav")
        .to_lowercase();

    match AudioFormat::from_str(&ext) {
        Some(AudioFormat::Wav) => write_wav(path, samples, sample_rate, num_channels),
        #[cfg(feature = "audio-ogg")]
        Some(AudioFormat::Ogg) => ogg::write_ogg(path, samples, sample_rate, num_channels),
        #[cfg(not(feature = "audio-ogg"))]
        Some(AudioFormat::Ogg) => Err(crate::Error::Audio(
            "OGG not enabled. Build with --features audio-ogg".to_string(),
        )),
        None => Err(crate::Error::Audio(format!("Unknown format: {}", ext))),
    }
}
