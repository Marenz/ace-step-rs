//! Audio processing: mel spectrogram, WAV I/O, resampling.
//!
//! The mel spectrogram uses the same parameters as the original ACE-Step:
//! - Sample rate: 44100 Hz
//! - FFT size: 2048
//! - Hop length: 512
//! - Mel bins: 128
//! - Frequency range: 40–16000 Hz
//! - Mel scale: Slaney

pub mod mel;
pub mod wav;

pub use mel::{MelConfig, MelSpectrogram};
pub use wav::{read_wav, write_wav};
