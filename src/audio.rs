//! Audio I/O utilities.
//!
//! WAV read/write at 48kHz stereo for ACE-Step v1.5.

mod wav;

pub use wav::{read_wav, write_wav};
