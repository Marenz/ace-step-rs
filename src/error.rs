//! Error types for ace-step-rs.

use std::fmt;

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Candle tensor/model error.
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),

    /// Tokenizer error.
    #[error("tokenizer: {0}")]
    Tokenizer(TokenizerError),

    /// Audio processing error (STFT, resampling, WAV I/O).
    #[error("audio: {0}")]
    Audio(String),

    /// Model weight loading error.
    #[error("weight loading: {0}")]
    WeightLoad(String),

    /// Invalid configuration.
    #[error("config: {0}")]
    Config(String),

    /// I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// HuggingFace Hub error.
    #[error("hf-hub: {0}")]
    HfHub(String),
}

/// Wrapper for tokenizer errors (tokenizers::Error doesn't impl std::error::Error).
#[derive(Debug)]
pub struct TokenizerError(pub String);

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<tokenizers::Error> for Error {
    fn from(error: tokenizers::Error) -> Self {
        Error::Tokenizer(TokenizerError(error.to_string()))
    }
}

impl From<hound::Error> for Error {
    fn from(error: hound::Error) -> Self {
        Error::Audio(error.to_string())
    }
}
