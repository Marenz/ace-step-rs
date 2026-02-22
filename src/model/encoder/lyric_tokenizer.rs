//! Lyric BPE tokenizer.
//!
//! Custom BPE tokenizer for lyrics with:
//! - Language detection per line → prefix token `[en]`, `[zh-cn]`, etc.
//! - Spaces replaced with `[SPACE]`
//! - Numbers/abbreviations expanded
//! - Start token `261` prepended
//! - Newlines become token `2`
//! - Vocabulary size: 6693
//!
//! The vocabulary is loaded from a `vocab.json` file distributed with
//! the model weights.

// TODO: implement VoiceBpeTokenizer
// TODO: implement language detection (can be simplified — just detect CJK vs Latin)

/// Special token IDs.
pub const START_TOKEN: u32 = 261;
pub const NEWLINE_TOKEN: u32 = 2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_tokens() {
        assert_eq!(START_TOKEN, 261);
        assert_eq!(NEWLINE_TOKEN, 2);
    }
}
