//! Condition encoder — combines text, lyrics, and timbre.
//!
//! Packs all condition sequences into a single sequence for cross-attention,
//! with valid tokens sorted to the front.

use candle_core::DType;
use candle_core::{Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

use super::lyric::AceStepLyricEncoder;
use super::timbre::AceStepTimbreEncoder;
use crate::config::AceStepConfig;

/// Pack two sequences by concatenating and sorting valid tokens first.
///
/// hidden1 [B, L1, D] + hidden2 [B, L2, D] → [B, L1+L2, D] with valid tokens first.
pub fn pack_sequences(
    hidden1: &Tensor,
    hidden2: &Tensor,
    mask1: &Tensor,
    mask2: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let hidden_cat = Tensor::cat(&[hidden1, hidden2], 1)?;
    let mask_cat = Tensor::cat(&[mask1, mask2], 1)?;

    let (b, l, d) = hidden_cat.dims3()?;

    // Sort so valid tokens (mask=1) come first via argsort descending
    let sort_idx = mask_cat.arg_sort_last_dim(false)?; // descending: 1s first

    // Gather hidden states
    let sort_idx_expanded = sort_idx.unsqueeze(2)?.expand((b, l, d))?.contiguous()?;
    let hidden_sorted = hidden_cat.contiguous()?.gather(&sort_idx_expanded, 1)?;

    // Create new mask based on valid counts
    let lengths = mask_cat.sum(1)?.to_dtype(DType::I64)?; // [B]
    let positions = Tensor::arange(0i64, l as i64, mask_cat.device())?.unsqueeze(0)?; // [1, L]
    let lengths_expanded = lengths.unsqueeze(1)?; // [B, 1]
    let new_mask = positions.broadcast_lt(&lengths_expanded)?;
    let new_mask = new_mask.to_dtype(mask1.dtype())?;

    Ok((hidden_sorted, new_mask))
}

/// Combined condition encoder matching Python `AceStepConditionEncoder`.
#[derive(Debug, Clone)]
pub struct AceStepConditionEncoder {
    text_projector: nn::Linear,
    lyric_encoder: AceStepLyricEncoder,
    timbre_encoder: AceStepTimbreEncoder,
}

impl AceStepConditionEncoder {
    pub fn new(
        cfg: &AceStepConfig,
        dtype: DType,
        dev: &candle_core::Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let text_projector = nn::linear_no_bias(
            cfg.text_hidden_dim,
            cfg.hidden_size,
            vb.pp("text_projector"),
        )?;
        let lyric_encoder = AceStepLyricEncoder::new(cfg, dtype, dev, vb.pp("lyric_encoder"))?;
        let timbre_encoder = AceStepTimbreEncoder::new(cfg, dtype, dev, vb.pp("timbre_encoder"))?;

        Ok(Self {
            text_projector,
            lyric_encoder,
            timbre_encoder,
        })
    }

    /// Encode all conditions and pack into a single sequence.
    ///
    /// Returns (encoder_hidden_states [B, S, D], encoder_attention_mask [B, S]).
    pub fn forward(
        &self,
        text_hidden_states: &Tensor,
        text_attention_mask: &Tensor,
        lyric_hidden_states: &Tensor,
        lyric_attention_mask: &Tensor,
        refer_audio_packed: &Tensor,
        refer_audio_order_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Project text
        let text_h = text_hidden_states.apply(&self.text_projector)?;

        // Encode lyrics
        let lyric_h = self
            .lyric_encoder
            .forward(lyric_hidden_states, lyric_attention_mask)?;

        // Encode timbre
        let (timbre_h, timbre_mask) = self
            .timbre_encoder
            .forward(refer_audio_packed, refer_audio_order_mask)?;
        let timbre_mask = timbre_mask.to_dtype(lyric_attention_mask.dtype())?;

        // Pack: lyrics + timbre, then + text
        let (packed, mask) =
            pack_sequences(&lyric_h, &timbre_h, lyric_attention_mask, &timbre_mask)?;
        let text_mask = text_attention_mask.to_dtype(mask.dtype())?;
        pack_sequences(&packed, &text_h, &mask, &text_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_pack_sequences() {
        let dev = Device::Cpu;
        let h1 = Tensor::ones((1, 3, 4), DType::F32, &dev).unwrap();
        let h2 = Tensor::ones((1, 2, 4), DType::F32, &dev).unwrap();
        let m1 = Tensor::new(&[[1.0f32, 1.0, 0.0]], &dev).unwrap();
        let m2 = Tensor::new(&[[1.0f32, 0.0]], &dev).unwrap();
        let (packed, mask) = pack_sequences(&h1, &h2, &m1, &m2).unwrap();
        assert_eq!(packed.dims(), &[1, 5, 4]);
        assert_eq!(mask.dims(), &[1, 5]);
        // 3 valid tokens total (2 from h1, 1 from h2)
        let mask_sum: f32 = mask.sum_all().unwrap().to_scalar().unwrap();
        assert_eq!(mask_sum, 3.0);
    }
}
