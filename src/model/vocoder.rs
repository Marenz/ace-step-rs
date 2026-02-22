//! ADaMoSHiFiGAN vocoder: mel spectrogram → audio waveform.
//!
//! Two-component architecture:
//!
//! ## ConvNeXt backbone
//! ```text
//! Input: [B, 128, T_mel]
//! Stem:   Conv1d(128, 128, k=7) + LayerNorm
//! Stage0: 3 × ConvNeXtBlock(128)
//! Down:   LayerNorm + Conv1d(128, 256, k=1)
//! Stage1: 3 × ConvNeXtBlock(256)
//! Down:   LayerNorm + Conv1d(256, 384, k=1)
//! Stage2: 9 × ConvNeXtBlock(384)
//! Down:   LayerNorm + Conv1d(384, 512, k=1)
//! Stage3: 3 × ConvNeXtBlock(512)
//! Final:  LayerNorm(512)
//! Output: [B, 512, T_mel]
//! ```
//!
//! ## HiFiGAN head
//! ```text
//! Input: [B, 512, T_mel]
//! conv_pre(512, 1024, k=13)
//! 7 upsample stages: rates=[4,4,2,2,2,2,2], product=512=hop_length
//! Each: SiLU + ConvTranspose1d + multi-kernel ResBlocks
//! conv_post → [B, 1, T_audio]
//! tanh
//! Output: T_audio = T_mel × 512
//! ```
//!
//! For stereo: the vocoder is called twice (once per channel).

// TODO: implement ConvNeXtEncoder (backbone)
// TODO: implement HiFiGANGenerator (head)
// TODO: implement ConvNeXtBlock
// TODO: implement ResBlock1

/// Vocoder configuration.
#[derive(Debug, Clone)]
pub struct VocoderConfig {
    /// ConvNeXt stage depths.
    pub backbone_depths: Vec<usize>,
    /// ConvNeXt stage dimensions.
    pub backbone_dims: Vec<usize>,
    /// HiFiGAN upsample rates (product must equal hop_length=512).
    pub upsample_rates: Vec<usize>,
    /// HiFiGAN upsample kernel sizes.
    pub upsample_kernel_sizes: Vec<usize>,
    /// HiFiGAN ResBlock kernel sizes.
    pub resblock_kernel_sizes: Vec<usize>,
    /// HiFiGAN initial channel count.
    pub upsample_initial_channel: usize,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            backbone_depths: vec![3, 3, 9, 3],
            backbone_dims: vec![128, 256, 384, 512],
            upsample_rates: vec![4, 4, 2, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11, 13],
            upsample_initial_channel: 1024,
        }
    }
}

impl VocoderConfig {
    /// Verify that upsample rates multiply to the hop length.
    pub fn verify(&self) -> crate::Result<()> {
        let product: usize = self.upsample_rates.iter().product();
        if product != 512 {
            return Err(crate::Error::Config(format!(
                "upsample_rates product is {product}, expected 512 (hop_length)"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_upsample_product() {
        let config = VocoderConfig::default();
        config.verify().unwrap();
    }

    #[test]
    fn backbone_dims_increasing() {
        let config = VocoderConfig::default();
        for i in 1..config.backbone_dims.len() {
            assert!(
                config.backbone_dims[i] > config.backbone_dims[i - 1],
                "dims should increase: {} vs {}",
                config.backbone_dims[i - 1],
                config.backbone_dims[i]
            );
        }
    }

    #[test]
    fn bad_upsample_rates_rejected() {
        let config = VocoderConfig {
            upsample_rates: vec![4, 4, 2, 2],
            ..Default::default()
        };
        assert!(config.verify().is_err());
    }
}
