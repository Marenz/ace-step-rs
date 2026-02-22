//! Configuration for ACE-Step v1.5 model.
//!
//! Matches the Python `AceStepConfig` defaults from the turbo variant.

use serde::{Deserialize, Serialize};

/// Attention layer type — alternating sliding window and full attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

/// Top-level model configuration matching the Python AceStepConfig.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AceStepConfig {
    // --- Core transformer ---
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub attention_bias: bool,

    // --- Sliding window ---
    pub use_sliding_window: bool,
    pub sliding_window: usize,
    pub layer_types: Vec<LayerType>,

    // --- Text encoder ---
    pub text_hidden_dim: usize,

    // --- Lyric encoder ---
    pub num_lyric_encoder_hidden_layers: usize,

    // --- Audio ---
    pub audio_acoustic_hidden_dim: usize,
    pub pool_window_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,

    // --- FSQ ---
    pub fsq_dim: usize,
    pub fsq_input_levels: Vec<usize>,
    pub fsq_input_num_quantizers: usize,

    // --- Timbre encoder ---
    pub timbre_hidden_dim: usize,
    pub num_timbre_encoder_hidden_layers: usize,
    pub timbre_fix_frame: usize,

    // --- Pooler / detokenizer ---
    pub num_attention_pooler_hidden_layers: usize,

    // --- Audio decoder ---
    pub num_audio_decoder_hidden_layers: usize,

    // --- Timestep sampling ---
    pub timestep_mu: f64,
    pub timestep_sigma: f64,
    pub data_proportion: f64,

    // --- Model variant ---
    pub model_version: String,
}

impl Default for AceStepConfig {
    fn default() -> Self {
        let num_hidden_layers = 24;
        // Default layer types: odd layers = sliding, even layers = full
        let layer_types = (0..num_hidden_layers)
            .map(|i| {
                if (i + 1) % 2 == 1 {
                    LayerType::SlidingAttention
                } else {
                    LayerType::FullAttention
                }
            })
            .collect();

        Self {
            vocab_size: 64003,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
            attention_bias: false,
            use_sliding_window: true,
            sliding_window: 128,
            layer_types,
            text_hidden_dim: 1024,
            num_lyric_encoder_hidden_layers: 8,
            audio_acoustic_hidden_dim: 64,
            pool_window_size: 5,
            in_channels: 192,
            patch_size: 2,
            fsq_dim: 2048,
            fsq_input_levels: vec![8, 8, 8, 5, 5, 5],
            fsq_input_num_quantizers: 1,
            timbre_hidden_dim: 64,
            num_timbre_encoder_hidden_layers: 4,
            timbre_fix_frame: 750,
            num_attention_pooler_hidden_layers: 2,
            num_audio_decoder_hidden_layers: 24,
            timestep_mu: -0.4,
            timestep_sigma: 1.0,
            data_proportion: 0.5,
            model_version: "turbo".to_string(),
        }
    }
}

impl AceStepConfig {
    /// Number of GQA groups (num_attention_heads / num_key_value_heads).
    pub fn num_key_value_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// VAE configuration for AutoencoderOobleck.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaeConfig {
    pub encoder_hidden_size: usize,
    pub downsampling_ratios: Vec<usize>,
    pub channel_multiples: Vec<usize>,
    pub decoder_channels: usize,
    pub decoder_input_channels: usize,
    pub audio_channels: usize,
    pub sampling_rate: u32,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            encoder_hidden_size: 128,
            downsampling_ratios: vec![2, 4, 4, 6, 10],
            channel_multiples: vec![1, 2, 4, 8, 16],
            decoder_channels: 128,
            decoder_input_channels: 64,
            audio_channels: 2,
            sampling_rate: 48000,
        }
    }
}

impl VaeConfig {
    /// Total hop length = product of all downsampling ratios.
    pub fn hop_length(&self) -> usize {
        self.downsampling_ratios.iter().product()
    }

    /// Latent frames per second.
    pub fn latent_fps(&self) -> f64 {
        self.sampling_rate as f64 / self.hop_length() as f64
    }
}

/// Pre-defined timestep schedules for turbo inference (fix_nfe=8).
pub struct TurboSchedule;

impl TurboSchedule {
    pub fn for_shift(shift: f64) -> Vec<f64> {
        match shift as u32 {
            1 => vec![1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
            2 => vec![
                1.0,
                0.9333333333333333,
                0.8571428571428571,
                0.7692307692307693,
                0.6666666666666666,
                0.5454545454545454,
                0.4,
                0.2222222222222222,
            ],
            3 => vec![
                1.0,
                0.9545454545454546,
                0.9,
                0.8333333333333334,
                0.75,
                0.6428571428571429,
                0.5,
                0.3,
            ],
            _ => Self::for_shift(3.0), // default to shift=3
        }
    }

    /// All valid timestep values across all shifts.
    pub fn valid_timesteps() -> &'static [f64] {
        &[
            1.0,
            0.9545454545454546,
            0.9333333333333333,
            0.9,
            0.875,
            0.8571428571428571,
            0.8333333333333334,
            0.7692307692307693,
            0.75,
            0.6666666666666666,
            0.6428571428571429,
            0.625,
            0.5454545454545454,
            0.5,
            0.4,
            0.375,
            0.3,
            0.25,
            0.2222222222222222,
            0.125,
        ]
    }

    /// Map a timestep value to the nearest valid timestep.
    pub fn nearest_valid(t: f64) -> f64 {
        *Self::valid_timesteps()
            .iter()
            .min_by(|a, b| ((*a - t).abs()).partial_cmp(&((*b - t).abs())).unwrap())
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = AceStepConfig::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.num_key_value_groups(), 2);
        assert_eq!(cfg.layer_types.len(), 24);
        // Layer 0 (idx 0): (0+1)%2 == 1 → sliding
        assert_eq!(cfg.layer_types[0], LayerType::SlidingAttention);
        // Layer 1 (idx 1): (1+1)%2 == 0 → full
        assert_eq!(cfg.layer_types[1], LayerType::FullAttention);
    }

    #[test]
    fn test_vae_config() {
        let vae = VaeConfig::default();
        assert_eq!(vae.hop_length(), 1920); // 2*4*4*6*10
        assert_eq!(vae.sampling_rate, 48000);
        assert!((vae.latent_fps() - 25.0).abs() < 0.01); // 48000/1920 = 25Hz
    }

    #[test]
    fn test_turbo_schedule() {
        let s3 = TurboSchedule::for_shift(3.0);
        assert_eq!(s3.len(), 8);
        assert_eq!(s3[0], 1.0);
        assert_eq!(s3[7], 0.3);
    }

    #[test]
    fn test_nearest_valid_timestep() {
        assert_eq!(TurboSchedule::nearest_valid(0.99), 1.0);
        assert_eq!(TurboSchedule::nearest_valid(0.5), 0.5);
        assert_eq!(TurboSchedule::nearest_valid(0.31), 0.3);
    }
}
