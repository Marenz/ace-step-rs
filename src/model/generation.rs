//! Top-level generation model combining all components.
//!
//! `AceStepConditionGenerationModel` orchestrates:
//! - Condition encoding (text + lyrics + timbre → packed sequence)
//! - Diffusion (noisy latents → velocity prediction via DiT)
//! - Flow matching inference (8-step turbo ODE)

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use super::encoder::condition::AceStepConditionEncoder;
use super::transformer::dit::AceStepDiTModel;
use crate::config::{AceStepConfig, TurboSchedule};

/// Output from audio generation.
pub struct GenerationOutput {
    /// Generated latents [B, T, acoustic_dim] at 25Hz.
    pub target_latents: Tensor,
}

/// Top-level generation model matching Python `AceStepConditionGenerationModel`.
#[derive(Debug, Clone)]
pub struct AceStepConditionGenerationModel {
    decoder: AceStepDiTModel,
    encoder: AceStepConditionEncoder,
    null_condition_emb: Tensor, // [1, 1, hidden_size]
    cfg: AceStepConfig,
}

impl AceStepConditionGenerationModel {
    pub fn new(
        cfg: &AceStepConfig,
        dtype: DType,
        dev: &candle_core::Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let decoder = AceStepDiTModel::new(cfg, dtype, dev, vb.pp("decoder"))?;
        let encoder = AceStepConditionEncoder::new(cfg, dtype, dev, vb.pp("encoder"))?;
        let null_condition_emb = vb.get((1, 1, cfg.hidden_size), "null_condition_emb")?;

        Ok(Self {
            decoder,
            encoder,
            null_condition_emb,
            cfg: cfg.clone(),
        })
    }

    /// Run the condition encoder only.
    pub fn encode_conditions(
        &self,
        text_hidden_states: &Tensor,
        text_attention_mask: &Tensor,
        lyric_hidden_states: &Tensor,
        lyric_attention_mask: &Tensor,
        refer_audio_packed: &Tensor,
        refer_audio_order_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.encoder.forward(
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
            refer_audio_packed,
            refer_audio_order_mask,
        )
    }

    /// Run a single DiT forward pass (for debugging/comparison).
    pub fn dit_forward(
        &self,
        xt: &Tensor,
        timestep: &Tensor,
        timestep_r: &Tensor,
        encoder_hidden_states: &Tensor,
        context_latents: &Tensor,
    ) -> Result<Tensor> {
        self.decoder.forward(
            xt,
            timestep,
            timestep_r,
            encoder_hidden_states,
            context_latents,
        )
    }

    /// Generate audio latents from text/lyrics/timbre conditions.
    ///
    /// Returns latents [B, T, 64] at 25Hz, ready for VAE decoding.
    pub fn generate_audio(
        &self,
        text_hidden_states: &Tensor,
        text_attention_mask: &Tensor,
        lyric_hidden_states: &Tensor,
        lyric_attention_mask: &Tensor,
        refer_audio_packed: &Tensor,
        refer_audio_order_mask: &Tensor,
        src_latents: &Tensor,
        chunk_masks: &Tensor,
        shift: f64,
        seed: Option<u64>,
    ) -> Result<GenerationOutput> {
        let dev = src_latents.device();
        let dtype = src_latents.dtype();
        let b = src_latents.dim(0)?;
        let t = src_latents.dim(1)?;

        // 1. Encode conditions
        let (enc_hidden, _enc_mask) = self.encoder.forward(
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
            refer_audio_packed,
            refer_audio_order_mask,
        )?;

        // 2. Build context: cat[src_latents, chunk_masks] → [B, T, 128]
        let context = Tensor::cat(&[src_latents, chunk_masks], 2)?;

        // 3. Initialize noise
        let noise = if let Some(_seed) = seed {
            // TODO: use seed for deterministic noise generation
            let cpu_dev = candle_core::Device::Cpu;
            Tensor::randn(
                0f32,
                1.0,
                (b, t, self.cfg.audio_acoustic_hidden_dim),
                &cpu_dev,
            )?
            .to_dtype(dtype)?
            .to_device(dev)?
        } else {
            Tensor::randn(0f32, 1.0, (b, t, self.cfg.audio_acoustic_hidden_dim), dev)?
                .to_dtype(dtype)?
        };

        // 4. Get timestep schedule
        let schedule = TurboSchedule::for_shift(shift);
        let num_steps = schedule.len();

        // 5. ODE integration loop
        let mut xt = noise;
        for step_idx in 0..num_steps {
            let t_curr = schedule[step_idx];
            let t_curr_tensor = Tensor::full(t_curr as f32, (b,), dev)?.to_dtype(dtype)?;

            // Forward through DiT
            let vt = self.decoder.forward(
                &xt,
                &t_curr_tensor,
                &t_curr_tensor, // timestep_r = t for standard inference
                &enc_hidden,
                &context,
            )?;

            // On final step: predict x0 directly
            if step_idx == num_steps - 1 {
                // x0 = xt - vt * t
                let t_expand = t_curr_tensor.unsqueeze(1)?.unsqueeze(2)?;
                xt = (xt - vt.broadcast_mul(&t_expand)?)?;
                break;
            }

            // ODE Euler step: xt = xt - vt * dt
            let t_next = schedule[step_idx + 1];
            let dt = (t_curr - t_next) as f32;
            let dt_tensor = Tensor::full(dt, (b, 1, 1), dev)?.to_dtype(dtype)?;
            xt = (xt - vt.broadcast_mul(&dt_tensor)?)?;
        }

        Ok(GenerationOutput { target_latents: xt })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_generation_model_placeholder() {
        // Integration test requires actual weights — tested via pipeline
    }
}
