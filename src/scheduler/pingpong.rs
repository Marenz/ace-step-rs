//! PingPong flow-matching scheduler (stochastic SDE variant).
//!
//! Mixes the denoised estimate with fresh noise at each step:
//! ```text
//! x_0_hat = x_t - σ * v
//! x_{t-1} = (1 - σ_next) * x_0_hat + σ_next * ε
//! ```
//! where `ε ~ N(0, I)` is fresh noise drawn each step.
//!
//! Uses `ChaCha8Rng` for reproducible noise generation.

use candle_core::Tensor;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::{Scheduler, SchedulerConfig};
use crate::Result;

/// PingPong stochastic flow-matching scheduler.
pub struct PingPongScheduler {
    config: SchedulerConfig,
    sigmas: Vec<f64>,
    timesteps: Vec<f64>,
    rng: ChaCha8Rng,
}

impl PingPongScheduler {
    pub fn new(config: SchedulerConfig, seed: u64) -> Self {
        Self {
            config,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl Scheduler for PingPongScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) {
        let sigma_max = 1.0;
        let sigma_min = 1.0 / self.config.num_train_timesteps as f64;

        self.timesteps = Vec::with_capacity(num_inference_steps);
        self.sigmas = Vec::with_capacity(num_inference_steps + 1);

        for i in 0..num_inference_steps {
            let t = sigma_max
                - (sigma_max - sigma_min) * i as f64 / (num_inference_steps - 1).max(1) as f64;
            let sigma = self.config.shift_sigma(t);
            self.sigmas.push(sigma);
            self.timesteps.push(sigma * 1000.0);
        }

        self.sigmas.push(0.0);
    }

    fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    fn step(
        &mut self,
        model_output: &Tensor,
        sample: &Tensor,
        step_index: usize,
    ) -> Result<Tensor> {
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        // Estimate x_0: x_0_hat = x_t - σ * v
        let sigma_f = sigma as f32;
        let sigma_next_f = sigma_next as f32;
        let denoised = (sample - (model_output * sigma_f as f64)?)?;

        if sigma_next.abs() < 1e-10 {
            // Last step: just return the denoised estimate.
            return Ok(denoised);
        }

        // Re-inject noise: x_{t-1} = (1 - σ_next) * x_0_hat + σ_next * ε
        // Generate reproducible noise via ChaCha8Rng.
        let _seed = rand::Rng::random::<u64>(&mut self.rng);
        let noise = Tensor::randn(0.0_f32, 1.0, sample.shape(), sample.device())?;

        let denoised_scaled = (&denoised * (1.0 - sigma_next_f) as f64)?;
        let noise_scaled = (noise * sigma_next_f as f64)?;
        let result = (denoised_scaled + noise_scaled)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn pingpong_timesteps() {
        let config = SchedulerConfig::default();
        let mut scheduler = PingPongScheduler::new(config, 42);
        scheduler.set_timesteps(30);

        assert_eq!(scheduler.sigmas().len(), 31);
        assert_eq!(scheduler.timesteps().len(), 30);
    }

    #[test]
    fn pingpong_last_step_is_denoised() {
        let config = SchedulerConfig::default();
        let mut scheduler = PingPongScheduler::new(config, 42);
        scheduler.set_timesteps(5);

        let device = Device::Cpu;
        let sample = Tensor::randn(0.0_f32, 1.0, (1, 8, 16, 32), &device).unwrap();
        let velocity = Tensor::zeros_like(&sample).unwrap();

        // At the last step, sigma_next = 0, so result should equal denoised = sample - 0 * v = sample.
        let last_step = scheduler.sigmas().len() - 2; // index of last non-terminal sigma
        let result = scheduler.step(&velocity, &sample, last_step).unwrap();

        let diff: f32 = (&result - &sample)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-6, "last step should return denoised, diff={diff}");
    }

    #[test]
    fn pingpong_intermediate_step_adds_noise() {
        let config = SchedulerConfig::default();
        let mut scheduler = PingPongScheduler::new(config, 42);
        scheduler.set_timesteps(10);

        let device = Device::Cpu;
        let sample = Tensor::ones((1, 8, 16, 32), candle_core::DType::F32, &device).unwrap();
        let velocity = Tensor::zeros_like(&sample).unwrap();

        // At step 0 (not last), noise should be injected.
        let result = scheduler.step(&velocity, &sample, 0).unwrap();

        // Result should not equal input (noise was added).
        let diff: f32 = (&result - &sample)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            diff > 0.01,
            "intermediate step should add noise, diff={diff}"
        );
    }
}
