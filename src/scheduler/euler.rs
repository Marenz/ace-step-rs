//! Euler flow-matching scheduler.
//!
//! First-order ODE integrator with omega mean-shifting.
//!
//! Update rule:
//! ```text
//! dx = (σ_next - σ) * v
//! m  = mean(dx)
//! dx' = (dx - m) * ω + m     // ω ≈ 1.073
//! x_next = x + dx'
//! ```

use candle_core::Tensor;

use super::{Scheduler, SchedulerConfig};
use crate::Result;

/// Euler flow-matching scheduler.
pub struct EulerScheduler {
    config: SchedulerConfig,
    sigmas: Vec<f64>,
    timesteps: Vec<f64>,
}

impl EulerScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
        }
    }
}

impl Scheduler for EulerScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) {
        let sigma_max = 1.0;
        let sigma_min = 1.0 / self.config.num_train_timesteps as f64;

        // Linear spacing from max to min, then shift.
        self.timesteps = Vec::with_capacity(num_inference_steps);
        self.sigmas = Vec::with_capacity(num_inference_steps + 1);

        for i in 0..num_inference_steps {
            let t = sigma_max
                - (sigma_max - sigma_min) * i as f64 / (num_inference_steps - 1).max(1) as f64;
            let sigma = self.config.shift_sigma(t);
            self.sigmas.push(sigma);
            self.timesteps.push(sigma * 1000.0);
        }

        // Terminal sigma = 0.
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
        let omega = self.config.omega();

        // dx = (σ_next - σ) * v
        let dt = sigma_next - sigma;
        let dx = (model_output * dt)?;

        // Omega mean-shifting: dx' = (dx - m) * ω + m
        let mean = dx
            .mean_all()?
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?;
        let scaled = (&dx - mean)? * omega;
        let dx = (scaled? + mean)?;

        // x_next = x + dx'
        let next = (sample + dx)?;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn timesteps_setup() {
        let config = SchedulerConfig::default();
        let mut scheduler = EulerScheduler::new(config);
        scheduler.set_timesteps(60);

        assert_eq!(scheduler.sigmas().len(), 61); // 60 + terminal
        assert_eq!(scheduler.timesteps().len(), 60);

        // First sigma should be close to 1.0 (shifted).
        assert!(
            scheduler.sigmas()[0] > 0.9,
            "first sigma = {}",
            scheduler.sigmas()[0]
        );

        // Terminal sigma should be 0.
        assert!(
            (scheduler.sigmas()[60]).abs() < 1e-10,
            "terminal sigma = {}",
            scheduler.sigmas()[60]
        );

        // Sigmas should be monotonically decreasing.
        for i in 0..60 {
            assert!(
                scheduler.sigmas()[i] > scheduler.sigmas()[i + 1],
                "sigmas not decreasing at {i}: {} vs {}",
                scheduler.sigmas()[i],
                scheduler.sigmas()[i + 1]
            );
        }
    }

    #[test]
    fn step_moves_toward_data() {
        let config = SchedulerConfig::default();
        let mut scheduler = EulerScheduler::new(config);
        scheduler.set_timesteps(10);

        let device = Device::Cpu;
        // Start from noise.
        let sample = Tensor::randn(0.0_f32, 1.0, (1, 8, 16, 32), &device).unwrap();
        // Velocity pointing toward zero (predicting noise = sample).
        let velocity = sample.clone();

        let result = scheduler.step(&velocity, &sample, 0).unwrap();

        // After one step, the result should have smaller magnitude than the input.
        let sample_norm: f32 = sample
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        let result_norm: f32 = result
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();

        assert!(
            result_norm < sample_norm,
            "expected smaller norm after step: {result_norm} vs {sample_norm}"
        );
    }
}
