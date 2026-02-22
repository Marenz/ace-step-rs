//! Heun flow-matching scheduler (second-order predictor-corrector).
//!
//! Each logical step uses two model evaluations:
//! 1. **Predictor**: Euler step to estimate next state
//! 2. **Corrector**: re-evaluate at predicted state, average derivatives
//!
//! The sigma array is doubled via `repeat_interleave(2)` so the step()
//! method is called twice per logical step — first for prediction,
//! then for correction.

use candle_core::Tensor;

use super::{Scheduler, SchedulerConfig};
use crate::Result;

/// Heun flow-matching scheduler.
pub struct HeunScheduler {
    config: SchedulerConfig,
    sigmas: Vec<f64>,
    timesteps: Vec<f64>,
    /// Stored state from predictor pass.
    prev_derivative: Option<Tensor>,
    prev_sample: Option<Tensor>,
    dt: Option<f64>,
}

impl HeunScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            prev_derivative: None,
            prev_sample: None,
            dt: None,
        }
    }

    /// Whether the given step index is a first-order (predictor) step.
    fn is_first_order(&self, step_index: usize) -> bool {
        step_index % 2 == 0
    }
}

impl Scheduler for HeunScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) {
        let sigma_max = 1.0;
        let sigma_min = 1.0 / self.config.num_train_timesteps as f64;

        // Base sigmas (num_inference_steps + 1 values, including terminal 0).
        let mut base_sigmas = Vec::with_capacity(num_inference_steps + 1);
        for i in 0..num_inference_steps {
            let t = sigma_max
                - (sigma_max - sigma_min) * i as f64 / (num_inference_steps - 1).max(1) as f64;
            base_sigmas.push(self.config.shift_sigma(t));
        }
        base_sigmas.push(0.0);

        // Interleave: repeat each sigma (except last) twice, then append terminal.
        // [s0, s1, s2, ..., sN, 0] → [s0, s0, s1, s1, ..., sN, sN, 0]
        self.sigmas = Vec::with_capacity(2 * num_inference_steps + 1);
        self.timesteps = Vec::with_capacity(2 * num_inference_steps);

        for &sigma in &base_sigmas[..num_inference_steps] {
            self.sigmas.push(sigma);
            self.sigmas.push(sigma);
            self.timesteps.push(sigma * 1000.0);
            self.timesteps.push(sigma * 1000.0);
        }
        self.sigmas.push(0.0);

        self.prev_derivative = None;
        self.prev_sample = None;
        self.dt = None;
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

        // Estimate x_0 from the velocity prediction.
        let denoised = (sample - (model_output * sigma)?)?;
        let derivative = if sigma.abs() > 1e-10 {
            ((sample - &denoised)? / sigma)?
        } else {
            model_output.clone()
        };

        if self.is_first_order(step_index) {
            // Predictor step: store state, take Euler step.
            let dt = sigma_next - sigma;
            self.prev_derivative = Some(derivative.clone());
            self.prev_sample = Some(sample.clone());
            self.dt = Some(dt);

            let dx = (&derivative * dt)?;
            let mean = dx
                .mean_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?;
            let scaled = (&dx - mean)? * omega;
            let dx = (scaled? + mean)?;
            Ok((sample + dx)?)
        } else {
            // Corrector step: average derivatives, apply to stored sample.
            let prev_deriv = self
                .prev_derivative
                .take()
                .expect("corrector without predictor");
            let prev_sample = self
                .prev_sample
                .take()
                .expect("corrector without predictor");
            let dt = self.dt.take().expect("corrector without predictor");

            let avg_derivative = ((&prev_deriv + &derivative)? * 0.5)?;
            let dx = (&avg_derivative * dt)?;
            let mean = dx
                .mean_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?;
            let scaled = (&dx - mean)? * omega;
            let dx = (scaled? + mean)?;
            Ok((prev_sample + dx)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn heun_timesteps_doubled() {
        let config = SchedulerConfig::default();
        let mut scheduler = HeunScheduler::new(config);
        scheduler.set_timesteps(20);

        // 20 logical steps → 40 model evaluations + 1 terminal sigma.
        assert_eq!(scheduler.timesteps().len(), 40);
        assert_eq!(scheduler.sigmas().len(), 41);

        // Pairs should be equal.
        for i in 0..20 {
            assert!(
                (scheduler.sigmas()[2 * i] - scheduler.sigmas()[2 * i + 1]).abs() < 1e-10,
                "sigma pair {i} not equal"
            );
        }
    }

    #[test]
    fn heun_predictor_corrector_pair() {
        let config = SchedulerConfig::default();
        let mut scheduler = HeunScheduler::new(config);
        scheduler.set_timesteps(10);

        let device = Device::Cpu;
        let sample = Tensor::randn(0.0_f32, 1.0, (1, 8, 16, 32), &device).unwrap();
        let velocity = Tensor::randn(0.0_f32, 1.0, (1, 8, 16, 32), &device).unwrap();

        // Step 0 (predictor) — should succeed and store state.
        let predicted = scheduler.step(&velocity, &sample, 0).unwrap();
        assert_eq!(predicted.dims(), &[1, 8, 16, 32]);

        // Step 1 (corrector) — should use stored state and not panic.
        let velocity2 = Tensor::randn(0.0_f32, 1.0, (1, 8, 16, 32), &device).unwrap();
        let corrected = scheduler.step(&velocity2, &predicted, 1).unwrap();
        assert_eq!(corrected.dims(), &[1, 8, 16, 32]);

        // After corrector, internal state should be cleared.
        assert!(scheduler.prev_derivative.is_none());
        assert!(scheduler.prev_sample.is_none());
    }
}
