//! Flow-matching diffusion schedulers.
//!
//! Three schedulers for the denoising loop, all using shifted flow-matching
//! sigma schedules with `shift=3.0`:
//!
//! - [`euler`] — first-order Euler integrator with omega mean-shifting
//! - [`heun`] — second-order predictor-corrector (2 model evals per step)
//! - [`pingpong`] — stochastic SDE variant with noise re-injection
//!
//! ## Flow-matching formulation
//!
//! The forward process interpolates: `x_t = (1 - σ) * x_0 + σ * ε`
//!
//! The model predicts velocity `v = x_1 - x_0` (direction from data to noise).
//!
//! The sigma schedule is shifted: `σ' = shift * σ / (1 + (shift - 1) * σ)`

pub mod euler;
pub mod heun;
pub mod pingpong;

use candle_core::Tensor;

/// Common interface for all schedulers.
pub trait Scheduler {
    /// Set up the timestep schedule for a given number of inference steps.
    fn set_timesteps(&mut self, num_inference_steps: usize);

    /// Return the current sigma values.
    fn sigmas(&self) -> &[f64];

    /// Return the timestep values (sigmas × 1000, for the model).
    fn timesteps(&self) -> &[f64];

    /// Perform one scheduler step.
    ///
    /// - `model_output`: predicted velocity from the transformer
    /// - `sample`: current noisy latent
    /// - `step_index`: which step we're on
    fn step(
        &mut self,
        model_output: &Tensor,
        sample: &Tensor,
        step_index: usize,
    ) -> crate::Result<Tensor>;
}

/// Configuration shared across all schedulers.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SchedulerConfig {
    /// Number of training timesteps (default: 1000).
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize,

    /// Sigma schedule shift factor (default: 3.0).
    #[serde(default = "default_shift")]
    pub shift: f64,

    /// Omega scaling factor for mean-shift (default: 10.0).
    #[serde(default = "default_omega_scale")]
    pub omega_scale: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: default_num_train_timesteps(),
            shift: default_shift(),
            omega_scale: default_omega_scale(),
        }
    }
}

impl SchedulerConfig {
    /// Compute the omega value from omega_scale via logistic function.
    ///
    /// `omega = 0.9 + 0.2 / (1 + exp(-0.1 * omega_scale))`
    pub fn omega(&self) -> f64 {
        0.9 + 0.2 / (1.0 + (-0.1 * self.omega_scale).exp())
    }

    /// Apply the shift to a raw sigma value.
    ///
    /// `σ' = shift * σ / (1 + (shift - 1) * σ)`
    pub fn shift_sigma(&self, sigma: f64) -> f64 {
        self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma)
    }
}

fn default_num_train_timesteps() -> usize {
    1000
}

fn default_shift() -> f64 {
    3.0
}

fn default_omega_scale() -> f64 {
    10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omega_at_default_scale() {
        let config = SchedulerConfig::default();
        let omega = config.omega();
        // At omega_scale=10.0: 0.9 + 0.2 / (1 + exp(-0.1 * 10.0))
        //   = 0.9 + 0.2 / (1 + exp(-1.0)) ≈ 0.9 + 0.2 / 1.3679 ≈ 1.0462
        assert!((omega - 1.046).abs() < 0.01, "omega = {omega}");
    }

    #[test]
    fn shift_sigma_identity_at_shift_1() {
        let config = SchedulerConfig {
            shift: 1.0,
            ..Default::default()
        };
        let sigma = 0.5;
        let shifted = config.shift_sigma(sigma);
        assert!(
            (shifted - sigma).abs() < 1e-10,
            "shift=1.0 should be identity, got {shifted}"
        );
    }

    #[test]
    fn shift_sigma_at_default() {
        let config = SchedulerConfig::default();
        // shift=3.0, sigma=0.5: 3.0 * 0.5 / (1 + 2.0 * 0.5) = 1.5 / 2.0 = 0.75
        let shifted = config.shift_sigma(0.5);
        assert!(
            (shifted - 0.75).abs() < 1e-10,
            "expected 0.75, got {shifted}"
        );
    }

    #[test]
    fn shift_sigma_endpoints() {
        let config = SchedulerConfig::default();
        // sigma=0 → 0
        assert!((config.shift_sigma(0.0)).abs() < 1e-10);
        // sigma=1 → shift * 1 / (1 + (shift-1)*1) = shift/shift = 1.0
        assert!((config.shift_sigma(1.0) - 1.0).abs() < 1e-10);
    }
}
