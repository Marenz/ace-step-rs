//! Generation manager — keeps the pipeline resident and queues requests.
//!
//! The manager owns one [`AceStepPipeline`] loaded either on GPU or CPU.
//! Callers submit [`GenerationRequest`]s which are processed sequentially.
//! On a CUDA OOM the manager offloads the pipeline to CPU and retries.
//!
//! # Example
//!
//! ```no_run
//! use ace_step_rs::manager::{GenerationManager, ManagerConfig};
//! use ace_step_rs::pipeline::GenerationParams;
//!
//! #[tokio::main]
//! async fn main() {
//!     let manager = GenerationManager::start(ManagerConfig::default()).await.unwrap();
//!     let audio = manager.generate(GenerationParams::default()).await.unwrap();
//! }
//! ```

use std::time::{Duration, Instant};

use candle_core::{DType, Device};
use tokio::sync::{mpsc, oneshot};

use crate::pipeline::{AceStepPipeline, GeneratedAudio, GenerationParams};
use crate::{Error, Result};

/// Configuration for the generation manager.
#[derive(Debug, Clone)]
pub struct ManagerConfig {
    /// CUDA device ordinal (0 = first GPU). Ignored when CUDA is unavailable.
    pub cuda_device: usize,

    /// Data type for model weights and activations.
    pub dtype: DType,

    /// Minimum free VRAM (bytes) required before attempting GPU generation.
    ///
    /// If free VRAM drops below this threshold the manager offloads to CPU
    /// *before* attempting generation (proactive offload). Set to 0 to disable
    /// proactive offload and rely only on OOM retry.
    ///
    /// Default: 2 GiB.
    pub min_free_vram_bytes: u64,

    /// Automatically unload the pipeline from VRAM after this duration of
    /// inactivity. The next generation request will reload it automatically.
    ///
    /// `None` disables auto-unload.
    ///
    /// Default: `None`.
    pub idle_unload_after: Option<Duration>,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            cuda_device: 0,
            dtype: DType::F32,
            min_free_vram_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB
            idle_unload_after: None,
        }
    }
}

/// Commands sent to the manager worker thread.
enum ManagerCommand {
    /// Generate audio from the given parameters.
    Generate {
        params: GenerationParams,
        reply: oneshot::Sender<Result<GeneratedAudio>>,
    },
    /// Drop the pipeline to free VRAM. Next generate will reload automatically.
    Unload { reply: oneshot::Sender<Result<()>> },
}

/// Handle for submitting generation requests to a running manager.
#[derive(Clone)]
pub struct GenerationManager {
    tx: mpsc::Sender<ManagerCommand>,
}

impl GenerationManager {
    /// Start the manager background task and return a handle for submitting requests.
    ///
    /// Loads the pipeline immediately on startup (downloading from HuggingFace if needed).
    /// Returns an error if the initial load fails.
    pub async fn start(config: ManagerConfig) -> Result<Self> {
        // Load pipeline on the calling thread (blocking) then hand off to the worker.
        // We use spawn_blocking because pipeline loading does synchronous I/O and heavy compute.
        let pipeline = tokio::task::spawn_blocking(move || -> Result<AceStepPipeline> {
            let device = preferred_device(config.cuda_device);
            tracing::info!(device = ?device, "loading ACE-Step pipeline");
            AceStepPipeline::load(&device, config.dtype)
        })
        .await
        .map_err(|join_error| Error::Manager(format!("pipeline load task panicked: {join_error}")))?
        .map_err(|e| Error::Manager(format!("pipeline load failed: {e}")))?;

        let (tx, rx) = mpsc::channel::<ManagerCommand>(64);

        tokio::task::spawn_blocking(move || run_manager(pipeline, config, rx));

        Ok(Self { tx })
    }

    /// Submit a generation request and wait for the result.
    pub async fn generate(&self, params: GenerationParams) -> Result<GeneratedAudio> {
        let (reply_tx, reply_rx) = oneshot::channel::<Result<GeneratedAudio>>();
        self.tx
            .send(ManagerCommand::Generate {
                params,
                reply: reply_tx,
            })
            .await
            .map_err(|_| Error::Manager("manager has shut down".into()))?;

        reply_rx
            .await
            .map_err(|_| Error::Manager("manager dropped reply channel".into()))?
    }

    /// Unload the pipeline to free VRAM. The next `generate` call will reload it.
    pub async fn unload(&self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel::<Result<()>>();
        self.tx
            .send(ManagerCommand::Unload { reply: reply_tx })
            .await
            .map_err(|_| Error::Manager("manager has shut down".into()))?;

        reply_rx
            .await
            .map_err(|_| Error::Manager("manager dropped reply channel".into()))?
    }
}

/// The manager loop — runs in a dedicated blocking thread.
///
/// Processes requests sequentially. On CUDA OOM, offloads to CPU and retries.
/// When unloaded, the pipeline is dropped and lazily reloaded on next generate.
///
/// When `idle_unload_after` is configured, the loop uses a timeout on the
/// receive so it can auto-unload the pipeline after a period of inactivity.
fn run_manager(
    pipeline: AceStepPipeline,
    config: ManagerConfig,
    mut rx: mpsc::Receiver<ManagerCommand>,
) {
    let mut pipeline: Option<AceStepPipeline> = Some(pipeline);
    let mut last_activity = Instant::now();

    loop {
        let command = match config.idle_unload_after {
            Some(idle_timeout) if pipeline.is_some() => {
                // Calculate remaining time until idle unload fires.
                let elapsed = last_activity.elapsed();
                if elapsed >= idle_timeout {
                    // Already past the deadline — unload now.
                    pipeline.take();
                    tracing::info!(
                        idle_secs = idle_timeout.as_secs(),
                        "idle timeout reached — pipeline unloaded to free VRAM"
                    );
                    // Fall through to blocking recv (no timeout needed while unloaded).
                    rx.blocking_recv()
                } else {
                    let remaining = idle_timeout - elapsed;
                    // Use a short poll loop since tokio mpsc doesn't have blocking_recv_timeout.
                    recv_with_timeout(&mut rx, remaining)
                }
            }
            _ => {
                // No idle timeout or pipeline already unloaded — block indefinitely.
                rx.blocking_recv()
            }
        };

        let Some(command) = command else {
            break; // Channel closed, shut down.
        };

        match command {
            ManagerCommand::Generate { params, reply } => {
                // Reload if unloaded.
                if pipeline.is_none() {
                    tracing::info!("pipeline not loaded — reloading");
                    let device = preferred_device(config.cuda_device);
                    match AceStepPipeline::load(&device, config.dtype) {
                        Ok(p) => pipeline = Some(p),
                        Err(e) => {
                            let _ = reply.send(Err(Error::Manager(format!(
                                "failed to reload pipeline: {e}"
                            ))));
                            continue;
                        }
                    }
                }

                let p = pipeline.take().unwrap();
                let (result, new_pipeline) = generate_with_retry(p, &config, params);
                pipeline = Some(new_pipeline);
                last_activity = Instant::now();
                let _ = reply.send(result);
            }
            ManagerCommand::Unload { reply } => {
                if pipeline.take().is_some() {
                    tracing::info!("pipeline unloaded — VRAM freed");
                    let _ = reply.send(Ok(()));
                } else {
                    tracing::info!("pipeline already unloaded");
                    let _ = reply.send(Ok(()));
                }
            }
        }
    }
    tracing::info!("generation manager shut down");
}

/// Blocking receive with a timeout. Polls with exponential backoff to avoid
/// busy-waiting while still responding promptly to new messages.
fn recv_with_timeout(
    rx: &mut mpsc::Receiver<ManagerCommand>,
    timeout: Duration,
) -> Option<ManagerCommand> {
    let deadline = Instant::now() + timeout;
    let mut sleep_ms = 10u64;
    let max_sleep_ms = 500u64;

    loop {
        match rx.try_recv() {
            Ok(command) => return Some(command),
            Err(mpsc::error::TryRecvError::Empty) => {
                if Instant::now() >= deadline {
                    return None; // Timed out — signal idle unload.
                }
                let remaining = deadline.duration_since(Instant::now());
                let sleep = Duration::from_millis(sleep_ms).min(remaining);
                std::thread::sleep(sleep);
                sleep_ms = (sleep_ms * 2).min(max_sleep_ms);
            }
            Err(mpsc::error::TryRecvError::Disconnected) => return None,
        }
    }
}

/// Try to generate. On CUDA OOM, offload to CPU and retry once.
///
/// Returns the result and the (possibly replaced) pipeline.
fn generate_with_retry(
    pipeline: AceStepPipeline,
    config: &ManagerConfig,
    params: GenerationParams,
) -> (Result<GeneratedAudio>, AceStepPipeline) {
    // Proactive offload: if free VRAM is below threshold, move to CPU first.
    let mut pipeline = maybe_proactive_offload(pipeline, config);

    match pipeline.generate(&params) {
        Ok(audio) => (Ok(audio), pipeline),
        Err(ref error) if is_oom_error(error) => {
            let error_msg = error.to_string();
            tracing::warn!(error = %error_msg, "CUDA OOM — offloading pipeline to CPU and retrying");
            match offload_to_cpu(pipeline, config) {
                Ok(mut cpu_pipeline) => {
                    let result = cpu_pipeline
                        .generate(&params)
                        .map_err(|e| Error::Manager(format!("generation failed even on CPU: {e}")));
                    (result, cpu_pipeline)
                }
                Err((offload_error, recovered_pipeline)) => {
                    (Err(offload_error), recovered_pipeline)
                }
            }
        }
        Err(error) => (Err(error), pipeline),
    }
}

/// If free VRAM is below the configured threshold, proactively offload to CPU.
fn maybe_proactive_offload(pipeline: AceStepPipeline, config: &ManagerConfig) -> AceStepPipeline {
    if config.min_free_vram_bytes == 0 || !matches!(pipeline.device(), Device::Cuda(_)) {
        return pipeline;
    }

    match free_vram_bytes() {
        Ok(free) if free < config.min_free_vram_bytes => {
            tracing::info!(
                free_mb = free / (1024 * 1024),
                threshold_mb = config.min_free_vram_bytes / (1024 * 1024),
                "free VRAM below threshold — offloading pipeline to CPU"
            );
            match offload_to_cpu(pipeline, config) {
                Ok(cpu_pipeline) => cpu_pipeline,
                Err((error, recovered_pipeline)) => {
                    tracing::warn!(%error, "proactive CPU offload failed, will try GPU anyway");
                    recovered_pipeline
                }
            }
        }
        Ok(free) => {
            tracing::debug!(free_mb = free / (1024 * 1024), "VRAM OK");
            pipeline
        }
        Err(error) => {
            tracing::warn!(%error, "could not query free VRAM, skipping proactive offload");
            pipeline
        }
    }
}

/// Offload the pipeline to CPU by reloading from cached weights.
///
/// On success returns the CPU pipeline.
/// On failure returns the original error and a recovered pipeline (reloaded on GPU).
#[allow(clippy::result_large_err)]
fn offload_to_cpu(
    pipeline: AceStepPipeline,
    config: &ManagerConfig,
) -> std::result::Result<AceStepPipeline, (Error, AceStepPipeline)> {
    match pipeline.reload_on_device(&Device::Cpu) {
        Ok(cpu_pipeline) => {
            tracing::info!("pipeline offloaded to CPU");
            Ok(cpu_pipeline)
        }
        Err(reload_error) => {
            // CPU reload failed. Try to recover by reloading on GPU.
            tracing::error!(%reload_error, "CPU offload failed, attempting GPU reload");
            let device = preferred_device(config.cuda_device);
            match AceStepPipeline::load(&device, config.dtype) {
                Ok(recovered) => Err((
                    Error::Manager(format!("CPU offload failed: {reload_error}")),
                    recovered,
                )),
                Err(gpu_error) => {
                    // Total failure — panic so the thread dies and the channel closes,
                    // which will surface as errors to all future callers.
                    panic!(
                        "both CPU offload and GPU reload failed: offload={reload_error}, gpu={gpu_error}"
                    );
                }
            }
        }
    }
}

/// Query free VRAM on the first CUDA device.
///
/// Returns an error if CUDA is not available or the query fails.
#[cfg(feature = "cuda")]
fn free_vram_bytes() -> std::result::Result<u64, String> {
    cudarc::runtime::result::get_mem_info()
        .map(|(free, _total)| free as u64)
        .map_err(|e| format!("cudaMemGetInfo failed: {e}"))
}

#[cfg(not(feature = "cuda"))]
fn free_vram_bytes() -> std::result::Result<u64, String> {
    Err("CUDA not compiled in".into())
}

/// Return the preferred device: CUDA if available, otherwise CPU.
pub fn preferred_device(cuda_ordinal: usize) -> Device {
    Device::cuda_if_available(cuda_ordinal).unwrap_or(Device::Cpu)
}

/// Return true if the error looks like a CUDA out-of-memory condition.
///
/// Candle surfaces CUDA errors as `candle_core::Error::Cuda(Box<dyn Error>)` whose
/// `Display` contains the cudarc error string. We match on substrings
/// rather than types because the concrete error type is not exported.
pub fn is_oom_error(error: &crate::Error) -> bool {
    let msg = error.to_string().to_lowercase();
    msg.contains("out of memory")
        || msg.contains("cudaerrormemorya") // cudaErrorMemoryAllocation
        || msg.contains("cuda_error_out_of_memory")
        || msg.contains("cublas_status_alloc_failed")
        || msg.contains("alloc failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_oom_error_matches_known_patterns() {
        fn make_err(msg: &str) -> crate::Error {
            crate::Error::Manager(msg.to_string())
        }

        assert!(is_oom_error(&make_err("CUDA out of memory")));
        assert!(is_oom_error(&make_err("cudaErrorMemoryAllocation")));
        assert!(is_oom_error(&make_err("CUDA_ERROR_OUT_OF_MEMORY")));
        assert!(is_oom_error(&make_err("alloc failed")));
        assert!(!is_oom_error(&make_err("shape mismatch")));
        assert!(!is_oom_error(&make_err("invalid index")));
    }

    #[test]
    fn test_manager_config_defaults() {
        let config = ManagerConfig::default();
        assert_eq!(config.cuda_device, 0);
        assert_eq!(config.dtype, DType::F32);
        assert_eq!(config.min_free_vram_bytes, 2 * 1024 * 1024 * 1024);
        assert!(config.idle_unload_after.is_none());
    }

    #[test]
    fn test_recv_with_timeout_returns_none_on_empty_channel() {
        let (_tx, mut rx) = mpsc::channel::<ManagerCommand>(1);
        let start = Instant::now();
        let result = recv_with_timeout(&mut rx, Duration::from_millis(50));
        assert!(result.is_none());
        assert!(start.elapsed() >= Duration::from_millis(50));
    }

    #[test]
    fn test_recv_with_timeout_returns_command_before_deadline() {
        let (tx, mut rx) = mpsc::channel::<ManagerCommand>(1);
        let (reply_tx, _reply_rx) = oneshot::channel();
        tx.blocking_send(ManagerCommand::Unload { reply: reply_tx }).unwrap();
        let result = recv_with_timeout(&mut rx, Duration::from_secs(5));
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), ManagerCommand::Unload { .. }));
    }
}
