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
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            cuda_device: 0,
            dtype: DType::F32,
            min_free_vram_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB
        }
    }
}

/// A submitted generation request.
struct PendingRequest {
    params: GenerationParams,
    reply: oneshot::Sender<Result<GeneratedAudio>>,
}

/// Handle for submitting generation requests to a running manager.
#[derive(Clone)]
pub struct GenerationManager {
    tx: mpsc::Sender<PendingRequest>,
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

        let (tx, rx) = mpsc::channel::<PendingRequest>(64);

        tokio::task::spawn_blocking(move || run_manager(pipeline, config, rx));

        Ok(Self { tx })
    }

    /// Submit a generation request and wait for the result.
    pub async fn generate(&self, params: GenerationParams) -> Result<GeneratedAudio> {
        let (reply_tx, reply_rx) = oneshot::channel::<Result<GeneratedAudio>>();
        self.tx
            .send(PendingRequest {
                params,
                reply: reply_tx,
            })
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
fn run_manager(
    mut pipeline: AceStepPipeline,
    config: ManagerConfig,
    mut rx: mpsc::Receiver<PendingRequest>,
) {
    while let Some(request) = rx.blocking_recv() {
        let (result, new_pipeline) = generate_with_retry(pipeline, &config, request.params);
        pipeline = new_pipeline;
        // Ignore send errors — caller may have timed out.
        let _ = request.reply.send(result);
    }
    tracing::info!("generation manager shut down");
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
    }
}
