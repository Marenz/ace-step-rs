//! End-to-end inference pipeline.
//!
//! Orchestrates:
//! 1. Model downloading from HuggingFace
//! 2. Text/lyric tokenization via Qwen3 tokenizer
//! 3. Text encoding via Qwen3-Embedding-0.6B
//! 4. Condition assembly (text + lyrics + timbre)
//! 5. Diffusion generation (8-step turbo ODE)
//! 6. VAE decoding to waveform
//! 7. WAV file output

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::config::{AceStepConfig, VaeConfig};
pub use crate::model::encoder::text::format_metas;
use crate::model::encoder::text::{self, Qwen3TextEncoder};
use crate::model::generation::AceStepConditionGenerationModel;
use crate::vae::OobleckDecoder;
use crate::{Error, Result};

/// Load `silence_latent.pt` — a PyTorch zip archive containing raw float32 data.
///
/// The file is a zip with entry `silence_latent/data/0` containing raw f32 bytes
/// for a tensor of shape [1, 64, 15000]. We transpose it to [1, 15000, 64] to
/// match the Python code: `torch.load(...).transpose(1, 2)`.
fn load_silence_latent(path: &std::path::Path, dtype: DType, device: &Device) -> Result<Tensor> {
    use std::io::Read;

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| Error::WeightLoad(format!("Failed to open silence_latent.pt zip: {e}")))?;

    let mut data_entry = archive
        .by_name("silence_latent/data/0")
        .map_err(|e| Error::WeightLoad(format!("Missing data entry in silence_latent.pt: {e}")))?;

    let mut raw_bytes = Vec::with_capacity(data_entry.size() as usize);
    data_entry
        .read_to_end(&mut raw_bytes)
        .map_err(|e| Error::WeightLoad(format!("Failed to read silence_latent data: {e}")))?;

    // Raw data is [1, 64, 15000] float32
    let tensor = Tensor::from_raw_buffer(&raw_bytes, DType::F32, &[1, 64, 15000], &Device::Cpu)?;

    // Transpose to [1, 15000, 64] matching Python: .transpose(1, 2)
    let tensor = tensor.transpose(1, 2)?.contiguous()?;

    // Move to target device and dtype
    let tensor = tensor.to_dtype(dtype)?.to_device(device)?;

    tracing::info!(
        "Loaded silence_latent: shape {:?}, dtype {:?}",
        tensor.shape(),
        tensor.dtype()
    );
    Ok(tensor)
}

/// Parameters for audio generation.
pub struct GenerationParams {
    /// Text caption describing the music style/genre.
    pub caption: String,
    /// Metadata string (e.g., "bpm: 120, key: C major, genre: jazz").
    pub metas: String,
    /// Lyrics text.
    pub lyrics: String,
    /// Language of the lyrics (e.g., "en", "zh").
    pub language: String,
    /// Duration in seconds.
    pub duration_s: f64,
    /// Shift parameter for turbo schedule (1, 2, or 3). Default: 3.
    pub shift: f64,
    /// Random seed. None = random.
    pub seed: Option<u64>,
    /// Custom src_latents [1, T, 64] — overrides default silence latent.
    /// When set, chunk_masks must also be set, and duration_s is ignored
    /// (T is inferred from the tensor).
    pub src_latents: Option<Tensor>,
    /// Custom chunk_masks [1, T, 64] — 1.0 = generate, 0.0 = keep from src_latents.
    pub chunk_masks: Option<Tensor>,
    /// Custom timbre reference latents [N, 750, 64].
    /// When None, uses silence (no timbre conditioning).
    pub refer_audio: Option<Tensor>,
    /// Batch assignment for timbre references [N].
    pub refer_order: Option<Tensor>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            caption: String::new(),
            metas: String::new(),
            lyrics: String::new(),
            language: "en".to_string(),
            duration_s: 30.0,
            shift: 3.0,
            seed: None,
            src_latents: None,
            chunk_masks: None,
            refer_audio: None,
            refer_order: None,
        }
    }
}

/// Generated audio output.
pub struct GeneratedAudio {
    /// Stereo waveform samples, interleaved [L, R, L, R, ...]
    pub samples: Vec<f32>,
    /// Sample rate (48000).
    pub sample_rate: u32,
    /// Number of channels (2 = stereo).
    pub channels: u16,
}

/// Paths to downloaded model files.
struct ModelPaths {
    /// Path to the turbo DiT model directory.
    dit_safetensors: std::path::PathBuf,
    /// Path to the VAE safetensors.
    vae_safetensors: std::path::PathBuf,
    /// Path to the Qwen3 text encoder safetensors.
    text_encoder_safetensors: std::path::PathBuf,
    /// Path to the Qwen3 tokenizer.json.
    tokenizer_json: std::path::PathBuf,
    /// Path to silence_latent.pt (precomputed VAE-encoded silence).
    silence_latent_pt: std::path::PathBuf,
}

/// Download model files from HuggingFace.
///
/// Downloads from `ACE-Step/Ace-Step1.5` repo:
/// - `acestep-v15-turbo/model.safetensors` — DiT weights
/// - `vae/diffusion_pytorch_model.safetensors` — VAE weights
/// - `Qwen3-Embedding-0.6B/model.safetensors` — text encoder weights
/// - `Qwen3-Embedding-0.6B/tokenizer.json` — tokenizer
fn download_models() -> Result<ModelPaths> {
    let api = Api::new().map_err(|e| Error::HfHub(e.to_string()))?;
    let repo = api.model("ACE-Step/Ace-Step1.5".to_string());

    tracing::info!("Downloading ACE-Step v1.5 models from HuggingFace...");

    let dit_safetensors = repo
        .get("acestep-v15-turbo/model.safetensors")
        .map_err(|e| Error::HfHub(format!("Failed to download DiT model: {e}")))?;

    let vae_safetensors = repo
        .get("vae/diffusion_pytorch_model.safetensors")
        .map_err(|e| Error::HfHub(format!("Failed to download VAE model: {e}")))?;

    let text_encoder_safetensors = repo
        .get("Qwen3-Embedding-0.6B/model.safetensors")
        .map_err(|e| Error::HfHub(format!("Failed to download text encoder: {e}")))?;

    let tokenizer_json = repo
        .get("Qwen3-Embedding-0.6B/tokenizer.json")
        .map_err(|e| Error::HfHub(format!("Failed to download tokenizer: {e}")))?;

    let silence_latent_pt = repo
        .get("acestep-v15-turbo/silence_latent.pt")
        .map_err(|e| Error::HfHub(format!("Failed to download silence_latent: {e}")))?;

    tracing::info!("All models downloaded successfully");

    Ok(ModelPaths {
        dit_safetensors,
        vae_safetensors,
        text_encoder_safetensors,
        tokenizer_json,
        silence_latent_pt,
    })
}

/// ACE-Step v1.5 inference pipeline.
pub struct AceStepPipeline {
    tokenizer: Tokenizer,
    text_encoder: Qwen3TextEncoder,
    generation_model: AceStepConditionGenerationModel,
    vae: OobleckDecoder,
    /// Precomputed VAE-encoded silence [1, 15000, 64] at 25Hz (10 min max).
    silence_latent: Tensor,
    cfg: AceStepConfig,
    device: Device,
    dtype: DType,
    /// Cached paths for reloading on a different device (e.g. CPU fallback).
    paths: ModelPaths,
}

impl AceStepPipeline {
    /// Load the pipeline, downloading model weights from HuggingFace if needed.
    pub fn load(device: &Device, dtype: DType) -> Result<Self> {
        let paths = download_models()?;
        Self::load_from_paths(paths, device, dtype)
    }

    /// Reload the pipeline on a different device.
    ///
    /// Uses the cached HF paths (already on disk) so no network I/O is needed.
    /// Intended for CPU fallback after a CUDA OOM.
    pub fn reload_on_device(self, device: &Device) -> Result<Self> {
        let Self {
            dtype,
            paths,
            // Destructure everything so all GPU tensors are dropped before
            // we allocate on the new device.
            tokenizer: _,
            text_encoder: _,
            generation_model: _,
            vae: _,
            silence_latent: _,
            cfg: _,
            device: _,
        } = self;
        Self::load_from_paths(paths, device, dtype)
    }

    /// The device this pipeline is currently loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Load from pre-downloaded model files (takes ownership to store for later reload).
    fn load_from_paths(paths: ModelPaths, device: &Device, dtype: DType) -> Result<Self> {
        // Enable TF32 tensor-core math for F32 matmuls on Ampere+ GPUs.
        // Same tradeoff PyTorch makes by default (10-bit mantissa vs 23-bit).
        #[cfg(feature = "cuda")]
        candle_core::cuda::set_gemm_reduced_precision_f32(true);

        let cfg = AceStepConfig::default();

        // Load tokenizer
        tracing::info!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(&paths.tokenizer_json)?;

        // Load text encoder (Qwen3-Embedding-0.6B)
        tracing::info!("Loading Qwen3 text encoder...");
        let qwen3_cfg = text::default_qwen3_config();
        let text_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&paths.text_encoder_safetensors], dtype, device)?
        };
        // Candle's Qwen3 model requests keys with "model." prefix (e.g. "model.embed_tokens.weight")
        // but the Qwen3-Embedding-0.6B safetensors stores keys without it (e.g. "embed_tokens.weight").
        // Rename to strip the "model." prefix from requested names.
        let text_vb =
            text_vb.rename_f(|name: &str| name.strip_prefix("model.").unwrap_or(name).to_string());
        let text_encoder = Qwen3TextEncoder::new(&qwen3_cfg, text_vb)?;

        // Load DiT generation model
        tracing::info!("Loading DiT generation model...");
        let dit_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&paths.dit_safetensors], dtype, device)?
        };
        let generation_model = AceStepConditionGenerationModel::new(&cfg, dtype, device, dit_vb)?;

        // Load VAE decoder
        tracing::info!("Loading VAE decoder...");
        let vae_cfg = VaeConfig::default();
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&paths.vae_safetensors], dtype, device)?
        };
        let vae = OobleckDecoder::new(&vae_cfg, vae_vb.pp("decoder"))?;

        // Load silence latent (precomputed VAE-encoded silence)
        tracing::info!("Loading silence latent...");
        let silence_latent = load_silence_latent(&paths.silence_latent_pt, dtype, device)?;

        tracing::info!("Pipeline loaded successfully");

        Ok(Self {
            tokenizer,
            text_encoder,
            generation_model,
            vae,
            silence_latent,
            cfg,
            device: device.clone(),
            dtype,
            paths,
        })
    }

    // --- Accessor methods ---

    /// Access the silence latent tensor [1, 15000, 64].
    pub fn silence_latent(&self) -> &Tensor {
        &self.silence_latent
    }

    /// The model config.
    pub fn config(&self) -> &AceStepConfig {
        &self.cfg
    }

    /// The dtype used for model weights and computation.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Access the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Encode text caption through Qwen3.
    pub fn encode_text(&mut self, caption_ids: &Tensor) -> Result<Tensor> {
        Ok(self.text_encoder.encode_text(caption_ids)?)
    }

    /// Get lyric token embeddings from Qwen3.
    pub fn embed_lyrics(&mut self, lyric_ids: &Tensor) -> Result<Tensor> {
        Ok(self.text_encoder.embed_lyrics(lyric_ids)?)
    }

    /// Run the condition encoder (text + lyrics + timbre → packed sequence).
    pub fn encode_conditions(
        &self,
        text_hidden: &Tensor,
        text_mask: &Tensor,
        lyric_hidden: &Tensor,
        lyric_mask: &Tensor,
        refer_audio: &Tensor,
        refer_order: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Ok(self.generation_model.encode_conditions(
            text_hidden,
            text_mask,
            lyric_hidden,
            lyric_mask,
            refer_audio,
            refer_order,
        )?)
    }

    /// Run a single DiT forward pass.
    pub fn dit_forward(
        &self,
        xt: &Tensor,
        timestep: &Tensor,
        timestep_r: &Tensor,
        encoder_hidden: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        Ok(self
            .generation_model
            .dit_forward(xt, timestep, timestep_r, encoder_hidden, context)?)
    }

    /// Run VAE decode on latents [B, 64, T] → waveform [B, 2, samples].
    /// Uses tiled decode for long sequences.
    pub fn vae_decode(&self, latents: &Tensor) -> Result<Tensor> {
        Ok(self.vae.tiled_decode(latents, 256, 16)?)
    }

    /// Tokenize and encode a caption string, returning (hidden_states, mask).
    /// Both tensors have batch dim 1.
    pub fn tokenize_and_encode_caption(
        &mut self,
        caption: &str,
        metas: &str,
    ) -> Result<(Tensor, Tensor)> {
        let prompt = text::format_caption_prompt(caption, metas);
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| Error::Tokenizer(crate::error::TokenizerError(e.to_string())))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<f32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as f32)
            .collect();
        let ids_t = Tensor::new(&ids[..], &self.device)?.unsqueeze(0)?;
        let mask_t = Tensor::new(&mask[..], &self.device)?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?;
        let hidden = self.text_encoder.encode_text(&ids_t)?;
        Ok((hidden, mask_t))
    }

    /// Tokenize and embed lyrics, returning (hidden_states, mask).
    /// Both tensors have batch dim 1.
    pub fn tokenize_and_embed_lyrics(
        &mut self,
        lyrics: &str,
        language: &str,
    ) -> Result<(Tensor, Tensor)> {
        let prompt = text::format_lyric_prompt(lyrics, language);
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| Error::Tokenizer(crate::error::TokenizerError(e.to_string())))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<f32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as f32)
            .collect();
        let ids_t = Tensor::new(&ids[..], &self.device)?.unsqueeze(0)?;
        let mask_t = Tensor::new(&mask[..], &self.device)?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?;
        let hidden = self.text_encoder.embed_lyrics(&ids_t)?;
        Ok((hidden, mask_t))
    }

    /// Generate audio from text and lyrics.
    pub fn generate(&mut self, params: &GenerationParams) -> Result<GeneratedAudio> {
        use std::time::Instant;
        let t0 = Instant::now();

        // 1. Format and tokenize text caption
        let caption_prompt = text::format_caption_prompt(&params.caption, &params.metas);
        let caption_encoding = self
            .tokenizer
            .encode(caption_prompt, true)
            .map_err(|e| Error::Tokenizer(crate::error::TokenizerError(e.to_string())))?;
        let caption_ids: Vec<u32> = caption_encoding.get_ids().to_vec();
        let caption_mask: Vec<f32> = caption_encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as f32)
            .collect();

        let caption_tensor = Tensor::new(&caption_ids[..], &self.device)?.unsqueeze(0)?;
        let caption_mask_tensor = Tensor::new(&caption_mask[..], &self.device)?.unsqueeze(0)?;

        // 2. Encode caption through Qwen3 (clear KV cache from any prior call)
        let t1 = Instant::now();
        self.text_encoder.clear_kv_cache();
        let text_hidden = self.text_encoder.encode_text(&caption_tensor)?;
        tracing::info!("Text encoding: {:.2}s", t1.elapsed().as_secs_f64());

        // 3. Format and tokenize lyrics
        let lyric_prompt = text::format_lyric_prompt(&params.lyrics, &params.language);
        let lyric_encoding = self
            .tokenizer
            .encode(lyric_prompt, true)
            .map_err(|e| Error::Tokenizer(crate::error::TokenizerError(e.to_string())))?;
        let lyric_ids: Vec<u32> = lyric_encoding.get_ids().to_vec();
        let lyric_mask: Vec<f32> = lyric_encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as f32)
            .collect();

        let lyric_tensor = Tensor::new(&lyric_ids[..], &self.device)?.unsqueeze(0)?;
        let lyric_mask_tensor = Tensor::new(&lyric_mask[..], &self.device)?.unsqueeze(0)?;

        // 4. Get lyric embeddings (raw token embeddings, NOT full encoder)
        let lyric_hidden = self.text_encoder.embed_lyrics(&lyric_tensor)?;

        // 5. Compute sequence length from duration, or infer from custom src_latents.
        let acoustic_dim = self.cfg.audio_acoustic_hidden_dim;

        let (src_latents, chunk_masks) =
            if let (Some(sl), Some(cm)) = (&params.src_latents, &params.chunk_masks) {
                (sl.clone(), cm.clone())
            } else {
                // Default: silence src_latents + all-ones chunk_masks (text2music).
                let t = (params.duration_s * 25.0) as usize;
                let src = self.silence_latent.i((.., ..t, ..))?.contiguous()?;
                let masks = Tensor::ones((1, t, acoustic_dim), self.dtype, &self.device)?;
                (src, masks)
            };
        let t = src_latents.dim(1)?;

        // 6. Timbre reference — custom or default silence.
        let (refer_audio, refer_order) =
            if let (Some(ra), Some(ro)) = (&params.refer_audio, &params.refer_order) {
                (ra.clone(), ro.clone())
            } else {
                let ra = self
                    .silence_latent
                    .i((.., ..self.cfg.timbre_fix_frame, ..))?
                    .contiguous()?;
                let ro = Tensor::zeros((1,), DType::I64, &self.device)?;
                (ra, ro)
            };

        // 8. Generate latents via diffusion
        let t2 = Instant::now();
        tracing::info!(
            "Generating {:.1}s of audio ({} frames)...",
            params.duration_s,
            t
        );
        let output = self.generation_model.generate_audio(
            &text_hidden,
            &caption_mask_tensor.to_dtype(self.dtype)?,
            &lyric_hidden,
            &lyric_mask_tensor.to_dtype(self.dtype)?,
            &refer_audio,
            &refer_order,
            &src_latents,
            &chunk_masks,
            params.shift,
            params.seed,
        )?;
        tracing::info!(
            "Diffusion (8 ODE steps): {:.2}s",
            t2.elapsed().as_secs_f64()
        );

        // 9. Decode latents to waveform via VAE
        // Latents: [B, T, 64] → transpose to [B, 64, T] for VAE
        // Uses chunked/tiled decode for long sequences to avoid VRAM OOM.
        let t3 = Instant::now();
        let latents = output.target_latents.transpose(1, 2)?.contiguous()?;
        let waveform = self.vae.tiled_decode(&latents, 256, 16)?;
        tracing::info!("VAE decode: {:.2}s", t3.elapsed().as_secs_f64());

        // 10. Peak normalization (matches Python: pred_wavs / peak.clamp(min=1.0))
        // Only scales down if any sample exceeds [-1, 1]
        let waveform = {
            let waveform_f32 = waveform.to_dtype(DType::F32)?;
            let abs_max = waveform_f32
                .abs()?
                .max(2)?
                .max(1)?
                .clamp(1.0f64, f64::MAX)?;
            let abs_max = abs_max.unsqueeze(1)?.unsqueeze(2)?;
            waveform_f32.broadcast_div(&abs_max)?
        };

        // 11. Convert to samples
        // waveform: [1, 2, samples] → interleaved stereo
        let waveform = waveform.squeeze(0)?; // [2, samples]
        let left: Vec<f32> = waveform.get(0)?.to_vec1()?;
        let right: Vec<f32> = waveform.get(1)?.to_vec1()?;

        let mut samples = Vec::with_capacity(left.len() * 2);
        for (l, r) in left.iter().zip(right.iter()) {
            samples.push(*l);
            samples.push(*r);
        }

        tracing::info!("Total generate: {:.2}s", t0.elapsed().as_secs_f64());

        Ok(GeneratedAudio {
            samples,
            sample_rate: 48000,
            channels: 2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_params_default() {
        let params = GenerationParams::default();
        assert_eq!(params.duration_s, 30.0);
        assert_eq!(params.shift, 3.0);
        assert_eq!(params.language, "en");
        assert!(params.seed.is_none());
    }
}
