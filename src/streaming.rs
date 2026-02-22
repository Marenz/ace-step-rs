//! Infinite streaming music generation.
//!
//! `StreamingGenerator` wraps `AceStepPipeline` and produces an endless stream
//! of audio by generating overlapping chunks. Between chunks, callers can
//! inject new lyrics, change style/genre instructions, or swap timbre reference
//! latents (pre-encoded).
//!
//! ## How it works
//!
//! The generator exploits ACE-Step's **repaint** mechanism. Each chunk after
//! the first carries over the last `overlap_frames` latent frames from the
//! previous chunk as `src_latents` context, with `chunk_masks = 0` in that
//! region ("keep this") and `1` in the new region ("generate here"). The DiT
//! sees the previous audio's latents through bidirectional attention and
//! produces a musically coherent continuation.
//!
//! At the waveform level, a short equal-power crossfade (default 100 ms)
//! smooths out any residual discontinuity from the VAE decoder's receptive
//! field.
//!
//! ## Audio injection (future)
//!
//! Injecting raw audio requires the VAE *encoder* (not yet implemented).
//! Once available, `ChunkRequest::inject_audio` will VAE-encode the audio
//! and place the resulting latents into `src_latents`.

use candle_core::{DType, IndexOp, Tensor};

use crate::audio::crossfade;
use crate::pipeline::{AceStepPipeline, GeneratedAudio, GenerationParams};
use crate::Result;

/// Configuration for the streaming generator.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Duration of each generated chunk in seconds. Default: 30.
    pub chunk_duration_s: f64,
    /// Overlap with previous chunk in seconds. Default: 8.
    /// Must be less than `chunk_duration_s`.
    pub overlap_s: f64,
    /// Audio-domain crossfade duration in milliseconds. Default: 100.
    /// Applied at the waveform level after VAE decode.
    pub crossfade_ms: u32,
    /// Shift parameter for the turbo schedule (1, 2, or 3). Default: 3.
    pub shift: f64,
    /// Default language for lyrics. Default: "en".
    pub language: String,

    // --- Feature toggles (for debugging / A-B testing) ---
    /// Use overlap repaint: carry previous latents as src_latents context.
    /// When false, each chunk is generated independently (no continuity).
    pub use_overlap: bool,
    /// Use timbre conditioning from previous chunk's latents.
    /// When false, uses silence for timbre reference (default text2music).
    pub use_timbre_from_prev: bool,
    /// Use audio-domain crossfade at chunk boundaries.
    /// When false, chunks are hard-concatenated (after trimming overlap).
    pub use_crossfade: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_duration_s: 30.0,
            overlap_s: 8.0,
            crossfade_ms: 100,
            shift: 3.0,
            language: "en".to_string(),
            use_overlap: true,
            use_timbre_from_prev: true,
            use_crossfade: false,
        }
    }
}

/// A request to influence the next generated chunk.
///
/// All fields are optional — `None` means "keep the previous value."
pub struct ChunkRequest {
    /// New caption (genre/style tags). `None` = keep previous.
    pub caption: Option<String>,
    /// New metadata (bpm, key, etc.). `None` = keep previous.
    pub metas: Option<String>,
    /// New lyrics for this chunk. `None` = keep previous.
    pub lyrics: Option<String>,
    /// New language. `None` = keep previous.
    pub language: Option<String>,
    /// Random seed for this chunk. `None` = random.
    pub seed: Option<u64>,
}

impl Default for ChunkRequest {
    fn default() -> Self {
        Self {
            caption: None,
            metas: None,
            lyrics: None,
            language: None,
            seed: None,
        }
    }
}

/// A chunk of generated audio returned by the streaming generator.
pub struct StreamChunk {
    /// New audio samples (stereo interleaved, 48 kHz).
    /// This is the *new* audio only (after crossfade with previous tail),
    /// ready to be appended to the output stream.
    pub audio: GeneratedAudio,
    /// Which chunk number this is (0-indexed).
    pub chunk_index: usize,
    /// The latent-space overlap frames carried to the next chunk.
    /// Useful for debugging; callers don't need to touch this.
    pub overlap_frames: usize,
}

/// Infinite streaming music generator.
///
/// Call [`next_chunk`] in a loop to produce an endless audio stream.
/// Between calls, inject new lyrics/instructions via [`ChunkRequest`].
pub struct StreamingGenerator {
    config: StreamConfig,

    // --- Persistent state across chunks ---
    /// Previous chunk's full diffusion output latents [1, T, 64].
    prev_latents: Option<Tensor>,
    /// Tail audio samples from the previous chunk (for crossfade).
    /// Length = crossfade_samples * channels interleaved values.
    prev_audio_tail: Option<Vec<f32>>,
    /// Current caption (persists across chunks unless overridden).
    caption: String,
    /// Current metadata (persists across chunks unless overridden).
    metas: String,
    /// Current lyrics (persists across chunks unless overridden).
    lyrics: String,
    /// Current language.
    language: String,

    /// Chunk counter.
    chunk_index: usize,
}

impl StreamingGenerator {
    /// Create a new streaming generator.
    ///
    /// `initial_caption` and `initial_lyrics` are used for the first chunk.
    pub fn new(
        config: StreamConfig,
        initial_caption: &str,
        initial_metas: &str,
        initial_lyrics: &str,
    ) -> Self {
        Self {
            language: config.language.clone(),
            config,
            prev_latents: None,
            prev_audio_tail: None,
            caption: initial_caption.to_string(),
            metas: initial_metas.to_string(),
            lyrics: initial_lyrics.to_string(),
            chunk_index: 0,
        }
    }

    /// Number of latent frames per chunk.
    fn chunk_frames(&self) -> usize {
        (self.config.chunk_duration_s * 25.0) as usize
    }

    /// Number of latent frames used as overlap context.
    fn overlap_frames(&self) -> usize {
        (self.config.overlap_s * 25.0) as usize
    }

    /// Number of *new* latent frames generated per chunk (after the first).
    fn new_frames(&self) -> usize {
        self.chunk_frames() - self.overlap_frames()
    }

    /// Number of per-channel audio samples for crossfade.
    fn crossfade_samples(&self) -> usize {
        // 48000 Hz * ms / 1000
        (48000.0 * self.config.crossfade_ms as f64 / 1000.0) as usize
    }

    /// Generate the next chunk of audio.
    ///
    /// `pipeline` is borrowed mutably because text encoding uses `&mut self`
    /// (Qwen3's KV cache). `request` optionally overrides caption/lyrics for
    /// this chunk.
    pub fn next_chunk(
        &mut self,
        pipeline: &mut AceStepPipeline,
        request: Option<ChunkRequest>,
    ) -> Result<StreamChunk> {
        tracing::info!("=== Streaming chunk {} ===", self.chunk_index);

        // Apply any overrides from the request
        if let Some(req) = &request {
            if let Some(ref c) = req.caption {
                tracing::info!("  caption override: {c}");
                self.caption = c.clone();
            }
            if let Some(ref m) = req.metas {
                tracing::info!("  metas override: {m}");
                self.metas = m.clone();
            }
            if let Some(ref l) = req.lyrics {
                tracing::info!("  lyrics override: {:?}", &l[..l.len().min(80)]);
                self.lyrics = l.clone();
            }
            if let Some(ref lang) = req.language {
                tracing::info!("  language override: {lang}");
                self.language = lang.clone();
            }
        }

        tracing::info!("  caption: {}", self.caption);
        tracing::info!("  metas: {}", self.metas);
        tracing::info!("  lyrics: {:?}", &self.lyrics[..self.lyrics.len().min(120)]);
        tracing::info!("  language: {}", self.language);
        tracing::info!(
            "  config: duration={:.0}s overlap={:.0}s crossfade={}ms shift={}",
            self.config.chunk_duration_s,
            self.config.overlap_s,
            self.config.crossfade_ms,
            self.config.shift,
        );

        let seed = request.as_ref().and_then(|r| r.seed);
        let device = pipeline.device().clone();
        let dtype = pipeline.dtype();
        let acoustic_dim = pipeline.config().audio_acoustic_hidden_dim;
        let timbre_fix_frame = pipeline.config().timbre_fix_frame;

        let is_first = self.prev_latents.is_none();
        let use_overlap = self.config.use_overlap && !is_first;
        let use_timbre = self.config.use_timbre_from_prev && !is_first;

        tracing::info!(
            "  is_first={is_first} use_overlap={use_overlap} use_timbre={use_timbre} use_crossfade={}",
            self.config.use_crossfade,
        );

        // Debug: log audio dimensions
        let t_frames = self.chunk_frames();
        let expected_audio_samples = t_frames * 1920 * 2; // latent frames * hop * stereo
        tracing::info!(
            "  DEBUG: chunk_frames={} expected_audio_samples={}",
            t_frames,
            expected_audio_samples
        );

        // --- Build src_latents and chunk_masks ---
        let (src_latents, chunk_masks) = if use_overlap {
            let prev = self.prev_latents.as_ref().unwrap();
            let overlap = self.overlap_frames();
            let new_len = self.new_frames();

            // Take last `overlap` frames from previous output as context
            let prev_t = prev.dim(1)?;
            let overlap_start = prev_t.saturating_sub(overlap);
            let overlap_latents = prev.i((.., overlap_start.., ..))?.contiguous()?;

            // Fill new region with silence
            let silence_new = pipeline
                .silence_latent()
                .i((.., ..new_len, ..))?
                .contiguous()?;

            // src_latents = [overlap_context | silence_new]
            let src = Tensor::cat(&[&overlap_latents, &silence_new], 1)?;

            // chunk_masks = [zeros (keep) | ones (generate)]
            let mask_keep = Tensor::zeros((1, overlap, acoustic_dim), dtype, &device)?;
            let mask_gen = Tensor::ones((1, new_len, acoustic_dim), dtype, &device)?;
            let masks = Tensor::cat(&[&mask_keep, &mask_gen], 1)?;

            (Some(src), Some(masks))
        } else {
            // No overlap: independent chunk (silence src_latents, all-ones mask)
            (None, None)
        };

        // --- Timbre reference from previous latents (or silence) ---
        let (refer_audio, refer_order) = if use_timbre {
            let prev = self.prev_latents.as_ref().unwrap();
            // Use up to timbre_fix_frame frames from previous output as timbre ref
            let prev_t = prev.dim(1)?;
            let ref_len = prev_t.min(timbre_fix_frame);
            let ref_latents = prev.i((.., ..ref_len, ..))?.contiguous()?;

            // Pad to timbre_fix_frame if needed
            let refer = if ref_len < timbre_fix_frame {
                let pad_len = timbre_fix_frame - ref_len;
                let silence_pad = pipeline
                    .silence_latent()
                    .i((.., ..pad_len, ..))?
                    .contiguous()?;
                Tensor::cat(&[&ref_latents, &silence_pad], 1)?
            } else {
                ref_latents
            };

            let order = Tensor::zeros((1,), DType::I64, &device)?;
            (Some(refer), Some(order))
        } else {
            // No timbre conditioning: use silence (default text2music)
            (None, None)
        };

        // --- Generate via pipeline ---
        let params = GenerationParams {
            caption: self.caption.clone(),
            metas: self.metas.clone(),
            lyrics: self.lyrics.clone(),
            language: self.language.clone(),
            duration_s: self.config.chunk_duration_s,
            shift: self.config.shift,
            seed,
            src_latents,
            chunk_masks,
            refer_audio,
            refer_order,
        };

        // We need the raw latents back for the next chunk's overlap.
        // Use generate() which does full pipeline including VAE decode.
        // But we also need the latents. So let's use generate_with_latents().
        let (audio, latents) = generate_with_latents(pipeline, &params)?;

        // --- Extract new audio and crossfade ---
        let output_audio = if is_first {
            // First chunk: output everything
            audio
        } else {
            // Subsequent chunks: extract only the new region's audio,
            // crossfade with previous tail.
            let overlap = self.overlap_frames();
            let overlap_audio_samples = overlap * 1920; // latent frames * hop_length
            let overlap_interleaved = overlap_audio_samples * 2; // stereo

            let new_samples = if audio.samples.len() > overlap_interleaved {
                &audio.samples[overlap_interleaved..]
            } else {
                // Edge case: entire chunk is overlap (shouldn't happen with sane config)
                &audio.samples[..]
            };

            // Save tail for next chunk's crossfade - BEFORE we crossfade this chunk
            let tail_len = self.crossfade_samples() * 2; // stereo interleaved
            let prev_tail = self.prev_audio_tail.clone(); // Save old tail for crossfade
            if new_samples.len() >= tail_len {
                let start = new_samples.len() - tail_len;
                self.prev_audio_tail = Some(new_samples[start..].to_vec());
            } else {
                self.prev_audio_tail = Some(new_samples.to_vec());
            }

            let merged = if self.config.use_crossfade {
                if let Some(ref old_tail) = prev_tail {
                    tracing::info!(
                        "  DEBUG: crossfading old_tail={} with new_samples={}",
                        old_tail.len(),
                        new_samples.len()
                    );
                    crossfade(old_tail, new_samples, self.crossfade_samples(), 2)
                } else {
                    new_samples.to_vec()
                }
            } else {
                new_samples.to_vec()
            };

            GeneratedAudio {
                samples: merged,
                sample_rate: 48000,
                channels: 2,
            }
        };

        // Debug: log output audio size
        tracing::info!(
            "  DEBUG: output_audio.samples={} is_first={}",
            output_audio.samples.len(),
            is_first
        );

        // --- Save state for next chunk ---
        self.prev_latents = Some(latents);

        // For first chunk, save tail after output
        if is_first {
            let tail_len = self.crossfade_samples() * 2;
            if output_audio.samples.len() >= tail_len {
                let start = output_audio.samples.len() - tail_len;
                self.prev_audio_tail = Some(output_audio.samples[start..].to_vec());
            } else {
                self.prev_audio_tail = Some(output_audio.samples.clone());
            }
            tracing::info!(
                "  DEBUG: saved tail for next chunk: {} samples",
                self.prev_audio_tail.as_ref().map(|t| t.len()).unwrap_or(0)
            );
        }

        let chunk = StreamChunk {
            audio: output_audio,
            chunk_index: self.chunk_index,
            overlap_frames: if is_first { 0 } else { self.overlap_frames() },
        };

        self.chunk_index += 1;
        Ok(chunk)
    }

    /// How many chunks have been generated so far.
    pub fn chunks_generated(&self) -> usize {
        self.chunk_index
    }

    /// Current caption being used.
    pub fn caption(&self) -> &str {
        &self.caption
    }

    /// Current lyrics being used.
    pub fn lyrics(&self) -> &str {
        &self.lyrics
    }

    /// Current metadata being used.
    pub fn metas(&self) -> &str {
        &self.metas
    }

    /// Current language being used.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Read-only access to the current config.
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Rate a chunk for A/B testing. Logs the rating alongside all generation
    /// parameters so you can correlate what settings sound good vs bad.
    ///
    /// `rating`: `-` = bad, `.` = neutral, `+` = good
    pub fn rate_chunk(&self, chunk_index: usize, rating: &str) {
        tracing::warn!(
            "RATING chunk={} rating={} caption={:?} metas={:?} lyrics={:?} lang={} \
             duration={:.0}s overlap={:.0}s crossfade={}ms shift={} \
             use_overlap={} use_timbre={} use_crossfade={}",
            chunk_index,
            rating,
            self.caption,
            self.metas,
            &self.lyrics[..self.lyrics.len().min(120)],
            self.language,
            self.config.chunk_duration_s,
            self.config.overlap_s,
            self.config.crossfade_ms,
            self.config.shift,
            self.config.use_overlap,
            self.config.use_timbre_from_prev,
            self.config.use_crossfade,
        );
    }

    /// Mutable access to the config (for toggling features between chunks).
    pub fn config_mut(&mut self) -> &mut StreamConfig {
        &mut self.config
    }
}

/// Generate audio AND return the raw diffusion latents [1, T, 64].
///
/// Runs the full pipeline stages manually (tokenize → encode → diffuse → VAE
/// decode) while also capturing the intermediate latents before VAE decoding.
/// The streaming generator needs these latents for overlap context.
fn generate_with_latents(
    pipeline: &mut AceStepPipeline,
    params: &GenerationParams,
) -> Result<(GeneratedAudio, Tensor)> {
    use std::time::Instant;
    let t0 = Instant::now();

    let device = pipeline.device().clone();
    let dtype = pipeline.dtype();
    let cfg = pipeline.config().clone();
    let acoustic_dim = cfg.audio_acoustic_hidden_dim;

    // 1. Tokenize and encode caption
    let t1 = Instant::now();
    let (text_hidden, caption_mask) =
        pipeline.tokenize_and_encode_caption(&params.caption, &params.metas)?;
    tracing::debug!("Text encoding: {:.2}s", t1.elapsed().as_secs_f64());

    // 2. Tokenize and embed lyrics
    let (lyric_hidden, lyric_mask) =
        pipeline.tokenize_and_embed_lyrics(&params.lyrics, &params.language)?;

    // 3. Build src_latents + chunk_masks
    let (src_latents, chunk_masks) =
        if let (Some(sl), Some(cm)) = (&params.src_latents, &params.chunk_masks) {
            (sl.clone(), cm.clone())
        } else {
            let t = (params.duration_s * 25.0) as usize;
            let src = pipeline.silence_latent().i((.., ..t, ..))?.contiguous()?;
            let masks = Tensor::ones((1, t, acoustic_dim), dtype, &device)?;
            (src, masks)
        };

    // 4. Timbre reference
    let (refer_audio, refer_order) =
        if let (Some(ra), Some(ro)) = (&params.refer_audio, &params.refer_order) {
            (ra.clone(), ro.clone())
        } else {
            let ra = pipeline
                .silence_latent()
                .i((.., ..cfg.timbre_fix_frame, ..))?
                .contiguous()?;
            let ro = Tensor::zeros((1,), DType::I64, &device)?;
            (ra, ro)
        };

    // 5. Encode conditions
    let (enc_hidden, _enc_mask) = pipeline.encode_conditions(
        &text_hidden,
        &caption_mask,
        &lyric_hidden,
        &lyric_mask,
        &refer_audio,
        &refer_order,
    )?;

    // 6. Build context: cat[src_latents, chunk_masks] → [B, T, 128]
    let context = Tensor::cat(&[&src_latents, &chunk_masks], 2)?;

    // 7. Diffusion ODE
    let t_frames = src_latents.dim(1)?;
    let b = 1usize;

    tracing::info!(
        "Generating chunk ({} frames, {:.1}s)...",
        t_frames,
        t_frames as f64 / 25.0
    );

    let noise = if let Some(_seed) = params.seed {
        let cpu_dev = candle_core::Device::Cpu;
        Tensor::randn(0f32, 1.0, (b, t_frames, acoustic_dim), &cpu_dev)?
            .to_dtype(dtype)?
            .to_device(&device)?
    } else {
        Tensor::randn(0f32, 1.0, (b, t_frames, acoustic_dim), &device)?.to_dtype(dtype)?
    };

    let schedule = crate::config::TurboSchedule::for_shift(params.shift);
    let num_steps = schedule.len();
    let mut xt = noise;

    let t_diff = Instant::now();
    for step_idx in 0..num_steps {
        let t_curr = schedule[step_idx];
        let t_curr_tensor = Tensor::full(t_curr as f32, (b,), &device)?.to_dtype(dtype)?;

        let vt =
            pipeline.dit_forward(&xt, &t_curr_tensor, &t_curr_tensor, &enc_hidden, &context)?;

        if step_idx == num_steps - 1 {
            let t_expand = t_curr_tensor.unsqueeze(1)?.unsqueeze(2)?;
            xt = (xt - vt.broadcast_mul(&t_expand)?)?;
            break;
        }

        let t_next = schedule[step_idx + 1];
        let dt = (t_curr - t_next) as f32;
        let dt_tensor = Tensor::full(dt, (b, 1, 1), &device)?.to_dtype(dtype)?;
        xt = (xt - vt.broadcast_mul(&dt_tensor)?)?;
    }
    tracing::info!(
        "Diffusion (8 ODE steps): {:.2}s",
        t_diff.elapsed().as_secs_f64()
    );

    // 8. Save latents before VAE decode [1, T, 64]
    let latents = xt.clone();

    // 9. VAE decode: [1, T, 64] → transpose → [1, 64, T] → waveform
    let t_vae = Instant::now();
    let vae_input = xt.transpose(1, 2)?.contiguous()?;
    let waveform = pipeline.vae_decode(&vae_input)?;
    tracing::info!("VAE decode: {:.2}s", t_vae.elapsed().as_secs_f64());

    // 10. Peak normalize
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

    // 11. Convert to interleaved stereo
    let waveform = waveform.squeeze(0)?;
    let left: Vec<f32> = waveform.get(0)?.to_vec1()?;
    let right: Vec<f32> = waveform.get(1)?.to_vec1()?;

    let mut samples = Vec::with_capacity(left.len() * 2);
    for (l, r) in left.iter().zip(right.iter()) {
        samples.push(*l);
        samples.push(*r);
    }

    tracing::info!("Total chunk: {:.2}s", t0.elapsed().as_secs_f64());

    let audio = GeneratedAudio {
        samples,
        sample_rate: 48000,
        channels: 2,
    };

    Ok((audio, latents))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_defaults() {
        let cfg = StreamConfig::default();
        assert_eq!(cfg.chunk_duration_s, 30.0);
        assert_eq!(cfg.overlap_s, 8.0);
        assert_eq!(cfg.crossfade_ms, 100);
        assert_eq!(cfg.shift, 3.0);
    }

    #[test]
    fn test_streaming_generator_frame_math() {
        let cfg = StreamConfig {
            chunk_duration_s: 30.0,
            overlap_s: 8.0,
            ..Default::default()
        };
        let sg = StreamingGenerator::new(cfg, "jazz", "", "hello");
        assert_eq!(sg.chunk_frames(), 750);
        assert_eq!(sg.overlap_frames(), 200);
        assert_eq!(sg.new_frames(), 550);
        // crossfade: 48000 * 100 / 1000 = 4800 samples per channel
        assert_eq!(sg.crossfade_samples(), 4800);
    }

    #[test]
    fn test_chunk_request_defaults() {
        let req = ChunkRequest::default();
        assert!(req.caption.is_none());
        assert!(req.lyrics.is_none());
        assert!(req.seed.is_none());
    }

    #[test]
    fn test_streaming_generator_state_updates() {
        let cfg = StreamConfig::default();
        let mut sg = StreamingGenerator::new(cfg, "jazz piano", "bpm: 120", "la la la");
        assert_eq!(sg.caption(), "jazz piano");
        assert_eq!(sg.lyrics(), "la la la");
        assert_eq!(sg.chunks_generated(), 0);

        // Simulate applying a request (without actually generating)
        let req = ChunkRequest {
            caption: Some("rock guitar".to_string()),
            lyrics: Some("yeah yeah".to_string()),
            ..Default::default()
        };
        // Apply manually (normally done inside next_chunk)
        if let Some(ref c) = req.caption {
            sg.caption = c.clone();
        }
        if let Some(ref l) = req.lyrics {
            sg.lyrics = l.clone();
        }
        assert_eq!(sg.caption(), "rock guitar");
        assert_eq!(sg.lyrics(), "yeah yeah");
    }
}
