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

    /// Total planned duration of the full track in seconds. Default: 0 (chunk duration only).
    /// When > 0, the model generates for this full duration but we only output the new portion.
    /// This lets the model plan structure (intro/verse/chorus) for the entire track.
    pub total_duration_s: f64,

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
    /// Auto-increment seed for each chunk to get more variation.
    /// When true, each chunk gets a different random seed.
    pub auto_seed: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_duration_s: 30.0,
            overlap_s: 8.0,
            crossfade_ms: 100,
            shift: 3.0,
            language: "en".to_string(),
            total_duration_s: 0.0,
            use_overlap: true,
            use_timbre_from_prev: true,
            use_crossfade: false,
            auto_seed: true, // Changed to true for variety
        }
    }
}

/// A request to influence the next generated chunk.
///
/// All fields are optional — `None` means "keep the previous value."
pub struct ChunkRequest {
    /// New caption (genre/style tags). `None` = keep previous.
    pub caption: Option<String>,
    /// New BPM. `None` = keep previous. `Some(None)` = unset (N/A).
    pub bpm: Option<Option<u32>>,
    /// New key/scale. `None` = keep previous. `Some(None)` = unset (N/A).
    pub key_scale: Option<Option<String>>,
    /// New time signature. `None` = keep previous.
    pub time_signature: Option<String>,
    /// New lyrics for this chunk. `None` = keep previous.
    pub lyrics: Option<String>,
    /// New language. `None` = keep previous.
    pub language: Option<String>,
    /// Random seed for this chunk. `None` = random.
    pub seed: Option<u64>,
    /// New step size in seconds (sliding window only). `None` = keep previous.
    pub step_s: Option<f64>,
    /// New position mode. `None` = keep previous.
    pub position_mode: Option<PositionMode>,
}

impl Default for ChunkRequest {
    fn default() -> Self {
        Self {
            step_s: None,
            position_mode: None,
            caption: None,
            bpm: None,
            key_scale: None,
            time_signature: None,
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
    /// How many "real" (non-padding) frames are in prev_latents.
    prev_latent_frames: usize,
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
            prev_latent_frames: 0,
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

    /// Total latent frames for the full track (for planning).
    fn total_frames(&self) -> usize {
        let duration = if self.config.total_duration_s > 0.0 {
            self.config.total_duration_s
        } else {
            self.config.chunk_duration_s
        };
        (duration * 25.0) as usize
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

        // If request provides an explicit seed use it; otherwise auto-generate
        // one when auto_seed is enabled (ensures variety between chunks).
        let seed = request.as_ref().and_then(|r| r.seed).or_else(|| {
            if self.config.auto_seed {
                Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64,
                )
            } else {
                None
            }
        });
        let device = pipeline.device().clone();
        let dtype = pipeline.dtype();
        let acoustic_dim = pipeline.config().audio_acoustic_hidden_dim;
        let timbre_fix_frame = pipeline.config().timbre_fix_frame;

        let is_first = self.prev_latents.is_none();
        let use_overlap = self.config.use_overlap && !is_first;
        let use_timbre = self.config.use_timbre_from_prev && !is_first;

        tracing::debug!(
            "chunk={} is_first={} use_overlap={} use_timbre={}",
            self.chunk_index,
            is_first,
            use_overlap,
            use_timbre
        );

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
        let use_total_duration = self.config.total_duration_s > 0.0;

        // For total duration mode: pad context to full track length
        // but only generate chunk-sized pieces
        let context_frames = if use_total_duration {
            self.total_frames()
        } else {
            self.chunk_frames()
        };

        // How much new audio we generate this chunk
        let new_len = if use_total_duration && !is_first {
            self.chunk_frames() - self.overlap_frames() // Still generate just the new portion
        } else {
            self.new_frames()
        };

        // Overlap from previous
        let overlap = self.overlap_frames();

        tracing::info!(
            "  DEBUG: context_frames={} new_len={} overlap={} use_total_duration={}",
            context_frames,
            new_len,
            overlap,
            use_total_duration
        );

        let (src_latents, chunk_masks) = if use_overlap {
            let prev = self.prev_latents.as_ref().unwrap();

            // Take last `overlap` frames from previous output as context
            // Use prev_latent_frames to know how many real frames we have
            let prev_t = self.prev_latent_frames;
            let overlap_start = prev_t.saturating_sub(overlap);
            let overlap_latents = prev.i((.., overlap_start.., ..))?.contiguous()?;

            // Pad context to full track length if using total_duration
            let padding_len = context_frames.saturating_sub(overlap + new_len);
            let silence_padding = pipeline
                .silence_latent()
                .i((.., ..padding_len, ..))?
                .contiguous()?;

            // Fill new region with silence
            let silence_new = pipeline
                .silence_latent()
                .i((.., ..new_len, ..))?
                .contiguous()?;

            // src_latents = [overlap_context | silence_new | silence_padding]
            let src = Tensor::cat(&[&overlap_latents, &silence_new, &silence_padding], 1)?;

            // chunk_masks = [zeros (keep) | ones (generate) | zeros (future)]
            let mask_keep = Tensor::zeros((1, overlap.min(prev_t), acoustic_dim), dtype, &device)?;
            let mask_gen = Tensor::ones((1, new_len, acoustic_dim), dtype, &device)?;
            let mask_future = Tensor::zeros((1, padding_len, acoustic_dim), dtype, &device)?;
            let masks = Tensor::cat(&[&mask_keep, &mask_gen, &mask_future], 1)?;

            (Some(src), Some(masks))
        } else {
            // No overlap: independent chunk (silence src_latents, all-ones mask)
            let silence = pipeline
                .silence_latent()
                .i((.., ..context_frames, ..))?
                .contiguous()?;
            let mask = Tensor::ones((1, context_frames, acoustic_dim), dtype, &device)?;
            (Some(silence), Some(mask))
        };

        // --- Timbre reference from previous latents (or silence) ---
        let (refer_audio, refer_order) = if use_timbre {
            let prev = self.prev_latents.as_ref().unwrap();
            // Use up to timbre_fix_frame frames from previous output as timbre ref
            let prev_t = self.prev_latent_frames;
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
        // When using overlap, generate only the new portion (not full chunk)
        let generation_duration = if is_first {
            self.config.chunk_duration_s
        } else if self.config.use_overlap {
            // Generate only new portion: chunk_duration - overlap
            (self.chunk_frames() - self.overlap_frames()) as f64 / 25.0
        } else {
            self.config.chunk_duration_s
        };

        tracing::debug!(
            "generation_duration={}s (is_first={}, use_overlap={})",
            generation_duration,
            is_first,
            self.config.use_overlap
        );

        let params = GenerationParams {
            caption: self.caption.clone(),
            metas: self.metas.clone(),
            lyrics: self.lyrics.clone(),
            language: self.language.clone(),
            duration_s: generation_duration,
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
        self.prev_latent_frames = self.chunk_frames(); // Real audio frames generated this chunk

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

    /// Reset generator state so the next chunk starts fresh (no overlap context).
    /// Caption/lyrics/config are preserved; only the latent history is cleared.
    pub fn restart(&mut self) {
        self.prev_latents = None;
        self.prev_latent_frames = 0;
        self.prev_audio_tail = None;
        self.chunk_index = 0;
        tracing::info!("StreamingGenerator: restarted (latent history cleared)");
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

// ── Sliding Window Generator ──────────────────────────────────────────────

/// Controls how the virtual song position advances between steps.
///
/// The position is used to compute `duration = pos + window_s` in metas,
/// which tells the model where it is in the musical arc.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum PositionMode {
    /// Position advances by `step_s` each step and wraps back to `step_s`
    /// when `pos + step_s + window_s` would exceed `max_duration_s`.
    /// Model experiences a slowly unfolding song arc.
    #[default]
    Advancing,
    /// Position stays fixed at `window_s / 2` every step.
    /// Model always thinks it's in the middle of a `window_s`-length track —
    /// no arc progression, pure continuation.
    Fixed,
}

/// Configuration for the sliding window generator.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Full context window in seconds — what the model sees each step.
    /// Should be a "musically complete" duration: 2–10 minutes.
    /// Default: 240s (4 min).
    pub window_s: f64,
    /// How many new seconds to generate per step. Default: 30s.
    /// Smaller = slower evolution, more continuity.
    /// Larger = faster progression, bigger musical changes.
    /// Must be < window_s.
    pub step_s: f64,
    /// Shift parameter for the turbo schedule. Default: 3.
    pub shift: f64,
    /// Default language. Default: "en".
    pub language: String,
    /// Auto-generate a new seed each step for variety. Default: true.
    pub auto_seed: bool,
    /// Maximum song duration in seconds — ACE-Step supports up to 600s (10 min).
    /// Only used in `Advancing` mode. Default: 600s.
    pub max_duration_s: f64,
    /// How the virtual song position changes between steps. Default: `Advancing`.
    pub position_mode: PositionMode,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            window_s: 240.0,
            step_s: 30.0,
            shift: 3.0,
            language: "en".to_string(),
            auto_seed: true,
            max_duration_s: 600.0,
            position_mode: PositionMode::default(),
        }
    }
}

/// A step of generated audio from the sliding window generator.
pub struct WindowStep {
    /// New audio samples for this step (stereo interleaved, 48 kHz).
    /// Length = step_s * 48000 * 2 samples.
    pub audio: GeneratedAudio,
    /// Step index (0-indexed).
    pub step_index: usize,
    /// The step size used for this step in seconds.
    pub step_s: f64,
    /// Window size in seconds.
    pub window_s: f64,
    /// Song position at the start of this step (seconds).
    pub pos_s: f64,
}

/// Infinite music generator using a sliding latent window.
///
/// Maintains a fixed-size window of latents (e.g. 4 minutes). Each step:
/// 1. Shifts the window left by `step_s`, dropping the oldest audio.
/// 2. Fills the tail with silence and sets `chunk_masks = 1` there.
/// 3. Runs the full DiT forward pass over the entire window.
/// 4. Outputs only the new tail audio.
///
/// The model always sees a full `window_s` of context, so it understands
/// its position in the musical arc (intro / middle / outro) and generates
/// accordingly. Step size controls the rate of musical evolution — small
/// steps give slow, gradual change; large steps allow bigger jumps.
///
/// All parameters (caption, lyrics, step size) can be changed between steps
/// via [`ChunkRequest`] and take effect on the very next generation.
pub struct SlidingWindowGenerator {
    pub config: SlidingWindowConfig,

    /// The sliding window of latents [1, window_frames, 64].
    /// `None` before the first step.
    window_latents: Option<Tensor>,

    /// How many frames at the start of the window are "real" generated audio
    /// vs silence padding (only relevant during the initial fill phase).
    filled_frames: usize,

    /// Current caption.
    caption: String,
    /// Current lyrics.
    lyrics: String,
    /// Current language.
    language: String,
    /// Current BPM (None = N/A in metas).
    bpm: Option<u32>,
    /// Current key/scale (None = N/A in metas).
    key_scale: Option<String>,
    /// Current time signature.
    time_signature: String,

    /// Logical song position in seconds. Starts at 0.
    /// Advances by `step_s` each step. Used as the "start" when computing
    /// `duration = current_pos_s + window_s` for metas.
    current_pos_s: f64,

    /// Step counter.
    step_index: usize,

    /// Tail of the previous step's audio for crossfading (stereo interleaved).
    /// Empty before the first step.
    prev_tail: Vec<f32>,

    /// Smoothed peak across steps for consistent loudness normalization.
    /// Avoids loudness jumps between independently-normalized steps.
    smoothed_peak: f32,
}

impl SlidingWindowGenerator {
    pub fn new(
        config: SlidingWindowConfig,
        initial_caption: &str,
        initial_bpm: Option<u32>,
        initial_key_scale: Option<&str>,
        initial_time_signature: &str,
        initial_lyrics: &str,
    ) -> Self {
        Self {
            language: config.language.clone(),
            config,
            window_latents: None,
            filled_frames: 0,
            caption: initial_caption.to_string(),
            lyrics: initial_lyrics.to_string(),
            bpm: initial_bpm,
            key_scale: initial_key_scale.map(|s| s.to_string()),
            time_signature: initial_time_signature.to_string(),
            current_pos_s: 0.0,
            step_index: 0,
            prev_tail: Vec::new(),
            smoothed_peak: 0.0,
        }
    }

    fn window_frames(&self) -> usize {
        (self.config.window_s * 25.0) as usize
    }

    fn step_frames(&self) -> usize {
        (self.config.step_s * 25.0) as usize
    }

    /// Hop size in audio samples per latent frame (stereo interleaved).
    const AUDIO_HOP: usize = 1920 * 2;

    pub fn caption(&self) -> &str {
        &self.caption
    }
    pub fn lyrics(&self) -> &str {
        &self.lyrics
    }
    pub fn bpm(&self) -> Option<u32> {
        self.bpm
    }
    pub fn key_scale(&self) -> Option<&str> {
        self.key_scale.as_deref()
    }
    pub fn language(&self) -> &str {
        &self.language
    }
    pub fn current_pos_s(&self) -> f64 {
        self.current_pos_s
    }
    pub fn steps_generated(&self) -> usize {
        self.step_index
    }

    /// Build the metas string for the current position.
    ///
    /// In `Advancing` mode: `duration = current_pos_s + window_s`, capped at `max_duration_s`.
    /// In `Fixed` mode: `duration = window_s` (model always sees the same arc position).
    fn current_metas(&self) -> String {
        let pos = match self.config.position_mode {
            PositionMode::Advancing => self.current_pos_s,
            PositionMode::Fixed => 0.0,
        };
        let duration = (pos + self.config.window_s).min(self.config.max_duration_s);
        tracing::info!(
            "  metas: pos={:.0}s duration={:.0}s mode={:?} bpm={:?} key={:?}",
            pos,
            duration,
            self.config.position_mode,
            self.bpm,
            self.key_scale,
        );
        crate::model::encoder::text::format_metas(
            self.bpm,
            Some(&self.time_signature),
            self.key_scale.as_deref(),
            Some(duration),
        )
    }

    /// Apply overrides from a request.
    fn apply_request(&mut self, req: &ChunkRequest) {
        if let Some(ref c) = req.caption {
            self.caption = c.clone();
        }
        if let Some(ref b) = req.bpm {
            self.bpm = *b;
        }
        if let Some(ref k) = req.key_scale {
            self.key_scale = k.clone();
        }
        if let Some(ref ts) = req.time_signature {
            self.time_signature = ts.clone();
        }
        if let Some(ref l) = req.lyrics {
            self.lyrics = l.clone();
        }
        if let Some(ref l) = req.language {
            self.language = l.clone();
        }
        if let Some(s) = req.step_s {
            self.config.step_s = s.clamp(1.0, self.config.window_s - 1.0);
            tracing::info!("step_s changed to {:.1}s", self.config.step_s);
        }
        if let Some(ref m) = req.position_mode {
            self.config.position_mode = m.clone();
            tracing::info!("position_mode changed to {:?}", self.config.position_mode);
        }
    }

    /// Generate the next step of audio.
    ///
    /// On the first call, generates the full window and outputs the first
    /// `step_s` of audio. On subsequent calls, shifts the window and
    /// regenerates only the new tail, outputting `step_s` of new audio.
    pub fn next_step(
        &mut self,
        pipeline: &mut AceStepPipeline,
        request: Option<ChunkRequest>,
    ) -> Result<WindowStep> {
        if let Some(ref req) = request {
            self.apply_request(req);
        }

        let seed = request.as_ref().and_then(|r| r.seed).or_else(|| {
            if self.config.auto_seed {
                Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64,
                )
            } else {
                None
            }
        });

        let device = pipeline.device().clone();
        let dtype = pipeline.dtype();
        let acoustic_dim = pipeline.config().audio_acoustic_hidden_dim;
        let timbre_fix_frame = pipeline.config().timbre_fix_frame;

        let window_frames = self.window_frames();
        let step_frames = self.step_frames();
        let is_first = self.window_latents.is_none();

        // In Advancing mode: wrap position before it would push duration past max_duration_s.
        // Jump to step_s (just past the beginning) so the model thinks it's
        // early in a fresh track rather than at second 0.
        if self.config.position_mode == PositionMode::Advancing
            && self.current_pos_s + self.config.step_s + self.config.window_s
                > self.config.max_duration_s
        {
            let new_pos = self.config.step_s;
            tracing::info!(
                "SlidingWindowGenerator: position wrap {:.0}s → {:.0}s (max={:.0}s)",
                self.current_pos_s,
                new_pos,
                self.config.max_duration_s
            );
            self.current_pos_s = new_pos;
        }

        tracing::info!(
            "=== Sliding window step {} (pos={:.0}s window={:.0}s step={:.0}s is_first={}) ===",
            self.step_index,
            self.current_pos_s,
            self.config.window_s,
            self.config.step_s,
            is_first
        );

        // Build src_latents and chunk_masks
        let (src_latents, chunk_masks) = if is_first {
            // First step: all silence, all generate
            let src = pipeline
                .silence_latent()
                .i((.., ..window_frames, ..))?
                .contiguous()?;
            let masks = Tensor::ones((1, window_frames, acoustic_dim), dtype, &device)?;
            (src, masks)
        } else {
            let prev = self.window_latents.as_ref().unwrap();

            // Shift: drop first step_frames, keep the rest, append silence tail
            let keep_frames = window_frames - step_frames;
            let kept = prev.i((.., step_frames.., ..))?.contiguous()?;
            let silence_tail = pipeline
                .silence_latent()
                .i((.., ..step_frames, ..))?
                .contiguous()?;
            let src = Tensor::cat(&[&kept, &silence_tail], 1)?;

            // Mask: 0 for kept region (model sees as context), 1 for new tail
            let mask_keep = Tensor::zeros((1, keep_frames, acoustic_dim), dtype, &device)?;
            let mask_gen = Tensor::ones((1, step_frames, acoustic_dim), dtype, &device)?;
            let masks = Tensor::cat(&[&mask_keep, &mask_gen], 1)?;

            (src, masks)
        };

        // Timbre: use the start of the current window as reference
        let (refer_audio, refer_order) = if !is_first {
            let prev = self.window_latents.as_ref().unwrap();
            let ref_len = self.filled_frames.min(timbre_fix_frame);
            let ref_latents = prev.i((.., ..ref_len, ..))?.contiguous()?;
            let refer = if ref_len < timbre_fix_frame {
                let pad = pipeline
                    .silence_latent()
                    .i((.., ..timbre_fix_frame - ref_len, ..))?
                    .contiguous()?;
                Tensor::cat(&[&ref_latents, &pad], 1)?
            } else {
                ref_latents
            };
            let order = Tensor::zeros((1,), DType::I64, &device)?;
            (Some(refer), Some(order))
        } else {
            (None, None)
        };

        let metas_for_step = self.current_metas();
        let params = GenerationParams {
            caption: self.caption.clone(),
            metas: metas_for_step,
            lyrics: self.lyrics.clone(),
            language: self.language.clone(),
            duration_s: self.config.window_s, // ignored — src_latents drives size
            shift: self.config.shift,
            seed,
            src_latents: Some(src_latents),
            chunk_masks: Some(chunk_masks),
            refer_audio,
            refer_order,
        };

        let (full_audio, new_window_latents) = generate_with_latents(pipeline, &params)?;

        // Extract only the new step's audio from the full window decode
        let step_audio_samples = step_frames * Self::AUDIO_HOP;
        let mut output_samples = if is_first {
            // First step: output first step_frames of audio
            full_audio.samples[..step_audio_samples.min(full_audio.samples.len())].to_vec()
        } else {
            // Subsequent steps: output last step_frames of audio (the new tail)
            let start = full_audio.samples.len().saturating_sub(step_audio_samples);
            full_audio.samples[start..].to_vec()
        };

        // --- Loudness normalization ------------------------------------------
        // Re-normalize the output slice independently — the full-window peak
        // normalize inside generate_with_latents uses the peak of the entire
        // 240s window, which can crush the new 30s tail to near-silence.
        //
        // We use a smoothed peak that decays slowly across steps so adjacent
        // steps have consistent loudness (no sudden jumps).
        let step_peak = output_samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        // Decay coefficient: smoothed_peak tracks toward step_peak.
        // Alpha=0.4 → fast attack (loud steps bring it up quickly),
        // slow release (quiet steps don't suddenly drop the volume).
        let alpha = if step_peak > self.smoothed_peak {
            0.6 // fast attack
        } else {
            0.15 // slow release
        };
        self.smoothed_peak = self.smoothed_peak + alpha * (step_peak - self.smoothed_peak);
        let norm_peak = self.smoothed_peak.max(1e-6);
        // Target 0.85 to leave some headroom.
        let scale = 0.85 / norm_peak;
        for s in &mut output_samples {
            *s *= scale;
        }

        // --- Crossfade with previous step -----------------------------------
        // Apply a short equal-power crossfade between the tail of the previous
        // step and the head of this step.  This eliminates hard-cut clicks and
        // smooths perceived style transitions at step boundaries.
        const CROSSFADE_MS: usize = 400;
        const CROSSFADE_SAMPLES: usize = 48000 * CROSSFADE_MS / 1000 * 2; // stereo
        if !self.prev_tail.is_empty() && output_samples.len() >= CROSSFADE_SAMPLES {
            let prev = &self.prev_tail;
            let fade_len = prev.len().min(CROSSFADE_SAMPLES);
            for i in 0..fade_len {
                // Equal-power crossfade: cos²/sin² ramp
                let t = i as f32 / fade_len as f32;
                let angle = t * std::f32::consts::FRAC_PI_2;
                let gain_prev = angle.cos();
                let gain_next = angle.sin();
                output_samples[i] =
                    gain_prev * prev[prev.len() - fade_len + i] + gain_next * output_samples[i];
            }
        }
        // Save tail for next step's crossfade
        if output_samples.len() >= CROSSFADE_SAMPLES {
            self.prev_tail = output_samples[output_samples.len() - CROSSFADE_SAMPLES..].to_vec();
        } else {
            self.prev_tail = output_samples.clone();
        }

        // Update window state
        self.window_latents = Some(new_window_latents);
        self.filled_frames = if is_first {
            window_frames
        } else {
            window_frames // always full after first step
        };

        let step_s = self.config.step_s;
        let window_s = self.config.window_s;
        let step_index = self.step_index;
        let pos_s = self.current_pos_s;

        // Advance position for the next step (no-op in Fixed mode)
        if self.config.position_mode == PositionMode::Advancing {
            self.current_pos_s += step_s;
        }
        self.step_index += 1;

        Ok(WindowStep {
            audio: GeneratedAudio {
                samples: output_samples,
                sample_rate: 48000,
                channels: 2,
            },
            step_index,
            step_s,
            window_s,
            pos_s,
        })
    }

    /// Reset: clear window state so the next step generates fresh.
    /// Caption/lyrics/config are preserved.
    pub fn restart(&mut self) {
        self.window_latents = None;
        self.filled_frames = 0;
        self.step_index = 0;
        self.current_pos_s = 0.0;
        tracing::info!("SlidingWindowGenerator: restarted");
    }

    /// Save current window state as a named snapshot.
    /// Returns `None` if no window has been generated yet.
    pub fn save_snapshot(&self) -> Option<WindowSnapshot> {
        self.window_latents.as_ref().map(|latents| WindowSnapshot {
            latents: latents.clone(),
            step_index: self.step_index,
            pos_s: self.current_pos_s,
        })
    }

    /// Restore a previously saved window snapshot.
    /// The next step will generate from this window context.
    pub fn load_snapshot(&mut self, snapshot: &WindowSnapshot) {
        self.window_latents = Some(snapshot.latents.clone());
        self.filled_frames = self.window_frames();
        self.step_index = snapshot.step_index;
        self.current_pos_s = snapshot.pos_s;
        // Reset crossfade state: no prev_tail from a different point in time.
        self.prev_tail.clear();
        self.smoothed_peak = 0.0;
        tracing::info!(
            "SlidingWindowGenerator: loaded snapshot (step_index={} pos={:.0}s)",
            self.step_index,
            self.current_pos_s,
        );
    }
}

/// A saved window state that can be restored later.
/// Use [`SlidingWindowGenerator::save_snapshot`] and
/// [`SlidingWindowGenerator::load_snapshot`] to jump back to any
/// point in the musical arc — e.g. back to an intro feel after an outro.
pub struct WindowSnapshot {
    /// The latent window at the time of saving [1, window_frames, 64].
    latents: Tensor,
    /// The step index at the time of saving.
    pub step_index: usize,
    /// The song position at the time of saving.
    pub pos_s: f64,
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

    #[test]
    fn test_sliding_window_config_defaults() {
        let cfg = SlidingWindowConfig::default();
        assert_eq!(cfg.window_s, 240.0);
        assert_eq!(cfg.step_s, 30.0);
        assert_eq!(cfg.shift, 3.0);
    }

    #[test]
    fn test_sliding_window_frame_math() {
        let cfg = SlidingWindowConfig {
            window_s: 240.0,
            step_s: 30.0,
            ..Default::default()
        };
        let sg = SlidingWindowGenerator::new(cfg, "jazz", None, None, "4/4", "hello");
        assert_eq!(sg.window_frames(), 6000); // 240 * 25
        assert_eq!(sg.step_frames(), 750); // 30 * 25
    }

    #[test]
    fn test_sliding_window_step_s_clamped() {
        let cfg = SlidingWindowConfig {
            window_s: 60.0,
            step_s: 10.0,
            ..Default::default()
        };
        let mut sg = SlidingWindowGenerator::new(cfg, "test", None, None, "4/4", "");
        // step_s larger than window should be clamped
        sg.apply_request(&ChunkRequest {
            step_s: Some(999.0),
            ..Default::default()
        });
        assert!(sg.config.step_s < sg.config.window_s);
        // step_s = 0 should clamp to 1
        sg.apply_request(&ChunkRequest {
            step_s: Some(0.0),
            ..Default::default()
        });
        assert_eq!(sg.config.step_s, 1.0);
    }
}
