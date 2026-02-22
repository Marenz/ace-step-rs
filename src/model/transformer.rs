//! ACE-Step diffusion transformer.
//!
//! A 24-layer single-stream DiT with:
//! - Linear self-attention (ReLU kernel, O(n·d²))
//! - Standard cross-attention (SDPA) to encoder context
//! - GLUMBConv feed-forward (gated depthwise-separable 1D conv)
//! - AdaLN-Single conditioning from timestep
//!
//! ## Forward pass flow
//!
//! ```text
//! encode():
//!   speaker_embeds → Linear(512, 2560) → [B, 1, 2560]
//!   text → Linear(768, 2560) → [B, S_text, 2560]
//!   lyrics → Embedding → Conformer → Linear(1024, 2560) → [B, S_lyric, 2560]
//!   context = cat([speaker, text, lyrics]) → [B, S_enc, 2560]
//!
//! decode():
//!   timestep → sinusoidal(256) → MLP(256→2560) → SiLU+Linear(2560→6*2560) → temb
//!   latent → PatchEmbed → [B, T, 2560]
//!   for block in 24 blocks:
//!     AdaLN → linear self-attn → gated residual
//!     → cross-attn to context → residual
//!     → AdaLN → GLUMBConv → gated residual
//!   → FinalLayer → unpatchify → [B, 8, 16, T]
//! ```

pub mod attention;
pub mod config;
pub mod glumbconv;
pub mod patch_embed;
pub mod rope;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

use crate::model::encoder::conformer::{ConformerConfig, ConformerEncoder};
use crate::Result;
use config::TransformerConfig;

/// Complete ACE-Step transformer (encode + decode).
#[allow(dead_code)]
pub struct AceStepTransformer {
    config: TransformerConfig,
    /// Patch embedding: [B, 8, 16, T] → [B, T, 2560]
    patch_embed: patch_embed::PatchEmbed1d,
    /// Timestep sinusoidal projection channels (256).
    time_proj_dim: usize,
    /// Timestep MLP: Linear(256, 2560) + SiLU + Linear(2560, 2560)
    timestep_linear1: candle_nn::Linear,
    timestep_linear2: candle_nn::Linear,
    /// t_block: SiLU + Linear(2560, 6*2560) → per-block temb
    t_block_linear: candle_nn::Linear,
    /// Speaker embedding: Linear(512, 2560)
    speaker_embedder: candle_nn::Linear,
    /// Text/genre embedding: Linear(768, 2560)
    genre_embedder: candle_nn::Linear,
    /// Lyric token embeddings: Embedding(6693, 1024)
    lyric_embs: candle_nn::Embedding,
    /// Lyric Conformer encoder (6-layer relative-position attention).
    lyric_encoder: ConformerEncoder,
    /// Lyric projection: Linear(1024, 2560)
    lyric_proj: candle_nn::Linear,
    /// RoPE
    rope: rope::RotaryEmbedding,
    /// Transformer blocks
    blocks: Vec<LinearTransformerBlock>,
    /// Final layer
    final_layer: FinalLayer,
}

/// Single transformer block.
pub struct LinearTransformerBlock {
    /// RMSNorm before self-attention.
    norm1: candle_nn::RmsNorm,
    /// Linear self-attention (ReLU kernel).
    self_attn: attention::LinearAttention,
    /// Standard cross-attention (SDPA).
    cross_attn: attention::CrossAttention,
    /// RMSNorm before FFN.
    norm2: candle_nn::RmsNorm,
    /// Gated depthwise conv FFN.
    ffn: glumbconv::GluMbConv,
    /// AdaLN-Single: [6, inner_dim] learned table.
    scale_shift_table: Tensor,
}

impl LinearTransformerBlock {
    pub fn load(vb: VarBuilder, config: &TransformerConfig) -> Result<Self> {
        let dim = config.inner_dim;
        let heads = config.num_attention_heads;
        let head_dim = config.attention_head_dim;
        let hidden = (dim as f64 * config.mlp_ratio) as usize;

        let norm1 = candle_nn::rms_norm(dim, 1e-6, vb.pp("norm1"))?;
        let self_attn = attention::LinearAttention::load(vb.pp("attn"), dim, heads, head_dim)?;
        let cross_attn = attention::CrossAttention::load(
            vb.pp("cross_attn"),
            dim,
            dim, // cross_attention_dim = inner_dim
            heads,
            head_dim,
        )?;
        let norm2 = candle_nn::rms_norm(dim, 1e-6, vb.pp("norm2"))?;
        let ffn = glumbconv::GluMbConv::load(vb.pp("ff"), dim, hidden)?;
        let scale_shift_table = vb.get((6, dim), "scale_shift_table")?;

        Ok(Self {
            norm1,
            self_attn,
            cross_attn,
            norm2,
            ffn,
            scale_shift_table,
        })
    }

    /// Forward pass for one block.
    ///
    /// - `hidden_states`: `[B, T, dim]`
    /// - `encoder_hidden_states`: `[B, S_enc, dim]`
    /// - `attention_mask`: `[B, T]`
    /// - `encoder_attention_mask`: `[B, S_enc]`
    /// - `rope_cos`, `rope_sin`: `[T, head_dim]`
    /// - `rope_cos_cross`, `rope_sin_cross`: `[S_enc, head_dim]`
    /// - `temb`: `[B, 6*dim]`
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        rope_cos_cross: &Tensor,
        rope_sin_cross: &Tensor,
        temb: &Tensor,
    ) -> Result<Tensor> {
        let (batch, _seq, dim) = hidden_states.dims3()?;

        // AdaLN: scale_shift_table[None] + temb.reshape(N, 6, dim)
        // scale_shift_table: [6, dim] → [1, 6, dim]
        let table = self.scale_shift_table.unsqueeze(0)?;
        let temb_reshaped = temb.reshape((batch, 6, dim))?;
        let modulation = (&table + &temb_reshaped)?;

        // Chunk into 6 params: [B, 1, dim] each
        let chunks = modulation.chunk(6, 1)?;
        let shift_msa = &chunks[0];
        let scale_msa = &chunks[1];
        let gate_msa = &chunks[2];
        let shift_mlp = &chunks[3];
        let scale_mlp = &chunks[4];
        let gate_mlp = &chunks[5];

        // 1. Norm + modulate for self-attention
        let norm_hidden = self.norm1.forward(hidden_states)?;
        // t2i_modulate: x * (1 + scale) + shift
        let modulated =
            (norm_hidden.broadcast_mul(&(scale_msa + 1.0)?))?.broadcast_add(shift_msa)?;

        // 2. Linear self-attention
        let attn_out = self
            .self_attn
            .forward(&modulated, attention_mask, rope_cos, rope_sin)?;

        // 3. Gated residual
        let hidden_states = (hidden_states + attn_out.broadcast_mul(gate_msa)?)?;

        // 4. Cross-attention (no gate, direct residual)
        let cross_out = self.cross_attn.forward(
            &hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
            rope_cos,
            rope_sin,
            rope_cos_cross,
            rope_sin_cross,
        )?;
        let hidden_states = (&hidden_states + cross_out)?;

        // 5. Norm + modulate for FFN
        let norm_hidden = self.norm2.forward(&hidden_states)?;
        let modulated =
            (norm_hidden.broadcast_mul(&(scale_mlp + 1.0)?))?.broadcast_add(shift_mlp)?;

        // 6. FFN + gated residual
        let ff_out = self.ffn.forward(&modulated)?;
        let hidden_states = (hidden_states + ff_out.broadcast_mul(gate_mlp)?)?;

        Ok(hidden_states)
    }
}

/// Final output layer: RMSNorm + modulate + linear + unpatchify.
pub struct FinalLayer {
    norm: candle_nn::RmsNorm,
    /// Linear(inner_dim, patch_h * patch_w * out_channels) = Linear(2560, 128)
    linear: candle_nn::Linear,
    /// [2, inner_dim] scale/shift table.
    scale_shift_table: Tensor,
    patch_size: [usize; 2],
    out_channels: usize,
}

impl FinalLayer {
    pub fn load(vb: VarBuilder, config: &TransformerConfig) -> Result<Self> {
        let dim = config.inner_dim;
        let out_size = config.patch_size[0] * config.patch_size[1] * config.out_channels;

        let norm = candle_nn::rms_norm(dim, 1e-6, vb.pp("norm_final"))?;
        let linear = candle_nn::linear(dim, out_size, vb.pp("linear"))?;
        let scale_shift_table = vb.get((2, dim), "scale_shift_table")?;

        Ok(Self {
            norm,
            linear,
            scale_shift_table,
            patch_size: config.patch_size,
            out_channels: config.out_channels,
        })
    }

    /// Forward: `[B, T, dim]` → `[B, out_channels, patch_h, T]`
    ///
    /// `embedded_timestep`: `[B, dim]` (the 2560-dim vector before t_block)
    pub fn forward(
        &self,
        x: &Tensor,
        embedded_timestep: &Tensor,
        output_length: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _dim) = x.dims3()?;

        // Modulation: scale_shift_table[None] + t[:, None]
        let table = self.scale_shift_table.unsqueeze(0)?; // [1, 2, dim]
        let t = embedded_timestep.unsqueeze(1)?; // [B, 1, dim]
        let modulation = table.broadcast_add(&t)?;
        let chunks = modulation.chunk(2, 1)?;
        let shift = &chunks[0]; // [B, 1, dim]
        let scale = &chunks[1]; // [B, 1, dim]

        // t2i_modulate: norm(x) * (1 + scale) + shift
        let x = self.norm.forward(x)?;
        let x = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)?;

        // Linear: [B, T, dim] → [B, T, patch_h * patch_w * out_ch]
        let x = self.linear.forward(&x)?;

        // Unpatchify: [B, T, 128] → [B, out_channels, patch_h, T]
        let patch_h = self.patch_size[0];
        let patch_w = self.patch_size[1];
        let out_ch = self.out_channels;

        // Reshape: [B, T, patch_h * patch_w * out_ch] → [B, 1, T, patch_h, patch_w, out_ch]
        let x = x.reshape((batch, 1, seq_len, patch_h, patch_w, out_ch))?;
        // Permute: nhwpqc → nchpwq
        let x = x.permute([0, 5, 3, 1, 4, 2])?; // [B, out_ch, patch_h, 1, patch_w, T]
                                                // Reshape to [B, out_ch, patch_h, T * patch_w]
        let x = x.reshape((batch, out_ch, patch_h, seq_len * patch_w))?;

        // Trim or pad to output_length
        if seq_len * patch_w > output_length {
            Ok(x.narrow(3, 0, output_length)?)
        } else {
            // Would need padding, but in practice seq_len >= output_length
            Ok(x)
        }
    }
}

/// Sinusoidal timestep embedding (matches diffusers `Timesteps`).
///
/// Produces a `[B, dim]` embedding from scalar timesteps.
/// Uses `flip_sin_to_cos=True, downscale_freq_shift=0` as in ACE-Step.
fn sinusoidal_timestep_embedding(
    timesteps: &Tensor,
    dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let half_dim = dim / 2;
    let exponent: Vec<f64> = (0..half_dim)
        .map(|i| -(i as f64) * (10000.0_f64).ln() / half_dim as f64)
        .collect();
    let exponent = Tensor::from_vec(exponent, (1, half_dim), device)?.to_dtype(DType::F32)?;
    let exponent = exponent.exp()?;

    // timesteps: [B] → [B, 1]
    let timesteps = timesteps.unsqueeze(1)?.to_dtype(DType::F32)?;
    let args = timesteps.broadcast_mul(&exponent)?; // [B, half_dim]

    // flip_sin_to_cos=True: [cos, sin] order
    let cos = args.cos()?;
    let sin = args.sin()?;
    let embedding = Tensor::cat(&[&cos, &sin], 1)?; // [B, dim]

    Ok(embedding)
}

impl AceStepTransformer {
    /// Load all weights from safetensors via VarBuilder.
    pub fn load(vb: VarBuilder, config: &TransformerConfig) -> Result<Self> {
        let dim = config.inner_dim;

        let patch_embed = patch_embed::PatchEmbed1d::load(
            vb.pp("proj_in"),
            config.in_channels,
            dim,
            config.patch_size,
        )?;

        // TimestepEmbedding: Linear(256, dim) + SiLU + Linear(dim, dim)
        let timestep_linear1 = candle_nn::linear(256, dim, vb.pp("timestep_embedder.linear_1"))?;
        let timestep_linear2 = candle_nn::linear(dim, dim, vb.pp("timestep_embedder.linear_2"))?;

        // t_block: SiLU + Linear(dim, 6*dim)
        let t_block_linear = candle_nn::linear(dim, 6 * dim, vb.pp("t_block.1"))?;

        let speaker_embedder =
            candle_nn::linear(config.speaker_embedding_dim, dim, vb.pp("speaker_embedder"))?;
        let genre_embedder =
            candle_nn::linear(config.text_embedding_dim, dim, vb.pp("genre_embedder"))?;

        let lyric_embs = candle_nn::embedding(
            config.lyric_encoder_vocab_size,
            config.lyric_hidden_size,
            vb.pp("lyric_embs"),
        )?;
        let conformer_config = ConformerConfig {
            output_size: config.lyric_hidden_size,
            ..Default::default()
        };
        let lyric_encoder = ConformerEncoder::load(vb.pp("lyric_encoder"), &conformer_config)?;
        let lyric_proj = candle_nn::linear(config.lyric_hidden_size, dim, vb.pp("lyric_proj"))?;

        let rope = rope::RotaryEmbedding::new(config.attention_head_dim, config.rope_theta);

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block =
                LinearTransformerBlock::load(vb.pp(format!("transformer_blocks.{i}")), config)?;
            blocks.push(block);
        }

        let final_layer = FinalLayer::load(vb.pp("final_layer"), config)?;

        Ok(Self {
            config: config.clone(),
            patch_embed,
            time_proj_dim: 256,
            timestep_linear1,
            timestep_linear2,
            t_block_linear,
            speaker_embedder,
            genre_embedder,
            lyric_embs,
            lyric_encoder,
            lyric_proj,
            rope,
            blocks,
            final_layer,
        })
    }

    /// Encode conditioning context (text + speaker + lyrics).
    ///
    /// Returns `(encoder_hidden_states, encoder_mask)`:
    /// - `encoder_hidden_states`: `[B, 1 + S_text + S_lyric, 2560]`
    /// - `encoder_mask`: `[B, 1 + S_text + S_lyric]`
    pub fn encode(
        &self,
        text_hidden_states: &Tensor, // [B, S_text, 768]
        text_mask: &Tensor,          // [B, S_text]
        speaker_embeds: &Tensor,     // [B, 512]
        _lyric_token_ids: &Tensor,   // [B, S_lyric]
        lyric_mask: &Tensor,         // [B, S_lyric]
    ) -> Result<(Tensor, Tensor)> {
        let (batch, _s_text, _) = text_hidden_states.dims3()?;
        let device = text_hidden_states.device();

        // Speaker: [B, 512] → [B, 1, 2560]
        let speaker_emb = self
            .speaker_embedder
            .forward(speaker_embeds)?
            .unsqueeze(1)?;
        let speaker_mask = Tensor::ones((batch, 1), DType::F32, device)?;

        // Genre/text: [B, S_text, 768] → [B, S_text, 2560]
        let text_emb = self.genre_embedder.forward(text_hidden_states)?;

        // Lyrics: [B, S_lyric] → Embedding → Conformer → projection
        let lyric_emb = self.lyric_embs.forward(_lyric_token_ids)?; // [B, S_lyric, 1024]
        let lyric_encoded = self.lyric_encoder.forward(&lyric_emb, lyric_mask)?; // [B, S_lyric, 1024]
        let lyric_proj = self.lyric_proj.forward(&lyric_encoded)?; // [B, S_lyric, 2560]

        // Concatenate: [speaker(1), text(S_text), lyrics(S_lyric)]
        let encoder_hidden_states = Tensor::cat(&[&speaker_emb, &text_emb, &lyric_proj], 1)?;

        let text_mask_f32 = text_mask.to_dtype(DType::F32)?;
        let lyric_mask_f32 = lyric_mask.to_dtype(DType::F32)?;
        let encoder_mask = Tensor::cat(&[&speaker_mask, &text_mask_f32, &lyric_mask_f32], 1)?;

        Ok((encoder_hidden_states, encoder_mask))
    }

    /// Run the denoising forward pass (decode only — encoder states pre-computed).
    ///
    /// - `latent`: `[B, 8, 16, T]` current noisy latent
    /// - `attention_mask`: `[B, T]`
    /// - `encoder_hidden_states`: `[B, S_enc, 2560]` from encode()
    /// - `encoder_mask`: `[B, S_enc]`
    /// - `timestep`: `[B]` timestep values
    /// - `output_length`: target temporal length T
    ///
    /// Returns predicted velocity `[B, 8, 16, T]`.
    pub fn decode(
        &self,
        latent: &Tensor,
        attention_mask: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_mask: &Tensor,
        timestep: &Tensor,
        output_length: usize,
    ) -> Result<Tensor> {
        let device = latent.device();
        let dtype = latent.dtype();

        // 1. Timestep embedding
        // sinusoidal(256) → MLP(256→2560) → [B, 2560]
        let t_emb = sinusoidal_timestep_embedding(timestep, self.time_proj_dim, device)?;
        let t_emb = t_emb.to_dtype(dtype)?;
        let embedded_timestep = self
            .timestep_linear1
            .forward(&t_emb)?
            .apply(&candle_nn::Activation::Silu)?;
        let embedded_timestep = self.timestep_linear2.forward(&embedded_timestep)?;

        // t_block: SiLU + Linear(2560, 6*2560)
        let temb = embedded_timestep.apply(&candle_nn::Activation::Silu)?;
        let temb = self.t_block_linear.forward(&temb)?; // [B, 6*2560]

        // 2. Patch embed: [B, 8, 16, T] → [B, T, 2560]
        let mut hidden_states = self.patch_embed.forward(latent)?;

        // 3. Compute RoPE tables
        let seq_len = hidden_states.dim(1)?;
        let enc_len = encoder_hidden_states.dim(1)?;
        let (rope_cos, rope_sin) = self.rope.compute_freqs(seq_len, dtype, device)?;
        let (rope_cos_cross, rope_sin_cross) = self.rope.compute_freqs(enc_len, dtype, device)?;

        // 4. Transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(
                &hidden_states,
                encoder_hidden_states,
                Some(attention_mask),
                Some(encoder_mask),
                &rope_cos,
                &rope_sin,
                &rope_cos_cross,
                &rope_sin_cross,
                &temb,
            )?;
        }

        // 5. Final layer: [B, T, 2560] → [B, 8, 16, T]
        let output = self
            .final_layer
            .forward(&hidden_states, &embedded_timestep, output_length)?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::config::TransformerConfig;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    #[test]
    fn default_config_matches_deployed() {
        let config = TransformerConfig::default();
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.inner_dim, 2560);
        assert_eq!(config.num_attention_heads, 20);
        assert_eq!(config.attention_head_dim, 128);
        assert_eq!(config.in_channels, 8);
        assert_eq!(config.out_channels, 8);
        assert_eq!(config.patch_size, [16, 1]);
    }

    #[test]
    fn sinusoidal_embedding_shape() {
        let device = Device::Cpu;
        let timesteps = Tensor::new(&[500.0_f32, 250.0], &device).unwrap();
        let emb = super::sinusoidal_timestep_embedding(&timesteps, 256, &device).unwrap();
        assert_eq!(emb.dims(), &[2, 256]);
    }

    #[test]
    fn sinusoidal_embedding_different_timesteps() {
        let device = Device::Cpu;
        let t1 = Tensor::new(&[100.0_f32], &device).unwrap();
        let t2 = Tensor::new(&[900.0_f32], &device).unwrap();
        let e1 = super::sinusoidal_timestep_embedding(&t1, 256, &device).unwrap();
        let e2 = super::sinusoidal_timestep_embedding(&t2, 256, &device).unwrap();

        let diff: f32 = (&e1 - &e2)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            diff > 0.01,
            "different timesteps should produce different embeddings"
        );
    }

    #[test]
    fn final_layer_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = TransformerConfig::default();
        let final_layer = super::FinalLayer::load(vb, &config).unwrap();

        let x = Tensor::randn(0.0_f32, 1.0, (1, 644, 2560), &device).unwrap();
        let t = Tensor::randn(0.0_f32, 1.0, (1, 2560), &device).unwrap();

        let output = final_layer.forward(&x, &t, 644).unwrap();
        assert_eq!(output.dims(), &[1, 8, 16, 644]);
    }

    #[test]
    fn single_block_forward() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = TransformerConfig {
            inner_dim: 256,
            num_attention_heads: 4,
            attention_head_dim: 64,
            mlp_ratio: 2.5,
            ..Default::default()
        };

        let block = super::LinearTransformerBlock::load(vb, &config).unwrap();

        let hidden = Tensor::randn(0.0_f32, 1.0, (1, 32, 256), &device).unwrap();
        let encoder = Tensor::randn(0.0_f32, 1.0, (1, 16, 256), &device).unwrap();
        let temb = Tensor::randn(0.0_f32, 1.0, (1, 6 * 256), &device).unwrap();

        let rope = super::rope::RotaryEmbedding::new(64, 1_000_000.0);
        let (cos, sin) = rope.compute_freqs(32, DType::F32, &device).unwrap();
        let (cos_cross, sin_cross) = rope.compute_freqs(16, DType::F32, &device).unwrap();

        let output = block
            .forward(
                &hidden, &encoder, None, None, &cos, &sin, &cos_cross, &sin_cross, &temb,
            )
            .unwrap();

        assert_eq!(output.dims(), &[1, 32, 256]);
    }

    #[test]
    fn load_full_transformer_random_weights() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Use small config for test speed
        let config = TransformerConfig {
            num_layers: 2,
            inner_dim: 128,
            num_attention_heads: 2,
            attention_head_dim: 64,
            mlp_ratio: 2.0,
            text_embedding_dim: 64,
            speaker_embedding_dim: 32,
            lyric_encoder_vocab_size: 100,
            lyric_hidden_size: 64,
            ..Default::default()
        };

        let transformer = super::AceStepTransformer::load(vb, &config).unwrap();

        // Test encode
        let text = Tensor::randn(0.0_f32, 1.0, (1, 8, 64), &device).unwrap();
        let text_mask = Tensor::ones((1, 8), DType::F32, &device).unwrap();
        let speaker = Tensor::zeros((1, 32), DType::F32, &device).unwrap();
        let lyric_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let lyric_mask = Tensor::ones((1, 4), DType::F32, &device).unwrap();

        let (enc_states, enc_mask) = transformer
            .encode(&text, &text_mask, &speaker, &lyric_ids, &lyric_mask)
            .unwrap();
        // 1 (speaker) + 8 (text) + 4 (lyrics) = 13
        assert_eq!(enc_states.dims(), &[1, 13, 128]);
        assert_eq!(enc_mask.dims(), &[1, 13]);

        // Test decode
        let latent = Tensor::randn(0.0_f32, 1.0, (1, 8, 16, 32), &device).unwrap();
        let attn_mask = Tensor::ones((1, 32), DType::F32, &device).unwrap();
        let timestep = Tensor::new(&[500.0_f32], &device).unwrap();

        let output = transformer
            .decode(&latent, &attn_mask, &enc_states, &enc_mask, &timestep, 32)
            .unwrap();
        assert_eq!(output.dims(), &[1, 8, 16, 32]);
    }
}
