//! Compare Rust pipeline intermediates against Python reference.
//!
//! Usage: cargo run --example compare --no-default-features
//!
//! Requires Python intermediates in /tmp/ace-step-compare/ (run scripts/dump_intermediates.py first).

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

fn load_ref(name: &str) -> HashMap<String, Tensor> {
    let path = format!("/tmp/ace-step-compare/{name}");
    candle_core::safetensors::load(&path, &Device::Cpu).unwrap_or_else(|e| {
        panic!("Failed to load {path}: {e}");
    })
}

fn compare(label: &str, rust: &Tensor, python: &Tensor) {
    let rust_f32 = rust
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
    let python_f32 = python
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();

    let rust_shape = rust_f32.dims().to_vec();
    let py_shape = python_f32.dims().to_vec();

    if rust_shape != py_shape {
        println!("  {label}: SHAPE MISMATCH rust={rust_shape:?} python={py_shape:?}");
        return;
    }

    let diff = (&rust_f32 - &python_f32).unwrap().abs().unwrap();
    let diff_flat = diff.flatten_all().unwrap();
    let max_diff: f32 = diff_flat.max(0).unwrap().to_scalar().unwrap();
    let mean_diff: f32 = diff_flat.mean_all().unwrap().to_scalar().unwrap();

    let r_flat = rust_f32.flatten_all().unwrap();
    let p_flat = python_f32.flatten_all().unwrap();
    let r_min: f32 = r_flat.min(0).unwrap().to_scalar().unwrap();
    let r_max: f32 = r_flat.max(0).unwrap().to_scalar().unwrap();
    let p_min: f32 = p_flat.min(0).unwrap().to_scalar().unwrap();
    let p_max: f32 = p_flat.max(0).unwrap().to_scalar().unwrap();

    let status = if max_diff < 1e-4 {
        "OK"
    } else if max_diff < 1e-2 {
        "CLOSE"
    } else {
        "MISMATCH"
    };

    println!(
        "  {label}: {status} max_diff={max_diff:.6} mean_diff={mean_diff:.6} shape={rust_shape:?} rust=[{r_min:.4},{r_max:.4}] py=[{p_min:.4},{p_max:.4}]"
    );
}

fn main() -> ace_step_rs::Result<()> {
    tracing_subscriber::fmt::init();

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Check intermediates exist
    if !std::path::Path::new("/tmp/ace-step-compare/01_tokens.safetensors").exists() {
        eprintln!(
            "ERROR: Run scripts/dump_intermediates.py first to generate Python reference data"
        );
        std::process::exit(1);
    }

    println!("=== Loading pipeline ===");
    let mut pipeline = ace_step_rs::pipeline::AceStepPipeline::load(&device, dtype)?;
    println!("Pipeline loaded\n");

    // === Stage 1: Tokenization ===
    println!("=== Stage 1: Tokenization ===");
    let ref_tokens = load_ref("01_tokens.safetensors");

    // Use same prompt as Python
    let caption = "A gentle acoustic guitar melody with warm piano chords, soft and peaceful";
    let metas = "bpm: 90, key: C major, genre: ambient, instruments: acoustic guitar, piano";
    let lyrics = "[verse]\nI'm sorry for the things I've said\nThe words that echoed in my head\n[chorus]\nForgive me now, I'll make it right\nI'll hold you close through every night\n";
    let language = "en";

    use ace_step_rs::model::encoder::text;
    let tokenizer = {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.model("ACE-Step/Ace-Step1.5".to_string());
        let tok_path = repo.get("Qwen3-Embedding-0.6B/tokenizer.json").unwrap();
        tokenizers::Tokenizer::from_file(&tok_path).unwrap()
    };

    let caption_prompt = text::format_caption_prompt(caption, metas);
    let caption_enc = tokenizer.encode(caption_prompt, true).unwrap();
    let caption_ids: Vec<u32> = caption_enc.get_ids().to_vec();
    let caption_mask: Vec<f32> = caption_enc
        .get_attention_mask()
        .iter()
        .map(|&m| m as f32)
        .collect();

    let lyric_prompt = text::format_lyric_prompt(lyrics, language);
    let lyric_enc = tokenizer.encode(lyric_prompt, true).unwrap();
    let lyric_ids: Vec<u32> = lyric_enc.get_ids().to_vec();
    let lyric_mask: Vec<f32> = lyric_enc
        .get_attention_mask()
        .iter()
        .map(|&m| m as f32)
        .collect();

    let caption_ids_t = Tensor::new(&caption_ids[..], &device)?.unsqueeze(0)?;
    let caption_mask_t = Tensor::new(&caption_mask[..], &device)?.unsqueeze(0)?;
    let lyric_ids_t = Tensor::new(&lyric_ids[..], &device)?.unsqueeze(0)?;
    let lyric_mask_t = Tensor::new(&lyric_mask[..], &device)?.unsqueeze(0)?;

    // Compare as f32 (Python saved ids as f32)
    let caption_ids_f32 = caption_ids_t.to_dtype(DType::F32)?;
    let lyric_ids_f32 = lyric_ids_t.to_dtype(DType::F32)?;
    compare("caption_ids", &caption_ids_f32, &ref_tokens["caption_ids"]);
    compare("caption_mask", &caption_mask_t, &ref_tokens["caption_mask"]);
    compare("lyric_ids", &lyric_ids_f32, &ref_tokens["lyric_ids"]);
    compare("lyric_mask", &lyric_mask_t, &ref_tokens["lyric_mask"]);

    // === Stage 2: Text encoding ===
    println!("\n=== Stage 2: Text encoding ===");
    let ref_text = load_ref("02_text_hidden.safetensors");
    let text_hidden = pipeline.encode_text(&caption_ids_t)?;
    compare("text_hidden", &text_hidden, &ref_text["text_hidden"]);

    // === Stage 3: Lyric embedding ===
    println!("\n=== Stage 3: Lyric embedding ===");
    let ref_lyric = load_ref("03_lyric_hidden.safetensors");
    let lyric_hidden = pipeline.embed_lyrics(&lyric_ids_t)?;
    compare("lyric_hidden", &lyric_hidden, &ref_lyric["lyric_hidden"]);

    // === Stage 4: Inputs (silence_latent) ===
    println!("\n=== Stage 4: Inputs ===");
    let ref_inputs = load_ref("04_inputs.safetensors");
    let silence_latent = pipeline.silence_latent();
    let t = 250usize; // 10s * 25Hz
    let src_latents = silence_latent.i((.., ..t, ..))?.contiguous()?;
    let refer_audio = silence_latent.i((.., ..750, ..))?.contiguous()?;
    compare("src_latents", &src_latents, &ref_inputs["src_latents"]);
    compare("refer_audio", &refer_audio, &ref_inputs["refer_audio"]);
    compare(
        "silence_latent_slice",
        &silence_latent.i((.., ..10, ..))?.contiguous()?,
        &ref_inputs["silence_latent_slice"],
    );

    // === Stage 5: Condition encoder ===
    println!("\n=== Stage 5: Condition encoder ===");
    let ref_enc = load_ref("05_encoder_out.safetensors");
    let refer_order = Tensor::zeros((1,), DType::I64, &device)?;
    let (enc_hidden, enc_mask) = pipeline.encode_conditions(
        &text_hidden,
        &caption_mask_t.to_dtype(dtype)?,
        &lyric_hidden,
        &lyric_mask_t.to_dtype(dtype)?,
        &refer_audio,
        &refer_order,
    )?;
    compare("encoder_hidden", &enc_hidden, &ref_enc["encoder_hidden"]);
    compare("encoder_mask", &enc_mask, &ref_enc["encoder_mask"]);

    // === Stage 6: Context ===
    println!("\n=== Stage 6: Context ===");
    let ref_ctx = load_ref("06_context.safetensors");
    let chunk_masks = Tensor::ones((1, t, 64), dtype, &device)?;
    let context = Tensor::cat(&[&src_latents, &chunk_masks], 2)?;
    compare("context_latents", &context, &ref_ctx["context_latents"]);

    // === Stage 7: Noise ===
    // We can't compare noise since seeds differ between Rust rand and Python torch.
    // But we can feed Python's noise to verify the ODE loop.
    println!("\n=== Stage 7: Noise (skipping - seed differs) ===");
    let ref_noise = load_ref("07_noise.safetensors");
    println!(
        "  Python noise range: [{:.4}, {:.4}]",
        ref_noise["noise"]
            .flatten_all()?
            .min(0)?
            .to_scalar::<f32>()?,
        ref_noise["noise"]
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?
    );

    // === Stage 8: ODE loop (using Python's noise for fair comparison) ===
    println!("\n=== Stage 8: ODE loop (using Python noise) ===");
    let noise = ref_noise["noise"].clone();
    let schedule = ace_step_rs::config::TurboSchedule::for_shift(3.0);
    let num_steps = schedule.len();
    let mut xt = noise;

    for step_idx in 0..num_steps {
        let t_curr = schedule[step_idx];
        let t_tensor = Tensor::full(t_curr as f32, (1,), &device)?.to_dtype(dtype)?;

        let vt = pipeline.dit_forward(&xt, &t_tensor, &t_tensor, &enc_hidden, &context)?;

        if step_idx == num_steps - 1 {
            let t_expand = t_tensor.unsqueeze(1)?.unsqueeze(2)?;
            xt = (&xt - vt.broadcast_mul(&t_expand)?)?;
        } else {
            let t_next = schedule[step_idx + 1];
            let dt = (t_curr - t_next) as f32;
            let dt_tensor = Tensor::full(dt, (1, 1, 1), &device)?.to_dtype(dtype)?;
            xt = (&xt - vt.broadcast_mul(&dt_tensor)?)?;
        }

        let ref_step = load_ref(&format!("08_ode_step_{step_idx:02}.safetensors"));
        compare(&format!("ode_step_{step_idx:02}_xt"), &xt, &ref_step["xt"]);
        compare(&format!("ode_step_{step_idx:02}_vt"), &vt, &ref_step["vt"]);
    }

    // === Stage 9: Final latents ===
    println!("\n=== Stage 9: Final latents ===");
    let ref_final = load_ref("09_final_latents.safetensors");
    compare("final_latents", &xt, &ref_final["final_latents"]);

    // === Stage 10: VAE decode ===
    println!("\n=== Stage 10: VAE decode ===");
    let ref_vae = load_ref("10_vae_input.safetensors");
    let vae_input = xt.transpose(1, 2)?.contiguous()?;
    compare("vae_input", &vae_input, &ref_vae["vae_input"]);

    let waveform = pipeline.vae_decode(&vae_input)?;
    let ref_wav = load_ref("11_waveform.safetensors");
    compare("waveform", &waveform, &ref_wav["waveform"]);

    println!("\n=== Done ===");
    Ok(())
}
