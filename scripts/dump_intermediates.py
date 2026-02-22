#!/usr/bin/env python3
"""Dump intermediate tensors from ACE-Step v1.5 pipeline for comparison with Rust.

Saves safetensors files to /tmp/ace-step-compare/ at each pipeline stage:
  01_tokens.safetensors       — tokenized caption/lyric IDs and masks
  02_text_hidden.safetensors  — Qwen3 text encoder output
  03_lyric_hidden.safetensors — Qwen3 lyric token embeddings
  04_inputs.safetensors       — src_latents, chunk_masks, refer_audio, refer_order
  05_encoder_out.safetensors  — condition encoder output
  06_context.safetensors      — context_latents (cat[src, masks])
  07_noise.safetensors        — initial noise
  08_ode_step_N.safetensors   — xt after each ODE step
  09_final_latents.safetensors — predicted x0
  10_vae_input.safetensors    — latents transposed for VAE
  11_waveform.safetensors     — decoded waveform
"""

import os
import sys
import time

import torch
import safetensors.torch as st
from pathlib import Path

# Setup
sys.path.insert(0, os.path.expanduser("~/repos/ACE-Step-1.5"))
OUT_DIR = Path("/tmp/ace-step-compare")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # Match Rust CPU run
SEED = 42
DURATION_S = 10.0
SHIFT = 3.0

# Same prompt as Rust example
CAPTION = "A gentle acoustic guitar melody with warm piano chords, soft and peaceful"
METAS = "bpm: 90, key: C major, genre: ambient, instruments: acoustic guitar, piano"
LYRICS = "[verse]\nI'm sorry for the things I've said\nThe words that echoed in my head\n[chorus]\nForgive me now, I'll make it right\nI'll hold you close through every night\n"
LANGUAGE = "en"

HF_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ACE-Step--Ace-Step1.5/"
    "snapshots/19671f406d603126926c1b7e2adc169acbcade22"
)


def save(name, tensors):
    """Save dict of tensors as safetensors."""
    path = OUT_DIR / name
    # Convert all to float32 for comparison
    out = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().float().contiguous()
        else:
            out[k] = torch.tensor(v, dtype=torch.float32)
    st.save_file(out, str(path))
    print(f"  Saved {path} ({', '.join(f'{k}: {v.shape}' for k, v in out.items())})")


def main():
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")

    # 1. Load tokenizer
    from tokenizers import Tokenizer
    tokenizer_path = os.path.join(HF_CACHE, "Qwen3-Embedding-0.6B/tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("Tokenizer loaded")

    # 2. Format and tokenize caption using the correct SFT_GEN_PROMPT template.
    #    Python reference: SFT_GEN_PROMPT.format(instruction, caption, parsed_meta)
    #    The triple-quoted SFT_GEN_PROMPT ends with a newline after <|endoftext|>.
    instruction = "Fill the audio semantic mask based on the given conditions:"
    caption_prompt = (
        f"# Instruction\n{instruction}\n\n"
        f"# Caption\n{CAPTION}\n\n"
        f"# Metas\n{METAS}<|endoftext|>\n"
    )
    caption_enc = tokenizer.encode(caption_prompt)
    caption_ids = caption_enc.ids
    caption_mask = caption_enc.attention_mask

    # 3. Format and tokenize lyrics using the correct _format_lyrics template.
    #    Python reference: f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"
    lyric_prompt = f"# Languages\n{LANGUAGE}\n\n# Lyric\n{LYRICS}<|endoftext|>"
    lyric_enc = tokenizer.encode(lyric_prompt)
    lyric_ids = lyric_enc.ids
    lyric_mask = lyric_enc.attention_mask

    caption_ids_t = torch.tensor([caption_ids], dtype=torch.long, device=DEVICE)
    caption_mask_t = torch.tensor([caption_mask], dtype=torch.float32, device=DEVICE)
    lyric_ids_t = torch.tensor([lyric_ids], dtype=torch.long, device=DEVICE)
    lyric_mask_t = torch.tensor([lyric_mask], dtype=torch.float32, device=DEVICE)

    save("01_tokens.safetensors", {
        "caption_ids": caption_ids_t,
        "caption_mask": caption_mask_t,
        "lyric_ids": lyric_ids_t,
        "lyric_mask": lyric_mask_t,
    })

    # 4. Load Qwen3 text encoder
    from transformers import Qwen3Model, Qwen3Config
    qwen3_path = os.path.join(HF_CACHE, "Qwen3-Embedding-0.6B")
    # No config.json in this dir, construct manually from safetensors inspection
    qwen3_config = Qwen3Config(
        vocab_size=151669,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )
    qwen3_model = Qwen3Model(qwen3_config).to(DEVICE).to(DTYPE)
    qwen3_weights = os.path.join(qwen3_path, "model.safetensors")
    qwen3_state = st.load_file(qwen3_weights)
    # Both safetensors and Qwen3Model use unprefixed keys (embed_tokens.*, layers.*, norm.*)
    missing, unexpected = qwen3_model.load_state_dict(qwen3_state, strict=False)
    if missing:
        print(f"  Warning: missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  Warning: unexpected keys: {unexpected[:5]}...")
    qwen3_model.eval()
    print("Qwen3 text encoder loaded")

    # 5. Text encoding
    with torch.inference_mode():
        text_out = qwen3_model(input_ids=caption_ids_t)
        text_hidden = text_out.last_hidden_state  # [1, T, 1024]

    save("02_text_hidden.safetensors", {"text_hidden": text_hidden})

    # 6. Lyric embeddings (raw token embeddings, not full encoder)
    with torch.inference_mode():
        lyric_hidden = qwen3_model.embed_tokens(lyric_ids_t)  # [1, T, 1024]

    save("03_lyric_hidden.safetensors", {"lyric_hidden": lyric_hidden})

    # 7. Load silence_latent
    silence_path = os.path.join(HF_CACHE, "acestep-v15-turbo/silence_latent.pt")
    silence_latent = torch.load(silence_path, weights_only=True).transpose(1, 2)
    silence_latent = silence_latent.to(DEVICE).to(DTYPE)
    print(f"Silence latent: {silence_latent.shape}")

    # 8. Prepare inputs
    T = int(DURATION_S * 25)  # 250 frames for 10s
    src_latents = silence_latent[:, :T, :].clone()
    chunk_masks = torch.ones(1, T, 64, dtype=DTYPE, device=DEVICE)
    refer_audio = silence_latent[:, :750, :].clone()
    refer_order = torch.zeros(1, dtype=torch.long, device=DEVICE)

    save("04_inputs.safetensors", {
        "src_latents": src_latents,
        "chunk_masks": chunk_masks,
        "refer_audio": refer_audio,
        "refer_order": refer_order,
        "silence_latent_slice": silence_latent[:, :10, :],  # first 10 frames for quick check
    })

    # 9. Load DiT model
    from acestep.models.turbo.modeling_acestep_v15_turbo import (
        AceStepConditionGenerationModel,
        AceStepConfig,
    )
    dit_path = os.path.join(HF_CACHE, "acestep-v15-turbo/model.safetensors")
    config = AceStepConfig()
    model = AceStepConditionGenerationModel(config).to(DEVICE).to(DTYPE)
    dit_state = st.load_file(dit_path)
    model.load_state_dict(dit_state)
    model.eval()
    print("DiT model loaded")

    # 10. Run condition encoder
    with torch.inference_mode():
        # Use prepare_condition which is what generate_audio calls
        attention_mask = torch.ones(1, T, dtype=DTYPE, device=DEVICE)
        is_covers = torch.zeros(1, dtype=torch.bool, device=DEVICE)

        encoder_hidden, encoder_mask, context_latents = model.prepare_condition(
            text_hidden_states=text_hidden,
            text_attention_mask=caption_mask_t.to(DTYPE),
            lyric_hidden_states=lyric_hidden,
            lyric_attention_mask=lyric_mask_t.to(DTYPE),
            refer_audio_acoustic_hidden_states_packed=refer_audio,
            refer_audio_order_mask=refer_order,
            hidden_states=src_latents,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
        )

    save("05_encoder_out.safetensors", {
        "encoder_hidden": encoder_hidden,
        "encoder_mask": encoder_mask,
    })
    save("06_context.safetensors", {"context_latents": context_latents})

    # 11. Generate noise (deterministic with seed)
    torch.manual_seed(SEED)
    # Python generates noise on CPU then moves to device
    noise_shape = (1, T, 64)
    noise = torch.randn(noise_shape, dtype=DTYPE, device=DEVICE)

    save("07_noise.safetensors", {"noise": noise})

    # 12. ODE loop
    SHIFT_TIMESTEPS = {
        1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
              0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
        3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334,
              0.75, 0.6428571428571429, 0.5, 0.3],
    }
    t_schedule = SHIFT_TIMESTEPS[SHIFT]
    num_steps = len(t_schedule)

    xt = noise.clone()
    with torch.inference_mode():
        for step_idx in range(num_steps):
            t_curr = t_schedule[step_idx]
            t_tensor = torch.full((1,), t_curr, dtype=DTYPE, device=DEVICE)

            # Forward through decoder (DiT)
            vt = model.decoder(
                hidden_states=xt,
                timestep=t_tensor,
                timestep_r=t_tensor,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
                context_latents=context_latents,
            )[0]

            if step_idx == num_steps - 1:
                # Final step: x0 = xt - vt * t
                t_expand = t_tensor.unsqueeze(-1).unsqueeze(-1)
                xt = xt - vt * t_expand
            else:
                # Euler step: xt = xt - vt * dt
                t_next = t_schedule[step_idx + 1]
                dt = t_curr - t_next
                xt = xt - vt * dt

            save(f"08_ode_step_{step_idx:02d}.safetensors", {
                "xt": xt,
                "vt": vt,
                "t": t_tensor,
            })
            print(f"  ODE step {step_idx}: t={t_curr:.4f}, xt range=[{xt.min():.4f}, {xt.max():.4f}]")

    save("09_final_latents.safetensors", {"final_latents": xt})

    # 13. VAE decode
    from diffusers.models import AutoencoderOobleck
    vae_path = os.path.join(HF_CACHE, "vae")
    vae = AutoencoderOobleck.from_pretrained(vae_path).to(DEVICE).to(DTYPE)
    vae.eval()
    print("VAE loaded")

    vae_input = xt.transpose(1, 2).contiguous()  # [1, 64, T]
    save("10_vae_input.safetensors", {"vae_input": vae_input})

    with torch.inference_mode():
        decoder_output = vae.decode(vae_input)
        waveform = decoder_output.sample  # [1, 2, samples]

    # Peak normalize
    peak = waveform.abs().amax(dim=[1, 2], keepdim=True)
    if torch.any(peak > 1.0):
        waveform = waveform / peak.clamp(min=1.0)

    save("11_waveform.safetensors", {"waveform": waveform})
    print(f"Waveform shape: {waveform.shape}, range: [{waveform.min():.4f}, {waveform.max():.4f}]")

    # Also save as WAV for listening comparison
    import scipy.io.wavfile as wavfile
    import numpy as np
    wav_path = OUT_DIR / "python_output.wav"
    wav_data = waveform.squeeze(0).cpu().numpy().T  # [samples, 2]
    wav_data = (wav_data * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(str(wav_path), 48000, wav_data)
    print(f"Saved {wav_path}")

    print(f"\nAll intermediates saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
