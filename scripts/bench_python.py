#!/usr/bin/env python3
"""Benchmark ACE-Step v1.5 Python pipeline — 10s generation."""

import os
import sys
import time

import torch
import safetensors.torch as st

sys.path.insert(0, os.path.expanduser("~/repos/ACE-Step-1.5"))

DEVICE = "cuda"
DTYPE = torch.float32
SEED = 42
DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
SHIFT = 3.0

CAPTION = "A gentle acoustic guitar melody with warm piano chords, soft and peaceful"
METAS = "bpm: 90, key: C major, genre: ambient, instruments: acoustic guitar, piano"
LYRICS = "[verse]\nWe wrote the code in Rust they said it couldn't be done\nTwo billion parameters underneath the sun\n[chorus]\nAce Step in Rust, we made the machine sing\nThree point four seconds, hear the future ring\n"
LANGUAGE = "en"

HF_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ACE-Step--Ace-Step1.5/"
    "snapshots/19671f406d603126926c1b7e2adc169acbcade22"
)


def main():
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(HF_CACHE, "Qwen3-Embedding-0.6B/tokenizer.json"))

    # Load Qwen3 text encoder
    from transformers import Qwen3Model, Qwen3Config
    qwen3_config = Qwen3Config(
        vocab_size=151669, hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, max_position_embeddings=40960, rms_norm_eps=1e-6, rope_theta=1000000.0,
    )
    qwen3_model = Qwen3Model(qwen3_config).to(DEVICE).to(DTYPE)
    qwen3_weights = os.path.join(HF_CACHE, "Qwen3-Embedding-0.6B/model.safetensors")
    qwen3_model.load_state_dict(st.load_file(qwen3_weights), strict=False)
    qwen3_model.eval()

    # Load DiT model
    from acestep.models.turbo.modeling_acestep_v15_turbo import (
        AceStepConditionGenerationModel, AceStepConfig,
    )
    dit_path = os.path.join(HF_CACHE, "acestep-v15-turbo/model.safetensors")
    config = AceStepConfig()
    model = AceStepConditionGenerationModel(config).to(DEVICE).to(DTYPE)
    model.load_state_dict(st.load_file(dit_path))
    model.eval()

    # Load VAE
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(os.path.join(HF_CACHE, "vae")).to(DEVICE).to(DTYPE)
    vae.eval()

    # Load silence_latent
    silence_latent = torch.load(
        os.path.join(HF_CACHE, "acestep-v15-turbo/silence_latent.pt"), weights_only=True
    ).transpose(1, 2).to(DEVICE).to(DTYPE)

    print("All models loaded. Starting benchmark...\n")

    # Warmup
    torch.cuda.synchronize()

    # === BENCHMARK START ===
    t_total = time.perf_counter()

    # 1. Tokenize
    instruction = "Fill the audio semantic mask based on the given conditions:"
    caption_prompt = f"# Instruction\n{instruction}\n\n# Caption\n{CAPTION}\n\n# Metas\n{METAS}<|endoftext|>\n"
    lyric_prompt = f"# Languages\n{LANGUAGE}\n\n# Lyric\n{LYRICS}<|endoftext|>"

    caption_enc = tokenizer.encode(caption_prompt)
    lyric_enc = tokenizer.encode(lyric_prompt)

    caption_ids_t = torch.tensor([caption_enc.ids], dtype=torch.long, device=DEVICE)
    caption_mask_t = torch.tensor([caption_enc.attention_mask], dtype=torch.float32, device=DEVICE)
    lyric_ids_t = torch.tensor([lyric_enc.ids], dtype=torch.long, device=DEVICE)
    lyric_mask_t = torch.tensor([lyric_enc.attention_mask], dtype=torch.float32, device=DEVICE)

    # 2. Text encoding
    t1 = time.perf_counter()
    with torch.inference_mode():
        text_hidden = qwen3_model(input_ids=caption_ids_t).last_hidden_state
        lyric_hidden = qwen3_model.embed_tokens(lyric_ids_t)
    torch.cuda.synchronize()
    print(f"Text encoding:          {time.perf_counter() - t1:.3f}s")

    # 3. Prepare inputs
    T = int(DURATION_S * 25)
    src_latents = silence_latent[:, :T, :].clone()
    chunk_masks = torch.ones(1, T, 64, dtype=DTYPE, device=DEVICE)
    refer_audio = silence_latent[:, :750, :].clone()
    refer_order = torch.zeros(1, dtype=torch.long, device=DEVICE)
    attention_mask = torch.ones(1, T, dtype=DTYPE, device=DEVICE)
    is_covers = torch.zeros(1, dtype=torch.bool, device=DEVICE)

    # 4. Condition encoder
    t2 = time.perf_counter()
    with torch.inference_mode():
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
    torch.cuda.synchronize()
    print(f"Condition encoder:      {time.perf_counter() - t2:.3f}s")

    # 5. ODE loop
    SHIFT_TIMESTEPS = {
        3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334,
              0.75, 0.6428571428571429, 0.5, 0.3],
    }
    t_schedule = SHIFT_TIMESTEPS[SHIFT]
    num_steps = len(t_schedule)

    torch.manual_seed(SEED)
    noise = torch.randn(1, T, 64, dtype=DTYPE, device=DEVICE)
    xt = noise.clone()

    t3 = time.perf_counter()
    with torch.inference_mode():
        for step_idx in range(num_steps):
            t_curr = t_schedule[step_idx]
            t_tensor = torch.full((1,), t_curr, dtype=DTYPE, device=DEVICE)

            vt = model.decoder(
                hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
                attention_mask=attention_mask, encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask, context_latents=context_latents,
            )[0]

            if step_idx == num_steps - 1:
                t_expand = t_tensor.unsqueeze(-1).unsqueeze(-1)
                xt = xt - vt * t_expand
            else:
                t_next = t_schedule[step_idx + 1]
                dt = t_curr - t_next
                xt = xt - vt * dt

    torch.cuda.synchronize()
    print(f"Diffusion (8 ODE steps): {time.perf_counter() - t3:.3f}s")

    # 6. VAE decode (tiled for long sequences)
    t4 = time.perf_counter()
    vae_input = xt.transpose(1, 2).contiguous()
    with torch.inference_mode():
        latent_frames = vae_input.shape[2]
        chunk_size = 256
        overlap = 16
        if latent_frames <= chunk_size:
            waveform = vae.decode(vae_input).sample
        else:
            stride = chunk_size - 2 * overlap
            num_steps = (latent_frames + stride - 1) // stride
            decoded_chunks = []
            upsample_factor = None
            for i in range(num_steps):
                core_start = i * stride
                core_end = min(core_start + stride, latent_frames)
                win_start = max(core_start - overlap, 0)
                win_end = min(core_end + overlap, latent_frames)
                chunk = vae_input[:, :, win_start:win_end]
                audio_chunk = vae.decode(chunk).sample
                if upsample_factor is None:
                    upsample_factor = audio_chunk.shape[2] / chunk.shape[2]
                added_start = core_start - win_start
                trim_start = round(added_start * upsample_factor)
                added_end = win_end - core_end
                trim_end = round(added_end * upsample_factor)
                end_idx = audio_chunk.shape[2] - trim_end if trim_end > 0 else audio_chunk.shape[2]
                decoded_chunks.append(audio_chunk[:, :, trim_start:end_idx])
            waveform = torch.cat(decoded_chunks, dim=2)
    torch.cuda.synchronize()
    print(f"VAE decode:             {time.perf_counter() - t4:.3f}s")

    # 7. Peak normalize
    peak = waveform.abs().amax(dim=[1, 2], keepdim=True)
    if torch.any(peak > 1.0):
        waveform = waveform / peak.clamp(min=1.0)

    total = time.perf_counter() - t_total
    print(f"\nTotal generation:       {total:.3f}s")
    print(f"Waveform: {waveform.shape}, range: [{waveform.min():.4f}, {waveform.max():.4f}]")


if __name__ == "__main__":
    main()
