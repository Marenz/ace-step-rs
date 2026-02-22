# AGENTS.md — ace-step-rs

## Project Overview

Pure Rust implementation of ACE-Step music generation using the candle ML framework. Loads original safetensors weights directly — no ONNX conversion. Targets CUDA and CPU backends.

## Build & Test

```bash
# Build (default features include CUDA)
cargo build

# Build CPU-only
cargo build --no-default-features

# Run all tests
cargo test --no-default-features

# Run tests for a specific module
cargo test --no-default-features audio::mel
cargo test --no-default-features scheduler::euler
```

## Architecture

```
text prompt → UMT5 encoder ──┐
                              ├→ cross-attention context [B, S, 2560]
lyrics → Conformer encoder ──┘
                              ↓
             DiT (24 blocks, flow matching, linear self-attention)
                              ↓
             DCAE decoder (latent → mel spectrogram)
                              ↓
             ADaMoSHiFiGAN vocoder (mel → audio waveform)
```

## Module Layout

```
src/
├── lib.rs              — crate root
├── error.rs            — Error enum
├── audio.rs            → audio/
│   ├── mel.rs          — STFT + mel filterbank
│   └── wav.rs          — WAV read/write
├── model.rs            → model/
│   ├── transformer.rs  → transformer/
│   │   ├── config.rs   — TransformerConfig
│   │   ├── attention.rs — LinearAttention + CrossAttention
│   │   ├── glumbconv.rs — GLU depthwise conv FFN
│   │   ├── patch_embed.rs — latent → token sequence
│   │   └── rope.rs     — rotary position embedding
│   ├── encoder.rs      → encoder/
│   │   ├── conformer.rs — Conformer lyric encoder
│   │   └── lyric_tokenizer.rs — BPE tokenizer
│   ├── dcae.rs         — DCAE latent decoder
│   └── vocoder.rs      — ADaMoSHiFiGAN
├── scheduler.rs        → scheduler/
│   ├── euler.rs        — Euler flow-matching
│   ├── heun.rs         — Heun predictor-corrector
│   └── pingpong.rs     — Stochastic SDE
└── pipeline.rs         — end-to-end inference
```

Module roots (e.g., `src/audio.rs`) contain `mod` declarations and re-exports. Never create `mod.rs` files.

## Code Conventions

- Rust edition 2024
- Use `candle_core::Result` internally, wrap in `crate::Result` at module boundaries
- Every module has `#[cfg(test)] mod tests` in the same file
- Tests use `--no-default-features` (CPU only) so they run anywhere
- Weight key paths must match the original Python checkpoint exactly
- Tensor shapes documented in comments as `[B, C, H, W]` notation
- Reference the Python source in `~/repos/ACE-Step/` for architecture details

## Key Constants

- Sample rate: 44100 Hz
- Hop length: 512
- Mel bins: 128
- Latent shape: [B, 8, 16, T] where T = ceil(duration * 44100 / 512 / 8)
- DCAE compression: 8× in both dimensions
- DiT: 24 layers, 2560 hidden, 20 heads × 128 head_dim

## Reference Repositories

- `~/repos/ACE-Step/` — original Python implementation (ground truth)
- `~/repos/lofi.nvim/` — Rust ONNX-based implementation (reference for pipeline logic)
- `~/repos/candle/` — candle ML framework (reference for API patterns)

## Common Pitfalls

- candle's `Conv2dConfig` has a single `stride` field — ACE-Step needs `[16, 1]` stride for patch embedding. Since height is always exactly 16, setting stride=16 works but only because the spatial dimension matches.
- Linear attention uses ReLU kernel, NOT softmax. Don't use candle's built-in attention.
- The Conformer uses relative-position attention (Espnet-style) — nothing like this exists in candle.
- Weight files use `diffusion_pytorch_model.safetensors` naming, not `model.safetensors`.
