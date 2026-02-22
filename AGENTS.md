# AGENTS.md — ace-step-rs

## Project Overview

Pure Rust implementation of ACE-Step v1.5 music generation using the candle ML framework. Loads original safetensors weights directly — no ONNX conversion. Targets CUDA and CPU backends.

## Build & Test

```bash
# Build CPU-only
cargo build --no-default-features

# Build with CUDA (full command — all env vars are required)
LIBRARY_PATH=/usr/lib64:$LIBRARY_PATH \
PATH="/usr/local/cuda-12.4/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.4 \
NVCC_CCBIN=/usr/bin/g++-13 \
CPLUS_INCLUDE_PATH="/tmp/cuda-shim" \
cargo build --release

# Run all tests (CPU-only, tests must work without GPU)
cargo test --no-default-features

# Run tests for a specific module
cargo test --no-default-features config
cargo test --no-default-features vae
cargo test --no-default-features model::transformer::attention

# Run examples with CUDA
LIBRARY_PATH=/usr/lib64:$LIBRARY_PATH \
PATH="/usr/local/cuda-12.4/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.4 \
NVCC_CCBIN=/usr/bin/g++-13 \
CPLUS_INCLUDE_PATH="/tmp/cuda-shim" \
cargo run --release --example generate
```

### CUDA build notes

- **LIBRARY_PATH=/usr/lib64** — required because `/usr/lib/libcuda.so` is 32-bit; the 64-bit one is in `/usr/lib64/`
- **NVCC_CCBIN=/usr/bin/g++-13** — glibc 2.42 has an `rsqrt` conflict with CUDA headers
- **CPLUS_INCLUDE_PATH="/tmp/cuda-shim"** — header shim at `/tmp/cuda-shim/bits/mathcalls.h` suppresses `__GLIBC_USE_IEC_60559_FUNCS_EXT_C23`
- **F32 only** — BF16/F16 get `CUDA_ERROR_NOT_FOUND "named symbol not found"` (candle kernel issue)
- GPU: RTX 3090 (24GB), sm_86

### Performance (RTX 3090, F32, cuDNN ConvTranspose1d + TF32 matmul)

#### 10 seconds

| Stage | Python (PyTorch) | Rust (candle) |
|---|---|---|
| Text encoding | 0.18s | 0.02s |
| Diffusion (8 ODE steps) | 0.41s | 0.38s |
| VAE decode | 0.15s | 0.18s |
| **Total** | **0.88s** | **0.59s** |

#### 30 seconds

| Stage | Python (PyTorch) | Rust (candle) |
|---|---|---|
| Text encoding | 0.14s | 0.02s |
| Diffusion (8 ODE steps) | 0.73s | 0.63s |
| VAE decode | 0.39s | 0.54s |
| **Total** | **1.38s** | **1.25s** |

#### 4 minutes (240s)

| Stage | Python (PyTorch) | Rust (candle) |
|---|---|---|
| Text encoding | 0.12s | 0.02s |
| Diffusion (8 ODE steps) | 5.81s | 6.99s |
| VAE decode | 3.05s | 4.72s |
| **Total** | **9.26s** | **12.04s** |

Uses a local candle fork with cuDNN ConvTranspose1d (via `ConvBackwardData`) + TF32 matmul.
The candle patch lives in `~/repos/candle/` and is referenced via `path = ...` in Cargo.toml.
Rust wins at ≤30s. At 4 min Python's Cutlass tensor-core kernels (cuDNN v9 engine API) give it an edge on VAE.

## Architecture (v1.5)

```
text caption → Qwen3-Embedding-0.6B → text_projector(1024→2048) ──┐
lyrics → Qwen3 token embeddings → lyric_encoder(8 layers) ────────┤
reference audio → timbre_encoder(4 layers) ────────────────────────┘
                                                                   ↓
                                          pack_sequences (sort valid tokens first)
                                                                   ↓
                     DiT (24 layers, GQA, alternating sliding/full, AdaLN)
                       + 8-step turbo ODE (CFG-free flow matching)
                                                                   ↓
                     AutoencoderOobleck VAE decoder (latent → 48kHz stereo)
                                                                   ↓
                                                              WAV output
```

## Module Layout

```
src/
├── lib.rs                          — crate root
├── error.rs                        — Error enum
├── config.rs                       — AceStepConfig, VaeConfig, TurboSchedule
├── audio.rs                        → audio/
│   └── wav.rs                      — WAV read/write at 48kHz stereo
├── model.rs                        → model/
│   ├── transformer.rs              → transformer/
│   │   ├── attention.rs            — RotaryEmbedding, RmsNorm, AceStepAttention (GQA), SiluMlp
│   │   ├── layers.rs               — AceStepEncoderLayer, AceStepDiTLayer (AdaLN)
│   │   ├── timestep.rs             — TimestepEmbedding (sinusoidal + MLP)
│   │   ├── mask.rs                 — create_4d_mask (full + sliding window)
│   │   └── dit.rs                  — AceStepDiTModel (patchify → transformer → unpatchify)
│   ├── encoder.rs                  → encoder/
│   │   ├── lyric.rs                — AceStepLyricEncoder (8-layer transformer)
│   │   ├── timbre.rs               — AceStepTimbreEncoder (4-layer, CLS extract + unpack)
│   │   └── condition.rs            — AceStepConditionEncoder, pack_sequences
│   ├── tokenizer.rs                → tokenizer/
│   │   ├── fsq.rs                  — ResidualFsq (stub, cover mode only)
│   │   ├── pooler.rs               — AttentionPooler (25Hz→5Hz)
│   │   └── detokenizer.rs          — AudioTokenDetokenizer (5Hz→25Hz)
│   └── generation.rs               — AceStepConditionGenerationModel (ODE loop)
├── vae.rs                          — OobleckDecoder (Snake1d α+β, weight_norm)
└── pipeline.rs                     — end-to-end inference (stub)
```

Module roots (e.g., `src/audio.rs`) contain `mod` declarations and re-exports. Never create `mod.rs` files.

## Code Conventions

- Rust edition 2024
- Use `candle_core::Result` internally
- Every module has `#[cfg(test)] mod tests` in the same file
- Tests use `--no-default-features` (CPU only) so they run anywhere
- Weight key paths must match the original Python checkpoint exactly
- Tensor shapes documented in comments as `[B, T, D]` notation
- Reference the Python source in `~/repos/ACE-Step-1.5/` for v1.5 architecture

## Key Constants (v1.5)

- Sample rate: 48000 Hz
- Hidden dim: 2048 (16 heads, 8 KV heads — GQA)
- Head dim: 128
- Latent: continuous 64-dim acoustic features at 25Hz
- DiT: 24 layers, alternating sliding window (128) + full attention
- Patch size: 2 (Conv1d patchifying)
- VAE: AutoencoderOobleck, hop=2048, strides [2,4,4,8,8]
- Turbo: 8 steps CFG-free, pre-defined timestep schedules
- Max duration: 10 min

## Spacebot Integration (generation_daemon)

`examples/generation_daemon.rs` is the primary integration point with spacebot.

**What it does:** Keeps the pipeline resident in VRAM across requests. Each client connection sends one JSON request line over a Unix socket and receives one JSON response line.

**Socket:** `/tmp/ace-step-gen.sock` (default). Override with `--socket`.

**Protocol:**
- Request: `{"caption":"...", "output":"/tmp/out.ogg", "duration_s":30, ...}` + newline
- Response success: `{"ok":true, "path":"...", "duration_s":30, "sample_rate":48000, "channels":2}` + newline
- Response error: `{"ok":false, "error":"..."}` + newline

**Build:**
```bash
LIBRARY_PATH=/usr/lib64:$LIBRARY_PATH PATH="/usr/local/cuda-12.4/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.4 NVCC_CCBIN=/usr/bin/g++-13 CPLUS_INCLUDE_PATH="/tmp/cuda-shim" \
cargo build --release --example generation_daemon --features audio-ogg
```

**Binary location:** `target/release/examples/generation_daemon`

**Systemd unit:** `~/.config/systemd/user/ace-step-gen.service` (enabled, auto-starts)

**Skill:** `~/.spacebot/skills/generate_music/SKILL.md` — instructs the worker to talk to the daemon socket via `socat`, with CLI binary fallback.

**Design rationale:** `stream_daemon` (also in this repo) is for live/continuous playback to speakers. `generation_daemon` is for one-shot file generation: user asks for a track, gets a file. The `GenerationManager` (`src/manager.rs`) handles the resident pipeline with OOM retry; `generation_daemon` just wraps it with a socket interface.

## Reference Repositories

- `~/repos/ACE-Step-1.5/` — Python v1.5 implementation (ground truth)
- `~/repos/candle/` — candle ML framework (reference for API patterns)
- `~/repos/ACE-Step/` — Python v1 implementation (old, archived)

## Common Pitfalls

- `use candle_core::IndexOp;` is required in any file using `.i()` tensor indexing
- `(tensor_expr)?` doesn't compile — `Tensor` doesn't implement `Try`. Break into: `let result = expr; result?`
- `.contiguous()` required after `.transpose()` before `matmul`, `gather`, `index_select`, `Conv1d`, `ConvTranspose1d`
- `.expand()` returns non-contiguous view — call `.contiguous()` before gather/select operations
- Snake1d in Oobleck has BOTH alpha AND beta (unlike DAC which only has alpha)
- `arg_sort_last_dim` returns I64 indices; comparisons require matching dtypes
- `VarBuilder::zeros(DType::F32, &dev)` for test weight creation
