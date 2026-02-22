# ace-step-rs

Pure Rust implementation of [ACE-Step v1.5](https://github.com/ACE-Step/ACE-Step) music generation using the [candle](https://github.com/huggingface/candle) ML framework. Loads original safetensors weights directly from HuggingFace — no ONNX conversion, no Python runtime.

Generates up to 10 minutes of stereo 48kHz audio from text captions and lyrics.

## Binaries

### `ace-step` — CLI

One-shot generation from the command line. Prints a JSON summary to stdout on success.

```bash
ace-step \
  --caption "upbeat jazz with piano and drums, bpm: 120, key: C major" \
  --lyrics "[verse]\nWalking down the street on a sunny day" \
  --duration 30 \
  --output output.ogg
```

### `generation-daemon` — Unix socket daemon

Keeps the pipeline resident in VRAM across requests. Each client sends one JSON request line and receives one JSON response line.

Socket: `/tmp/ace-step-gen.sock` (override with `--socket`).

```sh
echo '{"caption":"ambient piano","duration_s":20,"output":"/tmp/piano.ogg"}' \
  | socat - UNIX-CONNECT:/tmp/ace-step-gen.sock
```

Uses the `GenerationManager` internally — monitors VRAM, proactively offloads to CPU on low memory, retries on CUDA OOM. Exits on unrecoverable failure so systemd can restart.

Requires the `audio-ogg` feature.

## Library

```rust
use ace_step_rs::pipeline::{AceStepPipeline, GenerationParams};

fn main() -> ace_step_rs::Result<()> {
    let device = candle_core::Device::cuda_if_available(0)?;
    let mut pipeline = AceStepPipeline::load(&device, candle_core::DType::F32)?;

    let params = GenerationParams {
        caption: "upbeat jazz with piano and drums".to_string(),
        lyrics: "[verse]\nWalking down the street on a sunny day\n".to_string(),
        duration_s: 30.0,
        ..Default::default()
    };

    let audio = pipeline.generate(&params)?;
    ace_step_rs::audio::write_audio("output.wav", &audio.samples, audio.sample_rate, audio.channels)?;

    Ok(())
}
```

Model weights (~6GB) are downloaded automatically from [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) on first run and cached in `~/.cache/huggingface/`.

### Key modules

| Module | Description |
|--------|-------------|
| `pipeline` | End-to-end inference: text encoding → diffusion → VAE decode |
| `manager` | `GenerationManager` — keeps the pipeline resident, queues requests, VRAM monitoring + OOM retry |
| `radio` | `RadioStation` — whole-song generation with request queue, auto-duration from lyrics |
| `audio` | WAV/OGG/MP3 I/O |
| `vae` | AutoencoderOobleck decoder (latent → 48kHz stereo waveform) |

### Radio station

`RadioStation` manages a song request queue and generates complete tracks sequentially. Duration is auto-estimated from lyrics (8s per line, clamped to 100–600s). The `radio_daemon` example wires this to cpal audio output with gapless double-buffered playback, Unix socket control, and skip/queue/history commands.

## Architecture

```
text caption → Qwen3-Embedding-0.6B (full encoder) ──┐
                                                      ├→ packed condition sequence
lyrics → Qwen3-Embedding (embed only)                │
  → lyric encoder (8-layer transformer)              │
                                                      │
ref audio → timbre encoder (4-layer) ─────────────────┘
                                                      ↓
             DiT (24 layers, GQA, sliding window + full attn)
             flow matching, 8-step turbo ODE (CFG-free)
                                                      ↓
             AutoencoderOobleck VAE (latent → 48kHz stereo waveform)
```

~2B parameters total. Uses continuous 64-dim acoustic features at 25Hz, flow matching with an 8-step CFG-free turbo schedule.

## Building

### CPU only

```bash
cargo build --no-default-features
```

### CUDA (recommended for inference)

Requires CUDA toolkit 12.x and a compatible NVIDIA GPU.

```bash
cargo build --release --features cuda
```

For cuDNN-accelerated ConvTranspose1d (faster VAE decode):

```bash
cargo build --release --features cudnn
```

Requires a [candle fork](https://github.com/Marenz/candle/tree/fast-conv-transpose1d-no-cudnn) with the following upstream PRs:
- [public `Model::clear_kv_cache` for Qwen3](https://github.com/huggingface/candle/pull/3381) — needed to reset KV state between inference calls
- [cuDNN ConvTranspose1d](https://github.com/huggingface/candle/pull/3383) (optional) — 100x faster VAE decode vs the default CPU fallback kernel

Depending on your system, you may need additional environment variables for the CUDA build — see [AGENTS.md](AGENTS.md) for platform-specific notes.

### Metal (macOS)

```bash
cargo build --release --features metal
```

Note: Metal support is provided by candle but has not been tested with this project.

### Building the daemon

```bash
cargo build --release --bin generation-daemon --features cuda,audio-ogg
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `cuda` | yes | NVIDIA GPU acceleration via CUDA |
| `cudnn` | no | cuDNN-accelerated ConvTranspose1d (implies `cuda`) |
| `metal` | no | Apple GPU acceleration via Metal |
| `cli` | no | Audio playback + terminal input (cpal, rodio, crossterm) |
| `audio-ogg` | no | OGG/Vorbis encoding (required by `generation-daemon`) |
| `audio-mp3` | no | MP3 encoding |
| `audio-all` | no | All audio encoders |

## `GenerationParams`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `caption` | `String` | `""` | Style/genre description (e.g. "lo-fi hip hop, mellow piano") |
| `metas` | `String` | `""` | Metadata: `bpm`, `key`, `genre`, `instruments` |
| `lyrics` | `String` | `""` | Lyrics with section tags like `[verse]`, `[chorus]` |
| `language` | `String` | `"en"` | Lyric language code |
| `duration_s` | `f64` | `30.0` | Output duration in seconds (max 600) |
| `shift` | `f64` | `3.0` | Turbo schedule shift (1, 2, or 3) |
| `seed` | `Option<u64>` | `None` | Random seed for reproducibility |
| `src_latents` | `Option<Tensor>` | `None` | Source latents for repaint/inpainting |
| `chunk_masks` | `Option<Tensor>` | `None` | Mask for repaint (0 = keep, 1 = generate) |
| `refer_audio` | `Option<Tensor>` | `None` | Reference audio latents for timbre conditioning |

## Performance

Benchmarked on RTX 3090 (24GB), F32 + TF32 tensor cores. Uses the [cuDNN ConvTranspose1d patch](https://github.com/huggingface/candle/pull/3383).

| Duration | Python (PyTorch) | Rust (candle) | Ratio |
|----------|-----------------|---------------|-------|
| 10s | 0.88s | 0.59s | **1.5x faster** |
| 30s | 1.38s | 1.25s | **1.1x faster** |
| 4 min | 9.26s | 12.04s | 1.3x slower |

<details>
<summary>Per-stage breakdown (30s)</summary>

| Stage | Python | Rust |
|-------|--------|------|
| Text encoding | 0.14s | 0.02s |
| Diffusion (8 ODE steps) | 0.73s | 0.63s |
| VAE decode | 0.39s | 0.54s |

</details>

Rust wins at short/medium durations. At longer durations PyTorch's Cutlass tensor-core VAE kernels (cuDNN v9 engine API) give it an edge. Without the [candle patch](https://github.com/huggingface/candle/pull/3383), VAE decode is ~3s (100x slower ConvTranspose1d).

## Running Tests

Tests run on CPU only and don't require a GPU or downloaded weights:

```bash
cargo test --no-default-features
```

## License

MIT
