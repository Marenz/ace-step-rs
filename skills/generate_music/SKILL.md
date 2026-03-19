---
name: generate_music
description: Generate original music from a text description and optional lyrics using ACE-Step v1.5 (local GPU, ~1–12s for 30s of audio).
---

# generate_music

**DO THIS IMMEDIATELY — no exploration, no checking, no setup steps:**

```sh
ace-step-client \
  --caption "YOUR STYLE HERE" \
  --lyrics "YOUR LYRICS HERE" \
  --duration 30 \
  --output /path/to/workspace/music_output.mp3
```

The client auto-starts the daemon if it's not running (first invocation takes ~20s to load the model into VRAM, subsequent calls are fast). Audio data is streamed back over the socket and written locally by the client, so the output file is always accessible regardless of sandbox. It prints the output path on success. Then use `send_file` to deliver the audio file to the user.

**IMPORTANT:** The `--output` path MUST be inside your workspace directory so `send_file` can access it. Use your workspace path, not `/tmp/`.

**IMPORTANT:** Generation can take 30-180 seconds depending on duration and GPU load. Set `timeout_seconds` to at least 300 when calling the shell tool.

If the client exits non-zero, check stderr for the error message.

---

Generate original music locally using ACE-Step v1.5, a flow-matching diffusion model. Runs on the local RTX 3090 GPU. No API keys needed.

**Client binary:** `ace-step-client` (on PATH via tools/bin)
**Output format:** Use `.mp3` — the generator outputs MP3 directly, no post-conversion needed.

## Unicode / non-ASCII lyrics

**CRITICAL:** Always preserve the original Unicode characters in lyrics and captions. Never transliterate umlauts or accented characters to ASCII equivalents. For example:
- Write `Österreich`, NOT `Oesterreich`
- Write `schönes Stück`, NOT `schoenes Stueck`
- Write `Gemütlichkeit`, NOT `Gemuetlichkeit`
- Write `Straße`, NOT `Strasse`

The model was trained on real Unicode text and produces significantly better pronunciation when given proper characters.

## Client usage

```sh
ace-step-client [OPTIONS]

Options:
  --caption <TEXT>      Style description: genre, mood, tempo, instruments [required]
  --output <PATH>       Output file (.mp3, .ogg, .wav) [default: auto]
  --duration <SECS>     Duration in seconds [default: 30]
  --lyrics <TEXT>       Lyrics with [verse]/[chorus]/[bridge] tags; omit for instrumental
  --metas <TEXT>        Metadata e.g. "bpm: 120, key: C major"
  --language <CODE>     Lyrics language code [default: en]
  --shift <FLOAT>       ODE shift 1–3 [default: 3.0]
  --seed <INT>          Fixed seed for reproducibility
  --socket <PATH>       Socket path [default: ~/.spacebot/sockets/ace-step-gen.sock]
  --timeout-secs <INT>  Wait timeout [default: 300]
  --unload              Unload pipeline from VRAM instead of generating
  --no-autostart        Don't try to auto-start the daemon
```

## Caption writing guide

Be specific — genre, mood, tempo, instruments, vibe:

- `"upbeat electronic dance music, 128 BPM, four-on-the-floor kick, synth arpeggios, euphoric build"`
- `"melancholic lo-fi hip-hop, slow 70 BPM, dusty vinyl samples, soft piano, rain atmosphere"`
- `"fast punk rock, distorted guitars, driving drums, raw energy, 180 BPM"`
- `"cinematic orchestral trailer music, epic brass, driving strings, 140 BPM, intense build to climax"`

## Full example

```sh
ace-step-client \
  --caption "indie pop with dreamy synths, gentle vocals, 100 BPM, wistful and nostalgic" \
  --lyrics "[verse]\nNeon lights on rainy streets\nWhere the city never sleeps\n[chorus]\nWe were infinite, we were free\nJust the stars and you and me" \
  --metas "bpm: 100, key: G major" \
  --duration 45 \
  --output {workspace}/music_output.mp3
```

On success, prints the output path to stdout. Then:

```
send_file(file_path="{workspace}/music_output.mp3", caption="Here's your 45s indie pop track!")
```

## Free VRAM when not generating

```sh
ace-step-client --unload
```

The next generation request will reload automatically.

## Troubleshooting

- **"auto-start unavailable"** — the daemon isn't running and the client can't start it from inside the sandbox. Report this to the user and suggest they run `systemctl --user start ace-step-gen.service` on the host. Do NOT try to install or download anything yourself.
- **Timeout** — generation is taking too long. Use a shorter `--duration`.
- **Non-zero exit** — check stderr for the error message.
- **30s ≈ 1–2s on RTX 3090** — generation is fast, timeout errors are unlikely unless the daemon is overloaded.
