---
name: generate_music
description: Generate original music from a text description and optional lyrics using ACE-Step v1.5 (local GPU, ~1–12s for 30s of audio).
---

# generate_music

Generate original music locally using ACE-Step v1.5, a flow-matching diffusion model. Runs on the local RTX 3090 GPU. No API keys needed.

**Daemon socket:** `/tmp/ace-step-gen.sock`

The daemon keeps the model weights resident in VRAM across requests (no 2 GB reload per call). If the socket is not available, fall back to running the CLI binary directly (see Fallback section below).

## Workflow

1. Gather the required inputs from the user's request (see Parameters below).
2. Choose an output path under `/tmp/` with a `.ogg` extension (smaller, good for Telegram/Discord).
3. Send a JSON request to the daemon socket and read the response.
4. Use `send_file` to deliver the audio file to the user.
5. Report the generation time if available.

## Unicode / non-ASCII lyrics

**CRITICAL:** Always preserve the original Unicode characters in lyrics and captions. Never transliterate umlauts or accented characters to ASCII equivalents. For example:
- Write `Österreich`, NOT `Oesterreich`
- Write `schönes Stück`, NOT `schoenes Stueck`
- Write `Gemütlichkeit`, NOT `Gemuetlichkeit`
- Write `Straße`, NOT `Strasse`

The model was trained on real Unicode text and produces significantly better pronunciation when given proper characters. ASCII transliteration (ö→oe, ü→ue, ä→ae, ß→ss) will cause wrong pronunciation in the generated audio.

## Sending a request to the daemon

The daemon speaks line-delimited JSON over a Unix socket. Send one JSON line, read one JSON response line.

**Recommended method:** Write the JSON to a temp file first, then pipe it to socat. This avoids shell quoting issues with Unicode, newlines, and special characters in lyrics:

```sh
# 1. Write JSON request to a temp file using the file tool
# 2. Then pipe it to the daemon:
cat /tmp/music_request.json | socat -t 120 - UNIX-CONNECT:/tmp/ace-step-gen.sock
```

### Alternative: inline shell command (short ASCII-only requests)

For simple requests without lyrics or with ASCII-only text:

```sh
echo '{"caption":"upbeat jazz, 120 BPM","duration_s":30,"output":"/tmp/music_1234.ogg"}' \
  | socat -t 120 - UNIX-CONNECT:/tmp/ace-step-gen.sock
```

**Do NOT use inline echo for requests with non-ASCII lyrics** — use the file-based method above instead.

### Request JSON fields

| Field | Type | Default | Description |
|---|---|---|---|
| `caption` | string | **required** | Style description: genre, mood, tempo, instruments |
| `output` | string | auto `/tmp/ace-step-<ms>.ogg` | Output file path (.wav or .ogg) |
| `lyrics` | string | `""` | Lyrics with `[verse]`/`[chorus]`/`[bridge]` tags; `""` = instrumental |
| `metas` | string | `""` | Metadata: `"bpm: 128, key: A minor, genre: electronic"` |
| `language` | string | `"en"` | Lyrics language code (`"zh"` for Chinese) |
| `duration_s` | float | LM suggestion or 30.0 | Duration in seconds (1–600). If omitted and LM is running, the LM may suggest a duration based on the caption. |
| `shift` | float | `3.0` | ODE schedule shift (1–3); lower = more faithful, less variation |
| `seed` | int\|null | `null` | Fixed seed for reproducibility; `null` = random |

### Response JSON fields (success)

```json
{"ok": true, "path": "/tmp/music.ogg", "duration_s": 30.0, "sample_rate": 48000, "channels": 2}
```

### Response JSON fields (error)

```json
{"ok": false, "error": "generation failed: ..."}
```

## Commands

The daemon also accepts command messages to manage the pipeline.

### Unload (free VRAM)

Drops the pipeline from VRAM. The next generation request will reload it automatically (~10–20s reload time).

```sh
echo '{"command":"unload"}' | socat - UNIX-CONNECT:/tmp/ace-step-gen.sock
```

Response: `{"ok":true,"message":"pipeline unloaded"}`

Use this when VRAM is needed for other tasks (e.g. other GPU workloads). No need to restart the daemon — it stays running and reloads on demand.

## Caption writing guide

Be specific — genre, mood, tempo, instruments, vibe:

- `"upbeat electronic dance music, 128 BPM, four-on-the-floor kick, synth arpeggios, euphoric build"`
- `"melancholic lo-fi hip-hop, slow 70 BPM, dusty vinyl samples, soft piano, rain atmosphere"`
- `"fast punk rock, distorted guitars, driving drums, raw energy, 180 BPM"`
- `"cinematic orchestral trailer music, epic brass, driving strings, 140 BPM, intense build to climax"`

## Full example

Step 1 — Use the **file tool** to write the JSON request:

```json
{
  "caption": "indie pop with dreamy synths, gentle vocals, 100 BPM, wistful and nostalgic",
  "lyrics": "[verse]\nNeon lights on rainy streets\nWhere the city never sleeps\n[chorus]\nWe were infinite, we were free\nJust the stars and you and me",
  "metas": "bpm: 100, key: G major, genre: indie pop, instruments: synth, guitar, drums",
  "duration_s": 45,
  "output": "/tmp/music_indie.ogg"
}
```

Save this to `/tmp/music_request.json`.

Step 2 — Use the **shell tool** to send it to the daemon:

```sh
cat /tmp/music_request.json | socat -t 120 - UNIX-CONNECT:/tmp/ace-step-gen.sock
```

The response will be a JSON line like `{"ok":true,"path":"/tmp/music_indie.ogg",...}`.

## After generation

Always use `send_file` to deliver the audio to the user:

```
send_file(file_path="/tmp/music.ogg", caption="Here's your 30s cinematic trailer music!")
```

## Fallback: CLI binary (if daemon is not running)

If `socat` returns an error connecting to the socket, the daemon is not running. Fall back to the CLI binary:

**Binary:** `/home/marenz/Projects/ace-step-rs-no-cudnn/target/release/ace-step`

```sh
/home/marenz/Projects/ace-step-rs-no-cudnn/target/release/ace-step \
  --caption "cinematic orchestral trailer music, epic brass, 140 BPM" \
  --duration 30 \
  --output /tmp/music.ogg
```

The binary reloads 2 GB of weights each run (~10–20s extra on first call after a cold start). Output format is the same JSON line to stdout on success.

## Troubleshooting

- **Connection refused / no such file** — daemon not running. Use fallback CLI binary.
- **`ok: false`** — check the `error` field. Common causes: invalid caption, CUDA OOM, bad output path.
- **OGG not supported** — use `.wav` extension in the output path instead.
- **Long generation time** — reduce `duration_s`. 30s ≈ 1–2s on RTX 3090.
