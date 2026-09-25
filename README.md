# WhisperX Wyoming - High-Accuracy Speech-to-Text

Wyoming protocol server for WhisperX with superior accuracy and speed. Integrates seamlessly with Home Assistant for voice assistants.

## What This Is

Speech-to-text server using WhisperX with word-level timestamps, voice activity detection, and improved accuracy over standard Whisper implementations. Up to **3-4x faster** than faster-whisper after warm-up.

## Features

- ✅ **Superior accuracy** - 9.99% WER vs 11.98% for faster-whisper
- 🚀 **Fast inference** - 0.4-0.6s transcription after warm-up
- 🎯 **Word-level timestamps** - Precise alignment
- 🔊 **VAD built-in** - Voice activity detection filters noise
- 🌍 **Multi-language** - Supports all Whisper languages
- 🏠 **Home Assistant ready** - Wyoming protocol integration

## Quick Start

**GPU mode (recommended):**
```bash
docker run -d \
  --name whisperx \
  --gpus all \
  -p 10300:10300 \
  --restart unless-stopped \
  nullableeth/whisperx-wyoming:latest
```

**CPU mode:**
```bash
docker run -d \
  --name whisperx \
  -p 10300:10300 \
  -e WHISPER_DEVICE=cpu \
  -e WHISPER_COMPUTE_TYPE=int8 \
  --restart unless-stopped \
  nullableeth/whisperx-wyoming:latest
```

## Configuration

### Environment Variables (Recommended)

Configure via environment variables - no need to override the command:

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `WYOMING_URI` | `tcp://0.0.0.0:10300` | `tcp://host:port` | Wyoming server bind address |
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` | Model size (accuracy vs speed) |
| `WHISPER_LANGUAGE` | `en` | [See languages](#supported-languages) | Primary language code |
| `WHISPER_DEVICE` | `cuda` | `cuda`, `cpu` | Device to run inference on |
| `WHISPER_COMPUTE_TYPE` | `float16` | `float16`, `int8` | Precision (GPU: float16, CPU: int8) |

### Docker Compose (Recommended)
```yaml
services:
  whisperx:
    image: nullableeth/whisperx-wyoming:latest
    container_name: whisperx
    restart: unless-stopped
    ports:
      - "10300:10300"
    environment:
      WHISPER_MODEL: "base"          # tiny, base, small, medium, large-v2
      WHISPER_LANGUAGE: "en"         # Language code
      WHISPER_DEVICE: "cuda"         # cuda or cpu
      WHISPER_COMPUTE_TYPE: "float16" # float16 or int8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Command-Line Arguments (Alternative)

You can also use command-line arguments (env vars take precedence if both are set):
```bash
docker run -d \
  --name whisperx \
  --gpus all \
  -p 10300:10300 \
  nullableeth/whisperx-wyoming:latest \
  --model medium \
  --language en \
  --device cuda \
  --compute-type float16
```

## Home Assistant Integration

1. **Settings → Devices & Services → Add Integration**
2. Search **"Wyoming Protocol"**
3. Host: `<docker-host-ip>`, Port: `10300`
4. Configure voice assistant to use WhisperX for speech-to-text

## Available Models

Choose based on your accuracy vs speed needs:

| Model | Size | VRAM | Accuracy | Speed | Best For |
|-------|------|------|----------|-------|----------|
| `tiny` | 39M | ~1GB | ⭐⭐ | ⚡⚡⚡⚡⚡ | Testing, low-resource |
| `base` | 74M | ~1GB | ⭐⭐⭐ | ⚡⚡⚡⚡ | **Default, balanced** |
| `small` | 244M | ~2GB | ⭐⭐⭐⭐ | ⚡⚡⚡ | Better accuracy |
| `medium` | 769M | ~5GB | ⭐⭐⭐⭐⭐ | ⚡⚡ | High accuracy |
| `large-v2` | 1550M | ~10GB | ⭐⭐⭐⭐⭐⭐ | ⚡ | Best accuracy |
| `large-v3` | 1550M | ~10GB | ⭐⭐⭐⭐⭐⭐ | ⚡ | Latest, best overall |

**Recommendation**: Start with `base`, upgrade to `small` or `medium` if accuracy isn't sufficient.

## Supported Languages

Full language support via ISO 639-1 codes:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `en` | English | `es` | Spanish | `fr` | French |
| `de` | German | `it` | Italian | `pt` | Portuguese |
| `nl` | Dutch | `pl` | Polish | `tr` | Turkish |
| `ru` | Russian | `cs` | Czech | `ar` | Arabic |
| `zh` | Chinese | `ja` | Japanese | `ko` | Korean |
| `hi` | Hindi | `hu` | Hungarian | `sv` | Swedish |
| `da` | Danish | `no` | Norwegian | `fi` | Finnish |

**50+ languages total**. Full list: https://github.com/openai/whisper#available-models-and-languages

Set via `WHISPER_LANGUAGE` environment variable or `--language` flag.

## Compute Types

| Type | Description | Use Case |
|------|-------------|----------|
| `float16` | Half-precision floating point | **GPU inference (default)** |
| `int8` | 8-bit integer quantization | **CPU inference, lower VRAM** |
| `float32` | Full precision | Not recommended (slower, more VRAM) |

**Recommendation**: Use `float16` for GPU, `int8` for CPU.

## Performance

**GPU Mode (RTX 4070 + base model):**
- First transcription: ~2.5s (includes VAD model loading)
- Subsequent: ~0.4-0.6s per utterance
- VRAM usage: ~2-3GB

**GPU Mode (RTX 4070 + medium model):**
- First transcription: ~3s
- Subsequent: ~0.8-1.2s per utterance
- VRAM usage: ~5GB

**CPU Mode (Intel i7):**
- First transcription: ~8-12s
- Subsequent: ~3-5s per utterance
- RAM usage: ~4GB

**Comparison to faster-whisper:**
- ✅ **3-4x faster** on subsequent transcriptions
- ✅ **Better accuracy** - correctly handles technical terms, proper nouns
- ⚠️ **2s slower** on first transcription (VAD initialization)

## Example Configurations

### Maximum Accuracy (Large GPU)
```yaml
environment:
  WHISPER_MODEL: "large-v3"
  WHISPER_LANGUAGE: "en"
  WHISPER_DEVICE: "cuda"
  WHISPER_COMPUTE_TYPE: "float16"
```
*Requires ~10GB VRAM*

### Balanced Performance (Recommended)
```yaml
environment:
  WHISPER_MODEL: "base"
  WHISPER_LANGUAGE: "en"
  WHISPER_DEVICE: "cuda"
  WHISPER_COMPUTE_TYPE: "float16"
```
*Requires ~2GB VRAM*

### CPU-Only Mode
```yaml
environment:
  WHISPER_MODEL: "base"
  WHISPER_LANGUAGE: "en"
  WHISPER_DEVICE: "cpu"
  WHISPER_COMPUTE_TYPE: "int8"
# Remove deploy.resources section
```
*No GPU required*

### Multi-Language
```yaml
environment:
  WHISPER_MODEL: "small"
  WHISPER_LANGUAGE: "es"  # Spanish
  WHISPER_DEVICE: "cuda"
  WHISPER_COMPUTE_TYPE: "float16"
```
*Auto-detects if language doesn't match*

## Requirements

**GPU (Recommended):**
- NVIDIA GPU with CUDA 12.1+ support
- Minimum 2GB VRAM (base model)
- 5GB+ VRAM for medium/large models
- Docker with NVIDIA Container Toolkit (`--gpus all`)

**CPU (Fallback):**
- 4GB+ RAM
- Multi-core CPU recommended
- 5-10x slower than GPU

## Logs

Clean, timestamped logs for easy debugging:
```
2026-03-20 12:00:00 - INFO - Loading WhisperX base on cuda...
2026-03-20 12:00:02 - INFO - ✓ Model loaded
2026-03-20 12:00:02 - INFO - Starting server on tcp://0.0.0.0:10300
2026-03-20 12:00:15 - INFO - Transcribing 3.98s of audio...
2026-03-20 12:00:15 - INFO - ✓ Transcribed in 0.44s: 'turn on the living room lights'
```

## Troubleshooting

### Slow first transcription
**Normal behavior!** VAD model loads on first use (~2s overhead). All subsequent transcriptions are fast.

### GPU not detected
```bash
# Verify NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# Check container has GPU access
docker exec whisperx nvidia-smi

# Verify CUDA in container
docker exec whisperx python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Poor accuracy
- **Try larger model**: Upgrade from `base` → `small` → `medium`
- **Check language**: Set `WHISPER_LANGUAGE` to match spoken language
- **Audio quality**: Ensure clean audio input (16kHz recommended)
- **Microphone levels**: Verify levels in Home Assistant

### Empty transcriptions
- **Audio too short**: Minimum ~0.5s needed
- **Microphone muted**: Check HA microphone settings
- **Wyoming config**: Verify integration points to correct port

### High VRAM usage
- **Use smaller model**: Switch from `medium` → `small` → `base`
- **Use int8**: Set `WHISPER_COMPUTE_TYPE=int8`
- **CPU mode**: Set `WHISPER_DEVICE=cpu` (slower but no VRAM)

### Container won't start
```bash
# Check logs
docker logs whisperx

# Common issues:
# 1. Port already in use → change port
# 2. No GPU access → add --gpus all
# 3. CUDA version mismatch → update nvidia-docker
```

## Comparison to Alternatives

**vs faster-whisper:**
- ✅ WhisperX: 3-4x faster after warm-up
- ✅ WhisperX: Better accuracy (9.99% vs 11.98% WER)
- ⚠️ faster-whisper: Faster cold start (~500ms less)

**vs OpenAI Whisper:**
- ✅ WhisperX: Much faster (batched processing)
- ✅ WhisperX: Word-level timestamps
- ✅ WhisperX: Built-in VAD

**vs Vosk:**
- ✅ WhisperX: Better accuracy
- ✅ Vosk: Lower resource usage (~500MB RAM)
- ✅ WhisperX: Better with accents/dialects

## Known Issues

- First transcription takes ~2s extra for VAD initialization (normal)
- Lightning checkpoint upgrade message on startup (harmless, can be ignored)
- Requires GPU for optimal performance (CPU works but 5-10x slower)

## Benchmark Results

From community testing on LibriSpeech dataset:

| Model | WER (%) | CER (%) | Latency (ms) |
|-------|---------|---------|--------------|
| WhisperX (fp16, B=16) | 9.99 | 3.6 | 1,170-2,660 |
| FasterWhisper (fp16) | 11.98 | 4.7 | 2,580 |
| OpenAI Whisper | 10.8 | 4.3 | 10,800 |

*Lower is better for WER/CER, latency varies by audio length*

## Building From Source
```bash
git clone <your-repo>
cd whisperx-wyoming
docker build -t whisperx-wyoming:latest .
```

## Links

- [WhisperX GitHub](https://github.com/m-bain/whisperx)
- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [Home Assistant Voice](https://www.home-assistant.io/voice_control/)
- [Benchmark Source](https://www.reddit.com/r/LocalLLaMA/comments/1brqwun/)

## License

- WhisperX: MIT License
- Whisper models: Apache 2.0
- Wyoming Protocol: MIT License

## Support

For issues specific to this container, please open an issue on GitHub.
For WhisperX questions, see the [WhisperX documentation](https://github.com/m-bain/whisperx).
---

## Provenance

The original build context for this image was lost (built directly on the
media server, pushed straight to Docker Hub). This repository was
reconstructed on 2026-09-25 from the running image:

- source: `nullableeth/whisperx-wyoming:latest` (image config `sha256:e49af2dd91dfd1431ecd15d10fe7ca7e0eca452a202c512b805b31c3bf74d0ff`)
- `src/` extracted verbatim from `/app/wrapper.py` in the running container
- Dockerfile rebuilt from the image-config layer history, with previously
  unpinned dependencies pinned to the versions the shipped image contains

CI publishes to `ghcr.io/nullable-eth/whisperx-wyoming` (latest + sha tags on main,
version tags on `v*`).
