# syntax=docker/dockerfile:1
# Recovered 2026-09-25 from nullableeth/whisperx-wyoming:latest image-config
# history after the original build context (built on the media server) was
# lost. Faithful to the shipped image; unpinned deps pinned to the versions
# that image actually contains.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    build-essential \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

RUN pip3 install --no-cache-dir whisperx==3.8.2
RUN pip3 install --no-cache-dir wyoming==1.8.0 librosa

# Upgrade the bundled pyannote VAD checkpoint once at build time so every
# container start doesn't relive the lightning migration warning.
RUN python3 << 'PYEOF'
import logging
logging.getLogger('lightning.pytorch.utilities.migration.utils').setLevel(logging.ERROR)
try:
    from lightning.pytorch.utilities.upgrade_checkpoint import upgrade_checkpoint
    upgrade_checkpoint('/usr/local/lib/python3.10/dist-packages/whisperx/assets/pytorch_model.bin')
    print("Lightning checkpoint upgraded")
except Exception as e:
    print(f"Note: Could not upgrade checkpoint: {e}")
PYEOF

RUN python3 -c "import whisperx; print('WhisperX installed successfully')"

COPY src/whisperx_wyoming_wrapper.py /app/wrapper.py

ENV WYOMING_URI=tcp://0.0.0.0:10300 \
    WHISPER_MODEL=base \
    WHISPER_LANGUAGE=en \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float16

EXPOSE 10300
ENTRYPOINT ["python3", "/app/wrapper.py"]
