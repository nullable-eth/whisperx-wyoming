#!/usr/bin/env python3
import argparse
import asyncio
import logging
import time
import warnings
import os

# Suppress all warnings before imports
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import whisperx
import numpy as np
from functools import partial

from wyoming.info import Attribution, Info, AsrProgram, AsrModel, Describe
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

_LOGGER = logging.getLogger(__name__)

# Disable PyTorch warnings
torch.set_warn_always(False)

# Suppress specific loggers
logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
logging.getLogger('pyannote').setLevel(logging.ERROR)
logging.getLogger('whisperx').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class WhisperXEventHandler(AsyncEventHandler):
    """Event handler for WhisperX Wyoming protocol."""
    
    def __init__(
        self,
        wyoming_info: Info,
        model,
        model_name: str,
        language: str,
        device: str,
        compute_type: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.audio_buffer = bytes()
        self.sample_rate = 16000
        self.sample_width = 2
        self.channels = 1

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True
        
        if Transcribe.is_type(event.type):
            return True
            
        if AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            self.sample_rate = audio_start.rate
            self.sample_width = audio_start.width
            self.channels = audio_start.channels
            self.audio_buffer = bytes()
            return True
            
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.audio_buffer += chunk.audio
            return True
            
        if AudioStop.is_type(event.type):
            if not self.audio_buffer or len(self.audio_buffer) < 100:
                _LOGGER.warning(f"Empty audio buffer")
                await self.write_event(Transcript(text="").event())
                return True
            
            start_time = time.time()
            
            # Convert bytes to numpy array
            if self.sample_width == 2:
                audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                _LOGGER.error(f"Unsupported sample width: {self.sample_width}")
                await self.write_event(Transcript(text="").event())
                return True
            
            # Resample if needed
            if self.sample_rate != 16000:
                import librosa
                audio_np = librosa.resample(audio_np, orig_sr=self.sample_rate, target_sr=16000)
            
            duration = len(audio_np) / 16000
            _LOGGER.info(f"Transcribing {duration:.2f}s of audio...")
            
            try:
                # Transcribe with WhisperX
                result = self.model.transcribe(
                    audio_np, 
                    batch_size=16,
                    language=self.language,
                )
                
                # Extract text from result
                if isinstance(result, dict):
                    text = result.get("text", "").strip()
                    segments = result.get("segments", [])
                    if not text and segments:
                        text = " ".join([seg.get("text", "") for seg in segments]).strip()
                else:
                    text = str(result).strip()
                
                elapsed = time.time() - start_time
                
                _LOGGER.info(f"✓ Transcribed in {elapsed:.2f}s: '{text}'")
                
                await self.write_event(Transcript(text=text).event())
                
            except Exception as e:
                _LOGGER.error(f"Transcription error: {e}")
                await self.write_event(Transcript(text="").event())
            
            self.audio_buffer = bytes()
            return True
        
        return True

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', help='unix:// or tcp://')
    parser.add_argument('--model', help='WhisperX model (tiny, base, small, medium, large-v2)')
    parser.add_argument('--language', help='Language code')
    parser.add_argument('--device', help='Device (cuda or cpu)')
    parser.add_argument('--compute-type', help='Compute type (float16, int8)')
    args = parser.parse_args()

    # Environment variables take precedence, then command-line args, then defaults
    uri = os.environ.get('WYOMING_URI') or args.uri or 'tcp://0.0.0.0:10300'
    model = os.environ.get('WHISPER_MODEL') or args.model or 'base'
    language = os.environ.get('WHISPER_LANGUAGE') or args.language or 'en'
    device = os.environ.get('WHISPER_DEVICE') or args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    compute_type = os.environ.get('WHISPER_COMPUTE_TYPE') or args.compute_type or ('float16' if device == 'cuda' else 'int8')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if device == 'cuda' and not torch.cuda.is_available():
        _LOGGER.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'
        compute_type = 'int8'
    
    _LOGGER.info(f"Loading WhisperX {model} on {device}...")
    
    try:
        model_obj = whisperx.load_model(
            model,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        _LOGGER.info(f"✓ Model loaded")
    except Exception as e:
        _LOGGER.error(f"Failed to load model: {e}")
        raise
    
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="whisperx",
                description=f"WhisperX ASR ({model})",
                attribution=Attribution(
                    name="WhisperX",
                    url="https://github.com/m-bain/whisperx"
                ),
                installed=True,
                version=model,
                models=[
                    AsrModel(
                        name=model,
                        description=f"WhisperX {model}",
                        attribution=Attribution(
                            name="WhisperX",
                            url="https://github.com/m-bain/whisperx"
                        ),
                        installed=True,
                        languages=[language],
                        version="1.0",
                    )
                ],
            )
        ],
    )
    
    _LOGGER.info(f"Starting server on {uri}")
    
    server = AsyncServer.from_uri(uri)
    
    await server.run(
        partial(
            WhisperXEventHandler,
            wyoming_info,
            model_obj,
            model,
            language,
            device,
            compute_type,
        )
    )

if __name__ == "__main__":
    asyncio.run(main())
