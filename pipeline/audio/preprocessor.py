"""Audio preprocessing and normalization utilities.

Standardizes input audio into 16kHz mono PCM format preferred by modern ASR
engines (Whisper, Vosk, Wav2Vec) and provides audio chunking mechanisms.
"""

from __future__ import annotations
import shutil
import subprocess
import wave
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from pipeline.models import AudioMetadata

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Preprocesses and normalizes audio for speech-to-text models."""

    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path("/tmp/audio_transcriber_cache")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_available = shutil.which("ffmpeg") is not None

    def needs_normalization(self, metadata: AudioMetadata) -> bool:
        """Determines if the audio format diverges from optimal 16kHz mono PCM."""
        if metadata.format != "wav":
            return True
        if metadata.channels != self.TARGET_CHANNELS:
            return True
        if metadata.sample_rate != self.TARGET_SAMPLE_RATE:
            return True
        return False

    def normalize(self, input_path: Path, metadata: AudioMetadata) -> Path:
        """Normalizes audio to 16kHz, mono WAV format using ffmpeg if available.

        If ffmpeg is absent, returns the original path and warns.
        """
        if not self.needs_normalization(metadata):
            return input_path

        if not self.ffmpeg_available:
            logger.warning(
                "ffmpeg is not installed. Using raw audio directly without 16kHz mono resampling."
            )
            return input_path

        output_path = self.temp_dir / f"normalized_{input_path.stem}.wav"

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-i", str(input_path),
            "-ac", str(self.TARGET_CHANNELS),
            "-ar", str(self.TARGET_SAMPLE_RATE),
            "-sample_fmt", "s16",
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=60,
            )
            return output_path
        except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Failed to normalize audio via ffmpeg: {e}. Falling back to original.")
            return input_path

    def calculate_chunks(
        self, duration_seconds: float, chunk_length_seconds: float = 30.0, overlap_seconds: float = 2.0
    ) -> List[Tuple[float, float]]:
        """Calculates time intervals for chunking long audio files to avoid OOM."""
        chunks = []
        start = 0.0
        while start < duration_seconds:
            end = min(start + chunk_length_seconds, duration_seconds)
            chunks.append((round(start, 2), round(end, 2)))
            if end >= duration_seconds:
                break
            start += chunk_length_seconds - overlap_seconds
        return chunks

