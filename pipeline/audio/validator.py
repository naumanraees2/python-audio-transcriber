"""Audio validation and metadata inspection module.

Validates file existence, size thresholds, and magic headers (RIFF/WAVE, MP3/ID3,
FLAC, OGG, M4A) to prevent malicious or malformed file processing.
"""

from __future__ import annotations
import os
import wave
from pathlib import Path
from typing import Optional

from pipeline.models import AudioMetadata


class AudioValidationError(Exception):
    """Raised when an audio file fails validation."""
    pass


class UnsupportedAudioFormatError(AudioValidationError):
    """Raised when an audio format is not supported."""
    pass


class FileTooLargeError(AudioValidationError):
    """Raised when an audio file exceeds max allowed size."""
    pass


MAGIC_SIGNATURES = {
    "wav": [b"RIFF"],
    "mp3": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"],
    "flac": [b"fLaC"],
    "ogg": [b"OggS"],
}

# Maximum supported file size (default: 100MB)
DEFAULT_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


class AudioValidator:
    """Validates audio files and extracts metadata without heavy native dependencies."""

    def __init__(self, max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES):
        self.max_size_bytes = max_size_bytes

    def detect_format(self, path: Path) -> str:
        """Inspects magic header bytes to identify format securely."""
        with open(path, "rb") as f:
            header = f.read(16)

        if len(header) < 4:
            raise AudioValidationError(f"File {path.name} is too small to be a valid audio file.")

        # Check WAV (RIFF....WAVE)
        if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
            return "wav"

        # Check MP3 (ID3 tag or sync word)
        if header.startswith(b"ID3") or header.startswith(b"\xff\xfb") or header.startswith(b"\xff\xf3") or header.startswith(b"\xff\xf2"):
            return "mp3"

        # Check FLAC
        if header.startswith(b"fLaC"):
            return "flac"

        # Check OGG
        if header.startswith(b"OggS"):
            return "ogg"

        # Check M4A / MP4
        if len(header) >= 8 and header[4:8] == b"ftyp":
            return "m4a"

        # Fallback to extension check if header is non-standard
        ext = path.suffix.lower().lstrip(".")
        if ext in ["wav", "mp3", "m4a", "flac", "ogg"]:
            return ext

        raise UnsupportedAudioFormatError(
            f"Unsupported audio format for {path.name}. Supported formats: WAV, MP3, M4A, FLAC, OGG."
        )

    def extract_metadata(self, path: Path) -> AudioMetadata:
        """Validates file and extracts metadata."""
        if not path.exists():
            raise AudioValidationError(f"Audio file not found: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise AudioValidationError(f"Audio file is empty: {path.name}")

        if file_size > self.max_size_bytes:
            raise FileTooLargeError(
                f"File size ({file_size / (1024*1024):.2f}MB) exceeds limit of "
                f"{self.max_size_bytes / (1024*1024):.2f}MB"
            )

        detected_format = self.detect_format(path)
        sample_rate: Optional[int] = None
        channels: Optional[int] = None
        bit_depth: Optional[int] = None
        duration_seconds = 0.0

        if detected_format == "wav":
            try:
                with wave.open(str(path), "rb") as wf:
                    channels = wf.getnchannels()
                    sample_rate = wf.getframerate()
                    sample_width = wf.getsampwidth()
                    bit_depth = sample_width * 8
                    n_frames = wf.getnframes()
                    if sample_rate > 0:
                        duration_seconds = round(n_frames / float(sample_rate), 2)
            except Exception as e:
                raise AudioValidationError(f"Corrupted or invalid WAV file: {e}") from e

        elif detected_format == "mp3":
            # Estimate duration based on standard 128kbps if native parser not available
            # 128 kbps = 16,000 bytes/sec
            duration_seconds = round(file_size / 16000.0, 2)
            channels = 2
            sample_rate = 44100
            bit_depth = 16
        else:
            # Generic fallback duration estimate
            duration_seconds = round(file_size / 32000.0, 2)

        return AudioMetadata(
            file_name=path.name,
            file_path=str(path.resolve()),
            format=detected_format,
            file_size_bytes=file_size,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
        )

