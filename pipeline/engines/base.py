"""Abstract Base Class for Speech-to-Text engines.

Enables the Strategy Pattern so that different transcription backends (Local Whisper,
Cloud APIs, Vosk, or Mock engines) can be interchanged without altering pipeline logic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

from pipeline.models import TranscriptionOptions, TranscriptSegment


class TranscriptionEngine(ABC):
    """Abstract interface that all STT engine implementations must fulfill."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Name of the engine."""
        pass

    @abstractmethod
    def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> Tuple[str, List[TranscriptSegment]]:
        """Transcribes an audio file.

        Args:
            audio_path: Absolute or relative path to the verified audio file.
            options: Transcription configuration (language, model size, etc.).

        Returns:
            A tuple of (raw_text, list_of_segments).
        """
        pass

