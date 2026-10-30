"""Factory pattern for instantiating speech-to-text engines."""

from __future__ import annotations
from pipeline.engines.base import TranscriptionEngine
from pipeline.engines.mock_engine import MockSTTEngine
from pipeline.engines.whisper_local import LocalWhisperEngine
from pipeline.engines.whisper_api import WhisperAPIEngine


def get_engine(engine_name: str = "mock", model_size: str = "base") -> TranscriptionEngine:
    """Returns the requested TranscriptionEngine implementation."""
    engine_name = engine_name.lower().strip()
    if engine_name in ["mock", "test"]:
        return MockSTTEngine()
    elif engine_name in ["whisper", "whisper_local", "local"]:
        return LocalWhisperEngine(model_size=model_size)
    elif engine_name in ["whisper_api", "openai", "cloud"]:
        return WhisperAPIEngine()
    else:
        raise ValueError(
            f"Unknown engine '{engine_name}'. Available options: 'mock', 'whisper_local', 'whisper_api'."
        )

