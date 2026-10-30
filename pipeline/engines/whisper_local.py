"""Local Whisper STT engine wrapper.

Supports either 'faster-whisper' (optimized CTranslate2 backend) or standard 'openai-whisper'.
Provides lazy model loading and device selection (CUDA/CPU/MPS).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from pipeline.engines.base import TranscriptionEngine
from pipeline.models import TranscriptionOptions, TranscriptSegment, Word

logger = logging.getLogger(__name__)


class LocalWhisperEngine(TranscriptionEngine):
    """Transcription engine backed by local Whisper models."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._model = None
        self._backend = None

    @property
    def engine_name(self) -> str:
        return f"whisper_local_{self._backend or 'unloaded'}_{self.model_size}"

    def _load_model(self):
        """Lazy loader to prevent memory spikes until transcription is requested."""
        if self._model is not None:
            return

        # Try faster-whisper first (recommended for production: 4x speedup, lower VRAM)
        try:
            from faster_whisper import WhisperModel
            device = "cpu" if self.device == "auto" else self.device
            compute_type = "int8" if device == "cpu" else "float16"
            logger.info(f"Loading faster-whisper model '{self.model_size}' on {device} ({compute_type})...")
            self._model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
            self._backend = "faster_whisper"
            return
        except ImportError:
            pass

        # Fallback to official openai-whisper
        try:
            import whisper
            logger.info(f"Loading openai-whisper model '{self.model_size}'...")
            self._model = whisper.load_model(self.model_size)
            self._backend = "openai_whisper"
            return
        except ImportError:
            raise RuntimeError(
                "Neither 'faster-whisper' nor 'openai-whisper' is installed. "
                "Install with `pip install faster-whisper` or use the mock engine (`--engine mock`)."
            )

    def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> Tuple[str, List[TranscriptSegment]]:
        self._load_model()

        segments_out: List[TranscriptSegment] = []
        raw_text_parts: List[str] = []

        if self._backend == "faster_whisper":
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                language=options.language,
                word_timestamps=True,
            )
            for idx, seg in enumerate(segments_iter):
                words = []
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        words.append(
                            Word(
                                word=w.word,
                                start=round(w.start, 2),
                                end=round(w.end, 2),
                                confidence=round(w.probability, 2) if hasattr(w, "probability") else 1.0,
                            )
                        )
                segment = TranscriptSegment(
                    id=idx + 1,
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=seg.text.strip(),
                    confidence=round(getattr(seg, "avg_logprob", 0.0), 2),
                    words=words,
                )
                segments_out.append(segment)
                raw_text_parts.append(seg.text.strip())

        elif self._backend == "openai_whisper":
            result = self._model.transcribe(str(audio_path), language=options.language)
            for idx, seg in enumerate(result.get("segments", [])):
                segment = TranscriptSegment(
                    id=idx + 1,
                    start=round(seg.get("start", 0.0), 2),
                    end=round(seg.get("end", 0.0), 2),
                    text=seg.get("text", "").strip(),
                    confidence=round(seg.get("confidence", 0.95), 2),
                )
                segments_out.append(segment)
                raw_text_parts.append(seg.get("text", "").strip())

        return " ".join(raw_text_parts), segments_out

