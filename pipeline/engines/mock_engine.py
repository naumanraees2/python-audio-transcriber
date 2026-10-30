"""High-fidelity mock transcription engine for offline testing, CI/CD, and demonstrations.

Provides realistic timestamps, speaker diarization, word-level confidence scores,
and dialogue containing natural action items and PII for downstream verification.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import List, Tuple

from pipeline.engines.base import TranscriptionEngine
from pipeline.models import TranscriptionOptions, TranscriptSegment, Word


class MockSTTEngine(TranscriptionEngine):
    """Deterministic, zero-dependency mock STT engine."""

    SAMPLE_DIALOGUE = [
        (
            0.0,
            4.2,
            "Speaker A",
            "Good morning team. Welcome to our quarterly product review meeting.",
        ),
        (
            4.5,
            9.1,
            "Speaker B",
            "Thanks. I've updated the backend API service and the transcription latency decreased by 40 percent.",
        ),
        (
            9.4,
            14.8,
            "Speaker A",
            "That is fantastic news. Please send me the updated performance report by Friday 5 PM.",
        ),
        (
            15.0,
            20.3,
            "Speaker B",
            "I will also email the security audit results to john.doe@company.com and schedule the staging deployment.",
        ),
        (
            20.5,
            24.0,
            "Speaker A",
            "Great. Let's make sure we review the production checklist before Monday's release.",
        ),
    ]

    def __init__(self, simulated_latency: float = 0.05):
        self.simulated_latency = simulated_latency

    @property
    def engine_name(self) -> str:
        return "mock_engine"

    def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> Tuple[str, List[TranscriptSegment]]:
        """Simulates speech-to-text inference with realistic segment outputs."""
        if self.simulated_latency > 0:
            time.sleep(self.simulated_latency)

        segments: List[TranscriptSegment] = []
        raw_text_parts: List[str] = []

        for idx, (start, end, speaker, sentence) in enumerate(self.SAMPLE_DIALOGUE):
            # Generate word-level timing estimates
            words_in_sentence = sentence.split()
            word_objects: List[Word] = []
            if words_in_sentence:
                word_dur = (end - start) / len(words_in_sentence)
                w_curr = start
                for w in words_in_sentence:
                    word_objects.append(
                        Word(
                            word=w,
                            start=round(w_curr, 2),
                            end=round(w_curr + word_dur, 2),
                            confidence=0.98,
                        )
                    )
                    w_curr += word_dur

            segment = TranscriptSegment(
                id=idx + 1,
                start=start,
                end=end,
                text=sentence,
                confidence=0.97,
                speaker=speaker,
                words=word_objects,
            )
            segments.append(segment)
            raw_text_parts.append(sentence)

        full_raw_text = " ".join(raw_text_parts)
        return full_raw_text, segments

