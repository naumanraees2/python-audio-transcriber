"""Data models for the transcription pipeline.

Uses Pydantic for validation, schema enforcement, and JSON serialization.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class AudioMetadata(BaseModel):
    """Metadata describing the ingested audio file."""
    file_name: str
    file_path: str
    format: str
    file_size_bytes: int
    duration_seconds: float = 0.0
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None


class Word(BaseModel):
    """Word-level timestamp and confidence."""
    word: str
    start: float
    end: float
    confidence: float = 1.0


class TranscriptSegment(BaseModel):
    """A timestamped segment of transcribed text."""
    id: int
    start: float
    end: float
    text: str
    confidence: float = 1.0
    speaker: Optional[str] = None
    words: List[Word] = Field(default_factory=list)


class ActionItem(BaseModel):
    """Extracted action item for downstream workflows."""
    description: str
    trigger_keyword: str
    urgency: str = "normal"


class DownstreamAnalysis(BaseModel):
    """Structured outputs enriched for downstream LLMs/NLP systems."""
    summary: str
    key_points: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    cleaned_text: str
    word_count: int
    pii_masked_text: Optional[str] = None
    sentiment: str = "neutral"


class TranscriptionOptions(BaseModel):
    """Configuration options for a transcription run."""
    language: Optional[str] = "en"
    engine: str = "mock"  # "mock", "whisper_local", "whisper_api"
    model_size: str = "base"  # tiny, base, small, medium, large
    redact_pii: bool = False
    extract_actions: bool = True
    generate_summary: bool = True
    output_format: str = "json"  # "json", "srt", "vtt", "txt"


class TranscriptionResponse(BaseModel):
    """Standardized response payload for downstream consumers."""
    job_id: str
    status: str = "completed"
    engine_used: str
    audio_metadata: AudioMetadata
    processing_time_seconds: float
    real_time_factor: float
    raw_text: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    downstream: Optional[DownstreamAnalysis] = None
    error: Optional[str] = None

