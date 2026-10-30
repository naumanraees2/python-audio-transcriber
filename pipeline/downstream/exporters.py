"""Exporters to render transcription results into downstream formats (JSON, SRT, VTT, Markdown)."""

from __future__ import annotations
import json
from typing import List
from pipeline.models import TranscriptionResponse, TranscriptSegment


def format_timestamp_srt(seconds: float) -> str:
    """Formats seconds into SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Formats seconds into WebVTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class TranscriptExporter:
    """Formats transcription response into target output formats."""

    @staticmethod
    def to_json(response: TranscriptionResponse, indent: int = 2) -> str:
        return response.model_dump_json(indent=indent)

    @staticmethod
    def to_srt(segments: List[TranscriptSegment]) -> str:
        lines = []
        for seg in segments:
            lines.append(str(seg.id))
            lines.append(f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}")
            speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
            lines.append(f"{speaker_prefix}{seg.text}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_vtt(segments: List[TranscriptSegment]) -> str:
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines.append(f"{seg.id}")
            lines.append(f"{format_timestamp_vtt(seg.start)} --> {format_timestamp_vtt(seg.end)}")
            speaker_prefix = f"<v {seg.speaker}>" if seg.speaker else ""
            speaker_suffix = "</v>" if seg.speaker else ""
            lines.append(f"{speaker_prefix}{seg.text}{speaker_suffix}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_markdown(response: TranscriptionResponse) -> str:
        lines = [
            f"# Transcription Report",
            f"- **File**: `{response.audio_metadata.file_name}`",
            f"- **Duration**: {response.audio_metadata.duration_seconds}s",
            f"- **Engine**: `{response.engine_used}`",
            f"- **Processing Time**: {response.processing_time_seconds:.2f}s (RTF: {response.real_time_factor:.2f}x)",
            "",
        ]

        if response.downstream:
            lines.append("## Executive Summary")
            lines.append(f"{response.downstream.summary}")
            lines.append("")
            if response.downstream.action_items:
                lines.append("### Action Items")
                for act in response.downstream.action_items:
                    lines.append(f"- [ ] **[{act.urgency.upper()}]** {act.description}")
                lines.append("")

        lines.append("## Transcript")
        for seg in response.segments:
            speaker = f"**{seg.speaker}**" if seg.speaker else f"Segment {seg.id}"
            lines.append(f"- `[{seg.start:.1f}s - {seg.end:.1f}s]` {speaker}: {seg.text}")

        return "\n".join(lines)

