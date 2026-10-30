#!/usr/bin/env python3
"""CLI interface for audio transcription pipeline.

Usage:
    python transcribe.py --input sample.wav --format json
    python transcribe.py -i meeting.mp3 -f markdown -o report.md --redact-pii
    python transcribe.py -i audio.wav -f srt -o subtitles.srt
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline.orchestrator import TranscriptionPipeline
from pipeline.models import TranscriptionOptions
from pipeline.downstream.exporters import TranscriptExporter
from pipeline.audio.validator import AudioValidationError


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Audio Transcription & Downstream NLP Processing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Path to input audio file (e.g. WAV, MP3, M4A, FLAC, OGG)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to output file. Defaults to stdout.",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["json", "srt", "vtt", "markdown", "txt"],
        default="json",
        help="Export format for the transcribed and processed output.",
    )
    parser.add_argument(
        "-e", "--engine",
        choices=["mock", "whisper_local", "whisper_api"],
        default="mock",
        help="Speech-to-text backend engine.",
    )
    parser.add_argument(
        "-m", "--model-size",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (applicable to whisper_local).",
    )
    parser.add_argument(
        "-l", "--language",
        default="en",
        help="Spoken language code (e.g., 'en', 'es', 'fr', 'de').",
    )
    parser.add_argument(
        "--redact-pii",
        action="store_true",
        help="Mask personally identifiable information (emails, phone numbers, SSNs).",
    )
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Disable automatic action-item extraction.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable summary and key points extraction.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    options = TranscriptionOptions(
        language=args.language,
        engine=args.engine,
        model_size=args.model_size,
        redact_pii=args.redact_pii,
        extract_actions=not args.no_actions,
        generate_summary=not args.no_summary,
        output_format=args.format,
    )

    pipeline = TranscriptionPipeline()

    try:
        response = pipeline.process(args.input, options=options)
    except AudioValidationError as e:
        sys.stderr.write(f"\n[Validation Error] {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"\n[Pipeline Error] {e}\n")
        sys.exit(2)

    # Export formatting
    if args.format == "json":
        rendered = TranscriptExporter.to_json(response)
    elif args.format == "srt":
        rendered = TranscriptExporter.to_srt(response.segments)
    elif args.format == "vtt":
        rendered = TranscriptExporter.to_vtt(response.segments)
    elif args.format == "markdown":
        rendered = TranscriptExporter.to_markdown(response)
    else:  # txt
        rendered = response.raw_text

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        sys.stderr.write(f"\nSuccessfully wrote {args.format.upper()} output to {args.output}\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

