"""Central transcription pipeline orchestrator.

Coordinates ingestion, format validation, preprocessing, engine dispatch,
and downstream enrichment.
"""

from __future__ import annotations
import time
import uuid
import logging
from pathlib import Path

from pipeline.models import TranscriptionOptions, TranscriptionResponse
from pipeline.audio.validator import AudioValidator
from pipeline.audio.preprocessor import AudioPreprocessor
from pipeline.engines.factory import get_engine
from pipeline.downstream.processor import DownstreamProcessor

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    """End-to-end pipeline orchestrator."""

    def __init__(self, validator: AudioValidator | None = None):
        self.validator = validator or AudioValidator()
        self.preprocessor = AudioPreprocessor()
        self.downstream_processor = DownstreamProcessor()

    def process(self, audio_path: Path | str, options: TranscriptionOptions | None = None) -> TranscriptionResponse:
        """Executes the full transcription and downstream processing lifecycle."""
        start_time = time.time()
        options = options or TranscriptionOptions()
        path = Path(audio_path).resolve()
        job_id = str(uuid.uuid4())

        logger.info(f"Starting job {job_id} for file: {path.name}")

        # Step 1: Ingestion & Validation
        metadata = self.validator.extract_metadata(path)
        logger.info(
            f"Validated audio: format={metadata.format}, duration={metadata.duration_seconds}s, "
            f"size={metadata.file_size_bytes} bytes"
        )

        # Step 2: Preprocessing / Normalization
        processed_path = self.preprocessor.normalize(path, metadata)

        # Step 3: Engine Selection and Speech-to-Text Transcription
        engine = get_engine(options.engine, model_size=options.model_size)
        raw_text, segments = engine.transcribe(processed_path, options)

        # Step 4: Downstream Processing (PII masking, Action Items, Summarization)
        downstream = None
        if options.extract_actions or options.generate_summary or options.redact_pii:
            downstream = self.downstream_processor.analyze(
                raw_text=raw_text,
                segments=segments,
                redact_pii_flag=options.redact_pii,
            )

        # Step 5: Metrics and Assembly
        processing_time = round(time.time() - start_time, 3)
        duration = metadata.duration_seconds if metadata.duration_seconds > 0 else 0.01
        real_time_factor = round(processing_time / duration, 3)

        return TranscriptionResponse(
            job_id=job_id,
            status="completed",
            engine_used=engine.engine_name,
            audio_metadata=metadata,
            processing_time_seconds=processing_time,
            real_time_factor=real_time_factor,
            raw_text=raw_text,
            segments=segments,
            downstream=downstream,
        )

