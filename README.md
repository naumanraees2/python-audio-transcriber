# Audio Transcription & Downstream NLP Processing Pipeline

A production-grade, modular speech-to-text pipeline designed to ingest audio files (WAV, MP3, M4A, FLAC, OGG), transcribe speech into timestamped segments, and enrich the output for downstream NLP/LLM systems (action items, PII masking, executive summary, subtitles).

---

## Architecture Overview

```
                         +-----------------------------+
                         |      Audio File Input       |
                         |  (WAV, MP3, M4A, FLAC, OGG) |
                         +--------------+--------------+
                                        |
                                        v
                         +-----------------------------+
                         |   Audio Ingestion & Check   |
                         |  Magic Bytes, Size & Specs  |
                         +--------------+--------------+
                                        |
                                        v
                         +-----------------------------+
                         |  Audio Preprocessor/Resample|
                         |    (16kHz mono PCM target)  |
                         +--------------+--------------+
                                        |
                                        v
                         +-----------------------------+
                         |  Speech-to-Text Strategy    |
                         |  - Mock Engine (Dev/Test)   |
                         |  - Faster-Whisper (Local)   |
                         |  - OpenAI Whisper (Cloud)   |
                         +--------------+--------------+
                                        |
                                        v
                         +-----------------------------+
                         | Downstream NLP Enrichment   |
                         | - Action Item Extraction    |
                         | - PII / Compliance Redaction|
                         | - Executive Summarization   |
                         +--------------+--------------+
                                        |
                  +---------------------+---------------------+
                  |                     |                     |
                  v                     v                     v
         [Structured JSON]       [SRT / VTT Subs]     [Markdown Report]
```

---

## Key Engineering Decisions

1. **Pluggable Engine Strategy Pattern (`TranscriptionEngine`)**:
   - Transcribing audio shouldn't tightly couple your business logic to a single model provider.
   - We define an abstract interface implemented by:
     - `MockSTTEngine`: Deterministic, zero-latency testing engine.
     - `LocalWhisperEngine`: Supports `faster-whisper` (CTranslate2 INT8/FP16) or vanilla `openai-whisper`.
     - `WhisperAPIEngine`: Cloud REST API client for OpenAI / Groq / Deepgram.
2. **Robust Ingestion & Validation**:
   - Does not trust file extensions. Validates headers via binary magic bytes (`RIFF....WAVE`, `ID3`, `fLaC`, `OggS`).
   - Imforces strict file size thresholds (e.g. 100MB) to protect against Denial of Service (DoS) and out-of-memory crashes.
3. **Audio Preprocessing**:
   - Modern ASR models expect 16kHz mono 16-bit PCM. The preprocessor validates sample rate and channels, normalizing via ffmpeg when available.
4. **Downstream-First Data Modeling**:
   - Rather than returning a bare text blob, the pipeline outputs:
     - Word-level and segment-level timestamps.
     - Extracted action items with urgency tags.
     - PII masking (emails, phone numbers, SSNs) for GDPR/HIPAA compliance.
     - Executive summary and key bullet points.
5. **Dual Interface**:
   - **CLI Script (`transcribe.py`)**: For local scripting, cron jobs, and batch workflows.
   - **HTTP REST API (`service.py`)**: Zero-dependency standard library HTTP server + optional FastAPI app.

---

## Quickstart

### 1. Run via CLI Script

```bash
# Transcribe to JSON
python transcribe.py -i samples/valid_sample.wav -f json

# Transcribe to Markdown with PII redaction
python transcribe.py -i samples/valid_sample.wav -f markdown --redact-pii

# Export Subtitles (SRT or VTT)
python transcribe.py -i samples/valid_sample.wav -f srt -o subtitles.srt
```

### 2. Run via HTTP Service

```bash
# Start HTTP Service on port 8080
python service.py --port 8080

# Check health
curl http://127.0.0.1:8080/health

# Transcribe via JSON reference
curl -X POST http://127.0.0.1:8080/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "samples/valid_sample.wav", "engine": "mock", "redact_pii": true}'
```

### 3. Run Test Suite

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

