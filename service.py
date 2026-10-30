#!/usr/bin/env python3
"""HTTP Service for Audio Transcription Pipeline.

Provides REST API endpoints for transcribing audio files.
Includes:
- Standalone zero-dependency HTTP server (via standard library `http.server`)
- FastAPI ASGI application export for production deployments (e.g. Uvicorn/Gunicorn/Docker)

Endpoints:
- GET  /health      : Service health and engine status
- POST /transcribe  : Upload audio file (multipart) or JSON file reference
"""

import http.server
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from pipeline.orchestrator import TranscriptionPipeline
from pipeline.models import TranscriptionOptions
from pipeline.audio.validator import AudioValidationError, FileTooLargeError, UnsupportedAudioFormatError

logger = logging.getLogger("transcription_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

pipeline = TranscriptionPipeline()


class TranscriptionHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Zero-dependency HTTP request handler for transcription pipeline."""

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ["/health", "/", "/v1/health"]:
            self._send_json(200, {
                "status": "healthy",
                "service": "audio-transcription-pipeline",
                "version": "1.0.0",
                "supported_formats": ["wav", "mp3", "m4a", "flac", "ogg"],
                "supported_engines": ["mock", "whisper_local", "whisper_api"],
            })
        else:
            self._send_json(404, {"error": "Not Found", "path": self.path})

    def do_POST(self):
        if self.path not in ["/transcribe", "/v1/transcribe"]:
            self._send_json(404, {"error": "Endpoint not found"})
            return

        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length > 100 * 1024 * 1024:
            self._send_json(413, {"error": "Payload exceeds 100MB limit"})
            return

        # Case 1: JSON payload referencing existing audio file
        if "application/json" in content_type:
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON body: {e}"})
                return

            audio_path = payload.get("audio_path")
            if not audio_path:
                self._send_json(400, {"error": "Missing 'audio_path' field in JSON request"})
                return

            options = TranscriptionOptions(
                engine=payload.get("engine", "mock"),
                model_size=payload.get("model_size", "base"),
                language=payload.get("language", "en"),
                redact_pii=payload.get("redact_pii", False),
            )

            self._process_and_reply(Path(audio_path), options)

        # Case 2: Multipart/form-data audio file upload
        elif "multipart/form-data" in content_type:
            # Parse multipart boundary
            boundary = content_type.split("boundary=")[-1].strip()
            raw_data = self.rfile.read(content_length)

            # Extract file content simply from multipart bytes
            sep = f"--{boundary}".encode()
            parts = raw_data.split(sep)
            audio_bytes = None
            filename = "uploaded_audio.wav"
            engine_choice = "mock"

            for part in parts:
                if b'Content-Disposition: form-data;' in part and b'filename="' in part:
                    # Extract filename
                    try:
                        header_part, file_data = part.split(b"\r\n\r\n", 1)
                        if file_data.endswith(b"\r\n"):
                            file_data = file_data[:-2]
                        audio_bytes = file_data

                        for line in header_part.decode("latin1").split("\r\n"):
                            if "filename=" in line:
                                filename = line.split('filename="')[1].split('"')[0]
                    except Exception:
                        pass
                elif b'name="engine"' in part:
                    try:
                        _, val = part.split(b"\r\n\r\n", 1)
                        engine_choice = val.decode().strip()
                    except Exception:
                        pass

            if not audio_bytes:
                self._send_json(400, {"error": "No audio file found in multipart upload"})
                return

            suffix = Path(filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                temp_audio_path = Path(tmp.name)

            try:
                options = TranscriptionOptions(engine=engine_choice)
                self._process_and_reply(temp_audio_path, options)
            finally:
                if temp_audio_path.exists():
                    try:
                        temp_audio_path.unlink()
                    except Exception:
                        pass
        else:
            self._send_json(415, {"error": f"Unsupported Content-Type: {content_type}. Use multipart/form-data or application/json."})

    def _process_and_reply(self, audio_path: Path, options: TranscriptionOptions):
        try:
            response = pipeline.process(audio_path, options=options)
            self._send_json(200, response.model_dump())
        except UnsupportedAudioFormatError as e:
            self._send_json(415, {"error": str(e), "type": "UnsupportedAudioFormatError"})
        except FileTooLargeError as e:
            self._send_json(413, {"error": str(e), "type": "FileTooLargeError"})
        except AudioValidationError as e:
            self._send_json(400, {"error": str(e), "type": "AudioValidationError"})
        except Exception as e:
            logger.exception("Unexpected error during transcription")
            self._send_json(500, {"error": f"Internal transcription error: {str(e)}"})


def run_standalone_server(port: int = 8080, host: str = "0.0.0.0"):
    """Starts the standalone zero-dependency HTTP server."""
    server_address = (host, port)
    httpd = http.server.HTTPServer(server_address, TranscriptionHTTPRequestHandler)
    print(f"Transcription service running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        httpd.server_close()


# Optional FastAPI application for production deployments (FastAPI + Uvicorn)
try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="Speech-to-Text Transcription Service",
        description="Scalable audio transcription pipeline with downstream NLP processing",
        version="1.0.0",
    )

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": "audio-transcription-pipeline"}

    # Register upload endpoint if python-multipart is available
    try:
        import multipart  # check python-multipart

        @app.post("/transcribe")
        async def transcribe_endpoint(
            file: UploadFile = File(...),
            engine: str = Form("mock"),
            model_size: str = Form("base"),
            redact_pii: bool = Form(False),
        ):
            suffix = Path(file.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_path = Path(tmp.name)

            try:
                options = TranscriptionOptions(
                    engine=engine,
                    model_size=model_size,
                    redact_pii=redact_pii,
                )
                result = pipeline.process(temp_path, options=options)
                return result
            except AudioValidationError as e:
                raise HTTPException(status_code=400, detail=str(e))
            finally:
                if temp_path.exists():
                    temp_path.unlink()
    except ImportError:
        @app.post("/transcribe")
        def transcribe_json(payload: dict):
            audio_path = payload.get("audio_path")
            if not audio_path:
                raise HTTPException(status_code=400, detail="Missing 'audio_path'")
            options = TranscriptionOptions(
                engine=payload.get("engine", "mock"),
                redact_pii=payload.get("redact_pii", False),
            )
            return pipeline.process(Path(audio_path), options=options)

except ImportError:
    app = None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start Audio Transcription HTTP Service")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    args = parser.parse_args()

    run_standalone_server(port=args.port, host=args.host)
