"""Automated test suite for audio transcription pipeline."""

import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from http.server import HTTPServer
from pathlib import Path

from pipeline.orchestrator import TranscriptionPipeline
from pipeline.models import TranscriptionOptions
from pipeline.audio.validator import AudioValidator, AudioValidationError
from pipeline.downstream.processor import DownstreamProcessor
from pipeline.downstream.exporters import TranscriptExporter
from pipeline.engines.factory import get_engine
from service import TranscriptionHTTPRequestHandler
from tests.generate_samples import create_sine_wav


class TestTranscriptionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("/tmp/test_transcriber_assets")
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        cls.valid_wav = cls.test_dir / "test_valid.wav"
        create_sine_wav(cls.valid_wav, duration_seconds=3.0, sample_rate=16000)

        cls.corrupted_wav = cls.test_dir / "test_corrupted.wav"
        cls.corrupted_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEbad_data")

        cls.fake_wav = cls.test_dir / "test_fake.wav"
        cls.fake_wav.write_text("Not a real wave file content")

    def test_audio_metadata_extraction(self):
        validator = AudioValidator()
        meta = validator.extract_metadata(self.valid_wav)
        self.assertEqual(meta.format, "wav")
        self.assertEqual(meta.sample_rate, 16000)
        self.assertEqual(meta.channels, 1)
        self.assertAlmostEqual(meta.duration_seconds, 3.0, delta=0.1)

    def test_corrupted_audio_rejection(self):
        validator = AudioValidator()
        with self.assertRaises(AudioValidationError):
            validator.extract_metadata(self.corrupted_wav)

    def test_fake_audio_rejection(self):
        validator = AudioValidator()
        with self.assertRaises(AudioValidationError):
            validator.extract_metadata(self.fake_wav)

    def test_full_pipeline_run(self):
        pipeline = TranscriptionPipeline()
        options = TranscriptionOptions(engine="mock", redact_pii=True)
        response = pipeline.process(self.valid_wav, options=options)

        self.assertEqual(response.status, "completed")
        self.assertTrue(len(response.segments) > 0)
        self.assertIsNotNone(response.downstream)
        self.assertIn("EMAIL REDACTED", response.downstream.pii_masked_text)
        self.assertTrue(len(response.downstream.action_items) > 0)

    def test_action_item_extraction(self):
        processor = DownstreamProcessor()
        text = "I will deliver the report tomorrow. Please send me the spreadsheet. Everything looks great."
        actions = processor.extract_action_items(text)
        self.assertEqual(len(actions), 2)
        descriptions = [a.description for a in actions]
        self.assertTrue(any("I will deliver" in d for d in descriptions))
        self.assertTrue(any("Please send me" in d for d in descriptions))

    def test_exporters(self):
        pipeline = TranscriptionPipeline()
        response = pipeline.process(self.valid_wav)

        srt_output = TranscriptExporter.to_srt(response.segments)
        self.assertIn("-->", srt_output)
        self.assertIn("00:00:00,000", srt_output)

        vtt_output = TranscriptExporter.to_vtt(response.segments)
        self.assertTrue(vtt_output.startswith("WEBVTT"))

        md_output = TranscriptExporter.to_markdown(response)
        self.assertIn("# Transcription Report", md_output)

    def test_engine_factory(self):
        mock_engine = get_engine("mock")
        self.assertEqual(mock_engine.engine_name, "mock_engine")

        with self.assertRaises(ValueError):
            get_engine("invalid_engine_type")


class DummySocket:
    def __init__(self, request_bytes: bytes):
        import io
        self.rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "b" in mode:
            if "r" in mode:
                return self.rfile
            elif "w" in mode:
                return self.wfile
        import io
        return io.BytesIO()

    def sendall(self, data):
        self.wfile.write(data)


class TestHTTPService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("/tmp/test_service_assets")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.test_wav = cls.test_dir / "service_test.wav"
        create_sine_wav(cls.test_wav, duration_seconds=2.0)

    def _simulate_http_request(self, method: str, path: str, body: bytes = b"", headers: dict = None) -> Tuple[int, dict]:
        import io
        headers = headers or {}
        header_lines = [f"{method} {path} HTTP/1.1", f"Host: localhost"]
        for k, v in headers.items():
            header_lines.append(f"{k}: {v}")
        if body and "Content-Length" not in headers:
            header_lines.append(f"Content-Length: {len(body)}")
        header_str = "\r\n".join(header_lines) + "\r\n\r\n"
        req_bytes = header_str.encode("latin1") + body

        dummy_sock = DummySocket(req_bytes)

        class MockServer:
            def __init__(self):
                pass

        try:
            handler = TranscriptionHTTPRequestHandler(dummy_sock, ("127.0.0.1", 12345), MockServer())
        except Exception:
            pass

        output = dummy_sock.wfile.getvalue()
        if not output:
            return 500, {}

        # Parse response
        parts = output.split(b"\r\n\r\n", 1)
        resp_header = parts[0].decode("latin1")
        resp_body = parts[1] if len(parts) > 1 else b"{}"
        status_line = resp_header.split("\r\n")[0]
        status_code = int(status_line.split()[1])
        try:
            data = json.loads(resp_body.decode("utf-8"))
        except Exception:
            data = {"raw": resp_body.decode("latin1")}
        return status_code, data

    def test_health_endpoint(self):
        status, data = self._simulate_http_request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("wav", data.get("supported_formats", []))

    def test_transcribe_json_request(self):
        payload = json.dumps({"audio_path": str(self.test_wav), "engine": "mock"}).encode()
        status, data = self._simulate_http_request(
            "POST",
            "/transcribe",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "completed")
        self.assertEqual(data["audio_metadata"]["format"], "wav")
        self.assertTrue(len(data["segments"]) > 0)


if __name__ == "__main__":
    unittest.main()
