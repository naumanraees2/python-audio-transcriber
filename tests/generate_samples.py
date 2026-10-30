"""Generates synthetic audio test samples for automated testing and demonstrations."""

import math
import struct
import wave
from pathlib import Path


def create_sine_wav(
    file_path: Path,
    duration_seconds: float = 3.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    num_channels: int = 1,
):
    """Generates a pure sine tone 16-bit PCM WAV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_seconds * sample_rate)

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_samples):
            # 440Hz sine wave, amplitude 32767 * 0.5
            value = int(16384.0 * math.sin(2.0 * math.pi * frequency * (i / sample_rate)))
            packed = struct.pack("<h", value)
            for _ in range(num_channels):
                frames.extend(packed)

        wf.writeframes(frames)


def create_test_samples(samples_dir: Path):
    """Creates a suite of test files."""
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard valid 16kHz mono WAV
    create_sine_wav(samples_dir / "valid_sample.wav", duration_seconds=5.0, sample_rate=16000, num_channels=1)

    # 2. Stereo 44.1kHz WAV
    create_sine_wav(samples_dir / "stereo_sample.wav", duration_seconds=2.0, sample_rate=44100, num_channels=2)

    # 3. Corrupted WAV (truncated header)
    corrupted_path = samples_dir / "corrupted_sample.wav"
    corrupted_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    # 4. Text file disguised as audio
    fake_path = samples_dir / "not_audio.wav"
    fake_path.write_text("This is plain text pretending to be an audio file.")

    print(f"Generated test audio samples in {samples_dir}")


if __name__ == "__main__":
    create_test_samples(Path(__file__).parent.parent / "samples")

