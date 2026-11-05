"""WAV and raw signed-16-bit PCM decoding."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np


class AudioError(ValueError):
    """Raised for unsupported or malformed audio."""


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray
    sample_rate: int

    @property
    def seconds(self) -> float:
        return float(self.samples.size) / float(self.sample_rate)


def _pcm16_to_mono(data: bytes, channels: int) -> np.ndarray:
    if channels < 1:
        raise AudioError("channels must be >= 1")
    frame_bytes = channels * 2
    if not data:
        raise AudioError("audio body is empty")
    if len(data) % frame_bytes:
        raise AudioError("PCM16 byte count is not a complete audio frame")

    pcm = np.frombuffer(data, dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).astype(np.float32).mean(axis=1)
    else:
        pcm = pcm.astype(np.float32)
    return np.ascontiguousarray(pcm / 32768.0, dtype=np.float32)


def decode_audio(
    body: bytes,
    *,
    raw_sample_rate: int = 16000,
    raw_channels: int = 1,
) -> DecodedAudio:
    """Decode PCM16 WAV, or treat a non-RIFF body as raw little-endian PCM16."""
    if body.startswith(b"RIFF") or body.startswith(b"RIFX"):
        if body.startswith(b"RIFX"):
            raise AudioError("big-endian RIFX WAV is not supported")
        try:
            with wave.open(io.BytesIO(body), "rb") as wav_file:
                if wav_file.getcomptype() != "NONE":
                    raise AudioError("compressed WAV is not supported")
                if wav_file.getsampwidth() != 2:
                    raise AudioError("WAV must contain signed 16-bit PCM")
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                frames = wav_file.readframes(wav_file.getnframes())
        except (EOFError, wave.Error) as exc:
            raise AudioError(f"invalid WAV: {exc}") from exc
        if sample_rate <= 0:
            raise AudioError("WAV sample rate must be positive")
        return DecodedAudio(_pcm16_to_mono(frames, channels), sample_rate)

    if raw_sample_rate <= 0:
        raise AudioError("raw sample rate must be positive")
    return DecodedAudio(_pcm16_to_mono(body, raw_channels), raw_sample_rate)


def pcm16_wav_bytes(pcm: bytes, *, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap signed little-endian PCM16 bytes in a WAV container."""
    if (sample_rate <= 0) or (channels <= 0):
        raise AudioError("sample rate and channels must be positive")
    if len(pcm) % (2 * channels):
        raise AudioError("PCM16 byte count is not a complete audio frame")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()
