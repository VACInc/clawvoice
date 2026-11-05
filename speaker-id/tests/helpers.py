from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from speaker_embed.audio import pcm16_wav_bytes
from speaker_embed.config import SpeakerConfig
from speaker_embed.store import normalize
from wyoming_tee.config import TeeConfig


class QueueBackend:
    def __init__(self, vectors: Iterable[Iterable[float]], name: str = "fake-primary"):
        self.name = name
        self.store_key = name.replace("-", "_")
        self._vectors = deque(
            np.asarray(vector, dtype=np.float32) for vector in vectors
        )
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def ready(self) -> bool:
        return True

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        assert samples.ndim == 1
        assert sample_rate > 0
        self._loaded = True
        return normalize(self._vectors.popleft())


class MemoryStore:
    def __init__(self):
        self.values: dict[tuple[str, str], list[np.ndarray]] = {}

    def add(self, model_key: str, user: str, embedding: np.ndarray) -> int:
        values = self.values.setdefault((model_key, user), [])
        values.append(normalize(embedding))
        return len(values)

    def centroids(self, model_key: str) -> Mapping[str, np.ndarray]:
        result = {}
        for (stored_model, user), values in self.values.items():
            if stored_model == model_key:
                result[user] = normalize(np.mean(values, axis=0))
        return result

    def summary(self) -> Mapping[str, Mapping[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for (model, user), values in self.values.items():
            result.setdefault(model, {})[user] = len(values)
        return result


def synthetic_wav(
    *, duration_seconds: float = 0.5, sample_rate: int = 16000, frequency: float = 440.0
) -> bytes:
    sample_count = int(duration_seconds * sample_rate)
    phase = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = 0.25 * np.sin(2.0 * math.pi * frequency * phase)
    pcm = np.clip(waveform * 32767.0, -32768, 32767).astype("<i2").tobytes()
    return pcm16_wav_bytes(pcm, sample_rate=sample_rate)


def speaker_config(**overrides) -> SpeakerConfig:
    unused = Path("tests/unused").resolve()
    values = dict(
        host="127.0.0.1",
        port=8099,
        model_dir=unused,
        enrollment_dir=unused,
        redimnet_source_dir=unused,
        redimnet_checkpoint=unused,
        redimnet_checkpoint_sha256="0" * 64,
        device="auto",
        high_threshold=0.90,
        low_threshold=0.50,
        shadow=True,
        titanet_enabled=False,
        titanet_checkpoint=unused,
        titanet_checkpoint_sha256="0" * 64,
        titanet_high_threshold=0.90,
        titanet_low_threshold=0.50,
        allowed_enroll_users=frozenset({"alice"}),
        raw_sample_rate=16000,
        raw_channels=1,
        min_audio_ms=10,
        max_audio_seconds=5.0,
        max_body_bytes=1024 * 1024,
        verify_model_hashes=False,
        log_level="INFO",
    )
    values.update(overrides)
    return SpeakerConfig(**values)


def tee_config(real_port: int, **overrides) -> TeeConfig:
    values = dict(
        host="127.0.0.1",
        port=0,
        real_stt_host="127.0.0.1",
        real_stt_port=real_port,
        embed_url="http://127.0.0.1:1/identify",
        ha_webhook_url="",
        device_id="test-satellite",
        shadow=True,
        identify_timeout_seconds=1.0,
        webhook_timeout_seconds=1.0,
        webhook_retries=2,
        webhook_retry_base_seconds=0.01,
        webhook_retry_max_seconds=0.05,
        max_utterance_bytes=1024 * 1024,
        max_frame_bytes=2 * 1024 * 1024,
        max_header_bytes=64 * 1024,
        relay_buffer_bytes=64 * 1024,
        capture_user="",
        capture_dir=Path("tests/unused-captures").resolve(),
        shutdown_grace_seconds=1.0,
        log_level="INFO",
    )
    values.update(overrides)
    return TeeConfig(**values)
