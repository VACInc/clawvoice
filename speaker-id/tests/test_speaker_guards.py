"""Hermetic guards for thresholds, audio bounds, store, config, and device choice."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from speaker_embed.app import create_app, enroll, health, identify
from speaker_embed.audio import AudioError, DecodedAudio, decode_audio, pcm16_wav_bytes
from speaker_embed.config import SpeakerConfig
from speaker_embed.core import SpeakerEngine, Thresholds, decide
from speaker_embed.models import resolve_device
from speaker_embed.store import FlatFileEmbeddingStore, normalize

from .helpers import MemoryStore, QueueBackend, speaker_config, synthetic_wav


def test_http_dependency_security_floor() -> None:
    """Keep production and test installs past the reviewed HTTP parser fixes."""
    from pathlib import Path
    import re
    import aiohttp
    from packaging.version import Version

    root = Path(__file__).resolve().parents[1]
    selected = []
    for manifest in ("pyproject.toml", "requirements.txt", "requirements-test.txt"):
        versions = re.findall(r'\baiohttp==([0-9.]+)', (root / manifest).read_text())
        # CVE-2026-69244 and CVE-2026-111066: all entry points need the patch.
        assert len(versions) == 1
        assert Version(versions[0]) >= Version("3.14.4")
        selected.append(versions[0])
    assert len(set(selected)) == 1
    assert Version(aiohttp.__version__) >= Version("3.14.4")


class _Backend:
    def __init__(self, ready: bool = True, vectors=None, name: str = "primary"):
        self.name = name
        self.store_key = name
        self.ready = ready
        self.loaded = ready
        self._vectors = list(vectors or [[1.0, 0.0]])

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.asarray(self._vectors.pop(0), dtype=np.float32)


class _Boom:
    name = "titanet"
    store_key = "titanet"
    ready = True
    loaded = False

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        raise RuntimeError("boom")


class _Cuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _Torch:
    def __init__(self, available: bool):
        self.cuda = _Cuda(available)


def _audio(count: int, rate: int = 16000) -> DecodedAudio:
    return DecodedAudio(np.zeros(count, dtype=np.float32), rate)


def _engine(**overrides) -> SpeakerEngine:
    values = dict(
        primary=_Backend(),
        store=MemoryStore(),
        thresholds=Thresholds(0.5, 0.9),
        allowed_users=frozenset({"alice"}),
        shadow=True,
        min_audio_ms=300,
        max_audio_seconds=30.0,
    )
    values.update(overrides)
    return SpeakerEngine(**values)


def test_threshold_edges_and_candidate_omission() -> None:
    Thresholds(-1.0, 1.0)
    Thresholds(0.5, 0.5)
    with pytest.raises(ValueError):
        Thresholds(0.8, 0.2)
    with pytest.raises(ValueError):
        Thresholds(-1.1, 0.0)

    centroids = {"alice": np.asarray([1.0, 0.0], dtype=np.float64)}
    accepted = decide(np.asarray([1.0, 0.0]), centroids, Thresholds(0.0, 1.0))
    assert accepted.status == "accepted"
    assert accepted.as_dict()["candidate_user_id"] == "alice"

    on_low = decide(np.asarray([0.0, 1.0]), centroids, Thresholds(0.0, 0.5))
    assert on_low.status == "uncertain"
    assert on_low.candidate_user_id == "alice"

    below = decide(np.asarray([0.0, 1.0]), centroids, Thresholds(0.2, 0.5))
    assert below.status == "unknown"
    assert below.candidate_user_id == "alice"

    empty = decide(np.asarray([1.0, 0.0]), {}, Thresholds(-1.0, 1.0))
    assert empty.status == "unknown"
    assert "candidate_user_id" not in empty.as_dict()


def test_audio_duration_edges_nan_and_enroll_user() -> None:
    exact = _engine()
    exact.identify(_audio(4800), "kitchen")
    with pytest.raises(ValueError, match="at least"):
        exact.identify(_audio(4799), "kitchen")

    bounded = _engine(min_audio_ms=1, max_audio_seconds=0.3, primary=_Backend())
    bounded.identify(_audio(4800), "kitchen")
    with pytest.raises(ValueError, match="no longer"):
        bounded.identify(_audio(4801), "kitchen")

    noisy = _audio(4800)
    noisy.samples[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _engine().identify(noisy, "kitchen")

    enrolled = _engine(min_audio_ms=1, primary=_Backend(vectors=[[1.0, 0.0]]))
    assert enrolled.enroll(_audio(1600), "alice")["user"] == "alice"
    with pytest.raises(ValueError, match="unsupported"):
        enrolled.enroll(_audio(1600), "../x")
    with pytest.raises(PermissionError):
        enrolled.enroll(_audio(1600), "other")


def test_secondary_failure_does_not_override_and_enforced_follows_shadow() -> None:
    primary = _Backend(name="primary")
    store = MemoryStore()
    store.add("primary", "alice", np.asarray([1.0, 0.0]))
    engine = _engine(
        primary=primary,
        secondary=_Boom(),
        store=store,
        min_audio_ms=1,
        thresholds=Thresholds(0.5, 0.9),
    )
    result = engine.identify(_audio(1600), "kitchen")
    assert result["user_id"] == "alice"
    assert result["shadow_error"] == "RuntimeError"
    assert "shadow_score" not in result
    assert result["enforced"] is False

    engine.shadow = False
    engine.primary = _Backend(name="primary")
    again = engine.identify(_audio(1600), "kitchen")
    assert again["enforced"] is True


def test_health_status_and_http_code() -> None:
    ready = _engine(primary=_Backend(ready=True), min_audio_ms=1)
    assert ready.health()["status"] == "ok"
    missing = _engine(primary=_Backend(ready=False), min_audio_ms=1)
    assert missing.health()["status"] == "model_missing"
    assert missing.health()["ready"] is False


@pytest.mark.asyncio
async def test_http_device_rate_user_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)
    engine = _engine(min_audio_ms=10, max_audio_seconds=5.0, primary=_Backend())
    app = create_app(speaker_config(), engine)

    ok = make_mocked_request("POST", "/identify?device_id=" + ("d" * 128), app=app)
    ok._read_bytes = synthetic_wav(duration_seconds=0.05)
    response = await identify(ok)
    assert response.status == 200

    long_id = make_mocked_request("POST", "/identify?device_id=" + ("d" * 129), app=app)
    long_id._read_bytes = synthetic_wav(duration_seconds=0.05)
    with pytest.raises(web.HTTPBadRequest):
        await identify(long_id)

    slow = make_mocked_request(
        "POST", "/identify?device_id=kitchen&sample_rate=0", app=app
    )
    slow._read_bytes = b"\x00\x00"
    with pytest.raises(web.HTTPBadRequest) as caught:
        await identify(slow)
    assert "sample_rate must be positive" in caught.value.text

    missing_user = make_mocked_request("POST", "/enroll?user=", app=app)
    with pytest.raises(web.HTTPBadRequest):
        await enroll(missing_user)

    forbidden = make_mocked_request("POST", "/enroll?user=other", app=app)
    forbidden._read_bytes = synthetic_wav(duration_seconds=0.05)
    with pytest.raises(web.HTTPForbidden):
        await enroll(forbidden)

    live = await health(make_mocked_request("GET", "/health", app=app))
    assert live.status == 200
    down = _engine(primary=_Backend(ready=False), min_audio_ms=1)
    down_app = create_app(speaker_config(), down)
    dead = await health(make_mocked_request("GET", "/health", app=down_app))
    assert dead.status == 503


def test_decode_edges_and_wav_roundtrip() -> None:
    pcm = np.array([1, -1], dtype="<i2").tobytes()
    wav = pcm16_wav_bytes(pcm, sample_rate=8000)
    decoded = decode_audio(wav)
    assert decoded.sample_rate == 8000
    np.testing.assert_allclose(decoded.samples, np.array([1, -1], dtype=np.float32) / 32768.0)

    left = np.array([32767, -32768], dtype="<i2")
    right = np.array([0, 0], dtype="<i2")
    stereo = np.column_stack([left, right]).ravel().astype("<i2").tobytes()
    mixed = decode_audio(stereo, raw_sample_rate=8000, raw_channels=2)
    np.testing.assert_allclose(
        mixed.samples,
        np.array([16383.5, -16384.0], dtype=np.float32) / 32768.0,
    )

    with pytest.raises(AudioError, match="big-endian"):
        decode_audio(b"RIFX" + b"\x00" * 32)
    with pytest.raises(AudioError, match="empty"):
        decode_audio(b"", raw_sample_rate=16000)
    with pytest.raises(AudioError, match="frame"):
        decode_audio(b"\x00\x01\x02", raw_sample_rate=16000)
    with pytest.raises(AudioError, match="channels"):
        decode_audio(b"\x00\x00", raw_sample_rate=16000, raw_channels=0)
    with pytest.raises(AudioError, match="raw sample rate"):
        decode_audio(b"\x00\x00", raw_sample_rate=0)
    with pytest.raises(AudioError):
        pcm16_wav_bytes(b"\x00\x00", sample_rate=0)
    with pytest.raises(AudioError):
        pcm16_wav_bytes(b"", sample_rate=8000, channels=0)


def test_store_rejects_bad_vectors_paths_and_keeps_unit_centroids(tmp_path) -> None:
    with pytest.raises(ValueError):
        normalize(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError):
        normalize(np.array([np.inf, 0.0], dtype=np.float32))

    store = FlatFileEmbeddingStore(tmp_path)
    with pytest.raises(ValueError):
        store.add("../x", "alice", np.array([1.0, 0.0]))
    assert store.add("model", "alice", np.array([1.0, 0.0])) == 1
    assert store.add("model", "alice", np.array([0.0, 1.0])) == 2
    with pytest.raises(ValueError, match="dimension"):
        store.add("model", "alice", np.ones(3))

    np.save(tmp_path / "model--bad.npy", np.array([1.0, 0.0], dtype=np.float32))
    centroids = store.centroids("model")
    assert "bad" not in centroids
    assert abs(float(np.linalg.norm(centroids["alice"])) - 1.0) < 1e-5


def test_config_bools_users_thresholds_and_channels() -> None:
    assert SpeakerConfig.from_env({"SHADOW": "yes"}).shadow is True
    assert SpeakerConfig.from_env({"SHADOW": "off"}).shadow is False
    with pytest.raises(ValueError):
        SpeakerConfig.from_env({"SHADOW": "maybe"})
    with pytest.raises(ValueError):
        SpeakerConfig.from_env({"ALLOWED_ENROLL_USERS": " , "})
    with pytest.raises(ValueError):
        SpeakerConfig.from_env({"LOW_THRESHOLD": "0.8", "HIGH_THRESHOLD": "0.2"})
    equal = SpeakerConfig.from_env(
        {
            "LOW_THRESHOLD": "0.5",
            "HIGH_THRESHOLD": "0.5",
            "TITANET_LOW_THRESHOLD": "0.4",
            "TITANET_HIGH_THRESHOLD": "0.4",
        }
    )
    assert equal.low_threshold == 0.5
    assert equal.high_threshold == 0.5
    with pytest.raises(ValueError):
        SpeakerConfig.from_env({"RAW_CHANNELS": "9"})
    assert SpeakerConfig.from_env({"RAW_CHANNELS": "8"}).raw_channels == 8


def test_resolve_device_cpu_cuda_and_prefix() -> None:
    assert resolve_device(_Torch(True), "auto") == "cuda"
    assert resolve_device(_Torch(False), "auto") == "cpu"
    assert resolve_device(_Torch(False), "cpu") == "cpu"
    assert resolve_device(_Torch(True), "cuda:0") == "cuda:0"
    with pytest.raises(RuntimeError):
        resolve_device(_Torch(False), "cuda")
    with pytest.raises(RuntimeError):
        resolve_device(_Torch(False), "cuda:1")
