from __future__ import annotations

import asyncio
import json
import math

import numpy as np
import pytest
from aiohttp.test_utils import make_mocked_request

from speaker_embed.app import create_app, enroll, enrolled, health, identify
from speaker_embed.config import SpeakerConfig
from speaker_embed.core import SpeakerEngine, Thresholds, decide
from speaker_embed.models import resolve_device
from wyoming_tee.config import TeeConfig

from .helpers import (
    MemoryStore,
    QueueBackend,
    speaker_config,
    synthetic_wav,
)


@pytest.mark.asyncio
async def test_enroll_averages_samples_then_identify_enrolled_and_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    # This test runner forbids clean executor-thread shutdown. Production keeps
    # asyncio.to_thread so model inference never blocks the aiohttp loop.
    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)
    diagonal = math.sqrt(0.5)
    backend = QueueBackend(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [diagonal, diagonal],
            [-1.0, 0.0],
        ]
    )
    store = MemoryStore()
    engine = SpeakerEngine(
        primary=backend,
        store=store,
        thresholds=Thresholds(low=0.50, high=0.90),
        allowed_users=frozenset({"alice"}),
        shadow=True,
        min_audio_ms=10,
        max_audio_seconds=5.0,
    )
    app = create_app(speaker_config(), engine)
    wav = synthetic_wav()

    first_request = make_mocked_request("POST", "/enroll?user=alice", app=app)
    first_request._read_bytes = wav
    first = await enroll(first_request)
    assert first.status == 200
    assert json.loads(first.body)["samples"] == 1

    second_request = make_mocked_request("POST", "/enroll?user=alice", app=app)
    second_request._read_bytes = wav
    second = await enroll(second_request)
    assert second.status == 200
    assert json.loads(second.body)["samples"] == 2

    centroid = store.centroids(backend.store_key)["alice"]
    np.testing.assert_allclose(centroid, np.asarray([diagonal, diagonal]), atol=1e-6)

    enrolled_request = make_mocked_request(
        "POST", "/identify?device_id=kitchen", app=app
    )
    enrolled_request._read_bytes = wav
    enrolled_response = await identify(enrolled_request)
    enrolled_json = json.loads(enrolled_response.body)
    assert enrolled_response.status == 200
    assert enrolled_json["user_id"] == "alice"
    assert enrolled_json["status"] == "accepted"
    assert enrolled_json["enforced"] is False

    unknown_request = make_mocked_request(
        "POST", "/identify?device_id=kitchen", app=app
    )
    unknown_request._read_bytes = wav
    unknown_response = await identify(unknown_request)
    unknown_json = json.loads(unknown_response.body)
    assert unknown_response.status == 200
    assert unknown_json["user_id"] == "unknown"
    assert unknown_json["status"] == "unknown"


def test_threshold_bands_are_fail_closed() -> None:
    centroids = {"alice": np.asarray([1.0, 0.0], dtype=np.float32)}
    thresholds = Thresholds(low=0.50, high=0.80)

    accepted = decide(np.asarray([0.80, 0.60]), centroids, thresholds)
    assert accepted.user_id == "alice"
    assert accepted.status == "accepted"

    uncertain = decide(
        np.asarray([0.70, math.sqrt(1.0 - 0.70**2)]), centroids, thresholds
    )
    assert uncertain.user_id == "unknown"
    assert uncertain.candidate_user_id == "alice"
    assert uncertain.status == "uncertain"

    unknown = decide(
        np.asarray([0.40, math.sqrt(1.0 - 0.40**2)]), centroids, thresholds
    )
    assert unknown.user_id == "unknown"
    assert unknown.status == "unknown"


def test_auto_device_has_cpu_fallback() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    assert resolve_device(FakeTorch(), "auto") == "cpu"


def test_shadow_defaults_on_for_both_services() -> None:
    assert SpeakerConfig.from_env({}).shadow is True
    assert TeeConfig.from_env({}).shadow is True


@pytest.mark.asyncio
async def test_titanet_is_report_only_and_cannot_override_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)
    primary = QueueBackend([[1.0, 0.0]], name="primary")
    secondary = QueueBackend([[0.0, 1.0]], name="titanet")
    store = MemoryStore()
    store.add(primary.store_key, "alice", np.asarray([1.0, 0.0]))
    store.add(secondary.store_key, "alice", np.asarray([1.0, 0.0]))
    engine = SpeakerEngine(
        primary=primary,
        secondary=secondary,
        store=store,
        thresholds=Thresholds(0.5, 0.9),
        secondary_thresholds=Thresholds(0.5, 0.9),
        allowed_users=frozenset({"alice"}),
        min_audio_ms=10,
    )
    app = create_app(speaker_config(), engine)
    request = make_mocked_request("POST", "/identify?device_id=office", app=app)
    request._read_bytes = synthetic_wav()
    response = await identify(request)
    result = json.loads(response.body)
    assert result["user_id"] == "alice"
    assert result["shadow_score"]["user_id"] == "unknown"


@pytest.mark.asyncio
async def test_raw_pcm_health_and_enrolled_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)
    primary = QueueBackend([[1.0, 0.0]], name="primary")
    store = MemoryStore()
    store.add(primary.store_key, "alice", np.asarray([1.0, 0.0]))
    engine = SpeakerEngine(
        primary=primary,
        store=store,
        thresholds=Thresholds(0.5, 0.9),
        allowed_users=frozenset({"alice"}),
        min_audio_ms=10,
    )
    app = create_app(speaker_config(), engine)
    route_paths = {route.resource.canonical for route in app.router.routes()}
    assert route_paths == {"/health", "/enrolled", "/identify", "/enroll"}

    raw_request = make_mocked_request(
        "POST",
        "/identify?device_id=den&sample_rate=16000&channels=1",
        app=app,
    )
    raw_request._read_bytes = (b"\x10\x00\xf0\xff") * 800
    result = json.loads((await identify(raw_request)).body)
    assert result["user_id"] == "alice"

    health_result = json.loads(
        (await health(make_mocked_request("GET", "/health", app=app))).body
    )
    assert health_result["ready"] is True
    enrolled_result = json.loads(
        (await enrolled(make_mocked_request("GET", "/enrolled", app=app))).body
    )
    assert enrolled_result["enrolled"][primary.store_key]["alice"] == 1
