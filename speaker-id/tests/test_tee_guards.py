"""Hermetic guards for webhook retry, frame limits, utterance buffer, and tee config."""

from __future__ import annotations

import asyncio

import pytest
from wyoming.audio import AudioChunk
from wyoming.event import Event

from wyoming_tee.config import TeeConfig
from wyoming_tee.dispatch import IdentityDispatcher, Utterance, WebhookPoster
from wyoming_tee.protocol import WyomingFrameError, read_raw_event
from wyoming_tee.proxy import ConnectionTee, UtteranceBuffer

from .helpers import tee_config


class _Response:
    def __init__(self, status: int, *, body=None, headers=None):
        self.status = status
        self.headers = headers or {}
        self.body = body if body is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def read(self) -> bytes:
        return b""

    async def json(self, content_type=None):
        return self.body


class _Session:
    def __init__(self, statuses, *, headers=None, body=None):
        self.statuses = list(statuses)
        self.headers = headers or {}
        self.body = body if body is not None else {}
        self.calls = 0
        self.payloads = []

    def post(self, url, **kwargs):
        self.calls += 1
        self.payloads.append(kwargs.get("json"))
        index = min(self.calls - 1, len(self.statuses) - 1)
        return _Response(self.statuses[index], body=self.body, headers=self.headers)


class _Router:
    def __init__(self, embed_status: int, embed_body):
        self.embed_status = embed_status
        self.embed_body = embed_body
        self.urls: list[str] = []
        self.stamps: list[dict] = []

    def post(self, url, **kwargs):
        self.urls.append(url)
        if "identify" in url:
            return _Response(self.embed_status, body=self.embed_body)
        self.stamps.append(dict(kwargs.get("json") or {}))
        return _Response(200)


def _poster(statuses, **config):
    values = dict(
        ha_webhook_url="http://127.0.0.1/hook",
        webhook_retries=2,
        webhook_retry_base_seconds=0.01,
        webhook_retry_max_seconds=0.05,
    )
    values.update(config)
    session = _Session(statuses, headers=values.pop("headers", None))
    return WebhookPoster(session, tee_config(1, **values)), session


async def _no_sleep(delay):
    return None


async def _inline(function, *args, **kwargs):
    return function(*args, **kwargs)


def _reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


@pytest.mark.asyncio
async def test_webhook_status_retry_and_retry_after_cap(monkeypatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    ok, ok_session = _poster([200])
    assert await ok.post({"user_id": "alice"}) is True
    assert ok_session.calls == 1

    edge, _session = _poster([299])
    assert await edge.post({"user_id": "alice"}) is True

    three, three_session = _poster([300])
    assert await three.post({"user_id": "alice"}) is False
    assert three_session.calls == 1

    low, low_session = _poster([199])
    assert await low.post({"user_id": "alice"}) is False
    assert low_session.calls == 1

    missing, missing_session = _poster([404])
    assert await missing.post({"user_id": "alice"}) is False
    assert missing_session.calls == 1

    busy, busy_session = _poster([500, 500, 500])
    assert await busy.post({"user_id": "alice"}) is False
    assert busy_session.calls == 3

    limited, limited_session = _poster([429, 429, 429])
    assert await limited.post({"user_id": "alice"}) is False
    assert limited_session.calls == 3

    slept = []

    async def record(delay):
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record)
    clamped, _clamped_session = _poster(
        [429, 200],
        headers={"Retry-After": "10"},
        webhook_retries=2,
        webhook_retry_max_seconds=0.05,
    )
    assert await clamped.post({"user_id": "alice"}) is True
    assert slept == [0.05]

    empty = WebhookPoster( _Session([200]), tee_config(1, ha_webhook_url=""))
    assert await empty.post({"user_id": "alice"}) is False


@pytest.mark.asyncio
async def test_embed_results_and_capture_user(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(asyncio, "to_thread", _inline)
    utterance = Utterance((b"\x00\x00",), 16000, 2, 1)
    wide = Utterance((b"\x00\x00\x00",), 16000, 3, 1)

    async def run(status, body, *, shadow=True, capture=""):
        router = _Router(status, body)
        config = tee_config(
            1,
            ha_webhook_url="http://127.0.0.1/hook",
            shadow=shadow,
            capture_dir=tmp_path,
        )
        dispatcher = IdentityDispatcher(config, router)
        await dispatcher.process(
            wide if status == 0 else utterance,
            device_id="kitchen",
            utterance_ts=1.0,
            capture_user=capture,
        )
        return router

    skipped = await run(0, {})
    assert skipped.urls == []

    rejected = await run(300, {"user_id": "alice", "confidence": 1})
    assert rejected.stamps == []
    assert len(rejected.urls) == 1

    accepted = await run(200, {"user_id": "alice", "confidence": 1, "enforced": True})
    assert accepted.stamps[0]["user_id"] == "alice"
    assert accepted.stamps[0]["enforced"] is False

    odd = await run(200, ["alice"])
    assert odd.stamps == []

    text = await run(200, {"user_id": "alice", "confidence": "0.9"})
    assert text.stamps == []

    live = await run(200, {"user_id": "alice", "confidence": 0.5}, shadow=False)
    assert live.stamps[0]["enforced"] is False

    saved = IdentityDispatcher._write_capture(tmp_path, "alice", b"RIFF")
    assert saved.parent == tmp_path / "alice"
    assert saved.read_bytes() == b"RIFF"
    with pytest.raises(ValueError):
        IdentityDispatcher._write_capture(tmp_path, "../x", b"RIFF")


@pytest.mark.asyncio
async def test_frame_limits_types_and_null_data() -> None:
    header = b'{"type":"ping","data_length":null,"payload_length":0}\n'
    frame = await read_raw_event(
        _reader(header), max_header_bytes=1024, max_frame_bytes=4096
    )
    assert frame is not None
    assert frame.event.type == "ping"
    assert frame.event.data == {}

    with pytest.raises(WyomingFrameError):
        await read_raw_event(
            _reader(b'{"type":"ping","data_length":true,"payload_length":0}\n'),
            max_header_bytes=1024,
            max_frame_bytes=4096,
        )
    with pytest.raises(WyomingFrameError):
        await read_raw_event(
            _reader(b'{"type":1,"data_length":0,"payload_length":0}\n'),
            max_header_bytes=1024,
            max_frame_bytes=4096,
        )

    big = b'{"type":"ping","data_length":0,"payload_length":0}\n'
    with pytest.raises(WyomingFrameError):
        await read_raw_event(
            _reader(big), max_header_bytes=len(big) - 1, max_frame_bytes=10_000
        )

    heavy = b'{"type":"ping","data_length":40,"payload_length":0}\n'
    with pytest.raises(WyomingFrameError):
        await read_raw_event(
            _reader(heavy),
            max_header_bytes=10_000,
            max_frame_bytes=len(heavy) + 39,
        )

    blank = await read_raw_event(
        _reader(b'{"type":"ping","data":null}\n'),
        max_header_bytes=1024,
        max_frame_bytes=4096,
    )
    assert blank is not None
    assert blank.event.data == {}

    extra = b"[1]"
    listed = (
        b'{"type":"ping","data_length":'
        + str(len(extra)).encode()
        + b',"payload_length":0}\n'
        + extra
    )
    with pytest.raises(WyomingFrameError):
        await read_raw_event(
            _reader(listed), max_header_bytes=1024, max_frame_bytes=4096
        )


def test_buffer_exact_overflow_and_format() -> None:
    exact = UtteranceBuffer(4)
    exact.append(AudioChunk(rate=16000, width=2, channels=1, audio=b"abcd"))
    kept = exact.finish()
    assert kept is not None
    assert kept.pcm == b"abcd"
    assert kept.sample_rate == 16000

    overflow = UtteranceBuffer(4)
    overflow.append(AudioChunk(rate=16000, width=2, channels=1, audio=b"ab"))
    overflow.append(AudioChunk(rate=16000, width=2, channels=1, audio=b"cdef"))
    assert overflow.finish() is None

    mismatched = UtteranceBuffer(100)
    mismatched.start(16000, 2, 1)
    mismatched.append(AudioChunk(rate=16000, width=2, channels=1, audio=b"ab"))
    mismatched.append(AudioChunk(rate=8000, width=2, channels=1, audio=b"cd"))
    assert mismatched.finish() is None


def test_context_truncation_capture_and_transcribe_reset() -> None:
    class _Proxy:
        def __init__(self):
            self.config = tee_config(1, device_id="sat", capture_user="")

        def dispatch(self, *args, **kwargs):
            return None

    tee = ConnectionTee(_Proxy())
    tee.observe(Event(type="info", data={"device_id": "d" * 200}))
    assert tee.device_id == "d" * 128
    tee.observe(Event(type="info", data={"device_id": "   "}))
    assert tee.device_id == "d" * 128
    tee.observe(Event(type="info", data={"capture": "alice"}))
    assert tee.capture_user == "alice"
    tee.observe(Event(type="info", data={"capture": "../x"}))
    assert tee.capture_user == "alice"
    tee.observe(Event(type="transcribe", data={}))
    assert tee.capture_user == ""


def test_tee_config_bounds_and_urls(tmp_path) -> None:
    assert len(TeeConfig.from_env({"DEVICE_ID": "d" * 128}).device_id) == 128
    with pytest.raises(ValueError):
        TeeConfig.from_env({"DEVICE_ID": "d" * 129})
    assert TeeConfig.from_env({"CAPTURE_USER": "alice"}).capture_user == "alice"
    with pytest.raises(ValueError):
        TeeConfig.from_env({"CAPTURE_USER": "../x"})
    with pytest.raises(ValueError):
        TeeConfig.from_env({"EMBED_URL": "ftp://example.com/x"})
    assert TeeConfig.from_env({"HA_WEBHOOK_URL": ""}).ha_webhook_url == ""
    with pytest.raises(ValueError):
        TeeConfig.from_env({"HA_WEBHOOK_URL": "not-a-url"})
    assert TeeConfig.from_env({"SHADOW": "yes"}).shadow is True
    assert TeeConfig.from_env({"SHADOW": "off"}).shadow is False
    assert TeeConfig.from_env({"CAPTURE_DIR": str(tmp_path)}).capture_dir == tmp_path
