from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcribe
from wyoming.event import async_read_event, async_write_event

from wyoming_tee.dispatch import IdentityDispatcher, Utterance
from wyoming_tee.proxy import ConnectionTee, TeeProxy

from .helpers import tee_config


class MemoryWriter:
    def __init__(self):
        self.data = bytearray()
        self.eof = False
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def can_write_eof(self) -> bool:
        return True

    def write_eof(self) -> None:
        self.eof = True

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


def memory_reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


def exact_transcript_frame(text: str = "bytes stay exact") -> bytes:
    data = json.dumps({"text": text}, separators=(",", ":")).encode()
    # Deliberate key order and whitespace prove the tee did not reserialize it.
    header = (
        b'{ "payload_length": 0, "type": "transcript", "data_length": '
        + str(len(data)).encode()
        + b', "version": "1.10.0" }\n'
    )
    return header + data


async def utterance_wire_bytes(pcm: bytes) -> bytes:
    writer = MemoryWriter()
    await async_write_event(Transcribe(language="en").event(), writer)
    await async_write_event(AudioStart(rate=16000, width=2, channels=1).event(), writer)
    await async_write_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=pcm).event(), writer
    )
    await async_write_event(AudioStop().event(), writer)
    return bytes(writer.data)


class RecordingDispatcher:
    def __init__(self):
        self.calls: list[tuple[Utterance, str, float, str]] = []
        self.called = asyncio.Event()

    async def process(
        self,
        utterance: Utterance,
        *,
        device_id: str,
        utterance_ts: float,
        capture_user: str = "",
    ) -> None:
        self.calls.append((utterance, device_id, utterance_ts, capture_user))
        self.called.set()

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_loopback_transcript_wire_bytes_are_identical_with_tee() -> None:
    pcm = (b"\x01\x00\xff\xff") * 800
    client_wire = await utterance_wire_bytes(pcm)
    expected_transcript = exact_transcript_frame()
    dispatcher = RecordingDispatcher()
    proxy = TeeProxy(tee_config(10300), dispatcher)

    # Client -> tee -> STT: exact input frames are forwarded while the tee observes
    # the Wyoming audio events and dispatches a copy.
    stt_input = MemoryWriter()
    await proxy._relay_client_to_stt(
        memory_reader(client_wire), stt_input, ConnectionTee(proxy)
    )
    await asyncio.wait_for(dispatcher.called.wait(), timeout=1.0)
    assert bytes(stt_input.data) == client_wire

    parsed_types = []
    parsed_reader = memory_reader(bytes(stt_input.data))
    while True:
        event = await async_read_event(parsed_reader)
        if event is None:
            break
        parsed_types.append(event.type)

    # STT -> tee -> client: response bytes are copied as an opaque stream.
    client_output = MemoryWriter()
    await proxy._relay_stt_to_client(memory_reader(expected_transcript), client_output)
    await proxy.close()

    assert bytes(client_output.data) == expected_transcript
    assert parsed_types == [
        "transcribe",
        "audio-start",
        "audio-chunk",
        "audio-stop",
    ]
    assert dispatcher.calls[0][0].pcm == pcm
    assert dispatcher.calls[0][1] == "test-satellite"


class FakeResponse:
    def __init__(
        self,
        status: int,
        *,
        json_value: Optional[dict[str, Any]] = None,
        entered: Optional[asyncio.Event] = None,
        release: Optional[asyncio.Event] = None,
    ):
        self.status = status
        self.headers: dict[str, str] = {}
        self.json_value = json_value
        self.entered = entered
        self.release = release

    async def __aenter__(self) -> "FakeResponse":
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        pass

    async def read(self) -> bytes:
        return b""

    async def json(self, content_type=None) -> dict[str, Any]:
        assert self.json_value is not None
        return self.json_value


class FakeHttpSession:
    def __init__(self):
        self.ha_entered = asyncio.Event()
        self.release_ha = asyncio.Event()
        self.ha_succeeded = asyncio.Event()
        self.stamps: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs):
        if url.endswith("/identify"):
            return FakeResponse(
                200,
                json_value={
                    "user_id": "alice",
                    "confidence": 0.93,
                    "model": "fake",
                    "ms": 1.0,
                    "enforced": True,
                },
            )
        stamp = dict(kwargs["json"])
        self.stamps.append(stamp)
        if len(self.stamps) == 1:
            return FakeResponse(503, entered=self.ha_entered, release=self.release_ha)
        self.ha_succeeded.set()
        return FakeResponse(200)


@pytest.mark.asyncio
async def test_webhook_retries_while_transcript_path_remains_unblocked(monkeypatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 1234.0)
    real_sleep = asyncio.sleep

    async def yield_only(_delay):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", yield_only)
    expected = exact_transcript_frame("retry proof")
    fake_session = FakeHttpSession()
    config = tee_config(
        10300,
        embed_url="http://embed.test/identify",
        ha_webhook_url="http://ha.test/hook",
        webhook_retry_base_seconds=0.001,
        webhook_retry_max_seconds=0.001,
    )
    dispatcher = IdentityDispatcher(config, session=fake_session)  # type: ignore[arg-type]
    proxy = TeeProxy(config, dispatcher)
    pcm = (b"\x02\x00") * 1600

    # The stop frame schedules identity work; the relay itself completes without
    # awaiting embed or webhook I/O.
    original_to_thread = asyncio.to_thread

    async def inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    # The execution sandbox cannot shut down executor threads. Production uses
    # the real asyncio.to_thread path tested structurally above.
    asyncio.to_thread = inline_to_thread  # type: ignore[assignment]
    try:
        stt_input = MemoryWriter()
        await proxy._relay_client_to_stt(
            memory_reader(await utterance_wire_bytes(pcm)),
            stt_input,
            ConnectionTee(proxy),
        )
        await asyncio.wait_for(fake_session.ha_entered.wait(), timeout=1.0)

        client_output = MemoryWriter()
        await asyncio.wait_for(
            proxy._relay_stt_to_client(memory_reader(expected), client_output),
            timeout=0.1,
        )
        assert bytes(client_output.data) == expected
        assert not fake_session.release_ha.is_set()

        fake_session.release_ha.set()
        await asyncio.wait_for(fake_session.ha_succeeded.wait(), timeout=1.0)
        await proxy.close()
    finally:
        asyncio.to_thread = original_to_thread  # type: ignore[assignment]

    assert len(fake_session.stamps) == 2
    assert fake_session.stamps[0] == fake_session.stamps[1]
    stamp = fake_session.stamps[-1]
    assert stamp["device_id"] == "test-satellite"
    assert stamp["user_id"] == "alice"
    assert stamp["confidence"] == pytest.approx(0.93)
    assert stamp["enforced"] is False
    assert stamp["ts"] == 1234.0
