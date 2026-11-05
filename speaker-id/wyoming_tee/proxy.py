"""Bidirectional Wyoming wire proxy with a passive utterance tee."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Set

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

from .config import TeeConfig
from .dispatch import IdentityDispatcher, Utterance
from .protocol import WyomingFrameError, read_raw_event

_LOGGER = logging.getLogger(__name__)
_SAFE_CAPTURE_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class Dispatcher(Protocol):
    async def process(
        self,
        utterance: Utterance,
        *,
        device_id: str,
        utterance_ts: float,
        capture_user: str = "",
    ) -> None: ...

    async def close(self) -> None: ...


@dataclass
class UtteranceBuffer:
    max_bytes: int
    sample_rate: Optional[int] = None
    width: Optional[int] = None
    channels: Optional[int] = None
    chunks: list[bytes] = field(default_factory=list)
    size: int = 0
    overflowed: bool = False
    invalid_format: bool = False

    def start(self, sample_rate: int, width: int, channels: int) -> None:
        self.sample_rate = sample_rate
        self.width = width
        self.channels = channels
        self.chunks = []
        self.size = 0
        self.overflowed = False
        self.invalid_format = False

    def append(self, chunk: AudioChunk) -> None:
        if self.sample_rate is None:
            self.start(chunk.rate, chunk.width, chunk.channels)
        if (
            chunk.rate != self.sample_rate
            or chunk.width != self.width
            or chunk.channels != self.channels
        ):
            self.invalid_format = True
            return
        if self.overflowed:
            return
        if self.size + len(chunk.audio) > self.max_bytes:
            self.overflowed = True
            self.chunks = []
            self.size = 0
            return
        self.chunks.append(chunk.audio)
        self.size += len(chunk.audio)

    def finish(self) -> Optional[Utterance]:
        if (
            self.sample_rate is None
            or self.width is None
            or self.channels is None
            or not self.chunks
            or self.overflowed
            or self.invalid_format
        ):
            utterance = None
        else:
            utterance = Utterance(
                chunks=tuple(self.chunks),
                sample_rate=self.sample_rate,
                width=self.width,
                channels=self.channels,
            )
        self.sample_rate = None
        self.width = None
        self.channels = None
        self.chunks = []
        self.size = 0
        self.overflowed = False
        self.invalid_format = False
        return utterance


class ConnectionTee:
    def __init__(self, proxy: "TeeProxy"):
        self.proxy = proxy
        self.buffer = UtteranceBuffer(proxy.config.max_utterance_bytes)
        self.device_id = proxy.config.device_id
        self.capture_user = proxy.config.capture_user

    @staticmethod
    def _nested_value(data: Any, key: str) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        value = data.get(key)
        return value if isinstance(value, str) and value.strip() else None

    def _observe_context(self, event: Event) -> None:
        context = event.data.get("context")
        for source in (event.data, context):
            for key in ("device_id", "satellite_id", "site_id"):
                value = self._nested_value(source, key)
                if value:
                    self.device_id = value[:128]
                    break
            capture = self._nested_value(source, "capture")
            if capture:
                if _SAFE_CAPTURE_USER.fullmatch(capture):
                    self.capture_user = capture
                else:
                    _LOGGER.warning("Ignoring invalid capture user")

    def observe(self, event: Event) -> None:
        if event.type == "transcribe":
            # A per-turn capture request must not leak into later turns on a
            # persistent Wyoming connection.
            self.capture_user = self.proxy.config.capture_user
        self._observe_context(event)
        if AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            self.buffer.start(audio_start.rate, audio_start.width, audio_start.channels)
            return
        if AudioChunk.is_type(event.type):
            self.buffer.append(AudioChunk.from_event(event))
            return
        if AudioStop.is_type(event.type):
            utterance = self.buffer.finish()
            if utterance is not None:
                self.proxy.dispatch(
                    utterance,
                    device_id=self.device_id,
                    capture_user=self.capture_user,
                    utterance_ts=time.time(),
                )


class TeeProxy:
    def __init__(self, config: TeeConfig, dispatcher: Optional[Dispatcher] = None):
        self.config = config
        self.dispatcher: Dispatcher = dispatcher or IdentityDispatcher(config)
        self._server: Optional[asyncio.AbstractServer] = None
        self._background: Set[asyncio.Task[None]] = set()
        self._connections: Set[asyncio.Task[Any]] = set()

    @property
    def port(self) -> int:
        if self._server is None or not self._server.sockets:
            return self.config.port
        return int(self._server.sockets[0].getsockname()[1])

    def dispatch(
        self,
        utterance: Utterance,
        *,
        device_id: str,
        capture_user: str,
        utterance_ts: float,
    ) -> None:
        task = asyncio.create_task(
            self.dispatcher.process(
                utterance,
                device_id=device_id,
                capture_user=capture_user,
                utterance_ts=utterance_ts,
            ),
            name="speaker identity dispatch",
        )
        self._background.add(task)
        task.add_done_callback(self._background_done)

    def _background_done(self, task: asyncio.Task[None]) -> None:
        self._background.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            _LOGGER.error(
                "Speaker identity dispatch failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    @staticmethod
    async def _half_close(writer: asyncio.StreamWriter) -> None:
        if writer.can_write_eof():
            with contextlib.suppress(ConnectionError, RuntimeError):
                writer.write_eof()
                await writer.drain()

    async def _relay_client_to_stt(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        tee: ConnectionTee,
    ) -> None:
        while True:
            frame = await read_raw_event(
                reader,
                max_header_bytes=self.config.max_header_bytes,
                max_frame_bytes=self.config.max_frame_bytes,
            )
            if frame is None:
                await self._half_close(writer)
                return
            writer.write(frame.wire_bytes)
            await writer.drain()
            # Inspection happens only after the exact frame has entered the STT socket.
            tee.observe(frame.event)

    async def _relay_stt_to_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        while True:
            data = await reader.read(self.config.relay_buffer_bytes)
            if not data:
                await self._half_close(writer)
                return
            writer.write(data)
            await writer.drain()

    async def _handle_client(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        current = asyncio.current_task()
        if current is not None:
            self._connections.add(current)
        backend_writer: Optional[asyncio.StreamWriter] = None
        relay_tasks: Set[asyncio.Task[None]] = set()
        try:
            backend_reader, backend_writer = await asyncio.open_connection(
                self.config.real_stt_host,
                self.config.real_stt_port,
                limit=self.config.max_header_bytes + 1,
            )
            tee = ConnectionTee(self)
            upstream = asyncio.create_task(
                self._relay_client_to_stt(client_reader, backend_writer, tee),
                name="Wyoming client to STT",
            )
            downstream = asyncio.create_task(
                self._relay_stt_to_client(backend_reader, client_writer),
                name="Wyoming STT to client",
            )
            relay_tasks.update((upstream, downstream))
            done, pending = await asyncio.wait(
                relay_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            error = next(
                (
                    task.exception()
                    for task in done
                    if not task.cancelled() and task.exception() is not None
                ),
                None,
            )
            if error is not None:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise error
            if downstream in done:
                # The backend closed (normally after its final response). It can
                # no longer consume client frames, so end the other relay too.
                upstream.cancel()
                await asyncio.gather(upstream, return_exceptions=True)
            else:
                # A client half-close can still be followed by the backend's
                # final transcript, so allow the downstream direction to drain.
                await downstream
        except (ConnectionError, OSError, asyncio.IncompleteReadError) as exc:
            _LOGGER.warning("Wyoming connection ended: %s", type(exc).__name__)
        except WyomingFrameError:
            _LOGGER.warning("Rejected malformed Wyoming frame")
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("Unexpected Wyoming proxy error")
        finally:
            for relay_task in relay_tasks:
                if not relay_task.done():
                    relay_task.cancel()
            if relay_tasks:
                await asyncio.gather(*relay_tasks, return_exceptions=True)
            if backend_writer is not None:
                backend_writer.close()
                with contextlib.suppress(ConnectionError):
                    await backend_writer.wait_closed()
            client_writer.close()
            with contextlib.suppress(ConnectionError):
                await client_writer.wait_closed()
            if current is not None:
                self._connections.discard(current)

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.config.host,
            port=self.config.port,
            limit=self.config.max_header_bytes + 1,
        )
        _LOGGER.info(
            "Wyoming tee listening on %s:%d and forwarding to %s:%d",
            self.config.host,
            self.port,
            self.config.real_stt_host,
            self.config.real_stt_port,
        )

    async def run(self) -> None:
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._connections:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tuple(self._connections), return_exceptions=True),
                    timeout=self.config.shutdown_grace_seconds,
                )
            except asyncio.TimeoutError:
                for task in tuple(self._connections):
                    task.cancel()
                await asyncio.gather(*tuple(self._connections), return_exceptions=True)

        if self._background:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tuple(self._background), return_exceptions=True),
                    timeout=self.config.shutdown_grace_seconds,
                )
            except asyncio.TimeoutError:
                for task in tuple(self._background):
                    task.cancel()
                await asyncio.gather(*tuple(self._background), return_exceptions=True)
        await self.dispatcher.close()
