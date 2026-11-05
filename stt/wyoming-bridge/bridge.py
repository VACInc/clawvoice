#!/usr/bin/env python3
"""Wyoming ASR server that forwards each utterance to an OpenAI-compatible
/v1/audio/transcriptions HTTP endpoint (the Parakeet server).

Why: HA's Wyoming STT path went tee -> AGX Orin Parakeet (0.3-0.45 s post-stop).
The Parakeet HTTP service answers in ~0.08 s but only speaks HTTP.
This bridge lets the existing wyoming-tee keep its byte-for-byte relay and
speaker-id side channel while the real transcription moves to the fast GPU.

The transcript gets the shared "TARS" name fix (names.py, installed next to
this file from shim/names.py) before HA sees it, so
"Hey Tarz" reaches the conversation agent as "Hey TARS". TARS_NAME_FIX=0 in
the unit environment turns it off.

Usage: bridge.py --uri tcp://127.0.0.1:10302 --http http://127.0.0.1:31411/v1/audio/transcriptions
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import time
import wave
from pathlib import Path

import aiohttp
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

# Deployed: names.py sits beside bridge.py. In the repo: the shared copy in shim/.
_HERE = Path(__file__).resolve().parent
for _dir in (_HERE, _HERE.parents[1] / "shim"):
    if (_dir / "names.py").is_file():
        sys.path.insert(0, str(_dir))
        break
from names import fix_tars_name  # noqa: E402

_LOGGER = logging.getLogger("wyoming_http_bridge")


def build_info(model_name: str) -> Info:
    attribution = Attribution(name="NVIDIA NeMo", url="https://huggingface.co/nvidia")
    return Info(
        asr=[
            AsrProgram(
                name="parakeet-http",
                description="Parakeet via OpenAI-compatible HTTP",
                attribution=attribution,
                installed=True,
                version="1.0",
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=attribution,
                        installed=True,
                        languages=["en"],
                        version="1",
                    )
                ],
            )
        ]
    )


class BridgeHandler(AsyncEventHandler):
    def __init__(self, *args, http_url: str, model_name: str, session: aiohttp.ClientSession,
                 timeout: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.http_url = http_url
        self.model_name = model_name
        self.session = session
        self.timeout = timeout
        self.info = build_info(model_name)
        self.language: str | None = None
        self.rate = 16000
        self.width = 2
        self.channels = 1
        self.audio = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if Transcribe.is_type(event.type):
            self.language = Transcribe.from_event(event).language
            return True
        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self.rate, self.width, self.channels = start.rate, start.width, start.channels
            self.audio = bytearray()
            return True
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if chunk.rate:
                self.rate, self.width, self.channels = chunk.rate, chunk.width, chunk.channels
            self.audio.extend(chunk.audio)
            return True
        if AudioStop.is_type(event.type):
            text = await self._transcribe()
            await self.write_event(Transcript(text=text).event())
            self.audio = bytearray()
            return True
        return True

    async def _transcribe(self) -> str:
        if not self.audio:
            return ""
        t0 = time.perf_counter()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(self.channels)
            w.setsampwidth(self.width)
            w.setframerate(self.rate)
            w.writeframes(bytes(self.audio))
        form = aiohttp.FormData()
        form.add_field("file", buf.getvalue(), filename="utterance.wav", content_type="audio/wav")
        form.add_field("model", self.model_name)
        if self.language:
            form.add_field("language", self.language)
        try:
            async with self.session.post(self.http_url, data=form,
                                         timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    _LOGGER.error("STT HTTP %s: %s", resp.status, body)
                    return ""
                data = await resp.json()
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("STT HTTP request failed: %r", err)
            return ""
        heard = (data.get("text") or "").strip()
        text = fix_tars_name(heard)
        _LOGGER.info("transcribed %.2fs audio in %.3fs: %r%s",
                     len(self.audio) / max(1, self.rate * self.width * self.channels),
                     time.perf_counter() - t0, text, " (name fix applied)" if text != heard else "")
        return text


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="tcp://127.0.0.1:10302")
    ap.add_argument("--http", required=True, help="OpenAI-compatible /v1/audio/transcriptions URL")
    ap.add_argument("--model", default="parakeet-tdt-0.6b-v2")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = AsyncServer.from_uri(args.uri)
    connector = aiohttp.TCPConnector(limit=8, keepalive_timeout=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        _LOGGER.info("listening on %s -> %s (%s)", args.uri, args.http, args.model)
        await server.run(
            lambda reader, writer: BridgeHandler(
                reader, writer, http_url=args.http, model_name=args.model,
                session=session, timeout=args.timeout,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
