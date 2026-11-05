#!/usr/bin/env python3
"""HA voice route: the Wyoming bridge hands HA the name-fixed transcript.

Inputs are real Parakeet :31411 output from the 2026-11-02 HA route probe
(Chatterbox speech resampled to 16 kHz, the format HA's Wyoming stream
carries). The HTTP call is faked; no network, no live STT.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from wyoming.asr import Transcript  # noqa: E402
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402

import bridge  # noqa: E402  (finds the shared names.py in shim/)
import names  # noqa: E402

assert Path(names.__file__).resolve() == HERE.parents[2] / "shim" / "names.py"

HEARD = {
    "Hey Taris, turn on the kitchen lights.": "Hey TARS, turn on the kitchen lights.",
    "Thanks, Tarz.": "Thanks, TARS.",
    "Good morning, Tarz.": "Good morning, TARS.",
    "Hey, Tars, turn on the kitchen lights.": "Hey, TARS, turn on the kitchen lights.",
    "The stars are bright tonight, Taras.": "The stars are bright tonight, TARS.",
    "Hey Taurus, what time is it?": "Hey TARS, what time is it?",
    # Real words outside an address stay.
    "I love looking at the stars at night.": "I love looking at the stars at night.",
    "Give it five stars!": "Give it five stars!",
}


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.status = 200
        self._text = text

    async def json(self):
        return {"text": self._text}

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def post(self, url, data=None, timeout=None):
        self.calls += 1
        return FakeResponse(self.text)


class Reader:
    async def readline(self):
        return b""


class Writer:
    def write(self, data):
        pass

    def writelines(self, data):
        pass

    async def drain(self):
        pass

    def close(self):
        pass


async def run_turn(heard: str) -> str:
    session = FakeSession(heard)
    handler = bridge.BridgeHandler(
        Reader(), Writer(), http_url="http://stt.invalid/v1/audio/transcriptions",
        model_name="parakeet-tdt-0.6b-v2", session=session, timeout=5.0,
    )
    sent = []

    async def write_event(event):
        sent.append(event)

    handler.write_event = write_event
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x01" * 800).event())
    await handler.handle_event(AudioStop().event())
    assert session.calls == 1
    transcripts = [Transcript.from_event(e).text for e in sent if Transcript.is_type(e.type)]
    assert len(transcripts) == 1, sent
    return transcripts[0]


class BridgeNameFixTests(unittest.TestCase):
    def test_ha_gets_tars(self):
        for heard, want in HEARD.items():
            with self.subTest(heard=heard):
                self.assertEqual(asyncio.run(run_turn(heard)), want)

    def test_kill_switch(self):
        clean = {k: v for k, v in os.environ.items() if k not in ("TARS_NAME_FIX", "RTSHIM_NAME_FIX")}
        try:
            with mock.patch.dict(os.environ, {**clean, "TARS_NAME_FIX": "0"}, clear=True):
                importlib.reload(names)
                importlib.reload(bridge)
                self.assertEqual(asyncio.run(run_turn("Thanks, Tarz.")), "Thanks, Tarz.")
        finally:
            importlib.reload(names)
            importlib.reload(bridge)
        self.assertEqual(asyncio.run(run_turn("Thanks, Tarz.")), "Thanks, TARS.")


if __name__ == "__main__":
    unittest.main()
