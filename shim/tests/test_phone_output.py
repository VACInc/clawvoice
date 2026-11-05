#!/usr/bin/env python3
"""Hermetic phone output tests: 20 ms frames, short lead, TTS prefetch, barge-in
leftovers, and the hold while caller audio is stalled.

Regression for the 2026-11-02 local phone test call: the shim ran 1.5 s ahead of
playback, so after a barge-in up to 1.5 s of stale reply audio sat in the Gateway
queue and kept playing; a 4.4 s Gateway stall then burst stale audio past the
caller's interruption. No live STT, LLM, TTS, or websocket client. Energy VAD only.
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests.net_guard as net_guard

net_guard.install()


def setUpModule():
    net_guard.install()


os.environ["RTSHIM_VAD"] = "energy"

import numpy as np  # noqa: E402
from aiohttp.test_utils import make_mocked_request  # noqa: E402

import server  # noqa: E402
from phone import ulaw_decode, ulaw_encode  # noqa: E402
from server import (  # noqa: E402
    PHONE_FRAME_BYTES,
    PhoneFrameBuffer,
    PlayoutClock,
    RealtimeSession,
    phone_input_stalled,
)

FRAME = 0.02
LEAD = 0.1


def tone24(seconds: float, freq: float = 700.0) -> bytes:
    t = np.arange(int(24000 * seconds)) / 24000
    return (6000 * np.sin(2 * np.pi * freq * t)).astype("<i2").tobytes()


class TimedWS:
    def __init__(self) -> None:
        self.closed = False
        self.sent: list[tuple[float, dict]] = []

    async def send_json(self, event: dict) -> None:
        self.sent.append((asyncio.get_running_loop().time(), event))

    def audio(self) -> list[tuple[float, bytes]]:
        return [
            (t, base64.b64decode(e["delta"]))
            for t, e in self.sent
            if e.get("type") in {"response.output_audio.delta", "response.audio.delta"}
        ]


PHONE_UPDATE = {
    "type": "session.update",
    "session": {
        "type": "realtime",
        "audio": {
            "input": {"format": {"type": "audio/pcmu"}},
            "output": {"format": {"type": "audio/pcmu"}, "voice": "tars"},
        },
    },
}


async def phone_session(ws: TimedWS) -> RealtimeSession:
    session = RealtimeSession(make_mocked_request("GET", "/openai/v1/realtime?model=tars-local"), ws)
    await session.handle(PHONE_UPDATE)
    ws.sent.clear()
    return session


def run(coro):
    return asyncio.run(coro)


class Patched(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = (server.PHONE_OUTPUT_LEAD_SECONDS, server.PHONE_INPUT_STALL_SECONDS, server.PHONE_STALL_HOLD_MAX_SECONDS)
        server.PHONE_OUTPUT_LEAD_SECONDS = LEAD
        server.PHONE_INPUT_STALL_SECONDS = 0.1
        server.PHONE_STALL_HOLD_MAX_SECONDS = 5.0

    def tearDown(self) -> None:
        server.PHONE_OUTPUT_LEAD_SECONDS, server.PHONE_INPUT_STALL_SECONDS, server.PHONE_STALL_HOLD_MAX_SECONDS = self.saved


class FrameAndClockUnitTests(unittest.TestCase):
    def test_frame_buffer_splits_whole_frames_and_pads_tail_with_silence(self):
        buf = PhoneFrameBuffer()
        buf.extend(b"\x01" * 170)
        self.assertEqual(buf.pop_frame(final=False), b"\x01" * 160)
        self.assertIsNone(buf.pop_frame(final=False))
        self.assertEqual(buf.pop_frame(final=True), b"\x01" * 10 + b"\xff" * 150)
        self.assertIsNone(buf.pop_frame(final=True))

    def test_clock_holds_lead_and_restarts_without_debt_after_gap(self):
        clock = PlayoutClock(0.1)
        self.assertEqual(clock.delay(10.0), 0.0)
        for _ in range(5):
            clock.sent(10.0)  # 100 ms queued at t=10
        self.assertAlmostEqual(clock.unplayed(10.0), 0.1)
        self.assertAlmostEqual(clock.delay(10.0), 0.0)
        clock.sent(10.0)
        self.assertAlmostEqual(clock.delay(10.0), 0.02)
        self.assertEqual(clock.gaps, 0)
        # Line ran dry for 0.5 s: one gap, and the clock restarts at now (no burst).
        clock.sent(10.62)
        self.assertEqual(clock.gaps, 1)
        self.assertAlmostEqual(clock.max_gap, 0.5)
        self.assertAlmostEqual(clock.unplayed(10.62), 0.02)
        self.assertEqual(clock.frames, 7)

    def test_input_stall_needs_prior_input(self):
        self.assertFalse(phone_input_stalled(None, 100.0, stall=0.25))
        self.assertFalse(phone_input_stalled(99.8, 100.0, stall=0.25))
        self.assertTrue(phone_input_stalled(99.7, 100.0, stall=0.25))


class PhoneOutputTests(Patched):
    def test_frames_are_20ms_and_never_more_than_lead_ahead(self):
        async def checks():
            ws = TimedWS()
            session = await phone_session(ws)
            try:
                async def fake_tts(text):
                    yield tone24(0.5)  # TTS far faster than realtime

                async def fake_chat(messages, tools=None):
                    yield {"type": "content", "text": "Hello there."}

                session.pipeline.stream_tts_pcm = fake_tts
                session.pipeline.stream_chat = fake_chat
                await session.run_response("resp_pace")
                audio = ws.audio()
                self.assertEqual(len(audio), 25, len(audio))  # 0.5 s = 25 frames
                self.assertTrue(all(len(frame) == PHONE_FRAME_BYTES for _, frame in audio))
                start = audio[0][0]
                for k, (t, _) in enumerate(audio):
                    # Frame k plays at start + k*20ms; it may leave at most LEAD early.
                    self.assertGreaterEqual(t, start + k * FRAME - LEAD - 0.012, (k, t - start))
                self.assertLess(audio[-1][0] - start, 0.5)
                self.assertEqual(session.output_clock.gaps, 0)
                pcm = ulaw_decode(b"".join(frame for _, frame in audio))
                self.assertEqual(len(pcm), 4000)
            finally:
                await session.close()

        run(checks())

    def test_next_sentence_tts_is_prefetched_so_slow_tts_leaves_no_gap(self):
        async def checks():
            ws = TimedWS()
            session = await phone_session(ws)
            try:
                async def slow_tts(text):
                    await asyncio.sleep(0.15)  # per-sentence first-chunk latency > lead
                    yield tone24(0.3)

                async def fake_chat(messages, tools=None):
                    yield {"type": "content", "text": "First one. Second one. Third one."}

                session.pipeline.stream_tts_pcm = slow_tts
                session.pipeline.stream_chat = fake_chat
                await session.run_response("resp_prefetch")
                self.assertEqual(len(ws.audio()), 45)
                self.assertEqual(session.output_clock.gaps, 0, session.output_clock.max_gap)
            finally:
                await session.close()

        run(checks())

    def test_barge_in_leaves_at_most_the_lead_unplayed_and_stops_frames(self):
        async def checks():
            ws = TimedWS()
            session = await phone_session(ws)
            try:
                async def long_tts(text):
                    yield tone24(3.0)

                async def fake_chat(messages, tools=None):
                    yield {"type": "content", "text": "A long answer."}

                session.pipeline.stream_tts_pcm = long_tts
                session.pipeline.stream_chat = fake_chat
                await session.start_response()
                await asyncio.sleep(0.3)
                loop = asyncio.get_running_loop()
                # Caller audio keeps flowing (no stall); now the caller talks over TARS.
                session.last_input_at = loop.time()
                cut = loop.time()
                await session.cancel_active("barge_in", emit_no_active_error=False)
                audio = ws.audio()
                start = audio[0][0]
                sent_s = len(audio) * FRAME
                played_s = cut - start
                self.assertLessEqual(sent_s - played_s, LEAD + 0.03, (sent_s, played_s))
                count = len(audio)
                await asyncio.sleep(0.1)
                self.assertEqual(len(ws.audio()), count)
                types = [e.get("type") for _, e in ws.sent]
                self.assertIn("response.cancelled", types)
            finally:
                await session.close()

        run(checks())

    def test_output_holds_while_caller_audio_is_stalled_and_resumes_without_burst(self):
        async def checks():
            ws = TimedWS()
            session = await phone_session(ws)
            loop = asyncio.get_running_loop()
            try:
                async def fake_tts(text):
                    yield tone24(1.0)

                async def fake_chat(messages, tools=None):
                    yield {"type": "content", "text": "Hi."}

                session.pipeline.stream_tts_pcm = fake_tts
                session.pipeline.stream_chat = fake_chat
                wire = base64.b64encode(ulaw_encode(np.zeros(160, dtype=np.int16))).decode()
                await session.handle({"type": "input_audio_buffer.append", "audio": wire})
                ws.sent.clear()
                await session.start_response()
                await asyncio.sleep(0.2)
                before_stall = len(ws.audio())
                self.assertGreater(before_stall, 0)
                # Gateway stall: caller frames stopped 0.2 s ago (> 0.1 s stall limit).
                await asyncio.sleep(0.3)
                during = len(ws.audio())
                self.assertEqual(during, before_stall)  # held since 0.1 s after the last caller frame
                resumed = loop.time()
                await session.handle({"type": "input_audio_buffer.append", "audio": wire})
                await asyncio.sleep(0.05)
                after = [t for t, _ in ws.audio() if t >= resumed]
                # No catch-up burst for the stalled time: at most the lead plus the elapsed 50 ms.
                self.assertLessEqual(len(after), (LEAD + 0.05) / FRAME + 2, len(after))
                self.assertGreater(session.output_clock.held, 0.0)
            finally:
                await session.close()

        run(checks())

    def test_pcm16_talk_path_keeps_tts_chunks(self):
        async def checks():
            ws = TimedWS()
            session = RealtimeSession(make_mocked_request("GET", "/v1/realtime"), ws)
            try:
                pcm = tone24(0.1)

                async def fake_tts(text):
                    yield pcm

                async def fake_chat(messages, tools=None):
                    yield {"type": "content", "text": "Hi."}

                session.pipeline.stream_tts_pcm = fake_tts
                session.pipeline.stream_chat = fake_chat
                await session.run_response("resp_talk")
                self.assertEqual([frame for _, frame in ws.audio()], [pcm])
                self.assertIsNone(session.output_clock)
            finally:
                await session.close()

        run(checks())


if __name__ == "__main__":
    unittest.main()
