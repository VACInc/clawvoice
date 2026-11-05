#!/usr/bin/env python3
"""Hermetic tests for the post-STT "TARS" name fix.

Inputs are real Parakeet output: the 2026-11-02 phone call records ("Thank you,
Stars.", "Can you hear me, Tarz?", "Thanks, Nars.") and the live TTS -> Parakeet
probe from the same day ("Thank you Tors.", "Thanks, Taras.", "Hey Tarz, ...").
No live STT, LLM, TTS, or websocket client.
"""

from __future__ import annotations

import asyncio
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

from aiohttp.test_utils import make_mocked_request  # noqa: E402

import names  # noqa: E402
from names import fix_tars_name  # noqa: E402
from server import RealtimeSession  # noqa: E402

FIXED = {
    # 2026-11-02 phone call records (voice-call plugin transcript)
    "Thank you, Stars.": "Thank you, TARS.",
    "Can you hear me, Tarz?": "Can you hear me, TARS?",
    "Ah, you did good. Thanks, Nars.": "Ah, you did good. Thanks, TARS.",
    # 2026-11-02 live probe: Chatterbox "Tars" -> (phone mu-law) -> Parakeet
    "Thank you Tors.": "Thank you TARS.",
    "Thank you Tars.": "Thank you TARS.",
    "Thanks, Tarz. Though n": "Thanks, TARS. Though n",
    "Hey Tarz, what's the weather today?": "Hey TARS, what's the weather today?",
    "Hey Tars, what's the weather today?": "Hey TARS, what's the weather today?",
    "Tarz, turn off the office lights.": "TARS, turn off the office lights.",
    "Tars?": "TARS?",
    "Okay, Tars, set a timer for 10 minutes.": "Okay, TARS, set a timer for 10 minutes.",
    "Ah, you did good. Thanks, Taras.": "Ah, you did good. Thanks, TARS.",
    "Okay Terse, set a timer for 10 minutes.": "Okay TARS, set a timer for 10 minutes.",
    "Okay Terz, set a timer for 10 minutes.": "Okay TARS, set a timer for 10 minutes.",
    "Hey Taris, what's the weather today?": "Hey TARS, what's the weather today?",
    # Addressed forms of the real-word mishearings
    "Stars, turn off the lights.": "TARS, turn off the lights.",
    "Stars?": "TARS?",
    "Hey stars what's up": "Hey TARS what's up",
    "Okay, Stars, set a timer.": "Okay, TARS, set a timer.",
    "Good night, Star.": "Good night, TARS.",
    "Thanks Tar's.": "Thanks TARS.",
    "Thanks, Tar’s.": "Thanks, TARS.",
    "Is that right, Tsar?": "Is that right, TARS?",
    "tarrs": "TARS",
    "Ask tarz about it.": "Ask TARS about it.",
    # 2026-11-02 HA route probe: Chatterbox (two different voices) -> 16 kHz WAV
    # (what HA's Wyoming stream carries) -> Parakeet :31411
    "Hey, Tars, turn on the kitchen lights.": "Hey, TARS, turn on the kitchen lights.",
    "Tars, what's the weather to-morrow?": "TARS, what's the weather to-morrow?",
    "Thanks, Tarz.": "Thanks, TARS.",
    "Good morning, Tarz.": "Good morning, TARS.",
    "Hey Taris, turn on the kitchen lights.": "Hey TARS, turn on the kitchen lights.",
    "The stars are bright tonight, Tars.": "The stars are bright tonight, TARS.",
    "The stars are bright tonight, Taras.": "The stars are bright tonight, TARS.",
    "Hey Taurus, what time is it?": "Hey TARS, what time is it?",
    "Thanks, Taurus.": "Thanks, TARS.",
}

UNCHANGED = [
    "I love looking at the stars at night.",
    "The stars are really bright tonight.",
    "She is a rising star in the company.",
    "Give it five stars.",
    "We saw the sun, moon, stars.",
    "Stars, planets, and moons all orbit something.",
    "Hey, stars are out tonight.",
    "To go back, press star.",
    "Star.",
    "Tarzan was on TV.",
    "Starsky and Hutch.",
    "The roof needs tar.",
    "Keep it terse, please.",
    "TARS, you there?",
    "",
    # 2026-11-02 HA route probe controls (same 16 kHz Parakeet path)
    "I love looking at the stars at night",
    "Give it five stars!",
    "I'm a Taurus.",
    "Taurus is my sign, and Leo is hers.",
]


class NameFixTests(unittest.TestCase):
    def test_real_transcripts_become_tars(self):
        for heard, want in FIXED.items():
            with self.subTest(heard=heard):
                self.assertEqual(fix_tars_name(heard), want)

    def test_real_words_stay(self):
        for heard in UNCHANGED:
            with self.subTest(heard=heard):
                self.assertEqual(fix_tars_name(heard), heard)

    def test_list_guard_is_per_sentence(self):
        self.assertEqual(fix_tars_name("Sun, moon. Thanks, Stars."), "Sun, moon. Thanks, TARS.")
        self.assertEqual(fix_tars_name("A, b. Ok. Right, Stars."), "A, b. Ok. Right, TARS.")
        self.assertEqual(fix_tars_name("Yes. Sun, moon, stars!"), "Yes. Sun, moon, stars!")
        self.assertEqual(fix_tars_name("Right, stars."), "Right, TARS.")

    def test_second_pass_changes_nothing(self):
        # A route that already fixed the text (or a future double hop) must not drift.
        for text in list(FIXED.values()) + UNCHANGED:
            with self.subTest(text=text):
                self.assertEqual(fix_tars_name(fix_tars_name(text)), fix_tars_name(text))

    def test_env_kill_switches(self):
        import importlib
        from unittest import mock

        try:
            for env, want in (
                ({}, "Thanks, TARS."),
                ({"TARS_NAME_FIX": "0"}, "Thanks, Tarz."),
                ({"RTSHIM_NAME_FIX": "0"}, "Thanks, Tarz."),
                ({"TARS_NAME_FIX": " 0 "}, "Thanks, Tarz."),
                ({"TARS_NAME_FIX": "1", "RTSHIM_NAME_FIX": "1"}, "Thanks, TARS."),
            ):
                clean = {k: v for k, v in os.environ.items() if k not in ("TARS_NAME_FIX", "RTSHIM_NAME_FIX")}
                with self.subTest(env=env), mock.patch.dict(os.environ, {**clean, **env}, clear=True):
                    self.assertEqual(importlib.reload(names).fix_tars_name("Thanks, Tarz."), want)
        finally:
            importlib.reload(names)

    def test_switch_off(self):
        old = names.ENABLED
        names.ENABLED = False
        try:
            self.assertEqual(fix_tars_name("Can you hear me, Tarz?"), "Can you hear me, Tarz?")
        finally:
            names.ENABLED = old


class FakeWS:
    def __init__(self) -> None:
        self.closed = False
        self.sent: list[dict] = []

    async def send_json(self, event: dict) -> None:
        self.sent.append(event)


class SessionWiringTests(unittest.TestCase):
    """Both copies (Talk 8820, phone 8821) run this same commit path."""

    def test_commit_sends_fixed_transcript_to_client_and_llm(self):
        async def checks():
            for path in ("/v1/realtime", "/openai/v1/realtime?model=tars-local"):
                ws = FakeWS()
                session = RealtimeSession(make_mocked_request("GET", path), ws)
                try:
                    async def transcribe(pcm, sample_rate, model):
                        return "Thank you, Stars."

                    session.pipeline.transcribe = transcribe
                    session.input_transcription = {"model": "whisper-1"}
                    session.vad.voiced_ms = 900
                    session.input_buffer.extend(b"\x00\x01")
                    await session.commit_audio(auto_response=False)
                    self.assertEqual(session.messages[-1]["content"], "Thank you, TARS.")
                    done = [e for e in ws.sent if e.get("type") == "conversation.item.input_audio_transcription.completed"]
                    self.assertEqual(done[-1]["transcript"], "Thank you, TARS.")
                    created = [e for e in ws.sent if e.get("type") == "conversation.item.created"]
                    self.assertEqual(created[-1]["item"]["content"][0]["transcript"], "Thank you, TARS.")
                finally:
                    await session.close()

        asyncio.run(checks())


if __name__ == "__main__":
    unittest.main()
