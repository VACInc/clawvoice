#!/usr/bin/env python3
"""Hermetic tests for the TTS-stage name respelling (pronounce.py).

The expectations come from the 2026-11-05 live TTS -> Parakeet probe:
"VAC" rendered and was heard as "back" (both HA :5051 and realtime :31420),
"Vack" came back as "Vac", and "TARS" was transcribed as "Tarz". No live
TTS, STT, or network access.
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests.net_guard as net_guard

net_guard.install()

import asyncio  # noqa: E402

import pronounce  # noqa: E402
from pipeline import BackendConfig, Pipeline  # noqa: E402
from pronounce import fix_pronunciation  # noqa: E402


class _FakeResponse:
    status = 200

    def __init__(self) -> None:
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def iter_chunked(self, _size: int):
        async def gen():
            if False:  # pragma: no cover - empty stream, no audio needed
                yield b""
        return gen()


class _FakeHttp:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        return _FakeResponse()


class PronounceTest(unittest.TestCase):
    def test_vac_becomes_spoken_vack(self):
        self.assertEqual(fix_pronunciation("Hey VAC, it's me."), "Hey Vack, it's me.")
        self.assertEqual(fix_pronunciation("VAC?"), "Vack?")
        self.assertEqual(fix_pronunciation("Thanks Vac."), "Thanks Vack.")

    def test_tars_becomes_spoken_tarz(self):
        self.assertEqual(fix_pronunciation("This is TARS."), "This is Tars.")
        self.assertEqual(fix_pronunciation("Tars here."), "Tars here.")

    def test_possessive_and_plural_forms(self):
        self.assertEqual(fix_pronunciation("VAC's turn"), "Vack's turn")
        self.assertEqual(fix_pronunciation("TARS's notes"), "Tars's notes")

    def test_plain_words_untouched(self):
        for text in (
            "vacuum the rug",
            "the stars at night",
            "tar and tars",
            "VACUUM",
            "starvation",
        ):
            self.assertEqual(fix_pronunciation(text), text)

    def test_pipeline_sends_respelled_text(self):
        http = _FakeHttp()
        pipeline = Pipeline(http, BackendConfig())

        async def drain() -> None:
            async for _ in pipeline.stream_tts_pcm("Hey VAC, this is TARS. Good to hear you, VAC."):
                pass

        asyncio.run(drain())
        self.assertTrue(http.posts)
        sent = http.posts[0][1]["text"]
        self.assertIn("Vack", sent)
        self.assertIn("Tars", sent)
        self.assertNotIn("VAC", sent)

    def test_kill_switch(self):
        old = pronounce.ENABLED
        try:
            pronounce.ENABLED = False
            self.assertEqual(fix_pronunciation("Hey VAC, TARS here."), "Hey VAC, TARS here.")
        finally:
            pronounce.ENABLED = old

    def test_env_switch_parsed(self):
        old = os.environ.get("TTS_PRONOUNCE_FIX")
        try:
            os.environ["TTS_PRONOUNCE_FIX"] = "0"
            reloaded = importlib.reload(pronounce)
            self.assertFalse(reloaded.ENABLED)
            os.environ["TTS_PRONOUNCE_FIX"] = "1"
            reloaded = importlib.reload(pronounce)
            self.assertTrue(reloaded.ENABLED)
        finally:
            if old is None:
                os.environ.pop("TTS_PRONOUNCE_FIX", None)
            else:
                os.environ["TTS_PRONOUNCE_FIX"] = old
            importlib.reload(pronounce)


if __name__ == "__main__":
    unittest.main(verbosity=2)
