#!/usr/bin/env python3
"""Channel voice-note route: transcribe_note.py prints the name-fixed transcript.

Inputs are real Parakeet :31411 output from the 2026-11-02 probes (Chatterbox
speech as ogg/opus, m4a, and 16 kHz WAV). STT HTTP and ffmpeg are faked.
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests.net_guard as net_guard

net_guard.install()

import transcribe_note  # noqa: E402

HEARD = {
    "Hey Tarz, what time is it?": "Hey TARS, what time is it?",
    "Hey Taurus, what time is it?": "Hey TARS, what time is it?",
    "Hey Taris, turn on the kitchen lights.": "Hey TARS, turn on the kitchen lights.",
    "The stars are bright tonight, Taras.": "The stars are bright tonight, TARS.",
    "  Thanks, Tarz.  ": "Thanks, TARS.",
    "I love looking at the stars at night": "I love looking at the stars at night",
    "Give it five stars!": "Give it five stars!",
}


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_post(payload, seen):
    def post(req, timeout=None):
        seen.append(req)
        return FakeResp(json.dumps(payload).encode())

    return post


class ToWavTests(unittest.TestCase):
    def test_without_ffmpeg_sends_the_file_as_is(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            note = Path(tmp) / "voice.ogg"
            note.write_bytes(b"OggS-note")
            with mock.patch.object(transcribe_note.shutil, "which", return_value=None):
                self.assertEqual(transcribe_note.to_wav(str(note)), (b"OggS-note", "voice.ogg", "application/octet-stream"))

    def test_with_ffmpeg_asks_for_16k_mono_wav(self):
        calls = []

        def run(argv, **kw):
            calls.append(argv)
            return mock.Mock(stdout=b"RIFFwav")

        with mock.patch.object(transcribe_note.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                mock.patch.object(transcribe_note.subprocess, "run", run):
            self.assertEqual(transcribe_note.to_wav("n.ogg"), (b"RIFFwav", "note.wav", "audio/wav"))
        self.assertEqual(calls[0][0], "/usr/bin/ffmpeg")
        self.assertIn("16000", calls[0])
        self.assertEqual(calls[0][-2:], ["wav", "-"])


class TranscribeNoteTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(transcribe_note, "to_wav", return_value=(b"RIFF", "note.wav", "audio/wav"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_prints_tars(self):
        for heard, want in HEARD.items():
            seen = []
            with self.subTest(heard=heard):
                self.assertEqual(transcribe_note.transcribe("x.ogg", post=fake_post({"text": heard}, seen)), want)
                self.assertIn(b'name="model"', seen[0].data)
                self.assertIn(b"RIFF", seen[0].data)

    def test_other_text_keys(self):
        self.assertEqual(transcribe_note.transcribe("x", post=fake_post({"transcript": "Thanks, Tarz."}, [])), "Thanks, TARS.")

    def test_missing_text_fails_so_openclaw_falls_back(self):
        with self.assertRaises(ValueError):
            transcribe_note.transcribe("x", post=fake_post({"error": "boom"}, []))

    def test_main_exit_codes(self):
        def broken(req, timeout=None):
            raise OSError("refused")

        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(transcribe_note.urllib.request, "urlopen", broken), redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(transcribe_note.main(["t", "x.ogg"]), 1)
        self.assertEqual(out.getvalue(), "")
        with mock.patch.object(transcribe_note.urllib.request, "urlopen", fake_post({"text": "Hey Tarz."}, [])), \
                redirect_stdout(out):
            self.assertEqual(transcribe_note.main(["t", "x.ogg"]), 0)
        self.assertEqual(out.getvalue(), "Hey TARS.\n")
        empty = io.StringIO()
        with mock.patch.object(transcribe_note.urllib.request, "urlopen", fake_post({"text": ""}, [])), \
                redirect_stdout(empty):
            self.assertEqual(transcribe_note.main(["t", "x.ogg"]), 0)
        self.assertEqual(empty.getvalue(), "")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(transcribe_note.main(["t"]), 2)


if __name__ == "__main__":
    unittest.main()
