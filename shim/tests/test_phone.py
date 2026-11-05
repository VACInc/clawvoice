#!/usr/bin/env python3
"""Hermetic phone-line tests: mu-law codec, 8k<->24k resampling, numbers to words,
and the session wiring that carries Twilio mu-law in and out.

No live STT, LLM, TTS, or websocket client. Energy VAD only.
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

import pipeline  # noqa: E402
from phone import (  # noqa: E402
    Downsampler,
    PhoneDecoder,
    PhoneEncoder,
    Upsampler,
    int_to_words,
    numbers_to_words,
    ordinal_words,
    ulaw_decode,
    ulaw_encode,
    year_words,
)
from server import RealtimeSession  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def tone(freq: float, rate: int, seconds: float, amp: float = 8000.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def dominant_hz(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64) * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1 / rate)[int(np.argmax(spectrum))])


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


class MuLawTests(unittest.TestCase):
    def test_known_g711_vectors(self):
        self.assertEqual(ulaw_encode(np.array([0], dtype=np.int16)), b"\xff")
        self.assertEqual(ulaw_encode(np.array([-1], dtype=np.int16)), b"\x7e")
        self.assertEqual(ulaw_encode(np.array([32767], dtype=np.int16)), b"\x80")
        self.assertEqual(ulaw_encode(np.array([-32768], dtype=np.int16)), b"\x00")
        self.assertEqual(ulaw_encode(np.array([1000], dtype=np.int16)), b"\xce")
        self.assertEqual(ulaw_decode(b"\xff").tolist(), [0])
        self.assertEqual(ulaw_decode(b"\x80").tolist(), [32124])
        self.assertEqual(ulaw_decode(b"\x00").tolist(), [-32124])
        self.assertEqual(ulaw_decode(b"\xce").tolist(), [988])
        self.assertEqual(ulaw_decode(b"\x4e").tolist(), [-988])

    def test_matches_reference_codec_for_every_byte_and_ramp(self):
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                import audioop  # stdlib reference, gone in 3.13
        except ImportError:
            self.skipTest("audioop not available")
        every_byte = bytes(range(256))
        ref = np.frombuffer(audioop.ulaw2lin(every_byte, 2), dtype="<i2")
        self.assertEqual(ulaw_decode(every_byte).tolist(), ref.tolist())
        ramp = np.arange(-32768, 32768, 7, dtype=np.int32).astype(np.int16)
        self.assertEqual(ulaw_encode(ramp), audioop.lin2ulaw(ramp.astype("<i2").tobytes(), 2))

    def test_round_trip_stays_within_quantization_step(self):
        samples = np.arange(-32000, 32000, 13, dtype=np.int32).astype(np.int16)
        back = ulaw_decode(ulaw_encode(samples)).astype(np.int32)
        err = np.abs(back - samples.astype(np.int32))
        # G.711 step is 2**(exponent+3); relative error stays under ~6.5%.
        self.assertTrue(np.all(err <= np.maximum(12, np.abs(samples.astype(np.int32)) * 0.065)))
        every_byte = bytes(b for b in range(256) if b != 0x7F)  # 0x7f and 0xff both decode to 0
        self.assertEqual(ulaw_encode(ulaw_decode(every_byte)), every_byte)


class ResampleTests(unittest.TestCase):
    def test_upsample_triples_length_and_keeps_pitch(self):
        x = tone(1000, 8000, 0.5)
        y = Upsampler().process(x)
        self.assertEqual(len(y), 3 * len(x))
        self.assertAlmostEqual(dominant_hz(y, 24000), 1000, delta=5)
        self.assertAlmostEqual(rms(y[200:]) / rms(x), 1.0, delta=0.05)
        # The 8 kHz images (7 kHz, 9 kHz) must be filtered out.
        spectrum = np.abs(np.fft.rfft(y[200:].astype(np.float64)))
        freqs = np.fft.rfftfreq(len(y[200:]), 1 / 24000)
        image = spectrum[(freqs > 6500) & (freqs < 9500)].max()
        self.assertLess(image, spectrum.max() * 0.01)

    def test_downsample_thirds_length_and_blocks_alias(self):
        x = tone(1000, 24000, 0.5)
        y = Downsampler().process(x)
        self.assertEqual(len(y), len(x) // 3)
        self.assertAlmostEqual(dominant_hz(y, 8000), 1000, delta=5)
        self.assertAlmostEqual(rms(y[100:]) / rms(x), 1.0, delta=0.05)
        # 6 kHz would fold to 2 kHz at 8 kHz without the low-pass.
        alias = Downsampler().process(tone(6000, 24000, 0.5))
        self.assertLess(rms(alias[100:]), rms(tone(6000, 24000, 0.5)) * 0.02)

    def test_streaming_chunks_match_one_shot(self):
        x = tone(440, 24000, 0.3) + tone(1700, 24000, 0.3, amp=3000)
        whole = Downsampler().process(x)
        down = Downsampler()
        parts = []
        start = 0
        for size in (1, 2, 5, 480, 7, 1000, 3, 4096):
            parts.append(down.process(x[start : start + size]))
            start += size
        parts.append(down.process(x[start:]))
        self.assertEqual(np.concatenate(parts).tolist(), whole.tolist())

        p = tone(300, 8000, 0.3)
        whole_up = Upsampler().process(p)
        up = Upsampler()
        chunked = np.concatenate([up.process(p[i : i + 160]) for i in range(0, len(p), 160)])
        self.assertEqual(chunked.tolist(), whole_up.tolist())

    def test_phone_codec_round_trip_keeps_speech_band(self):
        x = tone(800, 24000, 0.4)
        wire = PhoneEncoder().process(x.astype("<i2").tobytes())
        self.assertEqual(len(wire), len(x) // 3)  # one mu-law byte per 8 kHz sample
        back = np.frombuffer(PhoneDecoder().process(wire), dtype="<i2")
        self.assertEqual(len(back), len(x))
        self.assertAlmostEqual(dominant_hz(back, 24000), 800, delta=5)
        self.assertAlmostEqual(rms(back[400:]) / rms(x), 1.0, delta=0.08)


class NumberWordsTests(unittest.TestCase):
    def test_int_words(self):
        cases = {
            0: "zero",
            7: "seven",
            13: "thirteen",
            40: "forty",
            42: "forty-two",
            100: "one hundred",
            105: "one hundred five",
            999: "nine hundred ninety-nine",
            1000: "one thousand",
            1250: "one thousand two hundred fifty",
            2_000_017: "two million seventeen",
            -3: "minus three",
        }
        for n, words in cases.items():
            self.assertEqual(int_to_words(n), words, n)
        self.assertEqual(ordinal_words(1), "first")
        self.assertEqual(ordinal_words(12), "twelfth")
        self.assertEqual(ordinal_words(20), "twentieth")
        self.assertEqual(ordinal_words(23), "twenty-third")
        self.assertEqual(ordinal_words(104), "one hundred fourth")
        self.assertEqual(year_words(1999), "nineteen ninety-nine")
        self.assertEqual(year_words(2026), "twenty twenty-six")
        self.assertEqual(year_words(2005), "two thousand five")
        self.assertEqual(year_words(1900), "nineteen hundred")
        self.assertEqual(year_words(1907), "nineteen oh seven")

    def test_spoken_forms(self):
        cases = {
            "Your order number is 495120387.": "Your order number is four nine five one two zero three eight seven.",
            "Confirmation 0042.": "Confirmation zero zero four two.",
            "Call 555-123-4567.": "Call five five five, one two three, four five six seven.",
            "Call (555) 123-4567 now": "Call five five five, one two three, four five six seven now",
            "Call +1 555.123.4567": "Call one, five five five, one two three, four five six seven",
            "See you at 3:05 pm.": "See you at three oh five p m.",
            "Meet at 10:30 A.M. sharp": "Meet at ten thirty a m sharp",
            "Open at 7:00.": "Open at seven o'clock.",
            "Open at 9:00 PM": "Open at nine p m",
            "It costs $12.50.": "It costs twelve dollars and fifty cents.",
            "Total $1,250 today": "Total one thousand two hundred fifty dollars today",
            "Fee $0.99, tip $1.": "Fee ninety-nine cents, tip one dollar.",
            "Pi is 3.14": "Pi is three point one four",
            "We have 1,000,000 users": "We have one million users",
            "About 42% done": "About forty-two percent done",
            "The 1st and 22nd": "The first and twenty-second",
            "Part AB12345 ships": "Part AB one two three four five ships",
            "Back in 2026 we had 15 apples": "Back in twenty twenty-six we had fifteen apples",
            "Room 007": "Room zero zero seven",
            "No digits here.": "No digits here.",
        }
        for text, spoken in cases.items():
            self.assertEqual(numbers_to_words(text), spoken, text)

    def test_bad_time_left_as_digits_words(self):
        self.assertEqual(numbers_to_words("score 99:99"), "score ninety-nine:ninety-nine")


class FakeWS:
    def __init__(self) -> None:
        self.closed = False
        self.sent: list[dict] = []

    async def send_json(self, event: dict) -> None:
        self.sent.append(event)


class PhoneSessionTests(unittest.TestCase):
    def test_ulaw_session_decodes_input_and_encodes_output(self):
        async def checks():
            ws = FakeWS()
            session = RealtimeSession(make_mocked_request("GET", "/openai/v1/realtime?model=tars-local"), ws)
            try:
                await session.handle(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "audio": {
                                "input": {"format": {"type": "audio/pcmu"}},
                                "output": {"format": {"type": "audio/pcmu"}, "voice": "tars"},
                            },
                        },
                    }
                )
                updated = ws.sent[-1]
                self.assertEqual(updated["type"], "session.updated", updated)
                self.assertEqual(updated["session"]["audio"]["input"]["format"], {"type": "audio/pcmu"})
                self.assertEqual(updated["session"]["audio"]["output"]["format"], {"type": "audio/pcmu"})
                self.assertEqual(updated["session"]["input_audio_format"], "g711_ulaw")
                decoder = session.input_decoder
                self.assertIsNotNone(decoder)
                # A later turn_detection-only update keeps the stream decoder (no filter glitch).
                await session.handle({"type": "session.update", "session": {"turn_detection": {"silence_duration_ms": 600}}})
                self.assertIs(session.input_decoder, decoder)

                wire = ulaw_encode(tone(1000, 8000, 0.1))  # 800 mu-law bytes
                ws.sent.clear()
                await session.handle({"type": "input_audio_buffer.append", "audio": base64.b64encode(wire).decode()})
                self.assertFalse(any(e.get("type") == "error" for e in ws.sent), ws.sent)
                self.assertEqual(len(session.input_buffer), 2400 * 2)  # PCM16 at 24 kHz inside

                sent_text = []

                async def fake_tts(text):
                    sent_text.append(text)
                    yield tone(1000, 24000, 0.1).astype("<i2").tobytes()

                async def fake_chat(messages, tools=None):
                    yield {"type": "content", "text": "Order 495120387 is ready."}

                session.pipeline.stream_tts_pcm = fake_tts
                session.pipeline.stream_chat = fake_chat
                ws.sent.clear()
                await session.run_response("resp_test")
                deltas = [e for e in ws.sent if e.get("type") in {"response.output_audio.delta", "response.audio.delta"}]
                self.assertTrue(deltas, [e.get("type") for e in ws.sent])
                total = sum(len(base64.b64decode(e["delta"])) for e in deltas)
                self.assertEqual(total, 800)  # 0.1 s at 8 kHz, one byte per sample
                pcm = ulaw_decode(b"".join(base64.b64decode(e["delta"]) for e in deltas))
                self.assertAlmostEqual(dominant_hz(pcm, 8000), 1000, delta=15)
            finally:
                await session.close()

        run(checks())

    def test_pcm16_session_unchanged(self):
        async def checks():
            ws = FakeWS()
            session = RealtimeSession(make_mocked_request("GET", "/v1/realtime"), ws)
            try:
                await session.handle({"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}}})
                self.assertIsNone(session.input_decoder)
                self.assertEqual(ws.sent[-1]["session"]["audio"]["input"]["format"], {"type": "audio/pcm", "rate": 24000})
                pcm = tone(500, 24000, 0.05).astype("<i2").tobytes()
                await session.handle({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
                self.assertEqual(bytes(session.input_buffer), pcm)

                # A session left on an unsupported format (failed update) must not buffer audio.
                session.input_rate = 16000
                session.input_buffer.clear()
                ws.sent.clear()
                await session.handle({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
                self.assertIn("requires pcm16@24000", ws.sent[-1]["error"]["message"])
                self.assertEqual(len(session.input_buffer), 0)
                session.input_rate = 24000

                async def fake_tts(text):
                    yield pcm

                async def fake_chat(messages, tools=None):
                    yield {"type": "content", "text": "Hi."}

                session.pipeline.stream_tts_pcm = fake_tts
                session.pipeline.stream_chat = fake_chat
                ws.sent.clear()
                await session.run_response("resp_pcm")
                deltas = [e for e in ws.sent if e.get("type") in {"response.output_audio.delta", "response.audio.delta"}]
                self.assertEqual(b"".join(base64.b64decode(e["delta"]) for e in deltas), pcm)
            finally:
                await session.close()

        run(checks())


class TtsNumberTests(unittest.TestCase):
    def test_tts_payload_spells_numbers(self):
        posted = []

        class FakeResp:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            @property
            def content(self):
                class C:
                    async def iter_chunked(self, n):
                        if False:
                            yield b""

                return C()

        class FakeHttp:
            def post(self, url, json=None, timeout=None):
                posted.append(json)
                return FakeResp()

        async def go(config):
            p = pipeline.Pipeline(FakeHttp(), config)
            async for _ in p.stream_tts_pcm("Your order is 495120387, due at 3:05 pm."):
                pass

        run(go(pipeline.BackendConfig(speak_numbers=True)))
        self.assertEqual(posted[-1]["text"], "Your order is four nine five one two zero three eight seven, due at three oh five p m.")
        run(go(pipeline.BackendConfig(speak_numbers=False)))
        self.assertEqual(posted[-1]["text"], "Your order is 495120387, due at 3:05 pm.")


if __name__ == "__main__":
    unittest.main()
