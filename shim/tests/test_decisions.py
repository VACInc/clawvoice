#!/usr/bin/env python3
"""Hermetic decisions: auth edges, noise filter, event routing, audio parse, tools.

No live STT, LLM, TTS, or websocket client. Energy VAD only.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests.net_guard as net_guard

net_guard.install()


def setUpModule():
    net_guard.install()


os.environ["RTSHIM_VAD"] = "energy"

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from server import (  # noqa: E402
    NOISE_MAX_VOICED_MS,
    RealtimeSession,
    ToolCallAccumulator,
    auth_log_value,
    build_app,
    event_type,
    extract_text_from_item,
    health,
    is_noise_transcript,
    output_pace_delay,
    parse_audio_format,
    websocket_handler,
)


class FakeWS:
    def __init__(self) -> None:
        self.closed = False
        self.sent: list[dict] = []

    async def send_json(self, event: dict) -> None:
        self.sent.append(event)


def run(coro):
    return asyncio.run(coro)


class AuthEdgeTests(unittest.TestCase):
    def test_no_tail_or_hash_for_any_length(self):
        for value in ("abcd", "abcde", "xxab\x7f", "\u00e9", "sekretVALUE"):
            got = auth_log_value(value)
            self.assertEqual(got, "[redacted]")
            self.assertNotIn(value[-2:], got)
        self.assertFalse(auth_log_value("abcd").startswith("#"))

    def test_scheme_split_and_invalid_scheme(self):
        tabbed = auth_log_value("Bearer\tsekretVALUE")
        self.assertEqual(tabbed, "Bearer [redacted]")
        self.assertNotIn("ALUE", tabbed)
        digit = auth_log_value("123abc secretvalue")
        self.assertEqual(digit, "[redacted]")
        long_scheme = "A" * 33
        self.assertEqual(auth_log_value(f"{long_scheme} secretvalue"), "[redacted]")
        self.assertEqual(auth_log_value("A" * 32 + " secretvalue"), "A" * 32 + " [redacted]")


class NoiseAndParseTests(unittest.TestCase):
    def test_noise_boundary_and_empty_normalized(self):
        self.assertTrue(is_noise_transcript("uh", NOISE_MAX_VOICED_MS - 1))
        self.assertFalse(is_noise_transcript("uh", NOISE_MAX_VOICED_MS))
        self.assertTrue(is_noise_transcript("***", 10))
        self.assertFalse(is_noise_transcript("kitchen lights", 10))

    def test_event_type_and_text_extract(self):
        self.assertEqual(event_type({"type": "session.update"}), "session.update")
        self.assertEqual(event_type({"type": 1}), "unknown")
        self.assertEqual(event_type({}), "unknown")
        self.assertEqual(extract_text_from_item({"content": "hi"}), "hi")
        self.assertEqual(
            extract_text_from_item(
                {
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "input_text", "text": "there"},
                        {"transcript": "mic"},
                        {"type": "text", "text": ""},
                        "skip",
                    ]
                }
            ),
            "hello\nthere\nmic",
        )
        self.assertEqual(extract_text_from_item({"content": 5}), "")

    def test_parse_audio_format_keys_and_default(self):
        self.assertEqual(parse_audio_format({"type": "audio/pcm", "rate": 22050}), ("pcm16", 22050))
        self.assertEqual(parse_audio_format({"type": "linear16", "sample_rate": 22050}), ("pcm16", 22050))
        self.assertEqual(parse_audio_format({"format": "pcm16", "sampleRate": 22050}), ("pcm16", 22050))
        self.assertEqual(parse_audio_format("audio/pcmu"), ("g711_ulaw", 8000))
        self.assertEqual(parse_audio_format("nope"), ("pcm16", 24000))
        self.assertEqual(parse_audio_format({"type": "g711_ulaw", "rate": 16000}), ("g711_ulaw", 8000))

    def test_tool_calls_skip_junk_and_finalize_in_index_order(self):
        acc = ToolCallAccumulator()
        emitted = acc.update(
            [
                "bad",
                {"index": 2, "id": "c2", "function": {"name": "second"}},
                {"function": {"name": "onlyname"}},
                {"index": 0, "function": {"arguments": "{\"q\":"}},
                {"index": 0, "function": {"arguments": "1}"}},
            ]
        )
        kinds = [item[0] for item in emitted]
        self.assertIn("added", kinds)
        self.assertEqual([state.index for state in acc.finalize()], [0, 2])
        self.assertEqual(acc.states[0].arguments, '{"q":1}')
        bare = ToolCallAccumulator()
        bare.update([{"id": "c", "function": {"name": "n"}}])
        self.assertIn(0, bare.states)
        name_only = ToolCallAccumulator()
        self.assertEqual(
            [item[0] for item in name_only.update([{"function": {"name": "onlyname"}}])],
            ["added"],
        )
        id_only = ToolCallAccumulator()
        self.assertEqual([item[0] for item in id_only.update([{"id": "c-only"}])], ["added"])
        empty = acc.states[2]
        empty.arguments = ""
        self.assertEqual(empty.chat_tool_call()["function"]["arguments"], "{}")


class RouteTests(unittest.TestCase):
    def test_app_routes_health_and_openai_path_guard(self):
        app = build_app()
        paths = {route.resource.canonical for route in app.router.routes()}
        self.assertEqual(paths, {"/healthz", "/v1/realtime", "/openai/realtime", "/openai/v1/realtime"})

        async def checks():
            ok = await health(make_mocked_request("GET", "/healthz"))
            self.assertEqual(ok.status, 200)
            with self.assertRaises(web.HTTPNotFound):
                await websocket_handler(make_mocked_request("GET", "/nope"))
            for path in ("/openai/realtime", "/v1/realtime", "/openai/v1/realtime"):
                # A completed websocket stream needs no transport or listener.
                class EmptyWS(FakeWS):
                    prepare = AsyncMock()

                    def __aiter__(self):
                        return self

                    async def __anext__(self):
                        raise StopAsyncIteration

                ws = EmptyWS()
                request = make_mocked_request("GET", path + "?model=gpt-realtime&api-key=sentinel-query-QRST", headers={
                    "Authorization": "Bearer sentinel-secret-ABCD",
                    "api-key": "sentinel-key-WXYZ",
                })
                with patch("server.web.WebSocketResponse", return_value=ws), patch("server.log") as log:
                    response = await websocket_handler(request)
                self.assertIs(response, ws)
                ws.prepare.assert_awaited_with(request)
                self.assertEqual(ws.sent[0]["type"], "session.created")
                logged = " ".join(call.args[0] for call in log.call_args_list)
                self.assertIn("authorization=Bearer [redacted]", logged)
                self.assertIn("api-key=[redacted]", logged)
                self.assertIn(f"path={path}?model=gpt-realtime&[redacted] ", logged)
                for fragment in ("ABCD", "WXYZ", "QRST", "sentinel-query"):
                    self.assertNotIn(fragment, logged)
                self.assertNotIn("sentinel-secret", logged)
                self.assertNotIn("sentinel-key", logged)

        run(checks())

    def test_event_router_and_commit_filters(self):
        async def checks():
            req = make_mocked_request("GET", "/v1/realtime?model=unit")
            ws = FakeWS()
            session = RealtimeSession(req, ws)
            try:
                self.assertTrue(session.create_response_enabled())
                session.turn_detection["create_response"] = False
                self.assertFalse(session.create_response_enabled())
                session.turn_detection.pop("create_response")
                self.assertTrue(session.create_response_enabled())

                await session.handle({"type": "session.update", "session": {"tools": [{"name": "a"}, "no", 1]}})
                self.assertEqual(session.tools, [{"name": "a"}])
                self.assertTrue(any(event.get("type") == "session.updated" for event in ws.sent))

                ws.sent.clear()
                await session.handle({"type": "input_audio_buffer.append"})
                self.assertIn("missing audio", ws.sent[-1]["error"]["message"])

                ws.sent.clear()
                await session.handle({"type": "input_audio_buffer.commit"})
                self.assertEqual(ws.sent[-1]["type"], "input_audio_buffer.committed")
                self.assertIsNone(ws.sent[-1]["item_id"])

                session.input_buffer.extend(b"xyz")
                await session.handle({"type": "input_audio_buffer.clear"})
                self.assertEqual(session.input_buffer, bytearray())
                self.assertEqual(ws.sent[-1]["type"], "input_audio_buffer.cleared")

                await session.handle(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "text", "text": "hello"}],
                        },
                    }
                )
                self.assertEqual(session.messages[-1]["content"], "hello")
                await session.handle(
                    {
                        "type": "conversation.item.create",
                        "item": {"type": "function_call_output", "call_id": "c1", "output": {"ok": 1}},
                    }
                )
                self.assertEqual(session.messages[-1]["role"], "tool")
                self.assertIn("ok", session.messages[-1]["content"])
                ws.sent.clear()
                await session.handle(
                    {"type": "conversation.item.create", "item": {"type": "function_call_output", "output": "x"}}
                )
                self.assertIn("call_id", ws.sent[-1]["error"]["message"])

                await session.handle({"type": "conversation.item.truncate", "item_id": "item9"})
                self.assertEqual(ws.sent[-1]["type"], "conversation.item.truncated")
                self.assertEqual(ws.sent[-1]["item_id"], "item9")

                session.start_response = AsyncMock()
                await session.handle({"type": "response.create", "event_id": "e1"})
                session.start_response.assert_awaited()
                session.cancel_active = AsyncMock()
                await session.handle({"type": "response.cancel", "event_id": "e2"})
                session.cancel_active.assert_awaited()

                ws.sent.clear()
                await session.handle({"type": "nope", "event_id": "e3"})
                self.assertEqual(ws.sent[-1]["error"]["code"], "unsupported_event")

                ws.sent.clear()
                await session.handle(
                    {"type": "session.update", "session": {"input_audio_format": {"type": "audio/pcm", "rate": 16000}}}
                )
                self.assertIn("unsupported input", ws.sent[-1]["error"]["message"])
                ws.sent.clear()
                await session.handle(
                    {
                        "type": "session.update",
                        "session": {"input_audio_format": "pcm16", "output_audio_format": {"type": "audio/pcm", "rate": 16000}},
                    }
                )
                self.assertIn("unsupported output", ws.sent[-1]["error"]["message"])

                async def transcribe(pcm, sample_rate, model):
                    return "thank you"

                session.pipeline.transcribe = transcribe
                session.input_transcription = {"model": "whisper-1"}
                session.vad.voiced_ms = 100
                session.input_buffer.extend(b"\x00\x01")
                before = len(session.messages)
                await session.commit_audio(auto_response=False)
                self.assertEqual(len(session.messages), before)

                async def real(pcm, sample_rate, model):
                    return "turn on the lights"

                session.pipeline.transcribe = real
                session.vad.voiced_ms = 800
                session.input_buffer.extend(b"\x00\x01")
                started = []

                async def start(event_id=None):
                    started.append(event_id)

                session.start_response = start
                await session.commit_audio(auto_response=True)
                self.assertEqual(session.messages[-1]["content"], "turn on the lights")
                self.assertEqual(started, [None])

                called = []

                async def boom(pcm, sample_rate, model):
                    called.append(True)
                    return "x"

                session.input_transcription = None
                session.pipeline.transcribe = boom
                session.input_buffer.extend(b"\x00\x01")
                await session.commit_audio(auto_response=False)
                self.assertEqual(called, [])
                self.assertEqual(session.messages[-1]["content"], "")
            finally:
                await session.close()

        run(checks())


class OutputPaceTests(unittest.TestCase):
    def test_pace_holds_lead_over_playback(self):
        second = 24000 * 2
        self.assertEqual(output_pace_delay(0, 0.0, lead=1.5), 0.0)
        self.assertEqual(output_pace_delay(second, 0.0, lead=1.5), 0.0)
        self.assertAlmostEqual(output_pace_delay(4 * second, 1.0, lead=1.5), 1.5)
        self.assertAlmostEqual(output_pace_delay(4 * second, 1.0, rate=48000, lead=0.5), 0.5)
        self.assertEqual(output_pace_delay(2 * second, 3.0, lead=1.5), 0.0)


if __name__ == "__main__":
    unittest.main()
