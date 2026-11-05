#!/usr/bin/env python3
"""TARS realtime shim: OpenAI Realtime wire protocol, local STT/LLM/TTS backends."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

import protocol as P
from names import fix_tars_name
from phone import PCM_RATE, PHONE_RATE, PhoneDecoder, PhoneEncoder
from pipeline import BackendConfig, Pipeline, SentenceSegmenter
from vad import EnergyVAD, SileroVAD


HOST = os.environ.get("RTSHIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("RTSHIM_PORT", "8820"))
DEFAULT_SILENCE_MS = int(os.environ.get("RTSHIM_VAD_SILENCE_MS", "500"))
DEBUG = os.environ.get("RTSHIM_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
VAD_KIND = os.environ.get("RTSHIM_VAD", "silero").strip().lower()
SILERO_MODEL = os.environ.get(
    "RTSHIM_SILERO_MODEL", str(Path(__file__).resolve().parent / "models" / "silero_vad.onnx")
)
SILERO_THRESHOLD = float(os.environ.get("RTSHIM_SILERO_THRESHOLD", "0.5"))
MIN_SPEECH_MS = int(os.environ.get("RTSHIM_VAD_MIN_SPEECH_MS", "200"))
LOUDNESS_RATIO = float(os.environ.get("RTSHIM_VAD_LOUDNESS_RATIO", "0"))
MAX_TURN_MS = int(os.environ.get("RTSHIM_VAD_MAX_TURN_MS", "3000"))
TURN_DROP_RATIO = float(os.environ.get("RTSHIM_VAD_TURN_DROP_RATIO", "0"))
# Audio kept before speech_started, like OpenAI prefix_padding_ms. Older buffered
# audio (TV, speech that never opened a turn) is dropped instead of glued onto the turn.
PREFIX_PADDING_MS = int(os.environ.get("RTSHIM_VAD_PREFIX_PADDING_MS", "300"))
# Whisper's usual output for breaths, clicks, and room noise. Dropped only when
# the turn holds little detected speech, so a real "thank you" mid-chat survives.
NOISE_TRANSCRIPTS = {"", "you", "thank you", "thanks for watching", "thank you for watching", "uh", "um", "hmm", "mm", "mhm", "huh", "ah", "oh"}
NOISE_MAX_VOICED_MS = int(os.environ.get("RTSHIM_NOISE_MAX_VOICED_MS", "1500"))


def log(message: str) -> None:
    print(f"[rtshim] {message}", flush=True)


def make_vad(*, sample_rate: int, threshold: float, silence_duration_ms: int) -> EnergyVAD | SileroVAD:
    if VAD_KIND == "silero":
        try:
            return SileroVAD(
                model_path=SILERO_MODEL,
                sample_rate=sample_rate,
                threshold=SILERO_THRESHOLD,
                silence_duration_ms=silence_duration_ms,
                min_speech_ms=MIN_SPEECH_MS,
                loudness_ratio=LOUDNESS_RATIO,
                max_turn_ms=MAX_TURN_MS,
                turn_drop_ratio=TURN_DROP_RATIO,
            )
        except Exception as exc:  # missing model or onnxruntime: keep serving on the energy gate
            log(f"silero vad unavailable ({exc}); falling back to energy vad")
    return EnergyVAD(sample_rate=sample_rate, threshold=threshold, silence_duration_ms=silence_duration_ms)


def is_noise_transcript(transcript: str, voiced_ms: int) -> bool:
    normalized = re.sub(r"[^a-z' ]+", "", transcript.lower()).strip()
    if not normalized:
        return True
    return normalized in NOISE_TRANSCRIPTS and voiced_ms < NOISE_MAX_VOICED_MS


_AUTH_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+._-]{0,31}$")
REDACTED = "[redacted]"


def auth_log_value(value: str | None) -> str:
    """Redact a credential for logs. Empty becomes '-'; only a safe scheme name survives."""
    if not isinstance(value, str):
        return "-"
    text = value.strip()
    if not text:
        return "-"
    parts = text.split(None, 1)
    if len(parts) == 2 and _AUTH_SCHEME.fullmatch(parts[0]):
        return f"{parts[0]} {REDACTED}"
    return REDACTED


# Query params whose values are routing hints, not credentials. Every other
# param (api-key, key, token, sig, ...) is logged as a bare placeholder.
LOG_SAFE_QUERY_KEYS = frozenset({"model", "deployment", "intent"})
_LOG_SAFE_QUERY_VALUE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")


def path_log_value(path: str, query: Any) -> str:
    """Request path plus a redacted query string for the connect log line."""
    items: list[str] = []
    for key, value in query.items():
        if key in LOG_SAFE_QUERY_KEYS:
            shown = value if isinstance(value, str) and _LOG_SAFE_QUERY_VALUE.fullmatch(value) else REDACTED
            items.append(f"{key}={shown}")
        else:
            items.append(REDACTED)
    return f"{path}?{'&'.join(items)}" if items else path


def event_type(event: dict[str, Any]) -> str:
    value = event.get("type")
    return value if isinstance(value, str) else "unknown"


def extract_text_from_item(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"input_text", "text"} and isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif isinstance(part.get("transcript"), str):
            parts.append(part["transcript"])
    return "\n".join(p for p in parts if p).strip()


# Clients buffer a few seconds at most (the Wear phone relay holds 64 frames); TTS runs
# far faster than realtime, so stay this far ahead of playback instead of bursting.
OUTPUT_LEAD_SECONDS = float(os.environ.get("VOICE_SHIM_OUTPUT_LEAD_SECONDS", "1.5"))


def output_pace_delay(sent_bytes: int, elapsed: float, *, rate: int = 24000, lead: float = OUTPUT_LEAD_SECONDS) -> float:
    ahead = sent_bytes / (rate * 2) - elapsed
    return max(0.0, ahead - lead)


# Phone path (voice-call plugin -> Twilio). The plugin keeps only ~160 ms in Twilio
# and queues the rest in the Gateway, and it skips the Twilio "clear" when a caller
# speaks in the first 250 ms of a reply. Whatever the shim sends early can therefore
# keep playing after a barge-in, so phone output stays only a short lead ahead and
# TTS is prefetched into the shim (dropped instantly on cancel) instead.
PHONE_FRAME_BYTES = 160  # one 20 ms G.711 frame at 8 kHz
PHONE_FRAME_SECONDS = PHONE_FRAME_BYTES / PHONE_RATE
ULAW_SILENCE = b"\xff"
PHONE_OUTPUT_LEAD_SECONDS = int(os.environ.get("RTSHIM_PHONE_OUTPUT_LEAD_MS", "300")) / 1000
# Twilio streams caller audio every 20 ms, silence included. When it stops, the
# Gateway is stalled: audio sent now only piles up and bursts out late, past any
# barge-in. Hold output until caller audio flows again (capped, so a client that
# legitimately stops sending input cannot wedge a reply).
PHONE_INPUT_STALL_SECONDS = int(os.environ.get("RTSHIM_PHONE_INPUT_STALL_MS", "250")) / 1000
PHONE_STALL_HOLD_MAX_SECONDS = int(os.environ.get("RTSHIM_PHONE_STALL_HOLD_MAX_MS", "6000")) / 1000


class PhoneFrameBuffer:
    """mu-law bytes in, whole 20 ms frames out; the tail is padded with silence."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def extend(self, wire: bytes) -> None:
        self.buf.extend(wire)

    def pop_frame(self, *, final: bool) -> bytes | None:
        if len(self.buf) >= PHONE_FRAME_BYTES:
            frame = bytes(self.buf[:PHONE_FRAME_BYTES])
            del self.buf[:PHONE_FRAME_BYTES]
            return frame
        if final and self.buf:
            frame = bytes(self.buf) + ULAW_SILENCE * (PHONE_FRAME_BYTES - len(self.buf))
            self.buf.clear()
            return frame
        return None


class PlayoutClock:
    """Estimates when audio already sent finishes playing on the line.

    Never builds debt: after a gap the clock restarts at now, so a late frame is
    not followed by a catch-up burst.
    """

    def __init__(self, lead: float) -> None:
        self.lead = lead
        self.play_until: float | None = None
        self.frames = 0
        self.gaps = 0
        self.max_gap = 0.0
        self.held = 0.0

    def delay(self, now: float) -> float:
        if self.play_until is None:
            return 0.0
        return max(0.0, self.play_until - now - self.lead)

    def unplayed(self, now: float) -> float:
        if self.play_until is None:
            return 0.0
        return max(0.0, self.play_until - now)

    def sent(self, now: float, seconds: float = PHONE_FRAME_SECONDS) -> None:
        if self.play_until is None:
            self.play_until = now
        elif self.play_until < now:
            gap = now - self.play_until
            self.gaps += 1
            self.max_gap = max(self.max_gap, gap)
            self.play_until = now
        self.play_until += seconds
        self.frames += 1


def phone_input_stalled(last_input_at: float | None, now: float, *, stall: float | None = None) -> bool:
    limit = PHONE_INPUT_STALL_SECONDS if stall is None else stall
    return last_input_at is not None and now - last_input_at > limit


# Wire formats the shim speaks. Everything inside runs at PCM16 24 kHz; mu-law is
# converted at the edge (Twilio phone calls through the voice-call plugin).
SUPPORTED_AUDIO_FORMATS = {("pcm16", PCM_RATE), ("g711_ulaw", PHONE_RATE)}


def audio_format_doc(fmt: str, rate: int) -> dict[str, Any]:
    if fmt == "g711_ulaw":
        return {"type": "audio/pcmu"}
    return {"type": "audio/pcm", "rate": rate}


def parse_audio_format(value: Any, *, default_rate: int = 24000) -> tuple[str, int]:
    if isinstance(value, dict):
        ftype = str(value.get("type") or value.get("format") or "").lower()
        rate = int(value.get("rate") or value.get("sample_rate") or value.get("sampleRate") or default_rate)
        if ftype in {"audio/pcm", "pcm16", "linear16"}:
            return "pcm16", rate
        if ftype in {"audio/pcmu", "g711_ulaw", "ulaw"}:
            return "g711_ulaw", 8000
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pcm16", "audio/pcm", "linear16"}:
            return "pcm16", default_rate
        if normalized in {"g711_ulaw", "audio/pcmu", "ulaw"}:
            return "g711_ulaw", 8000
    return "pcm16", default_rate


@dataclass
class ToolCallState:
    index: int
    item_id: str = field(default_factory=lambda: P.new_id("item"))
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    added: bool = False
    done: bool = False

    def item(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "type": "function_call",
            "call_id": self.call_id or self.item_id,
            "name": self.name,
            "arguments": self.arguments,
        }

    def chat_tool_call(self) -> dict[str, Any]:
        return {
            "id": self.call_id or self.item_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments or "{}",
            },
        }


class ToolCallAccumulator:
    def __init__(self) -> None:
        self.states: dict[int, ToolCallState] = {}

    def update(self, tool_calls: list[dict[str, Any]]) -> list[tuple[str, ToolCallState, str]]:
        emitted: list[tuple[str, ToolCallState, str]] = []
        for raw in tool_calls:
            if not isinstance(raw, dict):
                continue
            index = int(raw.get("index") or 0)
            state = self.states.setdefault(index, ToolCallState(index=index))
            if isinstance(raw.get("id"), str) and raw["id"]:
                state.call_id = raw["id"]
            fn = raw.get("function") if isinstance(raw.get("function"), dict) else {}
            if isinstance(fn.get("name"), str) and fn["name"]:
                state.name = fn["name"]
            if not state.added and (state.call_id or state.name):
                state.added = True
                emitted.append(("added", state, ""))
            arg_delta = fn.get("arguments")
            if isinstance(arg_delta, str) and arg_delta:
                state.arguments += arg_delta
                if not state.added:
                    state.added = True
                    emitted.append(("added", state, ""))
                emitted.append(("delta", state, arg_delta))
        return emitted

    def finalize(self) -> list[ToolCallState]:
        return [self.states[index] for index in sorted(self.states)]


class RealtimeSession:
    def __init__(self, request: web.Request, ws: web.WebSocketResponse) -> None:
        self.request = request
        self.ws = ws
        self.id = P.new_id("sess")
        self.model = request.query.get("model") or request.query.get("deployment") or "local"
        self.instructions = ""
        self.tools: list[dict[str, Any]] = []
        self.tool_choice: str | None = "auto"
        self.turn_detection: dict[str, Any] = {
            "type": "server_vad",
            "threshold": 0.5,
            "silence_duration_ms": DEFAULT_SILENCE_MS,
            "create_response": True,
        }
        self.input_audio_format = "pcm16"
        self.output_audio_format = "pcm16"
        self.input_rate = 24000
        self.output_rate = 24000
        self.input_transcription: dict[str, Any] | None = {"model": "whisper-1"}
        self.input_buffer = bytearray()  # always PCM16 at PCM_RATE
        self.input_decoder: PhoneDecoder | None = None
        self.vad = make_vad(
            sample_rate=PCM_RATE,
            threshold=float(self.turn_detection["threshold"]),
            silence_duration_ms=int(self.turn_detection["silence_duration_ms"]),
        )
        self.messages: list[dict[str, Any]] = []
        self.http = aiohttp.ClientSession()
        self.pipeline = Pipeline(self.http, BackendConfig())
        self.active_task: asyncio.Task[None] | None = None
        self.active_response_id: str | None = None
        self.cancel_requested = False
        self.closed = False
        self.last_input_at: float | None = None  # loop time of the last phone input frame
        self.output_clock: PlayoutClock | None = None  # phone playout of the latest reply
        self.speech_started_at: float | None = None

    async def close(self) -> None:
        self.closed = True
        if self.active_task and not self.active_task.done():
            await self.cancel_active("disconnect", emit_no_active_error=False)
        await self.http.close()

    async def send(self, event: dict[str, Any]) -> None:
        if self.ws.closed:
            return
        etype = event_type(event)
        log(f"out session={self.id} type={etype}")
        if DEBUG:
            log(f"out payload={json.dumps(event, ensure_ascii=False)}")
        await self.ws.send_json(event)

    async def send_error(self, message: str, *, event_id: str | None = None, code: str = "server_error") -> None:
        await self.send(P.error_event(message, event_id=event_id, code=code))

    def session_doc(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "realtime.session",
            "model": self.model,
            "instructions": self.instructions,
            "modalities": ["text", "audio"],
            "output_modalities": ["audio"],
            "voice": os.environ.get("RTSHIM_VOICE", "tars.wav"),
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "input_audio_transcription": self.input_transcription,
            "turn_detection": self.turn_detection,
            "audio": {
                "input": {
                    "format": audio_format_doc(self.input_audio_format, self.input_rate),
                    "transcription": self.input_transcription,
                    "turn_detection": self.turn_detection,
                },
                "output": {
                    "format": audio_format_doc(self.output_audio_format, self.output_rate),
                    "voice": os.environ.get("RTSHIM_VOICE", "tars.wav"),
                },
            },
            "tools": self.tools,
            "tool_choice": self.tool_choice,
        }

    async def handle(self, event: dict[str, Any]) -> None:
        etype = event_type(event)
        log(f"in session={self.id} type={etype}")
        if DEBUG:
            log(f"in payload={json.dumps(event, ensure_ascii=False)}")
        try:
            if etype == "session.update":
                await self.handle_session_update(event)
            elif etype == "input_audio_buffer.append":
                await self.handle_audio_append(event)
            elif etype == "input_audio_buffer.commit":
                await self.commit_audio(auto_response=self.create_response_enabled())
            elif etype == "input_audio_buffer.clear":
                await self.clear_audio()
            elif etype == "conversation.item.create":
                await self.handle_conversation_item_create(event)
            elif etype == "response.create":
                await self.start_response(event_id=event.get("event_id"))
            elif etype == "response.cancel":
                await self.cancel_active("client_cancelled", event_id=event.get("event_id"))
            elif etype == "conversation.item.truncate":
                # The voice-call plugin sends this right before it clears queued Twilio audio.
                since = ""
                if self.speech_started_at is not None:
                    since = f" since_speech_started_ms={(asyncio.get_running_loop().time() - self.speech_started_at) * 1000:.0f}"
                log(f"client truncate session={self.id} audio_end_ms={event.get('audio_end_ms')}{since}")
                await self.send({"type": "conversation.item.truncated", "event_id": P.new_id("evt"), "item_id": event.get("item_id")})
            else:
                await self.send_error(f"unsupported event type: {etype}", event_id=event.get("event_id"), code="unsupported_event")
        except Exception as exc:
            await self.send_error(str(exc), event_id=event.get("event_id"))

    async def handle_session_update(self, event: dict[str, Any]) -> None:
        session = event.get("session") if isinstance(event.get("session"), dict) else {}
        if isinstance(session.get("model"), str):
            self.model = session["model"]
        if "instructions" in session:
            self.instructions = str(session.get("instructions") or "")
        if isinstance(session.get("tools"), list):
            self.tools = [tool for tool in session["tools"] if isinstance(tool, dict)]
        if isinstance(session.get("tool_choice"), str):
            self.tool_choice = session["tool_choice"]

        audio = session.get("audio") if isinstance(session.get("audio"), dict) else {}
        audio_input = audio.get("input") if isinstance(audio.get("input"), dict) else {}
        audio_output = audio.get("output") if isinstance(audio.get("output"), dict) else {}

        if "format" in audio_input:
            self.input_audio_format, self.input_rate = parse_audio_format(audio_input.get("format"))
        if "format" in audio_output:
            self.output_audio_format, self.output_rate = parse_audio_format(audio_output.get("format"))
        if "input_audio_format" in session:
            self.input_audio_format, self.input_rate = parse_audio_format(session.get("input_audio_format"))
        if "output_audio_format" in session:
            self.output_audio_format, self.output_rate = parse_audio_format(session.get("output_audio_format"))
        if (self.input_audio_format, self.input_rate) not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"unsupported input audio format {self.input_audio_format}@{self.input_rate}; use pcm16@24000 or g711_ulaw@8000"
            )
        if (self.output_audio_format, self.output_rate) not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"unsupported output audio format {self.output_audio_format}@{self.output_rate}; use pcm16@24000 or g711_ulaw@8000"
            )
        if self.input_audio_format != "g711_ulaw":
            self.input_decoder = None
        elif self.input_decoder is None:
            self.input_decoder = PhoneDecoder()

        if "transcription" in audio_input:
            self.input_transcription = audio_input.get("transcription") if isinstance(audio_input.get("transcription"), dict) else None
        if "input_audio_transcription" in session:
            self.input_transcription = (
                session.get("input_audio_transcription") if isinstance(session.get("input_audio_transcription"), dict) else None
            )

        td = None
        if isinstance(audio_input.get("turn_detection"), dict):
            td = audio_input["turn_detection"]
        if isinstance(session.get("turn_detection"), dict):
            td = session["turn_detection"]
        if td:
            self.turn_detection.update(td)
            if "silence_duration_ms" not in self.turn_detection:
                self.turn_detection["silence_duration_ms"] = DEFAULT_SILENCE_MS
            self.vad.reconfigure(
                sample_rate=PCM_RATE,
                threshold=float(self.turn_detection.get("threshold", 0.5)),
                silence_duration_ms=int(self.turn_detection.get("silence_duration_ms", DEFAULT_SILENCE_MS)),
            )
        await self.send(P.session_event("session.updated", self.session_doc()))

    async def handle_audio_append(self, event: dict[str, Any]) -> None:
        if (self.input_audio_format, self.input_rate) not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError("input_audio_buffer.append requires pcm16@24000 or g711_ulaw@8000")
        audio_b64 = event.get("audio")
        if not isinstance(audio_b64, str):
            raise ValueError("input_audio_buffer.append missing audio")
        pcm = base64.b64decode(audio_b64)
        loop = asyncio.get_running_loop()
        if self.input_audio_format == "g711_ulaw":
            if self.input_decoder is None:
                self.input_decoder = PhoneDecoder()
            pcm = self.input_decoder.process(pcm)
            self.last_input_at = loop.time()
        self.input_buffer.extend(pcm)
        for vad_event in self.vad.feed(pcm):
            if vad_event.type == "speech_started":
                keep_from = max(0, vad_event.audio_ms - PREFIX_PADDING_MS) * PCM_RATE // 1000 * 2
                if keep_from:
                    del self.input_buffer[:keep_from]
                self.speech_started_at = loop.time()
                await self.send(
                    {
                        "type": "input_audio_buffer.speech_started",
                        "event_id": P.new_id("evt"),
                        "audio_start_ms": vad_event.audio_ms,
                    }
                )
                clock = self.output_clock
                unplayed_ms = clock.unplayed(loop.time()) * 1000 if clock else 0.0
                if self.active_task and not self.active_task.done():
                    await self.cancel_active("barge_in", emit_no_active_error=False)
                    log(
                        f"barge-in session={self.id} cancel_ms={(loop.time() - self.speech_started_at) * 1000:.0f}"
                        f" unplayed_ms={unplayed_ms:.0f}"
                    )
                elif unplayed_ms > 0:
                    log(f"barge-in session={self.id} reply already sent unplayed_ms={unplayed_ms:.0f}")
            elif vad_event.type == "speech_stopped":
                await self.send(
                    {
                        "type": "input_audio_buffer.speech_stopped",
                        "event_id": P.new_id("evt"),
                        "audio_end_ms": vad_event.audio_ms,
                    }
                )
                await self.commit_audio(auto_response=self.create_response_enabled())

    async def clear_audio(self) -> None:
        self.input_buffer.clear()
        self.vad.reset()
        await self.send({"type": "input_audio_buffer.cleared", "event_id": P.new_id("evt")})

    async def commit_audio(self, *, auto_response: bool) -> None:
        if not self.input_buffer:
            await self.send({"type": "input_audio_buffer.committed", "event_id": P.new_id("evt"), "item_id": None})
            return
        item_id = P.new_id("item")
        pcm = bytes(self.input_buffer)
        voiced_ms = self.vad.voiced_ms
        if isinstance(self.vad, SileroVAD):
            log(
                f"session={self.id} turn voiced_ms={voiced_ms} turn_ms={self.vad.turn_ms}"
                f" median_rms={self.vad.turn_median_rms():.4f} peak_rms={self.vad.turn_peak_rms:.4f}"
                f" background_rms={self.vad.ambient_rms:.4f}"
            )
        self.input_buffer.clear()
        self.vad.reset()
        await self.send({"type": "input_audio_buffer.committed", "event_id": P.new_id("evt"), "item_id": item_id})

        transcript = ""
        if self.input_transcription is not None:
            model = str(self.input_transcription.get("model") or "whisper-1")
            heard = await self.pipeline.transcribe(pcm, sample_rate=PCM_RATE, model=model)
            transcript = fix_tars_name(heard)
            if transcript != heard:
                log(f"session={self.id} stt name fix applied")

        item = {
            "id": item_id,
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "transcript": transcript}],
        }
        await self.send({"type": "conversation.item.created", "event_id": P.new_id("evt"), "item_id": item_id, "item": item})
        if self.input_transcription is not None:
            await self.send(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "event_id": P.new_id("evt"),
                    "item_id": item_id,
                    "content_index": 0,
                    "transcript": transcript,
                }
            )
            if is_noise_transcript(transcript, voiced_ms):
                log(f"session={self.id} dropped noise turn voiced_ms={voiced_ms} transcript={transcript.strip()[:40]!r}")
                return
        self.messages.append({"role": "user", "content": transcript})
        if auto_response:
            await self.start_response()

    async def handle_conversation_item_create(self, event: dict[str, Any]) -> None:
        raw_item = event.get("item")
        if not isinstance(raw_item, dict):
            raise ValueError("conversation.item.create missing item")
        item = dict(raw_item)
        item.setdefault("id", P.new_id("item"))
        item_type = item.get("type")
        if item_type == "message" and item.get("role") == "user":
            text = extract_text_from_item(item)
            self.messages.append({"role": "user", "content": text})
        elif item_type == "function_call_output":
            call_id = str(item.get("call_id") or "")
            if not call_id:
                raise ValueError("function_call_output missing call_id")
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
        await self.send({"type": "conversation.item.created", "event_id": P.new_id("evt"), "item_id": item["id"], "item": item})

    def create_response_enabled(self) -> bool:
        return self.turn_detection.get("create_response") is not False

    def chat_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        messages.extend(self.messages)
        return messages

    async def start_response(self, *, event_id: str | None = None) -> None:
        if self.active_task and not self.active_task.done():
            await self.send_error(
                "Conversation already has an active response in progress:",
                event_id=event_id,
                code="active_response",
            )
            return
        self.cancel_requested = False
        response_id = P.new_id("resp")
        self.active_response_id = response_id
        self.active_task = asyncio.create_task(self.run_response(response_id))

    async def cancel_active(
        self,
        reason: str,
        *,
        event_id: str | None = None,
        emit_no_active_error: bool = True,
    ) -> None:
        task = self.active_task
        if not task or task.done():
            if emit_no_active_error:
                await self.send_error(
                    "Cancellation failed: no active response found",
                    event_id=event_id,
                    code="no_active_response",
                )
            return
        self.cancel_requested = True
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self.active_task = None
        self.active_response_id = None

    async def emit_cancelled(self, response_id: str, reason: str = "cancelled") -> None:
        await self.send(P.response_done(response_id, "cancelled", [], reason=reason))
        await self.send(
            {
                "type": "response.cancelled",
                "event_id": P.new_id("evt"),
                "response_id": response_id,
                "response": {"id": response_id, "status": "cancelled"},
            }
        )

    async def wait_for_phone_line(self, clock: PlayoutClock) -> None:
        """Sleep until the next frame may go out: caller audio flowing and within the lead."""
        loop = asyncio.get_running_loop()
        held_since: float | None = None
        while not self.cancel_requested:
            now = loop.time()
            if phone_input_stalled(self.last_input_at, now) and (
                held_since is None or now - held_since < PHONE_STALL_HOLD_MAX_SECONDS
            ):
                if held_since is None:
                    held_since = now
                await asyncio.sleep(PHONE_FRAME_SECONDS)
                continue
            if held_since is not None:
                clock.held += now - held_since
                log(f"phone output held session={self.id} ms={(now - held_since) * 1000:.0f} reason=no_caller_audio")
                held_since = None
            delay = clock.delay(now)
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    async def phone_output(self, response_id: str, item_id: str, tts_queue: asyncio.Queue[str | None]) -> None:
        """Phone reply audio: TTS prefetched into the shim, sent as 20 ms frames a short lead ahead."""
        loop = asyncio.get_running_loop()
        frames = PhoneFrameBuffer()
        ready = asyncio.Event()
        produced_all = False
        clock = PlayoutClock(PHONE_OUTPUT_LEAD_SECONDS)
        self.output_clock = clock

        async def produce() -> None:
            nonlocal produced_all
            encoder = PhoneEncoder()
            try:
                while True:
                    sentence = await tts_queue.get()
                    if sentence is None:
                        return
                    async for pcm in self.pipeline.stream_tts_pcm(sentence):
                        if self.cancel_requested:
                            return
                        wire = encoder.process(pcm)
                        if wire:
                            frames.extend(wire)
                            ready.set()
            finally:
                produced_all = True
                ready.set()

        producer = asyncio.create_task(produce())
        status = "cancelled"
        try:
            while not self.cancel_requested:
                frame = frames.pop_frame(final=produced_all)
                if frame is None:
                    if produced_all:
                        status = "completed"
                        break
                    ready.clear()
                    await ready.wait()
                    continue
                await self.wait_for_phone_line(clock)
                if self.cancel_requested:
                    break
                await self.send(P.audio_delta(response_id, item_id, base64.b64encode(frame).decode("ascii")))
                clock.sent(loop.time())
        finally:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer
            if clock.frames or clock.held:
                log(
                    f"phone output session={self.id} response={response_id} status={status} frames={clock.frames}"
                    f" audio_ms={clock.frames * PHONE_FRAME_SECONDS * 1000:.0f} gaps={clock.gaps}"
                    f" max_gap_ms={clock.max_gap * 1000:.0f} held_ms={clock.held * 1000:.0f}"
                    f" lead_ms={clock.lead * 1000:.0f}"
                )

    async def run_response(self, response_id: str) -> None:
        assistant_item_id = P.new_id("item")
        assistant_item = {
            "id": assistant_item_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "audio", "transcript": ""}],
        }
        output_items: list[dict[str, Any]] = []
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
        tts_task: asyncio.Task[None] | None = None
        transcript_parts: list[str] = []
        segmenter = SentenceSegmenter()
        tool_calls = ToolCallAccumulator()
        assistant_item_added = False
        response_done_sent = False

        async def tts_worker() -> None:
            if self.output_audio_format == "g711_ulaw":
                await self.phone_output(response_id, assistant_item_id, tts_queue)
                return
            loop = asyncio.get_running_loop()
            started: float | None = None
            sent_bytes = 0  # PCM16 bytes at PCM_RATE
            while True:
                sentence = await tts_queue.get()
                if sentence is None:
                    return
                async for pcm in self.pipeline.stream_tts_pcm(sentence):
                    if self.cancel_requested:
                        return
                    if started is None:
                        started = loop.time()
                    delay = output_pace_delay(sent_bytes, loop.time() - started, rate=PCM_RATE)
                    if delay > 0:
                        await asyncio.sleep(delay)
                        if self.cancel_requested:
                            return
                    sent_bytes += len(pcm)
                    if not pcm:
                        continue
                    await self.send(P.audio_delta(response_id, assistant_item_id, base64.b64encode(pcm).decode("ascii")))

        async def send_response_done(
            status: str,
            output: list[dict[str, Any]],
            *,
            reason: str | None = None,
        ) -> None:
            nonlocal response_done_sent
            if response_done_sent:
                return
            await self.send(P.response_done(response_id, status, output, reason=reason))
            response_done_sent = True

        async def drain_tts_task(*, cancel: bool) -> None:
            if not tts_task:
                return
            if cancel and not tts_task.done():
                tts_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await tts_task

        async def next_chat_chunk(chat_iter: Any) -> dict[str, Any] | None:
            next_task = asyncio.create_task(chat_iter.__anext__())
            wait_tasks: set[asyncio.Task[Any]] = {next_task}
            if tts_task:
                wait_tasks.add(tts_task)
            done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            if tts_task in done:
                if not next_task.done():
                    next_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await next_task
                else:
                    with contextlib.suppress(StopAsyncIteration, Exception):
                        next_task.result()
                if tts_task.cancelled():
                    raise asyncio.CancelledError()
                exc = tts_task.exception()
                if exc is not None:
                    raise RuntimeError("TTS worker failed") from exc
                raise RuntimeError("TTS worker exited before LLM stream completed")
            try:
                return next_task.result()
            except StopAsyncIteration:
                return None

        try:
            await self.send(P.response_created(response_id))
            tts_task = asyncio.create_task(tts_worker())
            chat_iter = self.pipeline.stream_chat(self.chat_messages(), tools=self.tools).__aiter__()
            while True:
                chunk = await next_chat_chunk(chat_iter)
                if chunk is None:
                    break
                if self.cancel_requested:
                    raise asyncio.CancelledError()
                if chunk["type"] == "content":
                    text = chunk["text"]
                    if not assistant_item_added:
                        assistant_item_added = True
                        await self.send(P.output_item_added(response_id, assistant_item))
                    transcript_parts.append(text)
                    await self.send(P.assistant_transcript_delta(response_id, assistant_item_id, text))
                    for sentence in segmenter.feed(text):
                        await tts_queue.put(sentence)
                elif chunk["type"] == "tool_calls":
                    for action, state, delta in tool_calls.update(chunk["tool_calls"]):
                        if action == "added":
                            await self.send(P.output_item_added(response_id, state.item()))
                        elif action == "delta":
                            await self.send(P.function_args_delta(response_id, state.item(), delta))

            for sentence in segmenter.flush():
                await tts_queue.put(sentence)
            await tts_queue.put(None)
            if tts_task:
                await tts_task

            transcript = "".join(transcript_parts).strip()
            assistant_item["content"][0]["transcript"] = transcript
            finalized_tools = tool_calls.finalize()
            if finalized_tools:
                if transcript:
                    await self.send(P.audio_done(response_id, assistant_item_id))
                    await self.send(P.assistant_transcript_done(response_id, assistant_item_id, transcript))
                    await self.send(P.output_item_done(response_id, assistant_item))
                    output_items.append(assistant_item)
                for state in finalized_tools:
                    item = state.item()
                    if not state.done:
                        await self.send(P.function_args_done(response_id, item))
                        await self.send(P.output_item_done(response_id, item))
                        await self.send(P.conversation_item_done(item))
                    output_items.append(item)
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": transcript,
                        "tool_calls": [state.chat_tool_call() for state in finalized_tools],
                    }
                )
            else:
                if not assistant_item_added:
                    assistant_item_added = True
                    await self.send(P.output_item_added(response_id, assistant_item))
                await self.send(P.audio_done(response_id, assistant_item_id))
                await self.send(P.assistant_transcript_done(response_id, assistant_item_id, transcript))
                await self.send(P.output_item_done(response_id, assistant_item))
                output_items.append(assistant_item)
                if transcript:
                    self.messages.append({"role": "assistant", "content": transcript})
            await send_response_done("completed", output_items)
        except asyncio.CancelledError:
            await drain_tts_task(cancel=True)
            with contextlib.suppress(Exception):
                await send_response_done("cancelled", [], reason="cancelled")
            with contextlib.suppress(Exception):
                await self.send(
                    {
                        "type": "response.cancelled",
                        "event_id": P.new_id("evt"),
                        "response_id": response_id,
                        "response": {"id": response_id, "status": "cancelled"},
                    }
                )
            raise
        except Exception as exc:
            log(f"response failed session={self.id} response={response_id} error={exc!r}")
            await drain_tts_task(cancel=True)
            with contextlib.suppress(Exception):
                await self.send_error(str(exc))
            with contextlib.suppress(Exception):
                await send_response_done("failed", output_items, reason="server_error")
        finally:
            if not response_done_sent and not self.ws.closed:
                log(f"response ended without done session={self.id} response={response_id}")
                with contextlib.suppress(Exception):
                    await self.send_error("response task ended before response.done")
                with contextlib.suppress(Exception):
                    await send_response_done("failed", output_items, reason="server_error")
            if self.active_response_id == response_id:
                self.active_response_id = None
                self.active_task = None
                self.cancel_requested = False


# OpenClaw Azure GA (no azureApiVersion) dials /openai/v1/realtime; preview dials /openai/realtime.
REALTIME_PATHS = ("/v1/realtime", "/openai/realtime", "/openai/v1/realtime")


async def websocket_handler(request: web.Request) -> web.StreamResponse:
    if request.path not in REALTIME_PATHS:
        raise web.HTTPNotFound()
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
    await ws.prepare(request)
    log(
        "connect "
        f"path={path_log_value(request.path, request.query)} "
        f"authorization={auth_log_value(request.headers.get('Authorization'))} "
        f"api-key={auth_log_value(request.headers.get('api-key'))}"
    )
    session = RealtimeSession(request, ws)
    await session.send(P.session_event("session.created", session.session_doc()))
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    event = json.loads(msg.data)
                    if not isinstance(event, dict):
                        raise ValueError("event must be a JSON object")
                except Exception as exc:
                    await session.send_error(f"invalid JSON event: {exc}")
                    continue
                await session.handle(event)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                log(f"ws error session={session.id} error={ws.exception()}")
                break
    finally:
        await session.close()
        log(f"disconnect session={session.id}")
    return ws


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "clawvoice-shim"})


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/healthz", health)
    for path in REALTIME_PATHS:
        app.router.add_get(path, websocket_handler)
    return app


def main() -> None:
    app = build_app()
    log(f"listening ws://{HOST}:{PORT}/v1/realtime")
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
