"""Backend clients for Parakeet STT, DeepSeek Chat Completions, and Chatterbox TTS."""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import unicodedata
import wave
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from phone import numbers_to_words
from pronounce import fix_pronunciation


def _env_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class BackendConfig:
    stt_url: str = os.environ.get(
        "RTSHIM_STT_URL",
        "http://127.0.0.1:31411/v1/audio/transcriptions",
    )
    llm_url: str = os.environ.get(
        "RTSHIM_LLM_URL",
        "http://127.0.0.1:31320/v1/chat/completions",
    )
    tts_url: str = os.environ.get("RTSHIM_TTS_URL", "http://127.0.0.1:31420/tts")
    voice: str = os.environ.get("RTSHIM_VOICE", "tars.wav")
    speed: float = _env_float("RTSHIM_SPEED", "1.15")
    cfg_weight: float = _env_float("RTSHIM_CFG_WEIGHT", "0.4")
    exaggeration: float = _env_float("RTSHIM_EXAGGERATION", "1.0")
    # Qwen3.8 on a second GPU: survives the RMA-card crashes that take DeepSeek down.
    # Empty RTSHIM_LLM_FALLBACK_URL disables failover.
    llm_fallback_url: str = os.environ.get(
        "RTSHIM_LLM_FALLBACK_URL",
        "",
    )
    llm_connect_timeout: float = _env_float("RTSHIM_LLM_CONNECT_TIMEOUT", "3.0")
    llm_first_byte_timeout: float = _env_float("RTSHIM_LLM_FIRST_BYTE_TIMEOUT", "8.0")
    llm_temperature: float = _env_float("RTSHIM_LLM_TEMPERATURE", "0.6")
    llm_max_tokens: int = int(os.environ.get("RTSHIM_LLM_MAX_TOKENS", "240"))
    # Spell numerals the way people read them (order numbers digit by digit, $, times).
    speak_numbers: bool = os.environ.get("RTSHIM_SPEAK_NUMBERS", "1").strip().lower() not in {"0", "false", "no", "off"}


BRACKET_TAG_RE = re.compile(r"\[[a-zA-Z _-]{1,40}\]")
MAX_BACKEND_ERROR_BODY = 4000


def backend_log(message: str) -> None:
    print(f"[rtshim] {message}", flush=True)


async def raise_for_status_with_body(resp: aiohttp.ClientResponse, label: str) -> None:
    if resp.status < 400:
        return
    body = await resp.text()
    preview = body[:MAX_BACKEND_ERROR_BODY]
    backend_log(f"{label} non-200 status={resp.status} body={preview!r}")
    raise RuntimeError(f"{label} HTTP {resp.status}: {preview}")


_TTS_DROP_CATEGORIES = {"So", "Sk", "Cs", "Co", "Cn"}
_TTS_DROP_CHARS = {"‍", "⃣", "*", "_", "`", "#", "~"}


def speakable_text(text: str) -> str:
    """Drop emoji/symbols/markdown that Chatterbox vocalizes as groans.

    Returns "" when nothing pronounceable remains (e.g. a trailing lone emoji
    that the sentence segmenter split into its own chunk).
    """
    kept = []
    for ch in text:
        if ch in _TTS_DROP_CHARS or 0xFE00 <= ord(ch) <= 0xFE0F or 0x1F3FB <= ord(ch) <= 0x1F3FF:
            continue
        if unicodedata.category(ch) in _TTS_DROP_CATEGORIES:
            continue
        kept.append(ch)
    cleaned = re.sub(r"\s{2,}", " ", "".join(kept)).strip()
    if not any(ch.isalnum() for ch in cleaned):
        return ""
    return cleaned


def strip_emotion_tags(text: str) -> str:
    cleaned = BRACKET_TAG_RE.sub("", text)
    cleaned = cleaned.replace("[", "").replace("]", "")
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


class SentenceSegmenter:
    _enders = ".!?"
    _closers = "\"')]}”’"

    def __init__(self) -> None:
        self.buf = ""

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sentence = self.buf[:cut].strip()
            self.buf = self.buf[cut:].lstrip()
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> list[str]:
        tail = self.buf.strip()
        self.buf = ""
        return [tail] if tail else []

    def _find_cut(self) -> int | None:
        i = 0
        in_bracket_tag = False
        while i < len(self.buf):
            ch = self.buf[i]
            if ch == "[":
                in_bracket_tag = True
                i += 1
                continue
            if in_bracket_tag:
                if ch == "]":
                    in_bracket_tag = False
                i += 1
                continue
            if ch not in self._enders:
                i += 1
                continue
            if ch == "." and self._is_decimal_point(i):
                i += 1
                continue
            j = i + 1
            while j < len(self.buf) and self.buf[j] in self._enders:
                j += 1
            while j < len(self.buf) and self.buf[j] in self._closers:
                j += 1
            if j == len(self.buf) or self.buf[j].isspace():
                return j
            i = j
        return None

    def _is_decimal_point(self, i: int) -> bool:
        return 0 < i < len(self.buf) - 1 and self.buf[i - 1].isdigit() and self.buf[i + 1].isdigit()


def pcm16_to_wav(pcm16: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return buf.getvalue()


def models_url_from_chat_url(chat_url: str) -> str:
    parts = urlsplit(chat_url)
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    elif path.endswith("/completions"):
        path = path[: -len("/completions")]
    return urlunsplit((parts.scheme, parts.netloc, f"{path}/models", "", ""))


def chat_tools_from_realtime(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        if tool.get("type") != "function":
            continue
        function = {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or tool.get("input_schema") or {"type": "object", "properties": {}},
        }
        if isinstance(function["name"], str) and function["name"]:
            converted.append({"type": "function", "function": function})
    return converted


class WavPcmStripper:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.header_done = False
        self.raw_passthrough = False
        self.odd = b""

    def feed(self, chunk: bytes) -> list[bytes]:
        if not chunk:
            return []
        if self.raw_passthrough or self.header_done:
            return self._even_chunks(chunk)
        self.buf.extend(chunk)
        if len(self.buf) < 12:
            return []
        if not bytes(self.buf[:4]) == b"RIFF" or not bytes(self.buf[8:12]) == b"WAVE":
            self.raw_passthrough = True
            data = bytes(self.buf)
            self.buf.clear()
            return self._even_chunks(data)
        offset = 12
        while len(self.buf) >= offset + 8:
            chunk_id = bytes(self.buf[offset : offset + 4])
            size = int.from_bytes(self.buf[offset + 4 : offset + 8], "little", signed=False)
            data_start = offset + 8
            if chunk_id == b"data":
                self.header_done = True
                data = bytes(self.buf[data_start:])
                self.buf.clear()
                return self._even_chunks(data)
            offset = data_start + size + (size % 2)
        return []

    def flush(self) -> list[bytes]:
        out = [self.odd] if self.odd else []
        self.odd = b""
        return out

    def _even_chunks(self, data: bytes) -> list[bytes]:
        if self.odd:
            data = self.odd + data
            self.odd = b""
        if len(data) % 2:
            self.odd = data[-1:]
            data = data[:-1]
        return [data] if data else []


class Pipeline:
    def __init__(self, http: aiohttp.ClientSession, config: BackendConfig | None = None) -> None:
        self.http = http
        self.config = config or BackendConfig()
        self._models: dict[str, str] = {}
        self._model_lock = asyncio.Lock()

    async def discover_model(self, chat_url: str | None = None) -> str:
        chat_url = chat_url or self.config.llm_url
        if chat_url in self._models:
            return self._models[chat_url]
        async with self._model_lock:
            if chat_url in self._models:
                return self._models[chat_url]
            timeout = aiohttp.ClientTimeout(
                total=10,
                sock_connect=self.config.llm_connect_timeout,
                sock_read=self.config.llm_first_byte_timeout,
            )
            async with self.http.get(models_url_from_chat_url(chat_url), timeout=timeout) as resp:
                await raise_for_status_with_body(resp, "LLM models")
                body = await resp.json()
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, list) or not data:
                raise RuntimeError("LLM /v1/models returned no models")
            model = data[0].get("id") if isinstance(data[0], dict) else None
            if not isinstance(model, str) or not model:
                raise RuntimeError("LLM /v1/models first entry has no id")
            self._models[chat_url] = model
            return model

    async def transcribe(self, pcm16: bytes, *, sample_rate: int = 24000, model: str = "whisper-1") -> str:
        wav = pcm16_to_wav(pcm16, sample_rate)
        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
        async with self.http.post(self.config.stt_url, data=form, timeout=30) as resp:
            await raise_for_status_with_body(resp, "Parakeet STT")
            body = await resp.json()
        for key in ("text", "transcript", "transcription"):
            value = body.get(key) if isinstance(body, dict) else None
            if isinstance(value, str):
                return value.strip()
        return ""

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        fallback_url = self.config.llm_fallback_url
        if not fallback_url or fallback_url == self.config.llm_url:
            async for event in self._stream_chat_once(self.config.llm_url, messages, tools=tools):
                yield event
            return
        started = False
        try:
            async for event in self._stream_chat_once(self.config.llm_url, messages, tools=tools):
                started = True
                yield event
            return
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            # Once audio/tool events went out we cannot replay the turn on another model.
            if started:
                raise
            backend_log(f"primary LLM failed ({type(exc).__name__}: {exc}); falling back to {fallback_url}")
        async for event in self._stream_chat_once(fallback_url, messages, tools=tools):
            yield event

    async def _stream_chat_once(
        self,
        chat_url: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        model = await self.discover_model(chat_url)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.llm_max_tokens,
            # Both vLLM servers think by default; reasoning would eat the spoken-reply budget.
            # DeepSeek's template reads "thinking", Qwen's reads "enable_thinking".
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        }
        if tools:
            payload["tools"] = chat_tools_from_realtime(tools)
            payload["tool_choice"] = "auto"

        def events_from_line(raw_line: str) -> tuple[list[dict[str, Any]], bool]:
            line = raw_line.strip()
            if not line:
                return [], False
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                return [], True
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                return [], False
            if isinstance(data, dict) and data.get("error"):
                error = json.dumps(data["error"], ensure_ascii=False)[:MAX_BACKEND_ERROR_BODY]
                backend_log(f"DeepSeek chat error body={error!r}")
                raise RuntimeError(f"DeepSeek chat error: {error}")
            choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            events: list[dict[str, Any]] = []
            content = delta.get("content") or message.get("content") or ""
            if content:
                events.append({"type": "content", "text": content})
            tool_calls = delta.get("tool_calls") or message.get("tool_calls") or []
            if tool_calls:
                events.append({"type": "tool_calls", "tool_calls": tool_calls})
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                events.append({"type": "finish", "finish_reason": finish_reason})
            return events, False

        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=self.config.llm_connect_timeout,
            sock_read=self.config.llm_first_byte_timeout,
        )
        async with self.http.post(chat_url, json=payload, timeout=timeout) as resp:
            await raise_for_status_with_body(resp, "DeepSeek chat")
            buffer = ""
            saw_event = False
            saw_done = False
            async for chunk in resp.content.iter_any():
                buffer += chunk.decode("utf-8", "replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    events, done = events_from_line(line)
                    if done:
                        saw_done = True
                        return
                    for event in events:
                        saw_event = True
                        yield event
            if buffer.strip():
                events, done = events_from_line(buffer)
                saw_done = saw_done or done
                for event in events:
                    saw_event = True
                    yield event
            if not saw_event and not saw_done:
                backend_log("DeepSeek chat stream ended without response data")
                raise RuntimeError("DeepSeek chat stream ended without response data")

    async def stream_tts_pcm(self, text: str) -> AsyncIterator[bytes]:
        if self.config.speak_numbers:
            text = numbers_to_words(text)
        text = speakable_text(text)
        if not text:
            return
        text = fix_pronunciation(text)
        payload = {
            "text": text,
            "predefined_voice_id": self.config.voice,
            "speed": self.config.speed,
            "cfg_weight": self.config.cfg_weight,
            "exaggeration": self.config.exaggeration,
            "stream": True,
        }
        stripper = WavPcmStripper()
        async with self.http.post(self.config.tts_url, json=payload, timeout=None) as resp:
            await raise_for_status_with_body(resp, "Chatterbox TTS")
            async for chunk in resp.content.iter_chunked(4096):
                for pcm in stripper.feed(chunk):
                    yield pcm
            for pcm in stripper.flush():
                yield pcm
