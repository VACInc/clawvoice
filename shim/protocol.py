"""Small OpenAI Realtime-compatible event helpers for the TARS shim."""

from __future__ import annotations

import time
import uuid
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def now_unix() -> int:
    return int(time.time())


def session_event(kind: str, session: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": kind,
        "event_id": new_id("evt"),
        "session": session,
    }


def error_event(message: str, *, event_id: str | None = None, code: str = "server_error") -> dict[str, Any]:
    return {
        "type": "error",
        "event_id": new_id("evt"),
        "error": {
            "type": "server_error",
            "code": code,
            "message": message,
            **({"event_id": event_id} if event_id else {}),
        },
    }


def response_created(response_id: str) -> dict[str, Any]:
    return {
        "type": "response.created",
        "event_id": new_id("evt"),
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "status": "in_progress",
            "status_details": None,
            "output": [],
            "created_at": now_unix(),
        },
    }


def response_done(
    response_id: str,
    status: str,
    output: list[dict[str, Any]],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "response.done",
        "event_id": new_id("evt"),
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "status": status,
            "status_details": {"type": reason} if reason else None,
            "output": output,
            "created_at": now_unix(),
        },
    }


def output_item_added(response_id: str, item: dict[str, Any], *, output_index: int = 0) -> dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "event_id": new_id("evt"),
        "response_id": response_id,
        "output_index": output_index,
        "item": item,
        "item_id": item.get("id"),
    }


def output_item_done(response_id: str, item: dict[str, Any], *, output_index: int = 0) -> dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "event_id": new_id("evt"),
        "response_id": response_id,
        "output_index": output_index,
        "item": item,
        "item_id": item.get("id"),
    }


def conversation_item_done(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "conversation.item.done",
        "event_id": new_id("evt"),
        "item_id": item.get("id"),
        "item": item,
    }


def audio_delta(response_id: str, item_id: str, delta_b64: str) -> dict[str, Any]:
    return {
        "type": "response.audio.delta",
        "event_id": new_id("evt"),
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta_b64,
    }


def audio_done(response_id: str, item_id: str) -> dict[str, Any]:
    return {
        "type": "response.audio.done",
        "event_id": new_id("evt"),
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
    }


def assistant_transcript_delta(response_id: str, item_id: str, delta: str) -> dict[str, Any]:
    return {
        "type": "response.audio_transcript.delta",
        "event_id": new_id("evt"),
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }


def assistant_transcript_done(response_id: str, item_id: str, transcript: str) -> dict[str, Any]:
    return {
        "type": "response.audio_transcript.done",
        "event_id": new_id("evt"),
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "transcript": transcript,
    }


def function_args_delta(response_id: str, item: dict[str, Any], delta: str) -> dict[str, Any]:
    return {
        "type": "response.function_call_arguments.delta",
        "event_id": new_id("evt"),
        "response_id": response_id,
        "item_id": item.get("id"),
        "call_id": item.get("call_id"),
        "name": item.get("name"),
        "delta": delta,
    }


def function_args_done(response_id: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "response.function_call_arguments.done",
        "event_id": new_id("evt"),
        "response_id": response_id,
        "item_id": item.get("id"),
        "call_id": item.get("call_id"),
        "name": item.get("name"),
        "arguments": item.get("arguments", ""),
    }
