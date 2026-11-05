"""Raw Wyoming frame reader that preserves the original wire bytes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from wyoming.event import Event


class WyomingFrameError(ValueError):
    """Raised when a Wyoming wire frame is malformed or too large."""


@dataclass(frozen=True)
class RawWyomingFrame:
    event: Event
    wire_bytes: bytes
    header_bytes: bytes
    data_bytes: bytes
    payload_bytes: bytes


def _length(header: Dict[str, Any], name: str) -> int:
    value = header.get(name, 0)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WyomingFrameError(f"{name} must be a non-negative integer")
    return value


async def read_raw_event(
    reader: asyncio.StreamReader,
    *,
    max_header_bytes: int,
    max_frame_bytes: int,
) -> Optional[RawWyomingFrame]:
    """Read one Wyoming event while retaining its byte-for-byte representation."""
    header_bytes = await reader.readline()
    if not header_bytes:
        return None
    if len(header_bytes) > max_header_bytes:
        raise WyomingFrameError("Wyoming header exceeds MAX_HEADER_BYTES")
    if not header_bytes.endswith(b"\n"):
        raise WyomingFrameError("Wyoming header is not newline terminated")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WyomingFrameError("invalid Wyoming JSON header") from exc
    if not isinstance(header, dict) or not isinstance(header.get("type"), str):
        raise WyomingFrameError("Wyoming header must contain a string type")

    data_length = _length(header, "data_length")
    payload_length = _length(header, "payload_length")
    total = len(header_bytes) + data_length + payload_length
    if total > max_frame_bytes:
        raise WyomingFrameError("Wyoming frame exceeds MAX_FRAME_BYTES")
    data_bytes = await reader.readexactly(data_length) if data_length else b""
    payload_bytes = await reader.readexactly(payload_length) if payload_length else b""

    inline_data = header.get("data", {})
    if inline_data is None:
        inline_data = {}
    if not isinstance(inline_data, dict):
        raise WyomingFrameError("Wyoming data must be a JSON object")
    event_data = dict(inline_data)
    if data_bytes:
        try:
            extra_data = json.loads(data_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WyomingFrameError("invalid Wyoming extra data JSON") from exc
        if not isinstance(extra_data, dict):
            raise WyomingFrameError("Wyoming extra data must be a JSON object")
        event_data.update(extra_data)

    return RawWyomingFrame(
        event=Event(
            type=header["type"],
            data=event_data,
            payload=payload_bytes if payload_length else None,
        ),
        wire_bytes=header_bytes + data_bytes + payload_bytes,
        header_bytes=header_bytes,
        data_bytes=data_bytes,
        payload_bytes=payload_bytes,
    )
