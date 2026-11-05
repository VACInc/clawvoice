from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Event:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    payload: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "data": self.data}


class Eventable(ABC):
    @abstractmethod
    def event(self) -> Event: ...

    @staticmethod
    @abstractmethod
    def is_type(event_type: str) -> bool: ...


async def async_read_event(reader):
    json_line = await reader.readline()
    if not json_line:
        return None
    event_dict = json.loads(json_line)
    data_length = event_dict.get("data_length", 0) or 0
    if data_length:
        extra = json.loads(await reader.readexactly(data_length))
        data = event_dict.get("data", {})
        data.update(extra)
        event_dict["data"] = data
    payload_length = event_dict.get("payload_length", 0) or 0
    payload = await reader.readexactly(payload_length) if payload_length else None
    return Event(
        type=event_dict["type"], data=event_dict.get("data", {}), payload=payload
    )


async def async_write_event(event: Event, writer) -> None:
    header: Dict[str, Any] = {"type": event.type, "version": "1.10.0"}
    data_bytes = None
    if event.data:
        data_bytes = json.dumps(event.data, ensure_ascii=False).encode("utf-8")
        header["data_length"] = len(data_bytes)
    if event.payload:
        header["payload_length"] = len(event.payload)
    writer.write(json.dumps(header, ensure_ascii=False).encode() + b"\n")
    if data_bytes:
        writer.write(data_bytes)
    if event.payload:
        writer.write(event.payload)
    await writer.drain()
