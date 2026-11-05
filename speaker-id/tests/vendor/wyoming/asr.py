from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .event import Event, Eventable


@dataclass
class Transcribe(Eventable):
    name: Optional[str] = None
    language: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    @staticmethod
    def is_type(event_type: str) -> bool:
        return event_type == "transcribe"

    def event(self) -> Event:
        data: Dict[str, Any] = {}
        if self.name is not None:
            data["name"] = self.name
        if self.language is not None:
            data["language"] = self.language
        if self.context is not None:
            data["context"] = self.context
        return Event("transcribe", data)


@dataclass
class Transcript(Eventable):
    text: str

    @staticmethod
    def is_type(event_type: str) -> bool:
        return event_type == "transcript"

    def event(self) -> Event:
        return Event("transcript", {"text": self.text})
