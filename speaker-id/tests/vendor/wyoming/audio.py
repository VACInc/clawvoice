from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .event import Event, Eventable


@dataclass
class AudioChunk(Eventable):
    rate: int
    width: int
    channels: int
    audio: bytes
    timestamp: Optional[int] = None

    @staticmethod
    def is_type(event_type: str) -> bool:
        return event_type == "audio-chunk"

    def event(self) -> Event:
        return Event(
            "audio-chunk",
            {
                "rate": self.rate,
                "width": self.width,
                "channels": self.channels,
                "timestamp": self.timestamp,
            },
            self.audio,
        )

    @staticmethod
    def from_event(event: Event) -> "AudioChunk":
        return AudioChunk(
            event.data["rate"],
            event.data["width"],
            event.data["channels"],
            event.payload or b"",
            event.data.get("timestamp"),
        )


@dataclass
class AudioStart(Eventable):
    rate: int
    width: int
    channels: int
    timestamp: Optional[int] = None

    @staticmethod
    def is_type(event_type: str) -> bool:
        return event_type == "audio-start"

    def event(self) -> Event:
        return Event(
            "audio-start",
            {
                "rate": self.rate,
                "width": self.width,
                "channels": self.channels,
                "timestamp": self.timestamp,
            },
        )

    @staticmethod
    def from_event(event: Event) -> "AudioStart":
        return AudioStart(
            event.data["rate"],
            event.data["width"],
            event.data["channels"],
            event.data.get("timestamp"),
        )


@dataclass
class AudioStop(Eventable):
    timestamp: Optional[int] = None

    @staticmethod
    def is_type(event_type: str) -> bool:
        return event_type == "audio-stop"

    def event(self) -> Event:
        return Event("audio-stop", {"timestamp": self.timestamp})

    @staticmethod
    def from_event(event: Event) -> "AudioStop":
        return AudioStop(event.data.get("timestamp"))
