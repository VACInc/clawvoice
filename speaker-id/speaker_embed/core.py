"""Speaker enrollment, averaging, scoring, and threshold policy."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .audio import DecodedAudio
from .models import EmbeddingBackend
from .store import EmbeddingStore, normalize

_LOGGER = logging.getLogger(__name__)
_SAFE_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class Thresholds:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not (-1.0 <= self.low <= self.high <= 1.0):
            raise ValueError("thresholds must satisfy -1 <= low <= high <= 1")


@dataclass(frozen=True)
class Decision:
    user_id: str
    confidence: float
    status: str
    candidate_user_id: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "user_id": self.user_id,
            "confidence": round(float(self.confidence), 6),
            "status": self.status,
        }
        if self.candidate_user_id is not None:
            result["candidate_user_id"] = self.candidate_user_id
        return result


def decide(
    embedding: np.ndarray,
    centroids: Mapping[str, np.ndarray],
    thresholds: Thresholds,
) -> Decision:
    if not centroids:
        return Decision("unknown", 0.0, "unknown", None)
    query = normalize(embedding)
    scores = {
        user: float(np.dot(query, normalize(centroid)))
        for user, centroid in centroids.items()
    }
    user, score = max(scores.items(), key=lambda item: item[1])
    score = max(-1.0, min(1.0, score))
    if score >= thresholds.high:
        return Decision(user, score, "accepted", user)
    if score >= thresholds.low:
        return Decision("unknown", score, "uncertain", user)
    return Decision("unknown", score, "unknown", user)


class SpeakerEngine:
    def __init__(
        self,
        *,
        primary: EmbeddingBackend,
        store: EmbeddingStore,
        thresholds: Thresholds,
        allowed_users: frozenset[str],
        shadow: bool = True,
        secondary: Optional[EmbeddingBackend] = None,
        secondary_thresholds: Optional[Thresholds] = None,
        min_audio_ms: int = 300,
        max_audio_seconds: float = 30.0,
        trimmer=None,
    ):
        if trimmer is not None and not shadow:
            raise ValueError("speech trimming is SHADOW-only")
        self.trimmer = trimmer
        self.primary = primary
        self.secondary = secondary
        self.store = store
        self.thresholds = thresholds
        self.secondary_thresholds = secondary_thresholds or thresholds
        self.allowed_users = allowed_users
        self.shadow = shadow
        self.min_audio_ms = min_audio_ms
        self.max_audio_seconds = max_audio_seconds

    def _validate_audio(self, audio: DecodedAudio) -> None:
        if audio.seconds * 1000.0 < self.min_audio_ms:
            raise ValueError(f"audio must be at least {self.min_audio_ms} ms")
        if audio.seconds > self.max_audio_seconds:
            raise ValueError(
                f"audio must be no longer than {self.max_audio_seconds:g} seconds"
            )
        if not np.isfinite(audio.samples).all():
            raise ValueError("audio contains non-finite samples")

    def identify(self, audio: DecodedAudio, device_id: str) -> Dict[str, Any]:
        self._validate_audio(audio)
        original_ms = round(audio.seconds * 1000, 3)
        if self.trimmer is not None:
            audio = self.trimmer.trim(audio)
        started = time.perf_counter()
        embedding = self.primary.embed(audio.samples, audio.sample_rate)
        decision = decide(
            embedding,
            self.store.centroids(self.primary.store_key),
            self.thresholds,
        )
        result = decision.as_dict()
        result.update(
            {
                "device_id": device_id,
                "model": self.primary.name,
                "ms": round((time.perf_counter() - started) * 1000.0, 3),
                "enforced": not self.shadow,
                "vad": {"enabled": self.trimmer is not None, "input_ms": original_ms, "speech_ms": round(audio.seconds * 1000, 3)},
            }
        )

        if self.secondary is not None:
            try:
                secondary_embedding = self.secondary.embed(
                    audio.samples, audio.sample_rate
                )
                secondary_decision = decide(
                    secondary_embedding,
                    self.store.centroids(self.secondary.store_key),
                    self.secondary_thresholds,
                )
                shadow_result = secondary_decision.as_dict()
                shadow_result["model"] = self.secondary.name
                result["shadow_score"] = shadow_result
            except Exception as exc:  # secondary is never allowed to break primary
                _LOGGER.exception("TitaNet shadow scoring failed")
                result["shadow_error"] = type(exc).__name__
        return result

    def enroll(self, audio: DecodedAudio, user: str) -> Dict[str, Any]:
        self._validate_audio(audio)
        if not _SAFE_USER.fullmatch(user):
            raise ValueError("user contains unsupported characters")
        if user not in self.allowed_users:
            raise PermissionError(f"enrollment is not allowed for user {user!r}")

        started = time.perf_counter()
        primary_embedding = self.primary.embed(audio.samples, audio.sample_rate)
        secondary_embedding: Optional[np.ndarray] = None
        secondary_error: Optional[str] = None
        if self.secondary is not None:
            try:
                secondary_embedding = self.secondary.embed(
                    audio.samples, audio.sample_rate
                )
            except Exception as exc:
                _LOGGER.exception("TitaNet shadow enrollment failed")
                secondary_error = type(exc).__name__

        count = self.store.add(self.primary.store_key, user, primary_embedding)
        result: Dict[str, Any] = {
            "user": user,
            "samples": count,
            "model": self.primary.name,
            "ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        if (self.secondary is not None) and (secondary_embedding is not None):
            secondary_count = self.store.add(
                self.secondary.store_key, user, secondary_embedding
            )
            result["shadow_model"] = self.secondary.name
            result["shadow_samples"] = secondary_count
        elif secondary_error is not None:
            result["shadow_error"] = secondary_error
        return result

    def enrolled(self) -> Mapping[str, Mapping[str, int]]:
        return self.store.summary()

    def health(self) -> Dict[str, Any]:
        primary_ready = bool(self.primary.ready)
        result: Dict[str, Any] = {
            "status": "ok" if primary_ready else "model_missing",
            "ready": primary_ready,
            "shadow": self.shadow,
            "primary": {
                "model": self.primary.name,
                "ready": primary_ready,
                "loaded": bool(self.primary.loaded),
            },
            "titanet_enabled": self.secondary is not None,
        }
        if self.secondary is not None:
            result["secondary"] = {
                "model": self.secondary.name,
                "ready": bool(self.secondary.ready),
                "loaded": bool(self.secondary.loaded),
            }
        return result
