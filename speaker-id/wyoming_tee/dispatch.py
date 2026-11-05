"""Detached embedding request, capture, and HA webhook retry pipeline."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import aiohttp

from speaker_embed.audio import pcm16_wav_bytes

from .config import TeeConfig

_LOGGER = logging.getLogger(__name__)
_SAFE_CAPTURE_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class Utterance:
    chunks: tuple[bytes, ...]
    sample_rate: int
    width: int
    channels: int

    @property
    def pcm(self) -> bytes:
        return b"".join(self.chunks)

    def to_wav(self) -> bytes:
        if self.width != 2:
            raise ValueError("speaker identification requires PCM16 audio")
        return pcm16_wav_bytes(
            self.pcm, sample_rate=self.sample_rate, channels=self.channels
        )


class WebhookPoster:
    def __init__(self, session: aiohttp.ClientSession, config: TeeConfig):
        self.session = session
        self.url = config.ha_webhook_url
        self.timeout = aiohttp.ClientTimeout(total=config.webhook_timeout_seconds)
        self.retries = config.webhook_retries
        self.retry_base = config.webhook_retry_base_seconds
        self.retry_max = config.webhook_retry_max_seconds

    async def post(self, stamp: Mapping[str, Any]) -> bool:
        if not self.url:
            _LOGGER.warning("HA_WEBHOOK_URL is empty; identity stamp was not posted")
            return False
        for attempt in range(self.retries + 1):
            retryable = False
            retry_after: Optional[float] = None
            try:
                async with self.session.post(
                    self.url, json=dict(stamp), timeout=self.timeout
                ) as response:
                    await response.read()
                    if 200 <= response.status < 300:
                        return True
                    retryable = (
                        response.status in {408, 425, 429} or response.status >= 500
                    )
                    retry_header = response.headers.get("Retry-After")
                    if retry_header is not None:
                        try:
                            retry_after = max(0.0, float(retry_header))
                        except ValueError:
                            retry_after = None
                    _LOGGER.warning(
                        "HA webhook returned status %d (attempt %d/%d)",
                        response.status,
                        attempt + 1,
                        self.retries + 1,
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError):
                retryable = True
                _LOGGER.warning(
                    "HA webhook unavailable (attempt %d/%d)",
                    attempt + 1,
                    self.retries + 1,
                )

            if (not retryable) or (attempt >= self.retries):
                return False
            delay = min(self.retry_max, self.retry_base * (2**attempt))
            if retry_after is not None:
                delay = min(self.retry_max, retry_after)
            await asyncio.sleep(delay)
        return False


class IdentityDispatcher:
    """All work in this class runs outside the STT relay coroutine."""

    def __init__(
        self,
        config: TeeConfig,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.config = config
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "wyoming-tee/1.0"}
            )
        return self._session

    @staticmethod
    def _write_capture(root: Path, user: str, wav_bytes: bytes) -> Path:
        if not _SAFE_CAPTURE_USER.fullmatch(user):
            raise ValueError("capture user contains unsupported characters")
        user_dir = root / user
        user_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        path = user_dir / f"{timestamp}-{uuid.uuid4().hex}.wav"
        with path.open("xb") as output_file:
            output_file.write(wav_bytes)
            output_file.flush()
        return path

    async def process(
        self,
        utterance: Utterance,
        *,
        device_id: str,
        utterance_ts: float,
        capture_user: str = "",
    ) -> None:
        try:
            # Joining chunks and constructing a WAV can copy megabytes. Keep it
            # off the event loop that is relaying the STT transcript.
            wav_bytes = await asyncio.to_thread(utterance.to_wav)
        except Exception:
            _LOGGER.exception("Skipping identity for unsupported audio format")
            return

        if capture_user:
            try:
                path = await asyncio.to_thread(
                    self._write_capture,
                    self.config.capture_dir,
                    capture_user,
                    wav_bytes,
                )
                _LOGGER.info("Captured enrollment utterance at %s", path)
            except Exception:
                _LOGGER.exception("Could not save enrollment capture")

        session = await self._get_session()
        identify_timeout = aiohttp.ClientTimeout(
            total=self.config.identify_timeout_seconds
        )
        try:
            async with session.post(
                self.config.embed_url,
                params={"device_id": device_id},
                data=wav_bytes,
                headers={"Content-Type": "audio/wav"},
                timeout=identify_timeout,
            ) as response:
                if response.status < 200 or response.status >= 300:
                    await response.read()
                    _LOGGER.warning(
                        "Embedding service returned status %d", response.status
                    )
                    return
                result = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            _LOGGER.warning("Embedding service unavailable; identity stamp omitted")
            return
        if not isinstance(result, dict):
            _LOGGER.warning("Embedding service returned a non-object response")
            return

        user_id = result.get("user_id", "unknown")
        confidence = result.get("confidence", 0.0)
        if not isinstance(user_id, str) or not isinstance(confidence, (int, float)):
            _LOGGER.warning("Embedding response is missing user_id/confidence")
            return
        stamp = {
            "device_id": device_id,
            "user_id": user_id,
            "confidence": float(confidence),
            "ts": utterance_ts,
            "enforced": bool(result.get("enforced", False)) and not self.config.shadow,
        }
        await WebhookPoster(session, self.config).post(stamp)

    async def close(self) -> None:
        if self._owns_session and (self._session is not None):
            await self._session.close()
            self._session = None
