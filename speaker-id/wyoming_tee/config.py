"""Environment-backed wyoming-tee configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlparse

_SAFE_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    value = int(env.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(
    env: Mapping[str, str], name: str, default: float, minimum: float = 0.0
) -> float:
    value = float(env.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class TeeConfig:
    host: str
    port: int
    real_stt_host: str
    real_stt_port: int
    embed_url: str
    ha_webhook_url: str
    device_id: str
    shadow: bool
    identify_timeout_seconds: float
    webhook_timeout_seconds: float
    webhook_retries: int
    webhook_retry_base_seconds: float
    webhook_retry_max_seconds: float
    max_utterance_bytes: int
    max_frame_bytes: int
    max_header_bytes: int
    relay_buffer_bytes: int
    capture_user: str
    capture_dir: Path
    shutdown_grace_seconds: float
    log_level: str

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "TeeConfig":
        env = os.environ if environ is None else environ
        config = cls(
            host=env.get("TEE_HOST", "0.0.0.0"),
            port=_int(env, "TEE_PORT", 10301, minimum=1),
            real_stt_host=env.get("REAL_STT_HOST", "127.0.0.1"),
            real_stt_port=_int(env, "REAL_STT_PORT", 10300, minimum=1),
            embed_url=env.get("EMBED_URL", "http://127.0.0.1:8099/identify"),
            ha_webhook_url=env.get("HA_WEBHOOK_URL", ""),
            device_id=env.get("DEVICE_ID", "voice_satellite"),
            shadow=_bool(env.get("SHADOW", "1"), "SHADOW"),
            identify_timeout_seconds=_float(
                env, "IDENTIFY_TIMEOUT_SECONDS", 10.0, minimum=0.1
            ),
            webhook_timeout_seconds=_float(
                env, "WEBHOOK_TIMEOUT_SECONDS", 3.0, minimum=0.1
            ),
            webhook_retries=_int(env, "WEBHOOK_RETRIES", 4),
            webhook_retry_base_seconds=_float(env, "WEBHOOK_RETRY_BASE_SECONDS", 0.5),
            webhook_retry_max_seconds=_float(env, "WEBHOOK_RETRY_MAX_SECONDS", 5.0),
            max_utterance_bytes=_int(
                env, "MAX_UTTERANCE_BYTES", 16 * 1024 * 1024, minimum=1024
            ),
            max_frame_bytes=_int(
                env, "MAX_FRAME_BYTES", 32 * 1024 * 1024, minimum=1024
            ),
            max_header_bytes=_int(env, "MAX_HEADER_BYTES", 64 * 1024, minimum=256),
            relay_buffer_bytes=_int(env, "RELAY_BUFFER_BYTES", 64 * 1024, minimum=1024),
            capture_user=env.get("CAPTURE_USER", "").strip(),
            capture_dir=Path(env.get("CAPTURE_DIR", "data/captures"))
            .expanduser()
            .resolve(),
            shutdown_grace_seconds=_float(
                env, "SHUTDOWN_GRACE_SECONDS", 10.0, minimum=0.1
            ),
            log_level=env.get("LOG_LEVEL", "INFO").upper(),
        )
        if not config.device_id or len(config.device_id) > 128:
            raise ValueError("DEVICE_ID must contain 1-128 characters")
        if config.capture_user and not _SAFE_USER.fullmatch(config.capture_user):
            raise ValueError("CAPTURE_USER contains unsupported characters")
        for name, url, optional in (
            ("EMBED_URL", config.embed_url, False),
            ("HA_WEBHOOK_URL", config.ha_webhook_url, True),
        ):
            if optional and not url:
                continue
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{name} must be an absolute http(s) URL")
        return config
