"""Environment-backed configuration for speaker-embed."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


REDIMNET_REVISION = "9438b1e98018d831df15fde93c580faae0a28d63"
REDIMNET_CHECKPOINT_SHA256 = (
    "1e0716c2f351e79df054554f17c525b899045d2d05e4c839208b975e87d94e6a"
)
TITANET_REVISION = "0dc382f40121a5fbd34db10a2bb04d826c2be6a8"
TITANET_CHECKPOINT_SHA256 = (
    "e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3"
)


def _bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be 0/1, true/false, yes/no, or on/off")


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    value = int(env.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(
    env: Mapping[str, str], name: str, default: float, minimum: Optional[float] = None
) -> float:
    value = float(env.get(name, str(default)))
    if (minimum is not None) and (value < minimum):
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _path(env: Mapping[str, str], name: str, default: str) -> Path:
    return Path(env.get(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class SpeakerConfig:
    host: str
    port: int
    model_dir: Path
    enrollment_dir: Path
    redimnet_source_dir: Path
    redimnet_checkpoint: Path
    redimnet_checkpoint_sha256: str
    device: str
    high_threshold: float
    low_threshold: float
    shadow: bool
    titanet_enabled: bool
    titanet_checkpoint: Path
    titanet_checkpoint_sha256: str
    titanet_high_threshold: float
    titanet_low_threshold: float
    allowed_enroll_users: frozenset[str]
    raw_sample_rate: int
    raw_channels: int
    min_audio_ms: int
    max_audio_seconds: float
    max_body_bytes: int
    verify_model_hashes: bool
    log_level: str

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "SpeakerConfig":
        env = os.environ if environ is None else environ
        model_dir = _path(env, "MODEL_DIR", "models")
        enrollment_dir = _path(env, "ENROLLMENT_DIR", "data/enrollments")
        source_default = str(model_dir / f"redimnet-{REDIMNET_REVISION}")
        checkpoint_default = str(model_dir / "M-vb2+vox2+cnc-ft_mix.pt")
        titanet_default = str(model_dir / "speakerverification_en_titanet_large.nemo")
        users = frozenset(
            user.strip()
            for user in env.get("ALLOWED_ENROLL_USERS", "alice").split(",")
            if user.strip()
        )
        if not users:
            raise ValueError("ALLOWED_ENROLL_USERS must contain at least one user")

        low = _float(env, "LOW_THRESHOLD", 0.55)
        high = _float(env, "HIGH_THRESHOLD", 0.72)
        titan_low = _float(env, "TITANET_LOW_THRESHOLD", 0.50)
        titan_high = _float(env, "TITANET_HIGH_THRESHOLD", 0.65)
        for low_value, high_value, prefix in (
            (low, high, "ReDimNet"),
            (titan_low, titan_high, "TitaNet"),
        ):
            if not (-1.0 <= low_value <= high_value <= 1.0):
                raise ValueError(
                    f"{prefix} thresholds must satisfy -1 <= low <= high <= 1"
                )

        raw_channels = _int(env, "RAW_CHANNELS", 1, minimum=1)
        if raw_channels > 8:
            raise ValueError("RAW_CHANNELS must be <= 8")

        return cls(
            host=env.get("SPEAKER_HOST", "0.0.0.0"),
            port=_int(env, "SPEAKER_PORT", 8099, minimum=1),
            model_dir=model_dir,
            enrollment_dir=enrollment_dir,
            redimnet_source_dir=_path(env, "REDIMNET_SOURCE_DIR", source_default),
            redimnet_checkpoint=_path(env, "REDIMNET_CHECKPOINT", checkpoint_default),
            redimnet_checkpoint_sha256=env.get(
                "REDIMNET_CHECKPOINT_SHA256", REDIMNET_CHECKPOINT_SHA256
            ).lower(),
            device=env.get("DEVICE", "auto").strip().lower(),
            high_threshold=high,
            low_threshold=low,
            shadow=_bool(env.get("SHADOW", "1"), "SHADOW"),
            titanet_enabled=_bool(env.get("TITANET_ENABLED", "0"), "TITANET_ENABLED"),
            titanet_checkpoint=_path(env, "TITANET_CHECKPOINT", titanet_default),
            titanet_checkpoint_sha256=env.get(
                "TITANET_CHECKPOINT_SHA256", TITANET_CHECKPOINT_SHA256
            ).lower(),
            titanet_high_threshold=titan_high,
            titanet_low_threshold=titan_low,
            allowed_enroll_users=users,
            raw_sample_rate=_int(env, "RAW_SAMPLE_RATE", 16000, minimum=1),
            raw_channels=raw_channels,
            min_audio_ms=_int(env, "MIN_AUDIO_MS", 300, minimum=1),
            max_audio_seconds=_float(env, "MAX_AUDIO_SECONDS", 30.0, minimum=0.1),
            max_body_bytes=_int(env, "MAX_BODY_BYTES", 16 * 1024 * 1024, minimum=1024),
            verify_model_hashes=_bool(
                env.get("VERIFY_MODEL_HASHES", "1"), "VERIFY_MODEL_HASHES"
            ),
            log_level=env.get("LOG_LEVEL", "INFO").upper(),
        )
