"""aiohttp API for speaker embedding and enrollment."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from aiohttp import web

from .audio import AudioError, decode_audio
from .config import SpeakerConfig
from .core import SpeakerEngine, Thresholds
from .models import ReDimNetBackend, TitaNetBackend
from .store import FlatFileEmbeddingStore

_LOGGER = logging.getLogger(__name__)
ENGINE_KEY: web.AppKey[SpeakerEngine] = web.AppKey("engine", SpeakerEngine)
CONFIG_KEY: web.AppKey[SpeakerConfig] = web.AppKey("config", SpeakerConfig)


def build_engine(config: SpeakerConfig) -> SpeakerEngine:
    primary = ReDimNetBackend(
        source_dir=config.redimnet_source_dir,
        checkpoint=config.redimnet_checkpoint,
        checkpoint_sha256=config.redimnet_checkpoint_sha256,
        device=config.device,
        verify_hash=config.verify_model_hashes,
    )
    secondary = None
    if config.titanet_enabled:
        secondary = TitaNetBackend(
            checkpoint=config.titanet_checkpoint,
            checkpoint_sha256=config.titanet_checkpoint_sha256,
            device=config.device,
            verify_hash=config.verify_model_hashes,
        )
    trimmer = None
    if config.shadow and os.environ.get("SPEAKER_VAD_MODEL"):
        from .vad import SpeechTrimmer
        trimmer = SpeechTrimmer(Path(os.environ["SPEAKER_VAD_MODEL"]))
    return SpeakerEngine(
        trimmer=trimmer,
        primary=primary,
        secondary=secondary,
        store=FlatFileEmbeddingStore(config.enrollment_dir),
        thresholds=Thresholds(config.low_threshold, config.high_threshold),
        secondary_thresholds=Thresholds(
            config.titanet_low_threshold, config.titanet_high_threshold
        ),
        allowed_users=config.allowed_enroll_users,
        shadow=config.shadow,
        min_audio_ms=config.min_audio_ms,
        max_audio_seconds=config.max_audio_seconds,
    )


def _query_int(request: web.Request, name: str, default: int) -> int:
    raw = request.query.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise AudioError(f"{name} must be an integer") from exc
    if value < 1:
        raise AudioError(f"{name} must be positive")
    return value


async def _decode_request(request: web.Request):
    config = request.app[CONFIG_KEY]
    body = await request.read()
    return decode_audio(
        body,
        raw_sample_rate=_query_int(request, "sample_rate", config.raw_sample_rate),
        raw_channels=_query_int(request, "channels", config.raw_channels),
    )


async def health(request: web.Request) -> web.Response:
    result = request.app[ENGINE_KEY].health()
    return web.json_response(result, status=200 if result["ready"] else 503)


async def enrolled(request: web.Request) -> web.Response:
    return web.json_response({"enrolled": request.app[ENGINE_KEY].enrolled()})


async def identify(request: web.Request) -> web.Response:
    device_id = request.query.get("device_id", "").strip()
    if not device_id:
        raise web.HTTPBadRequest(
            text='{"error":"device_id query parameter is required"}',
            content_type="application/json",
        )
    if len(device_id) > 128:
        raise web.HTTPBadRequest(
            text='{"error":"device_id is too long"}',
            content_type="application/json",
        )
    try:
        audio = await _decode_request(request)
        result = await asyncio.to_thread(
            request.app[ENGINE_KEY].identify, audio, device_id
        )
    except (AudioError, ValueError) as exc:
        raise web.HTTPBadRequest(
            text=web.json_response({"error": str(exc)}).text,
            content_type="application/json",
        ) from exc
    except (FileNotFoundError, ImportError, OSError, RuntimeError) as exc:
        _LOGGER.exception("Identification backend unavailable")
        raise web.HTTPServiceUnavailable(
            text=web.json_response({"error": str(exc)}).text,
            content_type="application/json",
        ) from exc
    return web.json_response(result)


async def enroll(request: web.Request) -> web.Response:
    user = request.query.get("user", "").strip()
    if not user:
        raise web.HTTPBadRequest(
            text='{"error":"user query parameter is required"}',
            content_type="application/json",
        )
    try:
        audio = await _decode_request(request)
        result = await asyncio.to_thread(request.app[ENGINE_KEY].enroll, audio, user)
    except PermissionError as exc:
        raise web.HTTPForbidden(
            text=web.json_response({"error": str(exc)}).text,
            content_type="application/json",
        ) from exc
    except (AudioError, ValueError) as exc:
        raise web.HTTPBadRequest(
            text=web.json_response({"error": str(exc)}).text,
            content_type="application/json",
        ) from exc
    except (FileNotFoundError, ImportError, OSError, RuntimeError) as exc:
        _LOGGER.exception("Enrollment backend unavailable")
        raise web.HTTPServiceUnavailable(
            text=web.json_response({"error": str(exc)}).text,
            content_type="application/json",
        ) from exc
    return web.json_response(result)


def create_app(
    config: SpeakerConfig, engine: Optional[SpeakerEngine] = None
) -> web.Application:
    app = web.Application(client_max_size=config.max_body_bytes)
    app[CONFIG_KEY] = config
    app[ENGINE_KEY] = engine or build_engine(config)
    app.add_routes(
        [
            web.get("/health", health),
            web.get("/enrolled", enrolled),
            web.post("/identify", identify),
            web.post("/enroll", enroll),
        ]
    )
    return app
