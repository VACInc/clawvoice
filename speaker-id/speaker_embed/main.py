"""speaker-embed process entry point."""

from __future__ import annotations

import logging

from aiohttp import web

from .app import create_app
from .config import SpeakerConfig


def main() -> None:
    config = SpeakerConfig.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    web.run_app(
        create_app(config),
        host=config.host,
        port=config.port,
        print=None,
        access_log=logging.getLogger("speaker_embed.access"),
    )


if __name__ == "__main__":
    main()
