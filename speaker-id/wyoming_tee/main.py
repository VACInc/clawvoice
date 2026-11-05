"""wyoming-tee process entry point."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from .config import TeeConfig
from .proxy import TeeProxy


async def _run() -> None:
    config = TeeConfig.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    proxy = TeeProxy(config)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(stop_signal, stop_event.set)
    run_task = asyncio.create_task(proxy.run(), name="Wyoming tee server")
    stop_task = asyncio.create_task(stop_event.wait(), name="signal waiter")
    try:
        done, _ = await asyncio.wait(
            {run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if run_task in done:
            await run_task
    finally:
        run_task.cancel()
        stop_task.cancel()
        await asyncio.gather(run_task, stop_task, return_exceptions=True)
        await proxy.close()


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
