"""Use a tiny wire-compatible Wyoming fixture only in network-isolated test hosts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import tests.net_guard as net_guard

net_guard.install()


if importlib.util.find_spec("wyoming") is None:
    # Production always uses the pinned PyPI dependency. This fallback mirrors the
    # v1.10.0 Event/Audio/ASR primitives so the repository's tests can still run
    # in an offline sandbox that cannot install the wheel.
    sys.path.insert(0, str(Path(__file__).parent / "vendor"))
