"""Spoken-name respelling for the TTS stage.

Chatterbox reads "VAC" as "back" (no /v/) and reads the all-caps "TARS" token
with an unstable vowel ("TAARS" on some renders). Respelling both to
plain phonetic tokens before synthesis fixes the first outright and pins the
second:

    "VAC"  -> "Vack"   (rhymes with "back"; /v/ survives)
    "TARS" -> "Tars"   (real word, /tɑːrz/, the movie pronunciation)

Only whole-word Title/UPPER forms change, so ordinary words ("vacuum",
"tars", "the stars") are untouched. This is a text-stage rule, not a
transcript rule: it runs at TTS request time and never reaches a client,
a log, or the model.

The shim imports this module directly; the HA TTS wrapper
(tts/chatterbox-openai/server.py) imports an installed copy beside it.
TTS_PRONOUNCE_FIX=0 turns it off everywhere.
"""

from __future__ import annotations

import os
import re

ENABLED = os.environ.get("TTS_PRONOUNCE_FIX", "1").strip() != "0"

# Whole-word, exact-case alternatives: lowercase English words stay alone.
_RULES = (
    (re.compile(r"(?<!\w)(?:VAC|Vac)(?!\w)"), "Vack"),
    (re.compile(r"(?<!\w)(?:TARS|Tars)(?!\w)"), "Tars"),
)


def fix_pronunciation(text: str) -> str:
    """Respells the names VAC and TARS the way Chatterbox actually says them."""
    if not ENABLED or not text:
        return text
    for pattern, spoken in _RULES:
        text = pattern.sub(spoken, text)
    return text
