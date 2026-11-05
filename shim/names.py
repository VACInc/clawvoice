"""Post-STT fix for the name "TARS".

Parakeet has no hotword or prompt input, so it spells the name as a nearby
word: "Tarz", "Terz", "Taris", "Tars", "Tors", "Taras", "Terse", "Taurus",
"Stars", "Nars". Spellings that are not real words become "TARS" everywhere.
Real words ("stars", "star", "tar", "taurus") change only where they are
spoken TO TARS: the whole utterance, the first word before a comma, right
after a greeting ("hey", "thanks"), or as the last word after a single comma
("Can you hear me, Tarz?"). "I love the stars." stays as is.

This file is the one copy of the rule for every speech route: the realtime
shims (Talk 8820, phone 8821) import it here, the HA Wyoming bridge ships
a copy of this file, and channel voice notes run
transcribe_note.py. Each route applies it exactly once, at its STT exit.
fix_tars_name is idempotent, so a second pass changes nothing.

TARS_NAME_FIX=0 (or the older RTSHIM_NAME_FIX=0) turns the fix off.
"""

from __future__ import annotations

import os
import re

NAME = "TARS"
ENABLED = all(os.environ.get(key, "1").strip() != "0" for key in ("TARS_NAME_FIX", "RTSHIM_NAME_FIX"))

_W = r"[\w'’]"  # word characters, apostrophes included ("tar's")
# Never real words in a call: always the name.
_ALWAYS = re.compile(rf"(?<!{_W})(?:tars|tarz|tarrs|tarss|terz|taris)(?!{_W})", re.IGNORECASE)
# Real words or names: only fixed when TARS is being addressed. Seen in real
# call logs or the live Parakeet probes: Stars, Nars, Tors, Taras, Terse, Taurus.
_VAR = r"(?:stars?|nars|tors|taras|terse|taurus|taz|tsars?|czars?|tar['’]s|tar)"
_GREET = (
    r"(?:hey|hi|hello|yo|ok|okay|thanks|thank\s+you|cheers|bye|goodbye"
    r"|good\s+(?:morning|night|evening|afternoon)|sorry|please|alright|all\s+right)"
)
_END = r"(?=\s*(?:[.?!,;:]|$))"

# A lone "Star." is a phone keypad word, so the singular never counts as the whole utterance.
_WHOLE = re.compile(rf"^(\W*)(?!star\W*$){_VAR}(\W*)$", re.IGNORECASE)
_START = re.compile(rf"^(\s*){_VAR}(?=\s*[,?!])", re.IGNORECASE)
_AFTER_GREET = re.compile(rf"((?<!{_W}){_GREET}[\s,]+){_VAR}{_END}", re.IGNORECASE)
# "Hey Stars what's up": no comma needed after a greeting that only ever opens an address.
_AFTER_HEY = re.compile(rf"((?<!{_W})(?:hey|hi|hello|yo)\s+){_VAR}(?!{_W})", re.IGNORECASE)
_AFTER_COMMA = re.compile(rf"(,\s*){_VAR}(?=\s*(?:[.?!]|$))", re.IGNORECASE)
_SENTENCE_START = re.compile(r"[.?!]\s+")
_LIST = re.compile(r",\s*(?:and|or)\s", re.IGNORECASE)


def _lone_comma(text: str, comma_at: int) -> bool:
    """True when the comma at comma_at is the only comma in its sentence so far.

    "Thanks, Nars." is an address; "sun, moon, stars." is a list.
    """
    sentence = _SENTENCE_START.split(text[:comma_at])[-1]
    return "," not in sentence


def fix_tars_name(text: str) -> str:
    if not ENABLED or not text:
        return text
    out = _ALWAYS.sub(NAME, text)
    whole = _WHOLE.match(out)
    if whole:
        return f"{whole.group(1)}{NAME}{whole.group(2)}"
    if not _LIST.search(out):
        out = _START.sub(lambda m: m.group(1) + NAME, out)
    out = _AFTER_HEY.sub(lambda m: m.group(1) + NAME, out)
    out = _AFTER_GREET.sub(lambda m: m.group(1) + NAME, out)
    out = _AFTER_COMMA.sub(
        lambda m: m.group(1) + NAME if _lone_comma(out, m.start()) else m.group(0),
        out,
    )
    return out
