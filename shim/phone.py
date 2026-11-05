"""Phone-line helpers: G.711 mu-law codec, 8 kHz <-> 24 kHz resampling, numbers to words.

Twilio media streams carry 8 kHz mu-law. The shim keeps its whole pipeline
(VAD, STT, TTS pacing) at PCM16 24 kHz and converts only at the wire edge.
"""

from __future__ import annotations

import re

import numpy as np

PCM_RATE = 24000
PHONE_RATE = 8000
_FACTOR = PCM_RATE // PHONE_RATE

_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635


def ulaw_decode(data: bytes) -> np.ndarray:
    """G.711 mu-law bytes -> int16 samples."""
    u = ~np.frombuffer(data, dtype=np.uint8).astype(np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = (((mantissa << 3) + _ULAW_BIAS) << exponent) - _ULAW_BIAS
    return np.where(sign != 0, -magnitude, magnitude).astype(np.int16)


def ulaw_encode(samples: np.ndarray) -> bytes:
    """int16 samples -> G.711 mu-law bytes (Sun g711.c / CPython audioop rounding)."""
    x14 = np.asarray(samples, dtype=np.int32) >> 2
    negative = x14 < 0
    magnitude = np.minimum(np.where(negative, -x14, x14), _ULAW_CLIP >> 2) + (_ULAW_BIAS >> 2)
    segment = np.floor(np.log2(magnitude)).astype(np.int32) - 5
    uval = np.where(segment >= 8, 0x7F, (segment << 4) | ((magnitude >> (segment + 1)) & 0x0F))
    return (uval ^ np.where(negative, 0x7F, 0xFF)).astype(np.uint8).tobytes()


def _lowpass_taps(num_taps: int = 97, cutoff_hz: float = 3600.0, rate: int = PCM_RATE) -> np.ndarray:
    n = np.arange(num_taps) - (num_taps - 1) / 2
    fc = cutoff_hz / rate
    taps = 2 * fc * np.sinc(2 * fc * n) * np.hamming(num_taps)
    return taps / taps.sum()


_TAPS = _lowpass_taps()


class _StreamingFIR:
    def __init__(self, taps: np.ndarray = _TAPS) -> None:
        self.taps = taps
        self.history = np.zeros(len(taps) - 1, dtype=np.float64)

    def filter(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float64)
        buf = np.concatenate([self.history, x.astype(np.float64)])
        self.history = buf[-(len(self.taps) - 1) :]
        return np.convolve(buf, self.taps, mode="valid")


def _to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


class Upsampler:
    """Streaming 8 kHz -> 24 kHz PCM16 (zero-stuff x3, then low-pass)."""

    def __init__(self) -> None:
        self.fir = _StreamingFIR()

    def process(self, samples: np.ndarray) -> np.ndarray:
        stuffed = np.zeros(len(samples) * _FACTOR, dtype=np.float64)
        stuffed[::_FACTOR] = np.asarray(samples, dtype=np.float64) * _FACTOR
        return _to_int16(self.fir.filter(stuffed))


class Downsampler:
    """Streaming 24 kHz -> 8 kHz PCM16 (low-pass, then keep every third sample)."""

    def __init__(self) -> None:
        self.fir = _StreamingFIR()
        self.skip = 0  # samples to drop before the next kept one, carried across chunks

    def process(self, samples: np.ndarray) -> np.ndarray:
        filtered = self.fir.filter(np.asarray(samples))
        kept = filtered[self.skip :: _FACTOR]
        consumed = len(filtered) - self.skip
        self.skip = (-consumed) % _FACTOR if consumed > 0 else self.skip - len(filtered)
        return _to_int16(kept)


class PhoneDecoder:
    """mu-law 8 kHz wire bytes -> PCM16 24 kHz bytes."""

    def __init__(self) -> None:
        self.up = Upsampler()

    def process(self, data: bytes) -> bytes:
        return self.up.process(ulaw_decode(data)).astype("<i2").tobytes()


class PhoneEncoder:
    """PCM16 24 kHz bytes -> mu-law 8 kHz wire bytes."""

    def __init__(self) -> None:
        self.down = Downsampler()

    def process(self, pcm16: bytes) -> bytes:
        return ulaw_encode(self.down.process(np.frombuffer(pcm16, dtype="<i2")))


# ---------------------------------------------------------------------------
# Numbers to words. Chatterbox reads "495120387" as "four hundred ninety-five
# million..." and stumbles on "$12.50" or "3:05"; the LLM writes numerals.

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"))
_ORDINAL_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def _under_thousand(n: int) -> str:
    parts: list[str] = []
    if n >= 100:
        parts.append(f"{_ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else ""))
    elif n or not parts:
        parts.append(_ONES[n])
    return " ".join(parts)


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 1000:
        return _under_thousand(n)
    parts: list[str] = []
    for value, name in _SCALES:
        if n >= value:
            parts.append(f"{int_to_words(n // value)} {name}")
            n %= value
    if n:
        parts.append(_under_thousand(n))
    return " ".join(parts)


def ordinal_words(n: int) -> str:
    words = int_to_words(n)
    head, sep, last = words.rpartition(" ")
    if "-" in last:
        stem, _, unit = last.rpartition("-")
        last = f"{stem}-{_ORDINAL_IRREGULAR.get(unit, unit + 'th')}"
    elif last in _ORDINAL_IRREGULAR:
        last = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head}{sep}{last}"


def digits_to_words(digits: str) -> str:
    return " ".join(_ONES[int(d)] for d in digits if d.isdigit())


def year_words(n: int) -> str:
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{int_to_words(hi)} hundred"
    if 2000 <= n <= 2009:
        return int_to_words(n)
    return f"{int_to_words(hi)} {('oh ' + _ONES[lo]) if lo < 10 else int_to_words(lo)}"


def _money(m: re.Match[str]) -> str:
    dollars = int(m.group(1).replace(",", ""))
    cents = int(m.group(2)) if m.group(2) else 0
    if cents and not dollars:
        return f"{int_to_words(cents)} cent{'' if cents == 1 else 's'}"
    out = f"{int_to_words(dollars)} dollar{'' if dollars == 1 else 's'}"
    if cents:
        out += f" and {int_to_words(cents)} cent{'' if cents == 1 else 's'}"
    return out


def _time(m: re.Match[str]) -> str:
    hour, minute, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
    if hour > 24 or minute > 59:
        return m.group(0)
    if minute == 0:
        spoken = int_to_words(hour) + ("" if suffix else " o'clock")
    elif minute < 10:
        spoken = f"{int_to_words(hour)} oh {_ONES[minute]}"
    else:
        spoken = f"{int_to_words(hour)} {int_to_words(minute)}"
    if suffix:
        spoken += " " + ("a m" if suffix.lower().startswith("a") else "p m")
    return spoken


def _phone(m: re.Match[str]) -> str:
    groups = [g for g in m.groups() if g]
    return ", ".join(digits_to_words(g) for g in groups)


def _decimal(m: re.Match[str]) -> str:
    whole, frac = m.group(1), m.group(2)
    return f"{int_to_words(int(whole))} point {digits_to_words(frac)}"


def _ordinal(m: re.Match[str]) -> str:
    return ordinal_words(int(m.group(1)))


def _plain(m: re.Match[str]) -> str:
    digits = m.group(0)
    if len(digits) >= 4:
        if len(digits) == 4 and 1900 <= int(digits) <= 2099:
            return year_words(int(digits))
        return digits_to_words(digits)  # order, confirmation, part numbers
    if len(digits) > 1 and digits.startswith("0"):
        return digits_to_words(digits)
    return int_to_words(int(digits))


_MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?(?!\d)")
_TIME_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])(?:\s?([AaPp]\.[Mm]\.|[AaPp]\s?[Mm])(?![A-Za-z]))?")
_PHONE_RE = re.compile(r"(?<![\d])(?:\+?(1)[\s.-]?)?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-](\d{4})(?!\d)")
_GROUPED_RE = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{3})+(?![\d,]\d)")
_DECIMAL_RE = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.]\d)")
_PERCENT_RE = re.compile(r"(?<!\d)(\d+)\s?%")
_ORDINAL_RE = re.compile(r"(?<!\d)(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")


def numbers_to_words(text: str) -> str:
    """Rewrite numerals the way a person reads them aloud on the phone."""
    if not any(ch.isdigit() for ch in text):
        return text
    text = _MONEY_RE.sub(_money, text)
    text = _TIME_RE.sub(_time, text)
    text = _PHONE_RE.sub(_phone, text)
    text = _GROUPED_RE.sub(lambda m: int_to_words(int(m.group(0).replace(",", ""))), text)
    text = _DECIMAL_RE.sub(_decimal, text)
    text = _PERCENT_RE.sub(lambda m: f"{int_to_words(int(m.group(1)))} percent", text)
    text = _ORDINAL_RE.sub(_ordinal, text)
    # Letters glued to digits (AB12345) read better with a gap before the digits.
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)
    return _DIGITS_RE.sub(_plain, text)
