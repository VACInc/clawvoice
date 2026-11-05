#!/usr/bin/env python3
"""Transcribe one voice note with Parakeet and apply the shared TARS name fix.

OpenClaw runs this as the first audio entry in tools.media.models (type cli)
for channel voice notes (Telegram, Signal, Discord attachments, Discord voice
in stt-tts mode). It prints the transcript on stdout. Any failure exits
non-zero so OpenClaw falls through to the next entry (OpenAI
gpt-4o-transcribe). Inference runs on the GPU box, never here.

Usage: transcribe_note.py AUDIO_PATH
Env:   TARS_STT_URL (default http://127.0.0.1:31411/v1/audio/transcriptions)
       TARS_STT_TIMEOUT seconds (default 120)
       TARS_NAME_FIX=0 turns the name fix off (transcript passes through).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import fix_tars_name  # noqa: E402

STT_URL = os.environ.get("TARS_STT_URL", "http://127.0.0.1:31411/v1/audio/transcriptions")
TIMEOUT = float(os.environ.get("TARS_STT_TIMEOUT", "120"))
MODEL = "parakeet-tdt-0.6b-v2"


def to_wav(path: str) -> tuple[bytes, str, str]:
    """16 kHz mono WAV via ffmpeg (same format as the HA route); raw file if ffmpeg is absent."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        out = subprocess.run(
            [ffmpeg, "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", "-f", "wav", "-"],
            capture_output=True, timeout=TIMEOUT, check=True,
        ).stdout
        return out, "note.wav", "audio/wav"
    return Path(path).read_bytes(), Path(path).name, "application/octet-stream"


def multipart(fields: dict[str, str], name: str, filename: str, data: bytes, ctype: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in fields.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe(path: str, post=None) -> str:
    post = post or urllib.request.urlopen
    data, filename, ctype = to_wav(path)
    body, content_type = multipart({"model": MODEL}, "file", filename, data, ctype)
    req = urllib.request.Request(STT_URL, data=body, headers={"Content-Type": content_type})
    with post(req, timeout=TIMEOUT) as resp:
        payload = json.loads(resp.read())
    text = next((payload[k] for k in ("text", "transcript", "transcription") if isinstance(payload.get(k), str)), None)
    if text is None:
        raise ValueError("STT response has no text field")
    return fix_tars_name(text.strip())


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        text = transcribe(argv[1])
    except Exception as err:  # noqa: BLE001 - any failure hands off to the next OpenClaw entry
        print(f"transcribe_note: {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    sys.stdout.write(text + "\n" if text else "")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
