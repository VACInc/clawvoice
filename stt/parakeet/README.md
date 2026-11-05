# Parakeet STT server

Tiny FastAPI wrapper around NVIDIA `parakeet-tdt-0.6b-v2` (NeMo) that speaks the OpenAI `/v1/audio/transcriptions` API. ~0.1 s per utterance on a 3090.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 31411
curl -F file=@clip.wav http://127.0.0.1:31411/v1/audio/transcriptions
```

Bind it to your LAN address or put it behind a firewall rule: it has no auth.
