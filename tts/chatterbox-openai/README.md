# Chatterbox Turbo, OpenAI-compatible and streaming

`server.py` serves `POST /v1/audio/speech` (OpenAI TTS shape) on top of Resemble AI's Chatterbox Turbo, with real streaming: speech tokens are sampled one at a time and a sliding window is rendered to audio as they arrive, so the first audio leaves after ~20 tokens (~0.3 s on a 3090) instead of after the whole sentence. Chunks are crossfaded and level-matched, then encoded by one ffmpeg per request.

```bash
pip install chatterbox-tts flask torchaudio numpy   # plus ffmpeg on PATH
mkdir voices && cp myvoice.wav voices/              # 10-20 s clean reference clip = your cloned voice
VOICE_DIR=$PWD/voices DEFAULT_VOICE=myvoice python3 server.py   # :5051
curl -s http://127.0.0.1:5051/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"myvoice","input":"Hello there.","response_format":"mp3"}' -o out.mp3
```

Knobs: `TURBO_STREAM_FIRST_TOKENS` (20), `TURBO_STREAM_CHUNK_TOKENS` (25), `TURBO_STREAM_CONTEXT_TOKENS` (25), `TURBO_CHUNK_THRESHOLD_CHARS`, `TURBO_MAX_CHUNK_CHARS`. `"stream": false` or `wav`/`flac` uses the blocking path.

`pronounce.py` respells names the model says wrong before synthesis; edit `_RULES` for your names. `TTS_PRONOUNCE_FIX=0` turns it off.

Only clone voices you have the right to use.
