# Wyoming to HTTP STT bridge

Home Assistant talks Wyoming; Parakeet talks HTTP. This bridge is a Wyoming ASR server that buffers the audio, POSTs one WAV to any OpenAI-compatible `/v1/audio/transcriptions` endpoint and returns the transcript, with the agent-name fix from `shim/names.py` applied.

```bash
cp ../../shim/names.py .
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python bridge.py --uri tcp://0.0.0.0:10302 --http http://127.0.0.1:31411/v1/audio/transcriptions
```

In HA: Settings → Devices & services → Add → Wyoming Protocol → host:10302 (or point it at the speaker-ID tee, which forwards here).

Test: `python3 tests/test_bridge_names.py` (offline, needs `wyoming`).
