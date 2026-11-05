# Chatterbox-TTS-Server patch (the shim's TTS backend)

The shim calls a second TTS endpoint (`/tts`, port 31420): [devnen/Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server) at commit `915ae289`, with `chatterbox-server.patch` applied:

- accepts both `speed` and `speed_factor` on the OpenAI and custom endpoints
- speed change through ffmpeg `atempo` (keeps pitch, no artifacts) instead of librosa/resample
- `config.yaml`: CUDA, port 31420, cache paths

```bash
git clone https://github.com/devnen/Chatterbox-TTS-Server && cd Chatterbox-TTS-Server
git checkout 915ae289340e10c6047f27f47e22eae9bf350c32
git apply /path/to/chatterbox-server.patch
# follow upstream install, then: python server.py
```

You can also skip this server and point the shim at the streaming wrapper in `../chatterbox-openai` instead.
