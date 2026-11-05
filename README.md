# clawvoice

A local voice stack for OpenClaw and Home Assistant: wake-word speakers, the OpenClaw app/web UI Talk mode, real phone calls and Discord voice. One set of local models does all the listening, talking and speaking. The only cloud hop is when the local model hands a real task to your main agent (and the phone carrier for calls).

This is a writeup of a working home setup, not a product. Hostnames are placeholders (`gpu-box`, `claw-host`, `ha`). Swap in your own.

## What's in this repo

| Path | What it is |
|---|---|
| [`shim/`](shim/) | The realtime shim: OpenAI Realtime websocket API in front of local STT/LLM/TTS. Powers Talk, phone calls and Discord voice. |
| [`stt/parakeet/`](stt/parakeet/) | Parakeet-TDT behind an OpenAI-style `/v1/audio/transcriptions` API |
| [`stt/wyoming-bridge/`](stt/wyoming-bridge/) | Wyoming server for Home Assistant that forwards to any HTTP STT |
| [`tts/chatterbox-openai/`](tts/chatterbox-openai/) | Chatterbox Turbo as an OpenAI `/v1/audio/speech` API with true streaming |
| [`tts/chatterbox-server/`](tts/chatterbox-server/) | Small patch for Chatterbox-TTS-Server (the shim's `/tts` backend) |
| [`speaker-id/`](speaker-id/) | Speaker recognition: a byte-exact Wyoming tee plus a ReDimNet/TitaNet embedding service |
| [`llm/`](llm/) | The voice LLM: Qwen3.8 one-card vLLM recipe or DeepSeek V4.1 Flash |
| [`config/`](config/) | Every voice block of our `openclaw.json` (Talk, realtime, phone, Discord, TTS, voice notes, consult agents, hooks) |
| [`ha/`](ha/) | The HA side of the handoff tool: authenticated consult service + test-mode toggle |

## The pieces

| Job | Model (exact) | Runs on |
|---|---|---|
| Wake word | microWakeWord "Okay Nabu", on the satellite itself | Voice PE / Satellite 1 / ReSpeaker Lite (ESPHome) |
| Speech to text | `nvidia/parakeet-tdt-0.6b-v2` (NeMo), served by [`stt/parakeet`](stt/parakeet/) | RTX 3090 (shares the card with TTS) |
| Fast voice brain | Either `nvidia/Qwen3.8-Flash-Next-FP4-Local` (NVFP4, vLLM, one card, recipe in [`llm/`](llm/)) or DeepSeek V4.1 Flash (3 bpw, two cards). Any fast tool-calling model works. | RTX Pro 6000 Blackwell (1 for Qwen, 2 for DeepSeek) |
| Text to speech | `ResembleAI/chatterbox-turbo` with a cloned reference voice, via [`tts/`](tts/) | RTX 3090 |
| Turn detection | Silero VAD v5.1.2 (ONNX) | Claw host, inside the shim (CPU) |
| Speaker ID | ReDimNet `M-vb2+vox2+cnc-ft_mix` (primary) + NVIDIA TitaNet-L (second opinion) | GPU box (falls back to CPU) |
| Consult agents (the brain) | SOTA and/or top local models: Claude Opus 5.5 (cloud) for Talk and phone, DeepSeek V4.1 Flash (local) for HA speakers. | Claw host (Ryzen AI Max+ 395) |
| Voice-note fallback | OpenAI `gpt-4o-transcribe`, only if local Parakeet fails | cloud |
| Glue | [`shim/`](shim/): a local stand-in for the OpenAI Realtime websocket API | Claw host |

The trick that makes everything else work: **the shim**. OpenClaw's Talk mode, voice-call plugin and Discord voice all speak the OpenAI Realtime protocol. The shim speaks that same protocol on `ws://127.0.0.1:8820`, but behind it runs local STT → local LLM → local TTS. OpenClaw thinks it's talking to OpenAI. It isn't.

```
OpenClaw (Talk / voice-call / Discord)
   │  OpenAI Realtime websocket
   ▼
realtime-shim :8820  (Silero VAD, barge-in, name fix)
   ├─▶ Parakeet STT        gpu-box:31411
   ├─▶ local LLM (vLLM)    gpu-box:31320
   └─▶ Chatterbox TTS      gpu-box:31420
```

## Latency

Measured on our box, warm, models on RTX Pro 6000 / 3090 cards, clients on the same LAN.

| Stage | p50 | Notes |
|---|---|---|
| Parakeet STT, 2.4 s clip | 82 ms | 75 to 115 ms |
| LLM first token (DeepSeek V4.1 Flash) | 91 ms | short voice prompt, thinking off |
| LLM full one-sentence reply | 380 ms | |
| TTS first audio, HA wrapper (`:5051`, streaming) | 295 ms | |
| TTS one sentence, Chatterbox server (`:31420`, shim path) | 690 ms | whole sentence at once; the shim sends one sentence at a time |

End to end, from end of speech to first reply audio:

| Path | p50 |
|---|---|
| HA satellite (ReSpeaker Lite, hands-free, real wake word) | 0.6 s (0.60 / 0.63 / 1.24 s over 3 runs) |
| HA pipeline, injected audio (Qwen3.8 option) | 0.55 s |
| Talk shim (`:8820`), injected audio (Qwen3.8 option) | 0.76 s |

Add your VAD's end-of-speech wait on top (about 0.25 s with "aggressive"). The first turn after ~30 min idle paid about 1.5 s extra for cold STT and Gateway paths until we added the keep-warm cron.

## The backstop: fast local voice, smart agent behind it

The local model does all the talking because it's fast. It is not the brain for anything real. When a turn needs reasoning, current info, tools or household facts, the local model calls a consult agent.

**The consults run on top models: a SOTA cloud model (ours: Claude Opus 5.5) and/or a top-of-the-line local model (ours: DeepSeek V4.1 Flash on 2x RTX Pro 6000).** Don't back a voice stack with a small model. The fast voice model is only the mouth; the consult model is the brain, and it decides how smart the whole thing feels.

| Flow | Local model talks | Backstop |
|---|---|---|
| Talk (app / web UI) | shim :8820 | `brain: "agent-consult"` → `openclaw_agent_consult` → main agent |
| Phone | shim :8821 | `consultPolicy: "substantive"` → main agent, read-only tools |
| HA speakers | vLLM direct | `ask_claw` → `/hooks/agent`, async, answer in chat |

The consult agents and how hard they think, per flow:

```json5
{
  agents: {
    defaults: { model: { primary: "anthropic/claude-opus-5-5" } },
    list: [
      { id: "main", default: true, model: { primary: "anthropic/claude-opus-5-5" } },  // Talk + phone consults
      { id: "voice", model: { primary: "local-llm/voice-llm", fallbacks: [] },          // HA consults, locked down
        tools: { profile: "minimal", deny: ["*"] }, skills: [], contextInjection: "never" },
    ],
  },
  talk: { agentId: "main", consultThinkingLevel: "medium", consultFastMode: false,
          realtime: { brain: "agent-consult" } },
  plugins: { entries: { "voice-call": { config: {
    agentId: "main",
    realtime: { consultPolicy: "substantive", consultThinkingLevel: "low", toolPolicy: "safe-read-only" },
  } } } },
  hooks: { allowedAgentIds: ["main", "voice"], allowRequestSessionKey: false },
}
```

Talk thinks at `medium` because you're looking at a screen and can wait a beat. Phone uses `low` because dead air on a call feels broken. Full blocks: [`config/openclaw.voice.json5`](config/openclaw.voice.json5).

The local model gets a line telling it never to claim it did something unless the consult returned it (see Flow 2). Without that line, it happily says "done, reminder set" and does nothing.

For HA we point the hook at a locked-down `voice` agent on the local model instead, because the HA speaker accepts anyone in the room. If you trust everyone in your house, set the consult `agent_id` to your main agent and the smart model answers there too.

---

## Flow 1: Home Assistant speakers (wake word)

Hardware: Home Assistant Voice PE pucks (plugged into real speakers, since the built-in speaker is weak), a FutureProofHomes Satellite 1 and a ReSpeaker Lite. All run their vendors' stock ESPHome firmware, no custom YAML:

- Voice PE: [esphome/home-assistant-voice-pe](https://github.com/esphome/home-assistant-voice-pe)
- Satellite 1: [futureproofhomes/satellite1-esphome](https://github.com/futureproofhomes/satellite1-esphome)
- ReSpeaker Lite: [formatBCE/Respeaker-Lite-ESPHome-integration](https://github.com/formatBCE/Respeaker-Lite-ESPHome-integration)

Everything below is server side, so any HA Assist satellite works.

```
satellite (wake word on device)
  → HA Assist pipeline
  → STT: Wyoming → speaker-ID tee → Wyoming-to-HTTP bridge → Parakeet
  → Conversation agent: Extended OpenAI Conversation → local LLM (direct to vLLM)
       └─ tool: ask_claw → OpenClaw /hooks/agent (async, answer lands in chat)
  → TTS: OpenAI TTS custom component → Chatterbox wrapper (streaming)
  → satellite plays reply
```

Setup:

1. **STT.** Parakeet serves plain HTTP, so a small Wyoming-to-HTTP bridge sits in front of it for HA. HA's Wyoming integration points at the speaker-ID tee (see Flow 4), which forwards to the bridge.
2. **Conversation agent.** The Extended OpenAI Conversation HACS integration, pointed straight at vLLM. The system prompt holds persona + functions, then the exposed-device list, then the time as the last line (see gotchas).
3. **Expose fewer entities.** We cut exposed entities in half (1159 → 573). Every exposed entity goes into the prompt on every turn.
4. **The handoff tool.** One function, `ask_claw`, calls a HA `rest_command` that POSTs to OpenClaw's `/hooks/agent`. It replies "on it" right away; the real answer arrives in Telegram. It's async because an agent turn takes longer than anyone will wait at a speaker. We point it at the locked-down `voice` agent (see [the backstop](#the-backstop-fast-local-voice-smart-agent-behind-it)).
5. **Lock that hook down.** The hook targets a dedicated `voice` agent with every executable tool denied, no skills, no workspace context and no memory search. It runs one request at a time with a 60 s cooldown, accepts only a 1 to 2000 char question, and refuses any call without an authenticated HA admin context.
6. **TTS.** The `openai_tts` custom component pointed at the Chatterbox wrapper, with `stream_pipelining: true` so it speaks sentence one while the LLM is still writing sentence two.
7. **Satellite tuning.** Set "finished speaking detection" to aggressive.

Latency: about 0.6 s from end of speech to first reply audio (see [Latency](#latency)).

## Flow 2: OpenClaw app / Control UI Talk mode

Talk mode in the OpenClaw web UI and the apps, hands-free, with interrupts.

```
browser/app mic
  → Gateway (gateway-relay transport)
  → realtime-shim :8820 → Parakeet → local LLM → Chatterbox
  → when it needs more: openclaw_agent_consult → main OpenClaw agent
```

Config (`openclaw.json`):

```json
{
  "talk": {
    "agentId": "main",
    "consultThinkingLevel": "medium",
    "provider": "openai",
    "providers": {
      "openai": { "baseUrl": "http://gpu-box:5051/v1", "apiKey": "local", "model": "tts-1", "voice": "myvoice", "responseFormat": "wav" }
    },
    "realtime": {
      "mode": "realtime",
      "transport": "gateway-relay",
      "brain": "agent-consult",
      "provider": "openai",
      "model": "local",
      "providers": {
        "openai": { "azureEndpoint": "http://127.0.0.1:8820", "apiKey": "local-shim", "model": "local", "voice": "myvoice" }
      },
      "instructions": "Stay brief and natural. Use openclaw_agent_consult when deeper reasoning, current information, or OpenClaw tools are needed. Never say you set a reminder, changed a device, or know a household fact (people, pets, schedules, home state) unless openclaw_agent_consult returned it; for those requests always call openclaw_agent_consult instead of answering or declining yourself."
    }
  }
}
```

- `azureEndpoint` is how you point the OpenAI realtime provider at a local URL. The shim ignores the API key, but OpenClaw requires one, so put anything there.
- `brain: agent-consult` means the fast local model does the talking and calls your main agent as a tool for anything real. That instruction line is what keeps it from inventing "done, reminder set."
- `gateway-relay` means the browser never connects to the shim itself; the Gateway relays, so the shim stays on loopback.

## Flow 3: Phone calls

Call a real phone number and talk to your agent, or have it call you.

```
phone → Twilio number → Media Streams websocket
  → Tailscale Funnel (only /voice/* paths exposed)
  → OpenClaw voice-call plugin :3334
  → realtime-shim (phone copy) :8821 → Parakeet → local LLM → Chatterbox
  → consult main agent when needed (read-only tools)
```

Honest scope: Twilio carries the call audio, so the phone network leg is not local. The voice models are local; consults go to your main agent's model.

```json
{
  "plugins": { "entries": { "voice-call": { "enabled": true, "config": {
    "provider": "twilio",
    "fromNumber": "+1...", "inboundPolicy": "allowlist", "allowFrom": "+1...",
    "twilio": { "accountSid": "...", "authToken": "..." },
    "publicUrl": "https://<your-funnel-host>/voice/webhook",
    "serve": { "port": 3334, "path": "/voice/webhook" },
    "maxDurationSeconds": 900,
    "agentId": "main",
    "realtime": {
      "enabled": true, "provider": "openai",
      "toolPolicy": "safe-read-only", "consultPolicy": "substantive", "consultThinkingLevel": "low",
      "providers": { "openai": { "azureEndpoint": "http://127.0.0.1:8821", "apiKey": "local-shim", "model": "local", "voice": "myvoice" } }
    }
  } } } }
}
```

- **Run a second shim for phone.** Same code, port 8821, with `RTSHIM_VAD_MAX_TURN_MS=15000`. Phone audio is 8 kHz mu-law and phone turns run long; the Talk defaults cut people off.
- **Inbound allowlist.** Only your number gets through. Twilio webhook signatures and one-time stream tokens are enforced.
- **Funnel only the voice paths** (`/voice/webhook`, `/voice/stream/realtime`, `/voice/hold-music`), never the whole Gateway.

## Flow 4: Speaker ID ("who's talking")

```
HA Wyoming STT → wyoming-tee (byte-for-byte proxy) → real STT
                     └─ copy of the audio, after the last frame is sent → speaker-embed (ReDimNet)
                          → HA webhook → input_text.speaker_last_<device>
```

- `wyoming-tee` forwards every frame to the real STT untouched; it only starts the speaker work after the last audio frame is already on its way. Measured cost: zero added latency (transcripts identical, ~100 ms faster than direct in testing).
- `speaker-embed` compares the embedding against enrolled voices. At or above 0.72 it accepts the speaker, 0.55 to 0.72 counts as "uncertain" and below 0.55 is "unknown". Uncertain is treated as unknown (fail-closed). TitaNet-L scores in parallel for comparison only and never overrides.
- Enrollment: 5 to 10 real utterances per person, recorded at the satellites, not a studio mic.
- **Enforcement** lives in the HA function backends, never the prompt: identity rides with the turn, guests get the safe functions only, and an uncertain voice asking for something sensitive gets a human APPROVE/DENY in Telegram. Run with `SHADOW=1` (label only) while you tune thresholds, then flip it off.
- If the tee or embedder dies, STT keeps working and the turn is just treated as a guest.

## Bonus: voice notes and Discord

- Voice notes on Telegram/Signal/Discord: `tools.media.models` first entry is a CLI script that converts to 16 kHz WAV and posts to Parakeet. OpenAI transcription is only the fallback.
- Discord voice channels: `channels.discord.voice.realtime` points at the same 8820 shim.

## Running the shim

The shim lives in [`shim/`](shim/). Python 3.11+, no GPU needed (it only calls your model servers).

```bash
cd shim
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./fetch-model.sh          # pinned Silero VAD ONNX, sha256-checked
python3 server.py         # ws://127.0.0.1:8820/v1/realtime
```

Run it as a service with `clawvoice-shim.service` (put your `RTSHIM_*` settings in `/etc/clawvoice/shim.env`). For the phone copy, run a second instance with `RTSHIM_PORT=8821` and the phone settings below.

What's in it:

| File | Job |
|---|---|
| `server.py` | OpenAI Realtime websocket protocol: sessions, audio buffers, responses, cancel/barge-in, tool calls |
| `pipeline.py` | Calls to STT, the LLM (streaming, tool calls, optional fallback URL) and TTS |
| `vad.py` | Silero VAD turn detection with an energy fallback |
| `phone.py` | 8 kHz mu-law in/out and paced 20 ms output for phone calls |
| `names.py` | Fixes how STT spells your agent's name (edit the word lists for yours) |
| `pronounce.py` | Respells names the TTS says wrong, before synthesis |
| `transcribe_note.py` | One-shot voice-note transcription for `tools.media.models` |

Tests run offline: `python3 tests/test_names.py` etc.

## Config reference

**Every voice block of our `openclaw.json`, as it runs today (scrubbed): [`config/openclaw.voice.json5`](config/openclaw.voice.json5).** Models, both consult agents, Talk, realtime, phone, Discord voice, chat TTS, voice notes, hooks. The snippets below are pieces of it.

All hosts, tokens and IDs below are placeholders.

### OpenClaw: models and agents

Local model provider (OpenAI-compatible vLLM). Send both thinking flags (see gotchas):

```json
{
  "models": { "providers": { "local-llm": {
    "baseUrl": "http://gpu-box:31320/v1",
    "api": "openai-completions",
    "models": [{ "id": "voice-llm", "contextWindow": 262144,
      "params": { "thinking": false, "chat_template_kwargs": { "enable_thinking": false } } }]
  } } },
  "tools": { "toolSearch": { "enabled": true, "mode": "tools" } }
}
```

The locked-down `voice` agent that HA hands questions to. No tools, no skills, no context, no memory:

```json
{
  "agents": { "entries": {
    "voice": {
      "workspace": "~/voice-workspace",
      "model": { "primary": "local-llm/voice-llm", "fallbacks": [] },
      "runtime": { "type": "embedded" },
      "contextInjection": "never",
      "skills": [],
      "tools": { "profile": "minimal", "deny": ["*"], "elevated": { "enabled": false } },
      "memory": { "search": { "enabled": false } }
    }
  } },
  "hooks": {
    "enabled": true,
    "path": "/hooks",
    "token": "<long random>",
    "allowedAgentIds": ["voice"],
    "allowRequestSessionKey": false
  }
}
```

### OpenClaw: Discord voice and voice notes

```json
{
  "channels": { "discord": { "voice": {
    "enabled": true,
    "realtime": { "provider": "openai", "model": "local",
      "providers": { "openai": { "azureEndpoint": "http://127.0.0.1:8820", "apiKey": "local-shim", "model": "local", "voice": "myvoice" } } },
    "tts": { "provider": "openai",
      "providers": { "openai": { "baseUrl": "http://gpu-box:5051/v1", "apiKey": "local", "model": "tts-1", "speakerVoice": "myvoice", "responseFormat": "wav" } } }
  } } },
  "tools": { "media": { "models": [
    { "type": "cli", "command": "python3", "args": ["/path/to/transcribe_note.py", "{{MediaPath}}"], "capabilities": ["audio"], "timeoutSeconds": 150 },
    { "provider": "openai", "model": "gpt-4o-transcribe", "capabilities": ["audio"] }
  ] } }
}
```

`transcribe_note.py` converts the file with ffmpeg to 16 kHz mono WAV, POSTs it to Parakeet, applies the name fix, prints the text and exits non-zero on any failure so OpenClaw falls through to the next entry.

### Realtime shim (env)

Talk / Discord copy on 8820:

```bash
RTSHIM_HOST=127.0.0.1
RTSHIM_PORT=8820
RTSHIM_STT_URL=http://gpu-box:31411/v1/audio/transcriptions
RTSHIM_LLM_URL=http://gpu-box:31320/v1/chat/completions
RTSHIM_TTS_URL=http://gpu-box:31420/tts
RTSHIM_VOICE=myvoice.wav
RTSHIM_SPEED=1.15
RTSHIM_VAD=silero
RTSHIM_SILERO_THRESHOLD=0.5
RTSHIM_VAD_SILENCE_MS=500
RTSHIM_VAD_MIN_SPEECH_MS=200      # speech needed to open a turn or barge in
RTSHIM_NOISE_MAX_VOICED_MS=1500   # drop "um"/"thank you" turns shorter than this
RTSHIM_VAD_MAX_TURN_MS=3000       # force-commit turns that never go quiet (TV)
RTSHIM_VAD_PREFIX_PADDING_MS=300
```

Phone copy on 8821: same file, plus:

```bash
RTSHIM_PORT=8821
RTSHIM_VAD_MAX_TURN_MS=15000
RTSHIM_PHONE_OUTPUT_LEAD_MS=300      # max reply audio that can still play after a barge-in
RTSHIM_PHONE_INPUT_STALL_MS=250      # no caller frame this long = Gateway stalled, hold output
RTSHIM_PHONE_STALL_HOLD_MAX_MS=6000
```

### Home Assistant: conversation agent

Extended OpenAI Conversation (HACS), base URL `http://gpu-box:31320/v1`, temperature 0.9. Prompt template order matters:

```jinja
You are <name>, a voice assistant. <persona, 2 short spoken sentences max>
<function rules: when to call ask_claw, never claim actions you did not do>

Devices:
```csv
entity_id,name,state,aliases
{% for entity in exposed_entities -%}
{{ entity.entity_id }},{{ entity.name }},{{ entity.state }},{{ entity.aliases | join('/') }}
{% endfor -%}
```

Current Time: {{ now().strftime('%A %Y-%m-%d %H:%M') }}
```

If your model thinks by default and the integration can't send extra body fields, a fork that reads `extra_body` from the model string works:
`voice-llm?extra_body={"chat_template_kwargs":{"enable_thinking":false},"thinking":false}` (URL-encoded).

The `ask_claw` function the model sees:

```yaml
- spec:
    name: ask_claw
    description: Hand a question that needs real tools, memory or current info to the main assistant. The answer arrives in chat.
    parameters:
      type: object
      properties:
        question: { type: string }
      required: [question]
  function:
    type: script
    sequence:
      - service: rest_command.claw_voice_consult
        data: { question: "{{ question }}" }
```

### Home Assistant: the consult service (auth + admission)

Full code in [`ha/pyscript/voiceconsult.py`](ha/pyscript/voiceconsult.py). Registered from a Pyscript app with the native HA API (so the caller's real `ServiceCall.context` is used, not anything in the service data). Core of it:

```python
async def consult(call):
    user = await hass.auth.async_get_user(call.context.user_id) if call.context.user_id else None
    if user is None or not user.is_active or not user.is_admin:
        raise Unauthorized()
    if set(call.data) != {"question"}:
        raise HomeAssistantError("question only")
    q = call.data["question"]
    if not isinstance(q, str) or not q.strip() or len(q) > 2000:
        raise HomeAssistantError("1 to 2000 chars")
    if admission["busy"] or time.monotonic() < admission["until"]:
        raise HomeAssistantError("busy")
    admission.update(busy=True, until=time.monotonic() + 60)
    muted = hass.states.is_state("input_boolean.voice_consult_test_mode", "on")
    payload = {
        "message": ("[VOICE-CONSULT-TEST] " if muted else "[VOICE-CONSULT] ") + q.strip(),
        "agentId": "voice", "deliver": not muted,
        "channel": "telegram", "to": "<your chat id>",
        "timeoutSeconds": 45,
    }
    # POST to https://claw-host/hooks/agent with the hook token + an Idempotency-Key,
    # return the ACK to HA right away, then re-POST the same key with
    # "waitForCompletion": true in the background and release the slot on the terminal receipt.
```

### Home Assistant: STT wiring

Wyoming integration → `gpu-box:10301` (the speaker-ID tee) → tee forwards to the Wyoming-to-HTTP bridge on `127.0.0.1:10302` → bridge POSTs to Parakeet `:31411`. To drop speaker ID, point Wyoming straight at the bridge.

Speaker stamp automation:

```yaml
- id: speaker_id_stamp
  trigger:
    - platform: webhook
      webhook_id: <long random id>
      local_only: true
  action:
    - service: input_text.set_value
      target:
        entity_id: "input_text.speaker_last_{{ trigger.json.device_id | default('voice_satellite') }}"
      data:
        value: "{{ trigger.json.user_id }}|{{ trigger.json.status }}|{{ now().isoformat() }}"
```

### Speaker ID (env)

```bash
# speaker-embed
SPEAKER_PORT=8099
DEVICE=cuda
LOW_THRESHOLD=0.55
HIGH_THRESHOLD=0.72
SHADOW=0                  # 1 = label only, handy while tuning
ALLOWED_ENROLL_USERS=me
TITANET_ENABLED=1         # second opinion, never overrides

# wyoming-tee
TEE_PORT=10301
REAL_STT_HOST=127.0.0.1
REAL_STT_PORT=10302
EMBED_URL=http://127.0.0.1:8099/identify
HA_WEBHOOK_URL=http://ha:8123/api/webhook/<long random id>
SHADOW=0
```

Enroll: `curl --data-binary @clip.wav 'http://127.0.0.1:8099/enroll?user=me'` for each of 5 to 10 clips recorded at the satellites.

### Keep-warm (cron)

```cron
*/5 * * * * curl -s http://claw-host:18789/v1/chat/completions -H "Authorization: Bearer $LANE_TOKEN" -d '{"model":"openclaw:voice-lane","max_tokens":64,"messages":[{"role":"user","content":"ping"}]}' >/dev/null; curl -s -F file=@silence.wav http://gpu-box:31411/v1/audio/transcriptions >/dev/null
```

---

## Gotchas (the ones that actually bit)

- **Prompt order decides latency.** Static stuff first (persona, functions), device list next, current time LAST at minute precision. Put the time first and vLLM's prefix cache misses on every turn.
- **Keep Code Mode off for local models.** Its meta-tool broke tool calling on DeepSeek. Structured tool search (`tools.toolSearch = {"enabled": true, "mode": "tools"}`, or `"directory"`) works.
- **Thinking models: send both flags.** On vLLM, `chat_template_kwargs.enable_thinking: false` is what turns thinking off; `thinking: false` alone gets ignored. Send both. Thinking on a voice lane adds seconds.
- **STT mangles your agent's name.** Parakeet has no hotword input and writes "TARS" as Tarz/Taris/Taurus/Stars. One shared name-fix function runs at every STT exit (shim, HA bridge, voice notes), and the real words ("stars") only get rewritten when the agent is being addressed.
- **Silero VAD, not energy VAD.** Energy gates open turns on breaths, sniffs and clicks, and those cut the agent off mid-sentence. Also drop "thank you" / "um" turns shorter than ~1.5 s; those are Whisper-style noise hallucinations.
- **TV counts as speech.** Cap turn length (`RTSHIM_VAD_MAX_TURN_MS`) or background talk holds a turn open forever.
- **Phone barge-in:** the voice-call plugin queues reply audio in the Gateway, so audio sent early keeps playing after the caller interrupts. The phone shim sends 20 ms frames only ~300 ms ahead and holds output if caller frames stop arriving.
- **HA only streams text to TTS after 60 characters**, so short replies don't benefit from streaming. Time-to-first-token is the term that matters.
- **The voice model will call your handoff tool even after answering.** Have a test-mode toggle that makes `ask_claw` run without delivering before you benchmark, or your group chat fills with "437".
- **Keep it warm.** A cron ping every 5 min to the LLM lane and STT saves about 1.5 s on the first turn after idle.

## Hardware notes

What we run: an RTX Pro 6000 Blackwell for the LLM and a 3090 for STT + TTS, with OpenClaw on a separate Ryzen AI Max+ 395 box. That's overkill. Parakeet and Chatterbox Turbo fit fine on one 24 GB card. The LLM is the part that sets your latency: anything that gets first token under ~0.3 s on a ~700-token prompt with tools will feel snappy.
