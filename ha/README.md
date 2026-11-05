# Home Assistant pieces

- `pyscript/voiceconsult.py`: registers `rest_command.claw_voice_consult`, the only way the voice model can reach OpenClaw. Checks the caller is an active HA admin from the real service context, accepts only `{question}` (1 to 2000 chars), runs one request at a time with a 60 s cooldown, POSTs to OpenClaw `/hooks/agent` with an idempotency key, ACKs HA right away and confirms completion in the background. Unknown completion keeps the slot held (fail closed).
- `packages/voice_consult_test_mode.yaml`: an `input_boolean` that makes consults run with `deliver: false` while you benchmark, and turns itself off after 30 min and on HA start.

Pyscript app config (`pyscript/config.yaml`):

```yaml
apps:
  voiceconsult:
    url: https://claw-host/hooks/agent
    authorization: !secret claw_hook_authorization   # "Bearer <hooks.token>"
    chat_id: "<telegram chat id for answers>"
    agent_id: voice
```

Reload order after changes: `rest_command.reload`, then `pyscript.reload` with `global_ctx: apps.voiceconsult`.
