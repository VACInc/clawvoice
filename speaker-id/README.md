# Speaker ID for Home Assistant Voice

This directory contains two additive services:

- `speaker-embed`: lazy-loaded ReDimNet speaker embeddings, flat-file enrollment, cosine scoring, and an optional report-only TitaNet-L scorer.
- `wyoming-tee`: a byte-preserving Wyoming TCP proxy. It forwards every original frame to the real STT service, copies STT responses without parsing or reserialization, and dispatches speaker work only after the final audio frame is already on the STT socket.

`SHADOW=1` is the default in both services. Identity is reported to Home Assistant but never changes tool arguments or enables enforcement.

## Dependency choice and model pins

Python 3.10+ is required. The runtime dependency set is deliberately small: PyTorch/TorchAudio for ReDimNet and resampling, NumPy for the flat embedding store, aiohttp for both the HTTP server and non-blocking clients, and the official Wyoming package. Using aiohttp on both sides avoids a second web stack and keeps embedding/webhook I/O on the tee's event loop.

- ReDimNet source: official `IDRnD/ReDimNet` commit `9438b1e98018d831df15fde93c580faae0a28d63`.
- ReDimNet checkpoint: official mixed-domain `M-vb2+vox2+cnc-ft_mix.pt`, SHA-256 `1e0716c2f351e79df054554f17c525b899045d2d05e4c839208b975e87d94e6a`.
- Optional TitaNet-L: NVIDIA checkpoint at Hugging Face revision `0dc382f40121a5fbd34db10a2bb04d826c2be6a8`, SHA-256 `e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3`.
- Wyoming: `wyoming==1.10.0`.

`fetch_models.py` downloads immutable source/checkpoint revisions and verifies both checkpoint hashes. Models remain unloaded until the first enroll/identify request.

## CUDA deployment

Example layout is `/opt/speaker-id`; adjust paths in both units and env files if needed.

```sh
sudo useradd --system --home /opt/speaker-id --shell /usr/sbin/nologin speaker-id
sudo install -d -o speaker-id -g speaker-id /opt/speaker-id /opt/speaker-id/data /opt/speaker-id/models /etc/speaker-id
sudo cp -a . /opt/speaker-id/
sudo -u speaker-id python3 -m venv /opt/speaker-id/.venv

# Pick the official wheel index matching the installed NVIDIA driver. CUDA 12.8 example:
sudo -u speaker-id /opt/speaker-id/.venv/bin/pip install \
  torch==2.10.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
sudo -u speaker-id /opt/speaker-id/.venv/bin/pip install -r /opt/speaker-id/requirements.txt

sudo -u speaker-id /opt/speaker-id/.venv/bin/python /opt/speaker-id/fetch_models.py \
  --model-dir /opt/speaker-id/models
```

For the optional TitaNet shadow score, install `requirements-titanet.txt`, run `fetch_models.py --titanet`, and set `TITANET_ENABLED=1`. NeMo is intentionally not in the base install because it is much larger than the primary service dependencies.

Install configuration and units:

```sh
sudo cp /opt/speaker-id/speaker-embed.env.example /etc/speaker-id/speaker-embed.env
sudo cp /opt/speaker-id/wyoming-tee.env.example /etc/speaker-id/wyoming-tee.env
sudo chmod 600 /etc/speaker-id/*.env
# Edit REAL_STT_*, DEVICE_ID, and the long random HA_WEBHOOK_URL before starting.

sudo cp /opt/speaker-id/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now speaker-embed.service wyoming-tee.service
curl -fsS http://127.0.0.1:8099/health
```

Point the Home Assistant STT Wyoming endpoint at `HOST:TEE_PORT`. The proxy continues to forward STT when embedding or Home Assistant is unavailable; in that case no fresh stamp exists and HA must treat the turn as guest.

## API

Both WAV (mono or multichannel PCM16) and raw little-endian PCM16 are accepted. Raw requests use `sample_rate` and `channels` query parameters, defaulting to `RAW_SAMPLE_RATE`/`RAW_CHANNELS`.

```sh
curl --data-binary @sample.wav 'http://127.0.0.1:8099/enroll?user=alice'
curl --data-binary @sample.wav 'http://127.0.0.1:8099/identify?device_id=kitchen'
curl http://127.0.0.1:8099/enrolled
```

An accepted result has `status=accepted` and an enrolled `user_id`. Scores from `LOW_THRESHOLD` up to (but not including) `HIGH_THRESHOLD` return `user_id=unknown`, `status=uncertain`, and a report-only `candidate_user_id`. Scores below low return unknown. This keeps ambiguous turns fail-closed.

Enrollment appends normalized sample embeddings. Identification compares against the normalized average of every sample for each user. With TitaNet enabled, enrollment stores a separate TitaNet embedding set and `/identify` includes `shadow_score`; that result never overrides ReDimNet.

## Environment variables

### speaker-embed

| Variable | Default | Purpose |
|---|---:|---|
| `SPEAKER_HOST` | `0.0.0.0` | HTTP bind address |
| `SPEAKER_PORT` | `8099` | HTTP port |
| `MODEL_DIR` | `models` | Base model directory |
| `ENROLLMENT_DIR` | `data/enrollments` | NPY/JSON embedding files |
| `REDIMNET_SOURCE_DIR` | pinned directory under `MODEL_DIR` | Exact source tree |
| `REDIMNET_CHECKPOINT` | mixed M checkpoint under `MODEL_DIR` | Primary weights |
| `REDIMNET_CHECKPOINT_SHA256` | pinned hash | Expected primary hash |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N`; auto falls back to CPU |
| `VERIFY_MODEL_HASHES` | `1` | Verify checkpoint hashes before load |
| `LOW_THRESHOLD` | `0.55` | Below this is unknown |
| `HIGH_THRESHOLD` | `0.72` | At/above this accepts the best enrollment |
| `SHADOW` | `1` | Mark results `enforced: false` |
| `ALLOWED_ENROLL_USERS` | `alice` | Comma-separated enrollment allowlist |
| `TITANET_ENABLED` | `0` | Enable optional TitaNet-L report-only scoring |
| `TITANET_CHECKPOINT` | pinned file under `MODEL_DIR` | Local `.nemo` checkpoint |
| `TITANET_CHECKPOINT_SHA256` | pinned hash | Expected TitaNet hash |
| `TITANET_LOW_THRESHOLD` | `0.50` | TitaNet unknown boundary |
| `TITANET_HIGH_THRESHOLD` | `0.65` | TitaNet accept/report boundary |
| `RAW_SAMPLE_RATE` | `16000` | Raw PCM input rate |
| `RAW_CHANNELS` | `1` | Raw PCM channel count |
| `MIN_AUDIO_MS` | `300` | Reject shorter clips |
| `MAX_AUDIO_SECONDS` | `30` | Reject longer clips |
| `MAX_BODY_BYTES` | `16777216` | HTTP request limit |
| `LOG_LEVEL` | `INFO` | Python log level |

### wyoming-tee

| Variable | Default | Purpose |
|---|---:|---|
| `TEE_HOST` | `0.0.0.0` | Wyoming bind address |
| `TEE_PORT` | `10301` | Wyoming proxy port |
| `REAL_STT_HOST` | `127.0.0.1` | Unchanged real STT host |
| `REAL_STT_PORT` | `10300` | Unchanged real STT port |
| `EMBED_URL` | `http://127.0.0.1:8099/identify` | Identify endpoint |
| `HA_WEBHOOK_URL` | empty | Full long-random-ID HA webhook URL |
| `DEVICE_ID` | `voice_satellite` | Stamp key for this satellite |
| `SHADOW` | `1` | Force webhook `enforced: false` |
| `IDENTIFY_TIMEOUT_SECONDS` | `10` | Detached identify timeout |
| `WEBHOOK_TIMEOUT_SECONDS` | `3` | Per-attempt HA timeout |
| `WEBHOOK_RETRIES` | `4` | Retries after the first HA attempt |
| `WEBHOOK_RETRY_BASE_SECONDS` | `0.5` | Exponential retry base |
| `WEBHOOK_RETRY_MAX_SECONDS` | `5` | Retry delay cap |
| `MAX_UTTERANCE_BYTES` | `16777216` | In-memory tee cap; overflow only drops identity |
| `MAX_FRAME_BYTES` | `33554432` | Wyoming input frame safety cap |
| `MAX_HEADER_BYTES` | `65536` | Wyoming JSON header safety cap |
| `RELAY_BUFFER_BYTES` | `65536` | Raw STT-to-client copy size |
| `CAPTURE_USER` | empty | Save WAVs under `CAPTURE_DIR/<user>` |
| `CAPTURE_DIR` | `data/captures` | Enrollment capture directory |
| `SHUTDOWN_GRACE_SECONDS` | `10` | Background dispatch shutdown grace |
| `LOG_LEVEL` | `INFO` | Python log level |

The tee also recognizes `capture`, `device_id`, `satellite_id`, or `site_id` string fields in a Wyoming event or its `context`. Setting `capture: alice` for a turn is the protocol equivalent of capture mode without changing the STT frame stream.

## Enrollment and tuning

Capture 5–10 varied Alice utterances (distance, room, volume), then enroll every WAV. Keep `SHADOW=1` while collecting same-speaker and guest scores. Tune from observed distributions:

- Raise `HIGH_THRESHOLD` to reduce false accepts; this is the safety-first adjustment.
- Set `LOW_THRESHOLD` below high to create a useful uncertain review band.
- Do not lower high based only on clean enrollment recordings; satellite acoustics are the target domain.
- Tune TitaNet separately. Its score is diagnostic and must not be mixed numerically with ReDimNet.

After changing thresholds, restart only `speaker-embed.service`. Existing enrollment files remain valid as long as the pinned model/checkpoint does not change.

## Tests

The suite uses synthetic PCM and injected deterministic CPU embedders, so its test-only dependency file omits Torch, never downloads model weights, and needs no GPU. It covers enrollment averaging, accepted/uncertain/unknown bands, CPU fallback, report-only TitaNet, exact Wyoming transcript wire bytes, and HA retry while the STT response is deliberately allowed through first.

```sh
python3 -m venv .test-venv
.test-venv/bin/pip install -r requirements-test.txt
cd ../..
bash scripts/tests/mutation-guard/run.sh -- bash -c 'cd projects/speaker-id && .test-venv/bin/python -m pytest -q -p no:cacheprovider'
```

## Rollback

Repoint the Home Assistant STT Wyoming endpoint from `TEE_HOST:TEE_PORT` directly back to `REAL_STT_HOST:REAL_STT_PORT`. That immediately removes the tee from the transcript path; no enrollment data, webhook automation, or service needs to be removed. Leaving the additive services running is harmless, and stopping them is optional after STT has been repointed.
