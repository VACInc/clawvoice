#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 on ONE RTX PRO 6000 Blackwell, OpenAI-compatible on :31320.
# MODEL_DIR = local download of nvidia/Qwen3.8-Flash-Next-FP4-Local. GPU = device index or UUID.
set -euo pipefail
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the downloaded model}"
GPU="${GPU:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
exec docker run -d --init --name qwen38-voice --restart unless-stopped \
  --gpus "\"device=$GPU\"" --network host --ipc host --shm-size 32g \
  --memory 180g --memory-swap 180g --stop-timeout 120 \
  --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
  -v "$MODEL_DIR":/model:ro -v qwen38-cache:/cache -v "$HERE/serve.sh":/inner.sh:ro \
  -e HOST="${HOST:-127.0.0.1}" -e PORT="${PORT:-31320}" -e SERVED=voice-llm \
  -e MAXLEN="${MAXLEN:-262144}" -e SPEC_K=3 -e UTIL=0.975 -e HF_HUB_OFFLINE=1 \
  --entrypoint /bin/bash ghcr.io/local-inference-lab/vllm:release-10152026 /inner.sh
