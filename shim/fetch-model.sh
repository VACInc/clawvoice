#!/usr/bin/env bash
# Fetch the pinned Silero VAD v5 ONNX model into models/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
url="https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
sha="2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
mkdir -p models
if [ -f models/silero_vad.onnx ] && echo "$sha  models/silero_vad.onnx" | sha256sum -c --status; then
  echo "models/silero_vad.onnx already present"; exit 0
fi
curl -fsSL -o models/silero_vad.onnx.tmp "$url"
echo "$sha  models/silero_vad.onnx.tmp" | sha256sum -c -
mv models/silero_vad.onnx.tmp models/silero_vad.onnx
echo "fetched models/silero_vad.onnx"
