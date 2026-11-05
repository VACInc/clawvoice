# The voice LLM: pick one

Everything else in this repo talks to one OpenAI-compatible endpoint (`/v1/chat/completions`, served model id `voice-llm`). We've run both of these in production for voice. Either works.

| | Option A: Qwen3.8-Flash-Next | Option B: DeepSeek V4.1 Flash |
|---|---|---|
| Model | `nvidia/Qwen3.8-Flash-Next-FP4-Local` (NVFP4, QAD) | DeepSeek V4.1 Flash, 3 bpw EXL3 quant |
| GPUs | 1x RTX PRO 6000 Blackwell (96 GB) | 2x RTX PRO 6000 Blackwell |
| Server | vLLM (`ghcr.io/local-inference-lab/vllm:release-10152026`), MTP speculative decoding k=3 | EXL3 server, OpenAI-compatible |
| Context | 262k native (we stretched to 524k with yarn x2) | 1M+ |
| Tool calls | `--tool-call-parser qwen3_xml` | native |
| Why pick it | One card, fastest first token, best all-round score in our tests | Strongest reasoning; needs two cards |

## Option A: Qwen on one card

`qwen38/serve.sh` is the model card's single RTX PRO 6000 recipe (every env var and flag), with our served name and port. `run.sh` starts it:

```bash
huggingface-cli download nvidia/Qwen3.8-Flash-Next-FP4-Local --local-dir /models/qwen38-fp4
MODEL_DIR=/models/qwen38-fp4 GPU=0 ./qwen38/run.sh
curl -s http://127.0.0.1:31320/v1/models
```

The first start compiles kernels for a while; they're cached in the `qwen38-cache` volume after that.

Thinking: it thinks by default. For voice, send `"chat_template_kwargs": {"enable_thinking": false}` on every request (see the provider config in the main README).

## Option B: DeepSeek V4.1 Flash on two cards

We serve a community 3 bpw EXL3 quant across two PRO 6000s. That build is too custom to package here; any OpenAI-compatible server for DeepSeek V4.1 Flash that fits your VRAM works the same. Point `RTSHIM_LLM_URL`, the HA conversation agent and the OpenClaw provider at it, served as `voice-llm`.
