#!/usr/bin/env bash
# Container launcher: Qwen3.8-Flash-Next on ghcr.io/local-inference-lab/vllm:release-10152026,
# the nvidia/Qwen3.8-Flash-Next-FP4-Local model card's "NVIDIA RTX PRO 6000 Blackwell" two-step verbose recipe
# (every exported env var and every flag), model mounted at /model. Differences from the card: /model instead of the
# Hub id, --served-model-name voice-llm, port 31320, kernel caches pinned to the /cache volume (same paths the
# image launcher derives for PROFILE=qwen38-flash-next), HF offline, gpu util overridable (UTIL, card value 0.975).
set -u
export B12X_DENSE_SPLITK_TURBO=1 B12X_DYNAMIC_DETERMINISTIC_OUTPUT=0 B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1 \
  B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1 B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224 B12X_DYNAMIC_SPLIT_FAST_PREPARE=1 \
  B12X_DYNAMIC_SPLIT_LOW_SMEM=1 B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1 B12X_DYNAMIC_WORK_SOURCE=persistent_grid \
  B12X_W4A16_A4_PREFILL=1 B12X_DISK_BACKEND=io_uring B12X_MHC_PDL=1 B12X_PCIE_ONESHOT_BLOCK_LIMIT=4 \
  B12X_PCIE_ONESHOT_PDL=1 B12X_PCIE_ONESHOT_THREADS=512 CUDA_BINARY_LOADER_THREAD_COUNT=8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_MODULE_LOADING=LAZY CUTE_DSL_ARCH=sm_120a INSTANTTENSOR_BACKEND=BUFFERED NCCL_BUFFSIZE=2097152 \
  NCCL_CUMEM_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_MAX_NCHANNELS=16 NCCL_MIN_NCHANNELS=16 NCCL_NET_PLUGIN=spcx \
  NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL,LL128,Simple OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SAFETENSORS_FAST_GPU=1 TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION=0 TORCH_CUDA_ARCH_LIST=12.0a \
  VLLM_B12X_DENSE_ACTIVATION_MODE=auto VLLM_B12X_MOE_FP4_FORCE_A16=1 VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1 \
  VLLM_COMPUTE_NANS_IN_LOGITS=0 VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=1 VLLM_LM_HEAD_A16=1 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  VLLM_MTP_NVFP4_LM_HEAD=1 VLLM_MXFP8_LM_HEAD=0 VLLM_PCIE_ALLREDUCE_BACKEND=b12x VLLM_PLE_CPU_OFFLOAD=1 \
  VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0 VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT=1 VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=1 \
  VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_USE_FASTOKENS=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 TORCH_NCCL_USE_COMM_NONBLOCKING=0
# Kernel/JIT caches on the named /cache volume (same layout as the image launcher's derived dirs).
C=/cache/jit/b59b3584e47a07032d14ef4e0a14dd9c8d1f336f991c6068cf72610ed0a58696/qwen38-flash-next-f7436b2e9ff58db7
export XDG_CACHE_HOME=$C VLLM_CACHE_ROOT=$C/vllm VLLM_CACHE_DIR=$C/vllm TRITON_CACHE_DIR=$C/triton \
  TORCHINDUCTOR_CACHE_DIR=$C/torchinductor TORCH_EXTENSIONS_DIR=$C/torch-extensions CUDA_CACHE_PATH=$C/cuda \
  B12X_COMPILE_CACHE_DIR=$C/b12x/compile B12X_CUTE_COMPILE_CACHE_DIR=$C/b12x/cute SPARKINFER_COMPILE_CACHE_DIR=$C/b12x/compile \
  CUTE_DSL_CACHE_DIR=$C/cute-dsl FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/flash-attention-cute-dsl \
  FLASHINFER_WORKSPACE_BASE=$C/flashinfer TVM_CACHE_DIR=$C/tvm TVM_FFI_CACHE_DIR=$C/tvm-ffi CUPY_CACHE_DIR=$C/cupy \
  NUMBA_CACHE_DIR=$C/numba TILELANG_CACHE_DIR=$C/tilelang HF_HUB_OFFLINE=1
mkdir -p "$C" 2>/dev/null || true
# SPEC_K=0 is a no-MTP fallback (only if the MTP drafter fails to load); the card recipe is SPEC_K=3.
SPEC=()
if [ "${SPEC_K:-3}" != 0 ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"moe_backend\":\"b12x\",\"num_speculative_tokens\":${SPEC_K:-3}}")
fi
exec vllm serve /model \
    --served-model-name "${SERVED:-voice-llm}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-31320}" \
    --async-scheduling \
    --block-size 64 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --decode-context-parallel-size 1 \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --enable-chunked-prefill \
    --no-enable-flashinfer-autotune \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --gdn-decode-kernel b12x \
    --gpu-memory-utilization "${UTIL:-0.975}" \
    --kv-cache-dtype fp8 \
    --no-language-model-only \
    --linear-backend b12x \
    --load-format instanttensor \
    --mamba-cache-mode align \
    --mamba-ssm-cache-dtype auto \
    --max-cudagraph-capture-size 64 \
    --max-model-len "${MAXLEN:-262144}" \
    --max-num-batched-tokens 6019 \
    --max-num-seqs 16 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-gb 0.0 \
    --moe-backend b12x \
    --pipeline-parallel-size 1 \
    --quantization modelopt_mixed \
    --reasoning-parser qwen3 \
    "${SPEC[@]}" \
    --tensor-parallel-size 1 \
    --tool-call-parser qwen3_xml
