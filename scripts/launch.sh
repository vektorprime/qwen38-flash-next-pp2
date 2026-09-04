#!/usr/bin/env bash
set -uo pipefail
# Launch qwen38-flash-next with PP=2 + PLE CPU offload.
# Token input (pick one, never commit a token):
#   bash scripts/launch.sh /path/to/token.txt
#   HT=$(cat /path/to/token.txt) bash scripts/launch.sh
IMAGE="${IMAGE:-qwen38-flash-next:pp2-p1}"
GPU_IDS="${GPU_IDS:-0,1}"
PORT="${PORT:-8001}"

if [ -z "${HT:-}" ] && [ $# -ge 1 ] && [ -f "$1" ]; then
  HT=$(cat "$1")
fi
HT="${HT:-}"
if [ -z "$HT" ]; then
  echo "ERROR: set HT (token string) or pass a token file as first arg" >&2
  exit 1
fi
echo "token len: ${#HT}"

docker rm -f qwen38-flash-next 2>/dev/null

docker run --runtime nvidia -d --gpus "\"device=$GPU_IDS\"" \
  --ipc=host --shm-size=96g \
  --cap-add=SYS_PTRACE \
  --name qwen38-flash-next --restart always -p "$PORT":8000 \
  -v vllm-hf-cache:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --env "VLLM_BT_POOL=5" \
  --env "CUDA_DEVICE_ORDER=PCI_BUS_ID" \
  --env "HUGGING_FACE_HUB_TOKEN=$HT" \
  --env "VLLM_PLE_CPU_OFFLOAD=1" \
  --env "VLLM_GDN_DECODE_KERNEL=triton" \
  "$IMAGE" \
  serve cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4 \
  --served-model-name qwen38-flash-next-awq \
  --max-model-len auto \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 8 \
  --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float32 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --async-scheduling \
  --generation-config auto \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0,"presence_penalty":0.0}'
rc=$?
echo "run exit: $rc"
[ $rc -eq 0 ] && echo "watch: docker logs -f qwen38-flash-next"
exit $rc
