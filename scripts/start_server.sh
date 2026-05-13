#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server for FlashEvolve examples.
#
# Usage:
#   bash scripts/start_server.sh                 # defaults: Qwen3-8B on :8000
#   bash scripts/start_server.sh 8001            # custom port
#   bash scripts/start_server.sh 8000 Qwen/Qwen3-14B   # custom port + model
#
# Pin GPU visibility with CUDA_VISIBLE_DEVICES=<idx> in the calling shell
# if you are sharing the box.

set -euo pipefail

PORT="${1:-8000}"
MODEL="${2:-Qwen/Qwen3-8B}"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95
