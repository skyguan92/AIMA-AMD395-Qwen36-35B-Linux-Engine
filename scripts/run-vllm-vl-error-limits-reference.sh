#!/usr/bin/env bash
# Start the frozen vLLM server for media-IO and error/limit capture.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

RUNTIME_PYTHON="${AIMA_VLLM_PYTHON:?set AIMA_VLLM_PYTHON}"
MODEL_DIR="${AIMA_MODEL_DIR:?set AIMA_MODEL_DIR}"
MEDIA_ROOT="${AIMA_VL_MEDIA_ROOT:?set AIMA_VL_MEDIA_ROOT}"
RUN_DIR="${AIMA_VL_RUN_DIR:?set AIMA_VL_RUN_DIR}"
PORT="${AIMA_VL_PORT:-31004}"
HOST="127.0.0.1"

if [[ ! -x "${RUNTIME_PYTHON}" ]]; then
  echo "reference Python is not executable: ${RUNTIME_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "reference model is incomplete: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MEDIA_ROOT}" ]]; then
  echo "reference media root is missing: ${MEDIA_ROOT}" >&2
  exit 1
fi
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo "invalid reference port: ${PORT}" >&2
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -Eq ":${PORT}$"; then
  echo "reference port is already in use: ${PORT}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
LOG_PATH="${RUN_DIR}/vllm-vl-error-limits.log"
PID_PATH="${RUN_DIR}/vllm-vl-error-limits.pid"

nohup env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LANG=C.UTF-8 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONHASHSEED=0 \
  PYTORCH_ROCM_ARCH=gfx1151 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  ROCM_PATH=/opt/rocm \
  HIP_PATH=/opt/rocm \
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
  VLLM_IMAGE_FETCH_TIMEOUT=10 \
  VLLM_VIDEO_FETCH_TIMEOUT=30 \
  VLLM_VIDEO_LOADER_BACKEND=opencv \
  "${RUNTIME_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --served-model-name qwen36-vl-reference \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.95 \
  --attention-backend TRITON_ATTN \
  --mm-encoder-attn-backend TRITON_ATTN \
  --gdn-prefill-backend triton \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --no-language-model-only \
  --no-skip-mm-profiling \
  --limit-mm-per-prompt '{"image":16,"video":21}' \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --allowed-media-domains localhost 127.0.0.1 \
  --media-io-kwargs '{"video":{"fps":2.0,"video_backend":"opencv"}}' \
  --mm-processor-kwargs '{}' \
  --mm-processor-cache-gb 4 \
  --video-pruning-rate 0 \
  --load-format safetensors \
  --tensor-parallel-size 1 \
  >"${LOG_PATH}" 2>&1 < /dev/null &

server_pid=$!
printf '%s\n' "${server_pid}" >"${PID_PATH}"
printf 'error/limit reference started: pid=%s port=%s log=%s\n' \
  "${server_pid}" "${PORT}" "${LOG_PATH}"
