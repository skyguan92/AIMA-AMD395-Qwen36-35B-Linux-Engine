#!/usr/bin/env bash
# Start the exact native VL candidate for paired G4 measurements.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

BINARY="${AIMA_NATIVE_BINARY:?set AIMA_NATIVE_BINARY}"
MODEL_DIR="${AIMA_MODEL_DIR:?set AIMA_MODEL_DIR}"
MEDIA_ROOT="${AIMA_VL_MEDIA_ROOT:?set AIMA_VL_MEDIA_ROOT}"
VISION_IMAGE="${AIMA_VISION_ATTENTION_IMAGE:?set AIMA_VISION_ATTENTION_IMAGE}"
RUN_DIR="${AIMA_VL_RUN_DIR:?set AIMA_VL_RUN_DIR}"
PORT="${AIMA_VL_PORT:-31005}"
CACHE_MODE="${AIMA_VL_CACHE_MODE:-enabled}"
PREFIX_CACHE_MODE="${AIMA_VL_PREFIX_CACHE_MODE:-enabled}"
CONTEXT_TOKENS="${AIMA_VL_CONTEXT_TOKENS:-262143}"
CACHE_CAPACITY="${AIMA_VL_CACHE_CAPACITY:-262144}"
HOST="127.0.0.1"

if [[ ! -x "${BINARY}" ]]; then
  echo "native candidate is not executable: ${BINARY}" >&2
  exit 1
fi
BINARY_DIR="$(dirname "$(readlink -f "${BINARY}")")"
if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "native model is incomplete: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MEDIA_ROOT}" ]]; then
  echo "performance media root is missing: ${MEDIA_ROOT}" >&2
  exit 1
fi
for provider in libaima-fmha-aotriton.so libaima-fmha-ck.so \
                libaima-fmha-q16384-hybrid.so; do
  if [[ ! -f "${BINARY_DIR}/${provider}" ]]; then
    echo "native automatic provider is missing: ${BINARY_DIR}/${provider}" >&2
    exit 1
  fi
done
AOTRITON_RUNTIME="${BINARY_DIR}/libaotriton_v2.so.0.11.1"
AOTRITON_IMAGE_ROOT="${BINARY_DIR}/aotriton.images"
AOTRITON_IMAGE="${AOTRITON_IMAGE_ROOT}/amd-gfx11xx/flash/attn_fwd/FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2"
AOTRITON_RUNTIME_SHA256="e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5"
AOTRITON_IMAGE_SHA256="0f3a6a2f9dee6620443ee2145ee1f8257bde65a378589952840d99bf3d485c10"
if [[ ! -f "${AOTRITON_RUNTIME}" ]]; then
  echo "native AOTriton runtime is missing: ${AOTRITON_RUNTIME}" >&2
  exit 1
fi
if [[ "$(sha256sum "${AOTRITON_RUNTIME}" | awk '{print $1}')" != \
      "${AOTRITON_RUNTIME_SHA256}" ]]; then
  echo "native AOTriton runtime differs from the frozen artifact" >&2
  exit 1
fi
if [[ ! -f "${AOTRITON_IMAGE}" ]]; then
  echo "native AOTriton gfx1151 image is missing: ${AOTRITON_IMAGE}" >&2
  exit 1
fi
if [[ "$(sha256sum "${AOTRITON_IMAGE}" | awk '{print $1}')" != \
      "${AOTRITON_IMAGE_SHA256}" ]]; then
  echo "native AOTriton gfx1151 image differs from the frozen artifact" >&2
  exit 1
fi
mapfile -d '' -t aotriton_images < <(
  find "${AOTRITON_IMAGE_ROOT}" -type f -name '*.aks2' -print0 | sort -z
)
if (( ${#aotriton_images[@]} != 1 )) || \
   [[ "${aotriton_images[0]}" != "${AOTRITON_IMAGE}" ]]; then
  echo "native AOTriton closure must contain exactly the frozen image" >&2
  exit 1
fi
if [[ ! -f "${VISION_IMAGE}" ]]; then
  echo "native vision image is missing: ${VISION_IMAGE}" >&2
  exit 1
fi
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo "invalid VL performance port: ${PORT}" >&2
  exit 1
fi
if [[ "${CACHE_MODE}" != "enabled" && "${CACHE_MODE}" != "disabled" ]]; then
  echo "AIMA_VL_CACHE_MODE must be enabled or disabled" >&2
  exit 1
fi
if [[ "${PREFIX_CACHE_MODE}" != "enabled" && \
      "${PREFIX_CACHE_MODE}" != "disabled" ]]; then
  echo "AIMA_VL_PREFIX_CACHE_MODE must be enabled or disabled" >&2
  exit 1
fi
if [[ ! "${CONTEXT_TOKENS}" =~ ^[0-9]+$ ]] || \
   [[ ! "${CACHE_CAPACITY}" =~ ^[0-9]+$ ]]; then
  echo "native context and cache capacity must be decimal integers" >&2
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -Eq ":${PORT}$"; then
  echo "VL performance port is already in use: ${PORT}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
LOG_PATH="${RUN_DIR}/native-vl-performance.log"
PID_PATH="${RUN_DIR}/native-vl-performance.pid"
REPORT_PATH="${RUN_DIR}/native-weight-load.json"
cache_args=()
if [[ "${CACHE_MODE}" == "disabled" ]]; then
  cache_args+=(--disable-media-cache)
fi
if [[ "${PREFIX_CACHE_MODE}" == "disabled" ]]; then
  cache_args+=(--disable-prefix-cache)
fi

nohup setsid env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LANG=C.UTF-8 \
  HOME="${AIMA_VL_NATIVE_HOME:-${HOME}}" \
  "${BINARY}" serve \
  --model-dir "${MODEL_DIR}" \
  --context-tokens "${CONTEXT_TOKENS}" \
  --cache-capacity "${CACHE_CAPACITY}" \
  --vision-attention-image "${VISION_IMAGE}" \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --allowed-media-domain localhost \
  --allowed-media-domain 127.0.0.1 \
  --host "${HOST}" \
  --port "${PORT}" \
  --request-timeout-ms 600000 \
  --report "${REPORT_PATH}" \
  "${cache_args[@]}" \
  >"${LOG_PATH}" 2>&1 < /dev/null &

server_pid=$!
printf '%s\n' "${server_pid}" >"${PID_PATH}"
printf 'native VL performance candidate started: pid=%s port=%s media_cache=%s prefix_cache=%s log=%s\n' \
  "${server_pid}" "${PORT}" "${CACHE_MODE}" "${PREFIX_CACHE_MODE}" \
  "${LOG_PATH}"
