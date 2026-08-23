#!/usr/bin/env bash
# Capture adjacent text/VL decode pairs for the three exact G4 decode cells.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${AIMA_VL_DECODE_CONTROL_DIR:?set AIMA_VL_DECODE_CONTROL_DIR}"
RUN_INDEX="${AIMA_VL_DECODE_CONTROL_INDEX:-1}"
MANIFEST_PATH="${AIMA_VL_TEXT_CONTROL_MANIFEST:-${ROOT}/benchmarks/fixtures/vl-text-decode-control-v0.1.0/manifest.json}"
MATRIX_PATH="${AIMA_VL_MATRIX_PATH:-${ROOT}/benchmarks/fixtures/vl-performance-v0.1.0/comparable-matrix.json}"
MEDIA_ROOT="${AIMA_VL_MEDIA_ROOT:?set AIMA_VL_MEDIA_ROOT}"
PORT="${AIMA_VL_DECODE_CONTROL_PORT:-31007}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to reuse VL decode-control output: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ ! "${RUN_INDEX}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AIMA_VL_DECODE_CONTROL_INDEX must be a positive decimal integer" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST_PATH}" || ! -f "${MATRIX_PATH}" ]]; then
  echo "VL decode-control manifest or G4 matrix is missing" >&2
  exit 1
fi
if [[ ! -d "${MEDIA_ROOT}" || "${MEDIA_ROOT}" != /* ]]; then
  echo "AIMA_VL_MEDIA_ROOT must be an existing absolute directory" >&2
  exit 1
fi
jq -e '
  .schema == "aima-amd395-qwen36/vl-text-decode-control-manifest/v1" and
  .complete == true and (.controls | length) == 3 and
  (.balanced_orders | length) == 2 and
  (.integrity.algorithm == "sha256")' "${MANIFEST_PATH}" >/dev/null
jq -e '
  .schema == "aima-amd395-qwen36/vl-performance-comparable-matrix/v1" and
  .complete == true and (.cells | length) > 0 and
  (.integrity.algorithm == "sha256")' "${MATRIX_PATH}" >/dev/null

while IFS=$'\t' read -r control_id cell_id expected_prompt expected_output; do
  jq -e --arg cell "${cell_id}" --argjson output "${expected_output}" '
    [.cells[] |
      select(.cell_id == $cell and .output_tokens == $output)] |
    length == 1' "${MATRIX_PATH}" >/dev/null
  if [[ -z "${control_id}" || ! "${expected_prompt}" =~ ^[0-9]+$ ]]; then
    echo "VL decode-control mapping is malformed: ${control_id}" >&2
    exit 1
  fi
done < <(
  jq -r '.controls[] |
    [.control_id, .g4_cell_id, .expected_prompt_tokens,
     .expected_completion_tokens] | @tsv' "${MANIFEST_PATH}"
)

mkdir -p "${OUTPUT_DIR}/text/requests" "${OUTPUT_DIR}/vl/requests"
cp "${MANIFEST_PATH}" "${OUTPUT_DIR}/manifest.json"
cp "${MATRIX_PATH}" "${OUTPUT_DIR}/matrix.json"
active_pid=""

require_gpu_idle() {
  if fuser /dev/kfd >/dev/null 2>&1; then
    fuser -v /dev/kfd 2>&1
    return 75
  fi
}

stop_active() {
  if [[ -z "${active_pid}" ]] || \
     ! pgrep -g "${active_pid}" >/dev/null 2>&1; then
    active_pid=""
    return
  fi
  if kill -0 "${active_pid}" 2>/dev/null; then
    curl --silent --show-error --max-time 5 \
      -X POST "http://127.0.0.1:${PORT}/shutdown" \
      >/dev/null 2>&1 || true
  fi
  for _ in $(seq 1 120); do
    if ! pgrep -g "${active_pid}" >/dev/null 2>&1; then
      active_pid=""
      return
    fi
    sleep 0.5
  done
  kill -TERM -- "-${active_pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! pgrep -g "${active_pid}" >/dev/null 2>&1; then
      active_pid=""
      return
    fi
    sleep 0.5
  done
  kill -KILL -- "-${active_pid}" 2>/dev/null || true
  echo "forced VL decode-control server stop after graceful timeout" >&2
  active_pid=""
}
trap stop_active EXIT INT TERM

wait_ready() {
  local endpoint="$1"
  local log_path="$2"
  for _ in $(seq 1 900); do
    if ! kill -0 "${active_pid}" 2>/dev/null; then
      echo "VL decode-control candidate exited before readiness" >&2
      tail -n 100 "${log_path}" >&2 || true
      return 1
    fi
    if curl --silent --show-error --fail --max-time 2 \
      "${endpoint}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "VL decode-control candidate did not become ready within 900 seconds" >&2
  return 1
}

matrix_cell_field() {
  local cell_id="$1"
  local expression="$2"
  jq -er --arg cell "${cell_id}" \
    ".cells[] | select(.cell_id == \$cell) | ${expression}" \
    "${MATRIX_PATH}"
}

endpoint="http://127.0.0.1:${PORT}"
require_gpu_idle
AIMA_VL_RUN_DIR="${OUTPUT_DIR}" \
AIMA_VL_PORT="${PORT}" \
AIMA_VL_CACHE_MODE=disabled \
AIMA_VL_PREFIX_CACHE_MODE=disabled \
AIMA_VL_CONTEXT_TOKENS=262143 \
AIMA_VL_CACHE_CAPACITY=262144 \
  "${ROOT}/scripts/run-native-vl-performance-candidate.sh"
active_pid="$(<"${OUTPUT_DIR}/native-vl-performance.pid")"
wait_ready "${endpoint}" "${OUTPUT_DIR}/native-vl-performance.log"
curl --silent --show-error --fail --max-time 5 \
  "${endpoint}/health" >"${OUTPUT_DIR}/health.json"

jq -cn \
  '{model:"aima-amd395-qwen36-35b",
    messages:[{role:"user",
               content:"Reply with one token for symmetric warmup."}],
    temperature:0,max_tokens:1,stream:false}' | \
  curl --silent --show-error --fail --max-time 300 \
    -H "Content-Type: application/json" --data-binary @- \
    "${endpoint}/v1/chat/completions" >"${OUTPUT_DIR}/text-warmup.json"
jq -e '
  .usage.prompt_tokens == 21 and .usage.completion_tokens == 1 and
  .usage.total_tokens == 22 and .choices[0].finish_reason == "length"' \
  "${OUTPUT_DIR}/text-warmup.json" >/dev/null

order_index=0
if (( RUN_INDEX % 2 == 0 )); then
  order_index=1
fi
mapfile -t controls < <(
  jq -er --argjson index "${order_index}" '.balanced_orders[$index][]' \
    "${MANIFEST_PATH}"
)
printf '%s\n' "${controls[@]}" >"${OUTPUT_DIR}/request-order.txt"
if (( RUN_INDEX % 2 == 0 )); then
  pair_order="vl text"
else
  pair_order="text vl"
fi
printf '%s\n' "${pair_order}" >"${OUTPUT_DIR}/pair-order.txt"

for control_id in "${controls[@]}"; do
  cell_id="$(
    jq -er --arg id "${control_id}" \
      '.controls[] | select(.control_id == $id) | .g4_cell_id' \
      "${MANIFEST_PATH}"
  )"
  expected_prompt="$(
    jq -er --arg id "${control_id}" \
      '.controls[] | select(.control_id == $id) | .expected_prompt_tokens' \
      "${MANIFEST_PATH}"
  )"
  vl_request_path="$(matrix_cell_field "${cell_id}" '.request.path')"
  vl_padding="$(matrix_cell_field "${cell_id}" '.text_padding_tokens')"
  prompt_nonce="$(matrix_cell_field "${cell_id}" '.prompt_nonce')"
  completion_tokens="$(matrix_cell_field "${cell_id}" '.output_tokens')"
  text_request_path="$(
    jq -er --arg id "${control_id}" \
      '.controls[] | select(.control_id == $id) | .request.path' \
      "${MANIFEST_PATH}"
  )"
  text_padding="$(
    jq -er --arg id "${control_id}" \
      '.controls[] | select(.control_id == $id) | .text_padding_tokens' \
      "${MANIFEST_PATH}"
  )"

  capture_text() {
    python3 "${ROOT}/scripts/capture-vl-text-decode-control.py" \
      --endpoint "${endpoint}" \
      --request "${ROOT}/${text_request_path}" \
      --request-logical-path "${text_request_path}" \
      --output "${OUTPUT_DIR}/text/requests/${control_id}.json" \
      --model aima-amd395-qwen36-35b \
      --benchmark-id "${control_id}.control-${RUN_INDEX}" \
      --text-padding-tokens "${text_padding}" \
      --expected-prompt-tokens "${expected_prompt}" \
      --expected-completion-tokens "${completion_tokens}" \
      --server-pid "${active_pid}" \
      --timeout-seconds 3600
  }

  capture_vl() {
    python3 "${ROOT}/scripts/capture-vl-performance-request.py" \
      --endpoint "${endpoint}" \
      --request "${ROOT}/${vl_request_path}" \
      --media-root "${MEDIA_ROOT}" \
      --output "${OUTPUT_DIR}/vl/requests/${control_id}.json" \
      --model aima-amd395-qwen36-35b \
      --benchmark-id "${cell_id}.vl-control-${RUN_INDEX}" \
      --prompt-nonce "${prompt_nonce}" \
      --text-padding-tokens "${vl_padding}" \
      --expected-completion-tokens "${completion_tokens}" \
      --engine-role candidate \
      --server-pid "${active_pid}" \
      --timeout-seconds 3600
  }

  for role in ${pair_order}; do
    "capture_${role}"
  done
  jq -e --argjson prompt "${expected_prompt}" \
    '.complete == true and
     .response.usage.prompt_tokens == $prompt and
     .native_metrics.prompt_tokens == $prompt' \
    "${OUTPUT_DIR}/vl/requests/${control_id}.json" >/dev/null
done

stop_active
trap - EXIT INT TERM
printf '%s\n' "${RUN_INDEX}" >"${OUTPUT_DIR}/run-index.txt"
printf '%s\n' "$(sha256sum "${AIMA_NATIVE_BINARY}" | awk '{print $1}')" \
  >"${OUTPUT_DIR}/candidate-binary.sha256"
printf 'VL decode control captured: %s\n' "${OUTPUT_DIR}"
