#!/usr/bin/env bash
# Run one fresh-process text control for the three G4 VL decode cells.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${AIMA_VL_TEXT_CONTROL_DIR:?set AIMA_VL_TEXT_CONTROL_DIR}"
RUN_INDEX="${AIMA_VL_TEXT_CONTROL_INDEX:-1}"
MANIFEST_PATH="${AIMA_VL_TEXT_CONTROL_MANIFEST:-${ROOT}/benchmarks/fixtures/vl-text-decode-control-v0.1.0/manifest.json}"
PORT="${AIMA_VL_TEXT_CONTROL_PORT:-31006}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to reuse text-control output: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ ! "${RUN_INDEX}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AIMA_VL_TEXT_CONTROL_INDEX must be a positive decimal integer" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "G4 text-control manifest is missing: ${MANIFEST_PATH}" >&2
  exit 1
fi
jq -e '
  .schema == "aima-amd395-qwen36/vl-text-decode-control-manifest/v1" and
  .complete == true and (.controls | length) == 3 and
  (.balanced_orders | length) == 2 and
  (.integrity.algorithm == "sha256")' "${MANIFEST_PATH}" >/dev/null

while IFS=$'\t' read -r request_path expected_bytes expected_sha; do
  resolved="${ROOT}/${request_path}"
  if [[ ! -f "${resolved}" ]] || \
     [[ "$(wc -c <"${resolved}")" -ne "${expected_bytes}" ]] || \
     [[ "$(sha256sum "${resolved}" | awk '{print $1}')" != "${expected_sha}" ]]; then
    echo "text-control request binding failed: ${request_path}" >&2
    exit 1
  fi
done < <(
  jq -r '.controls[] | [.request.path, .request.bytes, .request.sha256] | @tsv' \
    "${MANIFEST_PATH}"
)

mkdir -p "${OUTPUT_DIR}/requests"
cp "${MANIFEST_PATH}" "${OUTPUT_DIR}/manifest.json"
active_pid=""

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
  echo "forced text-control server stop after graceful timeout" >&2
  active_pid=""
}
trap stop_active EXIT INT TERM

wait_ready() {
  local endpoint="$1"
  local log_path="$2"
  for _ in $(seq 1 900); do
    if ! kill -0 "${active_pid}" 2>/dev/null; then
      echo "text-control candidate exited before readiness" >&2
      tail -n 100 "${log_path}" >&2 || true
      return 1
    fi
    if curl --silent --show-error --fail --max-time 2 \
      "${endpoint}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "text-control candidate did not become ready within 900 seconds" >&2
  return 1
}

endpoint="http://127.0.0.1:${PORT}"
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
jq -e \
  '.usage.prompt_tokens == 21 and .usage.completion_tokens == 1 and
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

for control_id in "${controls[@]}"; do
  request_path="$(
    jq -er --arg id "${control_id}" \
      '.controls[] | select(.control_id == $id) | .request.path' \
      "${MANIFEST_PATH}"
  )"
  padding="$(
    jq -er --arg id "${control_id}" \
      '.controls[] | select(.control_id == $id) | .text_padding_tokens' \
      "${MANIFEST_PATH}"
  )"
  prompt_tokens="$(
    jq -er --arg id "${control_id}" \
      '.controls[] | select(.control_id == $id) | .expected_prompt_tokens' \
      "${MANIFEST_PATH}"
  )"
  completion_tokens="$(
    jq -er --arg id "${control_id}" \
      '.controls[] | select(.control_id == $id) | .expected_completion_tokens' \
      "${MANIFEST_PATH}"
  )"
  python3 "${ROOT}/scripts/capture-vl-text-decode-control.py" \
    --endpoint "${endpoint}" \
    --request "${ROOT}/${request_path}" \
    --request-logical-path "${request_path}" \
    --output "${OUTPUT_DIR}/requests/${control_id}.json" \
    --model aima-amd395-qwen36-35b \
    --benchmark-id "${control_id}.control-${RUN_INDEX}" \
    --text-padding-tokens "${padding}" \
    --expected-prompt-tokens "${prompt_tokens}" \
    --expected-completion-tokens "${completion_tokens}" \
    --server-pid "${active_pid}" \
    --timeout-seconds 3600
done

stop_active
trap - EXIT INT TERM
printf '%s\n' "${RUN_INDEX}" >"${OUTPUT_DIR}/run-index.txt"
printf '%s\n' "$(sha256sum "${AIMA_NATIVE_BINARY}" | awk '{print $1}')" \
  >"${OUTPUT_DIR}/candidate-binary.sha256"
printf 'VL text decode control captured: %s\n' "${OUTPUT_DIR}"
