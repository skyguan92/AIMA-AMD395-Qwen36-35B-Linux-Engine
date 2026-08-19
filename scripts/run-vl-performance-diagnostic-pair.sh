#!/usr/bin/env bash
# Run one alternating, fresh-process G4 image diagnostic pair.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${AIMA_VL_DIAGNOSTIC_DIR:?set AIMA_VL_DIAGNOSTIC_DIR}"
MEDIA_ROOT="${AIMA_VL_MEDIA_ROOT:?set AIMA_VL_MEDIA_ROOT}"
PAIR_INDEX="${AIMA_VL_PAIR_INDEX:-1}"
REFERENCE_PORT="${AIMA_VL_REFERENCE_PORT:-31004}"
CANDIDATE_PORT="${AIMA_VL_CANDIDATE_PORT:-31005}"
REQUEST_PATH="${AIMA_VL_REQUEST_PATH:-${ROOT}/benchmarks/fixtures/vl-performance-v0.1.0/requests/diagnostic-image-typical-output1.json}"
CELL_ID="${AIMA_VL_CELL_ID:-diagnostic-image-typical-output1}"
TEXT_PADDING_TOKENS="${AIMA_VL_TEXT_PADDING_TOKENS:-0}"
EXPECTED_COMPLETION_TOKENS="${AIMA_VL_EXPECTED_COMPLETION_TOKENS:-}"
CACHE_MODE="${AIMA_VL_CACHE_MODE:-enabled}"
PREFIX_CACHE_MODE="${AIMA_VL_PREFIX_CACHE_MODE:-disabled}"
TIMEOUT_SECONDS="${AIMA_VL_TIMEOUT_SECONDS:-3600}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to reuse diagnostic output: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MEDIA_ROOT}" || "${MEDIA_ROOT}" != /* ]]; then
  echo "AIMA_VL_MEDIA_ROOT must be an existing absolute directory" >&2
  exit 1
fi
if [[ ! -f "${REQUEST_PATH}" ]]; then
  echo "diagnostic request is missing: ${REQUEST_PATH}" >&2
  exit 1
fi
if [[ ! "${PAIR_INDEX}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AIMA_VL_PAIR_INDEX must be a positive decimal integer" >&2
  exit 1
fi
if [[ ! "${CELL_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "AIMA_VL_CELL_ID contains unsupported characters" >&2
  exit 1
fi
if [[ ! "${TEXT_PADDING_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "AIMA_VL_TEXT_PADDING_TOKENS must be a decimal integer" >&2
  exit 1
fi
if [[ -z "${EXPECTED_COMPLETION_TOKENS}" ]]; then
  EXPECTED_COMPLETION_TOKENS="$(jq -er '.max_tokens' "${REQUEST_PATH}")"
fi
if [[ ! "${EXPECTED_COMPLETION_TOKENS}" =~ ^[1-9][0-9]*$ ]] ||
   (( EXPECTED_COMPLETION_TOKENS > 1024 )); then
  echo "AIMA_VL_EXPECTED_COMPLETION_TOKENS must be in [1, 1024]" >&2
  exit 1
fi
if [[ "${CACHE_MODE}" != "enabled" && "${CACHE_MODE}" != "disabled" ]]; then
  echo "AIMA_VL_CACHE_MODE must be enabled or disabled" >&2
  exit 1
fi
if [[ "${PREFIX_CACHE_MODE}" != "enabled" &&
      "${PREFIX_CACHE_MODE}" != "disabled" ]]; then
  echo "AIMA_VL_PREFIX_CACHE_MODE must be enabled or disabled" >&2
  exit 1
fi
if [[ ! "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] ||
   (( TIMEOUT_SECONDS > 3600 )); then
  echo "AIMA_VL_TIMEOUT_SECONDS must be in [1, 3600]" >&2
  exit 1
fi
if [[ "${CACHE_MODE}" == "disabled" ]]; then
  MEDIA_CACHE_EXPECTATION=disabled
else
  MEDIA_CACHE_EXPECTATION=cold
fi
if [[ "${PREFIX_CACHE_MODE}" == "disabled" ]]; then
  PREFIX_CACHE_EXPECTATION=disabled
else
  PREFIX_CACHE_EXPECTATION=miss
fi

mkdir -p "${OUTPUT_DIR}"
active_pid=""
active_role=""

stop_active() {
  if [[ -z "${active_pid}" ]] || \
     ! pgrep -g "${active_pid}" >/dev/null 2>&1; then
    active_pid=""
    active_role=""
    return
  fi
  if [[ "${active_role}" == "candidate" ]] && \
     kill -0 "${active_pid}" 2>/dev/null; then
    curl --silent --show-error --max-time 5 \
      -X POST "http://127.0.0.1:${CANDIDATE_PORT}/shutdown" \
      >/dev/null 2>&1 || true
  else
    kill -TERM -- "-${active_pid}" 2>/dev/null || true
  fi
  for _ in $(seq 1 120); do
    if ! pgrep -g "${active_pid}" >/dev/null 2>&1; then
      active_pid=""
      active_role=""
      return
    fi
    sleep 0.5
  done
  kill -TERM -- "-${active_pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! pgrep -g "${active_pid}" >/dev/null 2>&1; then
      active_pid=""
      active_role=""
      return
    fi
    sleep 0.5
  done
  kill -KILL -- "-${active_pid}" 2>/dev/null || true
  echo "forced stop after graceful timeout: role=${active_role} pid=${active_pid}" >&2
  active_pid=""
  active_role=""
}
trap stop_active EXIT INT TERM

wait_ready() {
  local endpoint="$1"
  local log_path="$2"
  for _ in $(seq 1 900); do
    if ! kill -0 "${active_pid}" 2>/dev/null; then
      echo "${active_role} exited before readiness" >&2
      tail -n 80 "${log_path}" >&2 || true
      return 1
    fi
    if curl --silent --show-error --fail --max-time 2 \
      "${endpoint}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "${active_role} did not become ready within 900 seconds" >&2
  return 1
}

warm_text_path() {
  local endpoint="$1"
  local model="$2"
  local output="$3"
  jq -cn --arg model "${model}" \
    '{model:$model,
      messages:[{role:"user",
                 content:"Reply with one token for symmetric warmup."}],
      temperature:0,max_tokens:1,stream:false}' | \
    curl --silent --show-error --fail --max-time 300 \
      -H "Content-Type: application/json" --data-binary @- \
      "${endpoint}/v1/chat/completions" >"${output}"
  jq -e \
    '.usage.completion_tokens == 1 and (.choices | length) == 1' \
    "${output}" >/dev/null
}

capture_reference() {
  local run_dir="${OUTPUT_DIR}/reference"
  AIMA_VL_RUN_DIR="${run_dir}" \
  AIMA_VL_PORT="${REFERENCE_PORT}" \
  AIMA_VL_CACHE_MODE="${CACHE_MODE}" \
    "${ROOT}/scripts/run-vllm-vl-performance-reference.sh"
  active_pid="$(<"${run_dir}/vllm-vl-performance.pid")"
  active_role="reference"
  wait_ready "http://127.0.0.1:${REFERENCE_PORT}" \
    "${run_dir}/vllm-vl-performance.log"
  curl --silent --show-error --fail --max-time 5 \
    "http://127.0.0.1:${REFERENCE_PORT}/health" \
    >"${run_dir}/health.txt"
  warm_text_path "http://127.0.0.1:${REFERENCE_PORT}" \
    qwen36-vl-reference "${run_dir}/text-warmup.json"
  python3 "${ROOT}/scripts/capture-vl-performance-request.py" \
    --endpoint "http://127.0.0.1:${REFERENCE_PORT}" \
    --request "${REQUEST_PATH}" \
    --media-root "${MEDIA_ROOT}" \
    --output "${run_dir}/request.json" \
    --model qwen36-vl-reference \
    --benchmark-id "${CELL_ID}.pair-${PAIR_INDEX}" \
    --prompt-nonce "${CELL_ID}" \
    --text-padding-tokens "${TEXT_PADDING_TOKENS}" \
    --expected-completion-tokens "${EXPECTED_COMPLETION_TOKENS}" \
    --engine-role reference \
    --server-pid "${active_pid}" \
    --timeout-seconds "${TIMEOUT_SECONDS}" \
    --prometheus
  for _ in $(seq 1 100); do
    if [[ -f "${run_dir}/vllm-vl-stages.jsonl" ]] && \
       [[ "$(wc -l <"${run_dir}/vllm-vl-stages.jsonl")" -eq 1 ]]; then
      break
    fi
    sleep 0.1
  done
  if [[ ! -f "${run_dir}/vllm-vl-stages.jsonl" ]] || \
     [[ "$(wc -l <"${run_dir}/vllm-vl-stages.jsonl")" -ne 1 ]]; then
    echo "reference stage middleware did not publish exactly one record" >&2
    return 1
  fi
  stop_active
}

capture_candidate() {
  local run_dir="${OUTPUT_DIR}/candidate"
  AIMA_VL_RUN_DIR="${run_dir}" \
  AIMA_VL_PORT="${CANDIDATE_PORT}" \
  AIMA_VL_CACHE_MODE="${CACHE_MODE}" \
  AIMA_VL_PREFIX_CACHE_MODE="${PREFIX_CACHE_MODE}" \
    "${ROOT}/scripts/run-native-vl-performance-candidate.sh"
  active_pid="$(<"${run_dir}/native-vl-performance.pid")"
  active_role="candidate"
  wait_ready "http://127.0.0.1:${CANDIDATE_PORT}" \
    "${run_dir}/native-vl-performance.log"
  curl --silent --show-error --fail --max-time 5 \
    "http://127.0.0.1:${CANDIDATE_PORT}/health" \
    >"${run_dir}/health.json"
  warm_text_path "http://127.0.0.1:${CANDIDATE_PORT}" \
    aima-amd395-qwen36-35b "${run_dir}/text-warmup.json"
  python3 "${ROOT}/scripts/capture-vl-performance-request.py" \
    --endpoint "http://127.0.0.1:${CANDIDATE_PORT}" \
    --request "${REQUEST_PATH}" \
    --media-root "${MEDIA_ROOT}" \
    --output "${run_dir}/request.json" \
    --model aima-amd395-qwen36-35b \
    --benchmark-id "${CELL_ID}.pair-${PAIR_INDEX}" \
    --prompt-nonce "${CELL_ID}" \
    --text-padding-tokens "${TEXT_PADDING_TOKENS}" \
    --expected-completion-tokens "${EXPECTED_COMPLETION_TOKENS}" \
    --engine-role candidate \
    --server-pid "${active_pid}" \
    --timeout-seconds "${TIMEOUT_SECONDS}"
  stop_active
}

if (( PAIR_INDEX % 2 == 1 )); then
  order=(reference candidate)
else
  order=(candidate reference)
fi
printf '%s\n' "${order[*]}" >"${OUTPUT_DIR}/execution-order.txt"
for role in "${order[@]}"; do
  if [[ "${role}" == "reference" ]]; then
    capture_reference
  else
    capture_candidate
  fi
done
trap - EXIT INT TERM
python3 "${ROOT}/scripts/summarize-vl-performance-diagnostic.py" \
  --run-dir "${OUTPUT_DIR}" \
  --expected-output-tokens "${EXPECTED_COMPLETION_TOKENS}" \
  --expected-prefix-cache-lookup "${PREFIX_CACHE_EXPECTATION}" \
  --expected-media-cache-mode "${MEDIA_CACHE_EXPECTATION}"
printf 'VL performance diagnostic pair complete: %s\n' "${OUTPUT_DIR}"
