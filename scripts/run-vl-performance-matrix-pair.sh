#!/usr/bin/env bash
# Run one fresh-process, alternating pair for every frozen G4 matrix cell.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${AIMA_VL_MATRIX_DIR:?set AIMA_VL_MATRIX_DIR}"
MEDIA_ROOT="${AIMA_VL_MEDIA_ROOT:?set AIMA_VL_MEDIA_ROOT}"
PAIR_INDEX="${AIMA_VL_PAIR_INDEX:-1}"
MATRIX_PATH="${AIMA_VL_MATRIX_PATH:-${ROOT}/benchmarks/fixtures/vl-performance-v0.1.0/matrix.json}"
REFERENCE_PORT="${AIMA_VL_REFERENCE_PORT:-31004}"
CANDIDATE_PORT="${AIMA_VL_CANDIDATE_PORT:-31005}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to reuse matrix-pair output: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MEDIA_ROOT}" || "${MEDIA_ROOT}" != /* ]]; then
  echo "AIMA_VL_MEDIA_ROOT must be an existing absolute directory" >&2
  exit 1
fi
if [[ ! -f "${MATRIX_PATH}" ]]; then
  echo "G4 matrix manifest is missing: ${MATRIX_PATH}" >&2
  exit 1
fi
if [[ ! "${PAIR_INDEX}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AIMA_VL_PAIR_INDEX must be a positive decimal integer" >&2
  exit 1
fi
jq -e \
  '.complete == true and (.cells | length) > 0 and
   ([.process_groups[].process_group] == ["disabled", "enabled"])' \
  "${MATRIX_PATH}" >/dev/null

mkdir -p "${OUTPUT_DIR}"
cp "${MATRIX_PATH}" "${OUTPUT_DIR}/matrix.json"
active_pid=""
active_role=""
active_port=""

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
    active_role=""
    active_port=""
    return
  fi
  if [[ "${active_role}" == "candidate" ]] && \
     kill -0 "${active_pid}" 2>/dev/null; then
    curl --silent --show-error --max-time 5 \
      -X POST "http://127.0.0.1:${active_port}/shutdown" \
      >/dev/null 2>&1 || true
  else
    kill -TERM -- "-${active_pid}" 2>/dev/null || true
  fi
  for _ in $(seq 1 120); do
    if ! pgrep -g "${active_pid}" >/dev/null 2>&1; then
      active_pid=""
      active_role=""
      active_port=""
      return
    fi
    sleep 0.5
  done
  kill -TERM -- "-${active_pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! pgrep -g "${active_pid}" >/dev/null 2>&1; then
      active_pid=""
      active_role=""
      active_port=""
      return
    fi
    sleep 0.5
  done
  kill -KILL -- "-${active_pid}" 2>/dev/null || true
  echo "forced stop after graceful timeout: role=${active_role} pid=${active_pid}" >&2
  active_pid=""
  active_role=""
  active_port=""
}
trap stop_active EXIT INT TERM

wait_ready() {
  local endpoint="$1"
  local log_path="$2"
  for _ in $(seq 1 900); do
    if ! kill -0 "${active_pid}" 2>/dev/null; then
      echo "${active_role} exited before readiness" >&2
      tail -n 100 "${log_path}" >&2 || true
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

wait_stage_count() {
  local stage_path="$1"
  local expected="$2"
  for _ in $(seq 1 100); do
    if [[ -f "${stage_path}" ]] && \
       [[ "$(wc -l <"${stage_path}")" -eq "${expected}" ]]; then
      return 0
    fi
    sleep 0.1
  done
  echo "reference middleware stage count did not reach ${expected}" >&2
  return 1
}

matrix_cell_field() {
  local cell_id="$1"
  local expression="$2"
  jq -er --arg cell "${cell_id}" \
    ".cells[] | select(.cell_id == \$cell) | ${expression}" \
    "${MATRIX_PATH}"
}

capture_cells() {
  local role="$1"
  local process_group="$2"
  local run_dir="$3"
  local endpoint="$4"
  local model="$5"
  local order_index=0
  if (( PAIR_INDEX % 2 == 0 )); then
    order_index=1
  fi
  mapfile -t cells < <(
    jq -er --arg group "${process_group}" --argjson index "${order_index}" \
      '.process_groups[] | select(.process_group == $group) |
       .balanced_orders[$index][]' "${MATRIX_PATH}"
  )
  mkdir -p "${run_dir}/requests"
  printf '%s\n' "${cells[@]}" >"${run_dir}/request-order.txt"
  local stage_count=0
  for cell_id in "${cells[@]}"; do
    local logical_request
    local request_path
    local padding
    local prompt_nonce
    local output_tokens
    logical_request="$(matrix_cell_field "${cell_id}" '.request.path')"
    request_path="${ROOT}/${logical_request}"
    padding="$(matrix_cell_field "${cell_id}" '.text_padding_tokens')"
    prompt_nonce="$(matrix_cell_field "${cell_id}" '.prompt_nonce')"
    output_tokens="$(matrix_cell_field "${cell_id}" '.output_tokens')"
    if [[ ! -f "${request_path}" ]]; then
      echo "matrix request is missing: ${request_path}" >&2
      return 1
    fi
    capture_args=(
      --endpoint "${endpoint}"
      --request "${request_path}"
      --media-root "${MEDIA_ROOT}"
      --output "${run_dir}/requests/${cell_id}.json"
      --model "${model}"
      --benchmark-id "${cell_id}.pair-${PAIR_INDEX}"
      --prompt-nonce "${prompt_nonce}"
      --text-padding-tokens "${padding}"
      --expected-completion-tokens "${output_tokens}"
      --engine-role "${role}"
      --server-pid "${active_pid}"
      --timeout-seconds 3600
    )
    if [[ "${role}" == "reference" ]]; then
      capture_args+=(--prometheus)
    fi
    python3 "${ROOT}/scripts/capture-vl-performance-request.py" \
      "${capture_args[@]}"
    if [[ "${role}" == "reference" ]]; then
      stage_count=$((stage_count + 1))
      wait_stage_count "${run_dir}/vllm-vl-stages.jsonl" "${stage_count}"
    fi
  done
}

run_process_group() {
  local role="$1"
  local process_group="$2"
  local cache_mode="$3"
  local run_dir="${OUTPUT_DIR}/${process_group}/${role}"
  local port
  local endpoint
  local model
  mkdir -p "${run_dir}"
  require_gpu_idle
  if [[ "${role}" == "reference" ]]; then
    port="${REFERENCE_PORT}"
    model="qwen36-vl-reference"
    AIMA_VL_RUN_DIR="${run_dir}" \
    AIMA_VL_PORT="${port}" \
    AIMA_VL_CACHE_MODE="${cache_mode}" \
      "${ROOT}/scripts/run-vllm-vl-performance-reference.sh"
    active_pid="$(<"${run_dir}/vllm-vl-performance.pid")"
    active_role="reference"
    active_port="${port}"
    wait_ready "http://127.0.0.1:${port}" \
      "${run_dir}/vllm-vl-performance.log"
    curl --silent --show-error --fail --max-time 5 \
      "http://127.0.0.1:${port}/health" >"${run_dir}/health.txt"
  else
    port="${CANDIDATE_PORT}"
    model="aima-amd395-qwen36-35b"
    AIMA_VL_RUN_DIR="${run_dir}" \
    AIMA_VL_PORT="${port}" \
    AIMA_VL_CACHE_MODE="${cache_mode}" \
    AIMA_VL_PREFIX_CACHE_MODE=disabled \
      "${ROOT}/scripts/run-native-vl-performance-candidate.sh"
    active_pid="$(<"${run_dir}/native-vl-performance.pid")"
    active_role="candidate"
    active_port="${port}"
    wait_ready "http://127.0.0.1:${port}" \
      "${run_dir}/native-vl-performance.log"
    curl --silent --show-error --fail --max-time 5 \
      "http://127.0.0.1:${port}/health" >"${run_dir}/health.json"
  fi
  endpoint="http://127.0.0.1:${port}"
  warm_text_path "${endpoint}" "${model}" "${run_dir}/text-warmup.json"
  capture_cells "${role}" "${process_group}" "${run_dir}" \
    "${endpoint}" "${model}"
  stop_active
}

if (( PAIR_INDEX % 2 == 1 )); then
  roles=(reference candidate)
else
  roles=(candidate reference)
fi
printf '%s\n' "${roles[*]}" >"${OUTPUT_DIR}/execution-order.txt"
for role in "${roles[@]}"; do
  for process_group in disabled enabled; do
    run_process_group "${role}" "${process_group}" "${process_group}"
  done
done
trap - EXIT INT TERM
python3 "${ROOT}/scripts/summarize-vl-performance-matrix-pair.py" \
  --pair-dir "${OUTPUT_DIR}" \
  --matrix "${MATRIX_PATH}"
printf 'VL performance matrix pair captured: %s\n' "${OUTPUT_DIR}"
