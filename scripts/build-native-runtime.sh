#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
OUT_DIR="${OUT_DIR:-${ROOT}/build/native}"
OUTPUT="${OUT_DIR}/aima-engine-native"
LAUNCHER="${OUT_DIR}/aima-engine-launcher"
ICU_STATIC_DIR="${ICU_STATIC_DIR:-/usr/lib/x86_64-linux-gnu}"
PREFILL_CONTEXTS="${PREFILL_CONTEXTS:-1024:2048:4096:7168:7680:8191:8192:16384:32768}"
prefill_closure_dir() {
  local context="$1"
  if [[ "${context}" == "8192" ]]; then
    printf '%s/native/aot/gfx1151/q8192-output2' "${ROOT}"
  else
    printf '%s/native/aot/gfx1151/q%s-output1' "${ROOT}" "${context}"
  fi
}

DEFAULT_AOT_MANIFESTS=""
DEFAULT_PREFILL_SCHEDULES=""
DEFAULT_PREFILL_AOT_MANIFESTS=""
IFS=: read -r -a PREFILL_CONTEXT_LIST <<< "${PREFILL_CONTEXTS}"
for context in "${PREFILL_CONTEXT_LIST[@]}"; do
  closure_dir="$(prefill_closure_dir "${context}")"
  manifest="${closure_dir}/manifest.json"
  schedule="${closure_dir}/prefill-schedule.json"
  DEFAULT_AOT_MANIFESTS="${DEFAULT_AOT_MANIFESTS:+${DEFAULT_AOT_MANIFESTS}:}${manifest}"
  DEFAULT_PREFILL_SCHEDULES="${DEFAULT_PREFILL_SCHEDULES:+${DEFAULT_PREFILL_SCHEDULES}:}${schedule}"
  DEFAULT_PREFILL_AOT_MANIFESTS="${DEFAULT_PREFILL_AOT_MANIFESTS:+${DEFAULT_PREFILL_AOT_MANIFESTS}:}${manifest}"
done
AOT_MANIFEST_Q8192="$(prefill_closure_dir 8192)/manifest.json"
AOT_MANIFESTS="${AOT_MANIFESTS:-${DEFAULT_AOT_MANIFESTS}}"
AOT_REGISTRY_CPP="${OUT_DIR}/aot_registry.cpp"
AOT_OBJECT_PLAN="${OUT_DIR}/aot_objects.tsv"
DECODE_SCHEDULE="${DECODE_SCHEDULE:-${ROOT}/native/aot/gfx1151/q8192-output2/decode-schedule.json}"
DECODE_AOT_MANIFEST="${DECODE_AOT_MANIFEST:-${AOT_MANIFEST_Q8192}}"
DECODE_REGISTRY_CPP="${OUT_DIR}/decode_schedule_registry.cpp"
if [[ -n "${PREFILL_SCHEDULE:-}" ]]; then
  PREFILL_SCHEDULES="${PREFILL_SCHEDULE}"
  PREFILL_AOT_MANIFESTS="${PREFILL_AOT_MANIFEST:-${AOT_MANIFEST_Q8192}}"
else
  PREFILL_SCHEDULES="${PREFILL_SCHEDULES:-${DEFAULT_PREFILL_SCHEDULES}}"
  PREFILL_AOT_MANIFESTS="${PREFILL_AOT_MANIFESTS:-${DEFAULT_PREFILL_AOT_MANIFESTS}}"
fi
PREFILL_REGISTRY_CPP="${OUT_DIR}/prefill_schedule_registry.cpp"
OBJCOPY="${OBJCOPY:-/usr/bin/objcopy}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${ROOT}" rev-parse HEAD)}"
if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=normal)" ]]; then
  SOURCE_COMMIT="${SOURCE_COMMIT}-dirty"
fi

for archive in libicui18n.a libicuuc.a libicudata.a; do
  if [[ ! -f "${ICU_STATIC_DIR}/${archive}" ]]; then
    echo "missing static ICU archive: ${ICU_STATIC_DIR}/${archive}" >&2
    exit 1
  fi
done

python3 "${ROOT}/scripts/generate-native-layout.py" --check
mkdir -p "${OUT_DIR}"
IFS=: read -r -a AOT_MANIFEST_PATHS <<< "${AOT_MANIFESTS}"
AOT_MANIFEST_ARGS=()
for manifest in "${AOT_MANIFEST_PATHS[@]}"; do
  if [[ ! -f "${manifest}" ]]; then
    echo "missing native AOT manifest: ${manifest}" >&2
    exit 1
  fi
  AOT_MANIFEST_ARGS+=(--manifest "${manifest}")
done
python3 "${ROOT}/scripts/generate-native-aot-registry.py" \
  "${AOT_MANIFEST_ARGS[@]}" \
  --output-cpp "${AOT_REGISTRY_CPP}" \
  --output-plan "${AOT_OBJECT_PLAN}"
python3 "${ROOT}/scripts/generate-native-decode-registry.py" \
  --schedule "${DECODE_SCHEDULE}" \
  --aot-manifest "${DECODE_AOT_MANIFEST}" \
  --output-cpp "${DECODE_REGISTRY_CPP}"
IFS=: read -r -a PREFILL_SCHEDULE_PATHS <<< "${PREFILL_SCHEDULES}"
IFS=: read -r -a PREFILL_MANIFEST_PATHS <<< "${PREFILL_AOT_MANIFESTS}"
if [[ "${#PREFILL_SCHEDULE_PATHS[@]}" -ne "${#PREFILL_MANIFEST_PATHS[@]}" ]]; then
  echo "prefill schedule and manifest counts differ" >&2
  exit 1
fi
PREFILL_REGISTRY_ARGS=()
for index in "${!PREFILL_SCHEDULE_PATHS[@]}"; do
  PREFILL_REGISTRY_ARGS+=(
    --schedule "${PREFILL_SCHEDULE_PATHS[$index]}"
    --aot-manifest "${PREFILL_MANIFEST_PATHS[$index]}"
  )
done
python3 "${ROOT}/scripts/generate-native-decode-registry.py" \
  --phase prefill \
  "${PREFILL_REGISTRY_ARGS[@]}" \
  --output-cpp "${PREFILL_REGISTRY_CPP}"

AOT_OBJECTS=()
while IFS=$'\t' read -r image_path object_name image_name; do
  object_path="${OUT_DIR}/${object_name}"
  (
    cd "$(dirname "${image_path}")"
    "${OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64 \
      "${image_name}" "${object_path}"
  )
  "${OBJCOPY}" \
    --rename-section .data=.rodata,alloc,load,readonly,data,contents \
    "${object_path}"
  AOT_OBJECTS+=("${object_path}")
done < "${AOT_OBJECT_PLAN}"

"${HIPCC}" -O3 -DNDEBUG -DU_STATIC_IMPLEMENTATION -std=c++17 --offload-arch=gfx1151 \
  -DAIMA_SOURCE_COMMIT=\"${SOURCE_COMMIT}\" \
  -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -fno-gpu-rdc \
  -fno-rtlib-add-rpath -ffunction-sections -fdata-sections -pthread \
  -I "${ROOT}/native/include" \
  -I "${ROOT}/native/generated" \
  "${ROOT}/native/src/main.cpp" \
  "${ROOT}/native/src/aot_kernel.hip.cpp" \
  "${ROOT}/native/src/aot_registry_probe.hip.cpp" \
  "${ROOT}/native/src/decode_schedule_probe.cpp" \
  "${ROOT}/native/src/bf16_gemm.hip.cpp" \
  "${ROOT}/native/src/bf16_wvsplitk.hip.cpp" \
  "${ROOT}/native/src/native_derived_weights.hip.cpp" \
  "${ROOT}/native/src/native_decode_bindings.hip.cpp" \
  "${ROOT}/native/src/native_decode_executor.hip.cpp" \
  "${ROOT}/native/src/native_decode_runner.hip.cpp" \
  "${ROOT}/native/src/native_decode_invocation.cpp" \
  "${ROOT}/native/src/native_decode_workspace.hip.cpp" \
  "${ROOT}/native/src/native_prefill_invocation.cpp" \
  "${ROOT}/native/src/native_prefill_workspace.hip.cpp" \
  "${ROOT}/native/src/native_prefill_gemm_plans.hip.cpp" \
  "${ROOT}/native/src/native_resident_engine.hip.cpp" \
  "${ROOT}/native/src/native_chat_protocol.cpp" \
  "${ROOT}/native/src/native_media.cpp" \
  "${ROOT}/native/src/native_multimodal_cache.cpp" \
  "${ROOT}/native/src/native_doctor.cpp" \
  "${ROOT}/native/src/native_http_server.cpp" \
  "${ROOT}/native/src/native_pointwise.hip.cpp" \
  "${ROOT}/native/src/native_full_prefill.hip.cpp" \
  "${ROOT}/native/src/native_full_attention.hip.cpp" \
  "${ROOT}/native/src/native_full_layer.hip.cpp" \
  "${ROOT}/native/src/native_linear_layer.hip.cpp" \
  "${ROOT}/native/src/native_linear_prefill.hip.cpp" \
  "${ROOT}/native/src/native_moe_prefill.hip.cpp" \
  "${ROOT}/native/src/native_layer_oracle.hip.cpp" \
  "${ROOT}/native/src/native_lm_head.hip.cpp" \
  "${ROOT}/native/src/native_lm_head_certificate.hip.cpp" \
  "${AOT_REGISTRY_CPP}" \
  "${DECODE_REGISTRY_CPP}" \
  "${PREFILL_REGISTRY_CPP}" \
  "${ROOT}/native/src/sha256.cpp" \
  "${ROOT}/native/src/native_tokenizer.cpp" \
  "${ROOT}/native/src/native_weight_store.hip.cpp" \
  "${ROOT}/benchmarks/shape-lab/native/src/torch_owned_safetensors_loader.hip.cpp" \
  -x none \
  "${AOT_OBJECTS[@]}" \
  "${ICU_STATIC_DIR}/libicui18n.a" \
  "${ICU_STATIC_DIR}/libicuuc.a" \
  "${ICU_STATIC_DIR}/libicudata.a" \
  -lhipblaslt -ldl -Wl,--gc-sections -Wl,--exclude-libs,ALL -Wl,-z,noexecstack \
  -Wl,-z,origin -Wl,--enable-new-dtags -Wl,-rpath,'$ORIGIN/../lib' \
  -o "${OUTPUT}"

"${OUTPUT}" --version
sha256sum "${OUTPUT}"
ldd "${OUTPUT}"

"${CC:-gcc}" -O2 -DNDEBUG -std=c11 -static -s \
  -Wl,-z,noexecstack \
  "${ROOT}/native/src/portable_launcher.c" \
  -o "${LAUNCHER}"
if readelf -l "${LAUNCHER}" | grep -Fq INTERP || \
   readelf -d "${LAUNCHER}" 2>/dev/null | grep -Fq NEEDED; then
  echo "portable launcher is not fully static" >&2
  exit 1
fi
file "${LAUNCHER}"
sha256sum "${LAUNCHER}"
