#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
OBJCOPY="${OBJCOPY:-/usr/bin/objcopy}"
OUT_DIR="${OUT_DIR:-${ROOT}/build/native}"
OUTPUT="${OUT_DIR}/native-vl-language-layer0-probe"
Q1024_DIR="${ROOT}/native/aot/gfx1151/q1024-output1"
Q8192_DIR="${ROOT}/native/aot/gfx1151/q8192-output2"
AOT_REGISTRY_CPP="${OUT_DIR}/vl-layer0-aot-registry.cpp"
AOT_OBJECT_PLAN="${OUT_DIR}/vl-layer0-aot-objects.tsv"
DECODE_REGISTRY_CPP="${OUT_DIR}/vl-layer0-decode-registry.cpp"
PREFILL_REGISTRY_CPP="${OUT_DIR}/vl-layer0-prefill-registry.cpp"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${ROOT}" rev-parse HEAD)}"
if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=normal)" ]]; then
  SOURCE_COMMIT="${SOURCE_COMMIT}-dirty"
fi

python3 "${ROOT}/scripts/generate-native-layout.py" --check
mkdir -p "${OUT_DIR}"
python3 "${ROOT}/scripts/generate-native-aot-registry.py" \
  --manifest "${Q1024_DIR}/manifest.json" \
  --output-cpp "${AOT_REGISTRY_CPP}" \
  --output-plan "${AOT_OBJECT_PLAN}"
python3 "${ROOT}/scripts/generate-native-decode-registry.py" \
  --schedule "${Q8192_DIR}/decode-schedule.json" \
  --aot-manifest "${Q8192_DIR}/manifest.json" \
  --output-cpp "${DECODE_REGISTRY_CPP}"
python3 "${ROOT}/scripts/generate-native-decode-registry.py" \
  --phase prefill \
  --schedule "${Q1024_DIR}/prefill-schedule.json" \
  --aot-manifest "${Q1024_DIR}/manifest.json" \
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

"${HIPCC}" -O3 -DNDEBUG -std=c++17 --offload-arch=gfx1151 \
  -DAIMA_SOURCE_COMMIT=\"${SOURCE_COMMIT}\" \
  -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -fno-gpu-rdc \
  -fno-rtlib-add-rpath -ffunction-sections -fdata-sections -pthread \
  -Wall -Wextra -Wpedantic \
  -I "${ROOT}/native/include" \
  -I "${ROOT}/native/generated" \
  "${ROOT}/native/tools/vl_language_layer0_oracle_probe.hip.cpp" \
  "${ROOT}/native/src/aot_kernel.hip.cpp" \
  "${ROOT}/native/src/bf16_gemm.hip.cpp" \
  "${ROOT}/native/src/native_derived_weights.hip.cpp" \
  "${ROOT}/native/src/native_decode_bindings.hip.cpp" \
  "${ROOT}/native/src/native_decode_executor.hip.cpp" \
  "${ROOT}/native/src/native_decode_invocation.cpp" \
  "${ROOT}/native/src/native_decode_workspace.hip.cpp" \
  "${ROOT}/native/src/native_prefill_invocation.cpp" \
  "${ROOT}/native/src/native_prefill_workspace.hip.cpp" \
  "${ROOT}/native/src/native_prefill_gemm_plans.hip.cpp" \
  "${ROOT}/native/src/native_vl_logical_projections.hip.cpp" \
  "${ROOT}/native/src/native_linear_prefill.hip.cpp" \
  "${ROOT}/native/src/native_moe_prefill.hip.cpp" \
  "${ROOT}/native/src/native_pointwise.hip.cpp" \
  "${ROOT}/native/src/native_layer_oracle.hip.cpp" \
  "${ROOT}/native/src/native_lm_head.hip.cpp" \
  "${ROOT}/native/src/native_weight_store.hip.cpp" \
  "${ROOT}/native/src/sha256.cpp" \
  "${ROOT}/benchmarks/shape-lab/native/src/torch_owned_safetensors_loader.hip.cpp" \
  "${AOT_REGISTRY_CPP}" \
  "${DECODE_REGISTRY_CPP}" \
  "${PREFILL_REGISTRY_CPP}" \
  -x none \
  "${AOT_OBJECTS[@]}" \
  -lhipblaslt \
  -Wl,--gc-sections -Wl,--exclude-libs,ALL -Wl,-z,noexecstack \
  -o "${OUTPUT}"

"${OUTPUT}" --help 2>/dev/null || test "$?" -eq 2
sha256sum "${OUTPUT}"
