#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
OUT_DIR="${OUT_DIR:-${ROOT}/build/native}"
OUTPUT="${OUT_DIR}/native-vision-block-suffix-probe"

python3 "${ROOT}/scripts/generate-native-visual-layout.py" --check
mkdir -p "${OUT_DIR}"
"${HIPCC}" -O3 -DNDEBUG -std=c++17 --offload-arch=gfx1151 \
  -fno-gpu-rdc -fno-rtlib-add-rpath -pthread \
  -Wall -Wextra -Wpedantic -Werror \
  -I "${ROOT}/native/include" \
  -I "${ROOT}/native/generated" \
  "${ROOT}/native/tools/vision_block_suffix_oracle_probe.hip.cpp" \
  "${ROOT}/native/src/native_vision_block_suffix.hip.cpp" \
  "${ROOT}/native/src/bf16_gemm.hip.cpp" \
  "${ROOT}/native/src/native_weight_store.hip.cpp" \
  "${ROOT}/native/src/sha256.cpp" \
  "${ROOT}/benchmarks/shape-lab/native/src/torch_owned_safetensors_loader.hip.cpp" \
  -lhipblaslt -Wl,-z,noexecstack \
  -o "${OUTPUT}"

"${OUTPUT}" --help 2>/dev/null || test "$?" -eq 2
sha256sum "${OUTPUT}"
