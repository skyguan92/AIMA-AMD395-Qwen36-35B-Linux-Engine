#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
OUT_DIR="${OUT_DIR:-${ROOT}/build/native}"
OUTPUT="${OUT_DIR}/native-vision-aot-attention-probe"

mkdir -p "${OUT_DIR}"
"${HIPCC}" -O3 -DNDEBUG -std=c++17 --offload-arch=gfx1151 \
  -fno-gpu-rdc -fno-rtlib-add-rpath -pthread \
  -Wall -Wextra -Wpedantic -Werror \
  -I "${ROOT}/native/include" \
  "${ROOT}/native/tools/vision_aot_attention_oracle_probe.hip.cpp" \
  "${ROOT}/native/src/native_vision_aot_attention.hip.cpp" \
  "${ROOT}/native/src/aot_kernel.hip.cpp" \
  "${ROOT}/native/src/sha256.cpp" \
  -Wl,-z,noexecstack \
  -o "${OUTPUT}"

"${OUTPUT}" --help 2>/dev/null || test "$?" -eq 2
sha256sum "${OUTPUT}"
