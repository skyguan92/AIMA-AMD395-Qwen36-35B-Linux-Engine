#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
OUT_DIR="${OUT_DIR:-${ROOT}/build/native}"
OUTPUT="${OUT_DIR}/native-vl-recompute-w-u-aot-probe"

mkdir -p "${OUT_DIR}"
"${HIPCC}" -O3 -DNDEBUG -std=c++17 --offload-arch=gfx1151 \
  -fno-rtlib-add-rpath -Wall -Wextra -Wpedantic \
  -I "${ROOT}/native/include" \
  "${ROOT}/native/tools/vl_recompute_w_u_aot_probe.hip.cpp" \
  "${ROOT}/native/src/aot_kernel.hip.cpp" \
  -o "${OUTPUT}"
printf '%s\n' "${OUTPUT}"
