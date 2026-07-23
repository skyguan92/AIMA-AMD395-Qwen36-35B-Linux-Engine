#!/usr/bin/env bash
set -euo pipefail

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
OUT=${OUT:-"$ROOT/build/native/aot-kernel-probe"}

mkdir -p "$(dirname "$OUT")"
"$ROCM_PATH/bin/hipcc" \
  -std=c++17 \
  -O3 \
  --offload-arch=gfx1151 \
  -I"$ROOT/native/include" \
  "$ROOT/native/src/aot_kernel.hip.cpp" \
  "$ROOT/native/src/aot_kernel_probe.hip.cpp" \
  -o "$OUT"
printf '%s\n' "$OUT"
