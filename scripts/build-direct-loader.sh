#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
OUT_DIR="${OUT_DIR:-${ROOT}/build/native}"
SOURCE="${ROOT}/benchmarks/shape-lab/native/src/torch_owned_safetensors_loader.hip.cpp"
OUTPUT="${OUT_DIR}/libtorch_owned_safetensors_loader.so"

mkdir -p "${OUT_DIR}"
"${HIPCC}" -O3 -std=c++17 --offload-arch=gfx1151 -shared -fPIC -pthread \
  "${SOURCE}" -o "${OUTPUT}"
sha256sum "${OUTPUT}"
