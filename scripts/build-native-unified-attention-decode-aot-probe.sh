#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
OUT_DIR="${OUT_DIR:-${ROOT}/build/native}"
OUTPUT="${OUT_DIR}/native-unified-attention-decode-aot-probe"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${ROOT}" rev-parse HEAD)}"
if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=normal)" ]]; then
  SOURCE_COMMIT="${SOURCE_COMMIT}-dirty"
fi

python3 "${ROOT}/scripts/generate-native-layout.py" --check
mkdir -p "${OUT_DIR}"
"${HIPCC}" -O3 -DNDEBUG -std=c++17 --offload-arch=gfx1151 \
  -DAIMA_SOURCE_COMMIT=\"${SOURCE_COMMIT}\" \
  -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -fno-gpu-rdc \
  -fno-rtlib-add-rpath -ffunction-sections -fdata-sections -pthread \
  -Wall -Wextra -Wpedantic \
  -I "${ROOT}/native/include" \
  -I "${ROOT}/native/generated" \
  "${ROOT}/native/tools/unified_attention_decode_aot_probe.hip.cpp" \
  "${ROOT}/native/src/aot_kernel.hip.cpp" \
  "${ROOT}/native/src/sha256.cpp" \
  -Wl,--gc-sections -Wl,--exclude-libs,ALL -Wl,-z,noexecstack \
  -o "${OUTPUT}"

"${OUTPUT}" --help 2>/dev/null || test "$?" -eq 2
sha256sum "${OUTPUT}"
