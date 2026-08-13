#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-g++}"
OUT_DIR="${OUT_DIR:-${ROOT}/build/native}"
OUTPUT="${OUT_DIR}/native-vl-mrope-probe"

mkdir -p "${OUT_DIR}"
"${CXX}" -O2 -DNDEBUG -std=c++17 \
  -Wall -Wextra -Wpedantic -Werror \
  -I "${ROOT}/native/include" \
  "${ROOT}/native/tools/vl_mrope_oracle_probe.cpp" \
  "${ROOT}/native/src/native_mrope.cpp" \
  "${ROOT}/native/src/sha256.cpp" \
  -Wl,-z,noexecstack \
  -o "${OUTPUT}"

"${OUTPUT}" --help 2>/dev/null || test "$?" -eq 2
sha256sum "${OUTPUT}"
