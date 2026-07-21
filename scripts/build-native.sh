#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CK_COMMIT="6667a9021713f794a2c9aee4696c19f6cf376235"
CK_DIR="${CK_DIR:?set CK_DIR to an AMD Composable Kernel checkout}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
CXX="${CXX:-g++}"
OUT_DIR="${OUT_DIR:-${ROOT}/build/native}"
SRC="${ROOT}/benchmarks/shape-lab/native/src"

actual_commit="$(git -C "${CK_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${CK_COMMIT}" ]]; then
  echo "Composable Kernel commit mismatch: expected ${CK_COMMIT}, got ${actual_commit}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

common=(
  "${HIPCC}" -std=c++17 -O3 -fPIC --offload-arch=gfx1151
  -DCK_TILE_FMHA_FWD_FAST_EXP2=0
  -I "${CK_DIR}/include"
  -I "${CK_DIR}/example/ck_tile/01_fmha"
  -shared
)

"${common[@]}" \
  "${SRC}/qrt_ck_fmha_q8192_provider.cpp" \
  "${SRC}/fmha_fwd_api.cpp" \
  "${SRC}/fmha_fwd_gfx1151_d256_bf16_f32out.cpp" \
  -o "${OUT_DIR}/libqrt_ck_fmha_q8192_provider.so"

"${common[@]}" \
  "${SRC}/qrt_ck_fmha_q8192_kv16384_bottom_right_provider.cpp" \
  "${SRC}/fmha_fwd_api.cpp" \
  "${SRC}/fmha_fwd_gfx1151_d256_bf16_f32out.cpp" \
  -o "${OUT_DIR}/libqrt_ck_fmha_q8192_kv16384_bottom_right_provider.so"

"${HIPCC}" -O3 -std=c++17 --offload-arch=gfx1151 -shared -fPIC -pthread \
  "${SRC}/torch_owned_striped_image_loader.hip.cpp" \
  -o "${OUT_DIR}/libtorch_owned_striped_image_loader.so"

"${CXX}" -O3 -std=c++17 -pthread \
  "${SRC}/striped_image_builder.cpp" \
  -o "${OUT_DIR}/striped_image_builder"

sha256sum \
  "${OUT_DIR}/libqrt_ck_fmha_q8192_provider.so" \
  "${OUT_DIR}/libqrt_ck_fmha_q8192_kv16384_bottom_right_provider.so" \
  "${OUT_DIR}/libtorch_owned_striped_image_loader.so" \
  "${OUT_DIR}/striped_image_builder"
