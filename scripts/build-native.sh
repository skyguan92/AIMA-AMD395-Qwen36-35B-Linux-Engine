#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CK_COMMIT="6667a9021713f794a2c9aee4696c19f6cf376235"
CK_DIR="${CK_DIR:?set CK_DIR to an AMD Composable Kernel checkout}"
AOTRITON_ROOT="${AOTRITON_ROOT:?set AOTRITON_ROOT to the qualified distribution root containing include/ and lib/}"
AOTRITON_SONAME="libaotriton_v2.so.0.11.1"
AOTRITON_LIBRARY="${AOTRITON_LIBRARY:-${AOTRITON_ROOT}/lib/${AOTRITON_SONAME}}"
AOTRITON_LIBRARY_SHA256="e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
CXX="${CXX:-g++}"
OUT_DIR="${OUT_DIR:-${ROOT}/build/native}"
SRC="${ROOT}/benchmarks/shape-lab/native/src"

actual_commit="$(git -C "${CK_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${CK_COMMIT}" ]]; then
  echo "Composable Kernel commit mismatch: expected ${CK_COMMIT}, got ${actual_commit}" >&2
  exit 2
fi
if [[ ! -f "${AOTRITON_ROOT}/include/aotriton/flash.h" ||
      ! -f "${AOTRITON_LIBRARY}" ]]; then
  echo "qualified AOTriton headers or library are missing under ${AOTRITON_ROOT}" >&2
  exit 2
fi
actual_aotriton_sha256="$(sha256sum "${AOTRITON_LIBRARY}" | awk '{print $1}')"
if [[ "${actual_aotriton_sha256}" != "${AOTRITON_LIBRARY_SHA256}" ]]; then
  echo "AOTriton library mismatch: expected ${AOTRITON_LIBRARY_SHA256}, got ${actual_aotriton_sha256}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

PACKED_GQA_IMAGE="${ROOT}/native/aot/gfx1151/q16384-hybrid-packed-gqa/kernels/215d96facb2fc98b-_packed_gqa_mha_fwd.hsaco"
FIRST_WINDOW_PACKED_GQA_IMAGE="${ROOT}/native/aot/gfx1151/q16384-first-window-packed-gqa/kernels/4c6b00554f1a7b65-_packed_gqa_mha_fwd.hsaco"
PACKED_GQA_OBJECT="${OUT_DIR}/q16384-packed-gqa-image.o"
if [[ ! -f "${PACKED_GQA_IMAGE}" ||
      ! -f "${FIRST_WINDOW_PACKED_GQA_IMAGE}" ]]; then
  echo "qualified q16384 packed-GQA image set is incomplete" >&2
  exit 2
fi
"${CC:-gcc}" -c \
  -DAIMA_Q16384_PACKED_GQA_HSACO_PATH=\"${PACKED_GQA_IMAGE}\" \
  -DAIMA_Q8192_PACKED_GQA_HSACO_PATH=\"${FIRST_WINDOW_PACKED_GQA_IMAGE}\" \
  "${ROOT}/native/src/native_q16384_packed_gqa_image.S" \
  -o "${PACKED_GQA_OBJECT}"

common=(
  "${HIPCC}" -std=c++17 -O3 -fPIC --offload-arch=gfx1151
  -fno-rtlib-add-rpath
  -DCK_TILE_FMHA_FWD_FAST_EXP2=0
  -I "${CK_DIR}/include"
  -I "${CK_DIR}/example/ck_tile/01_fmha"
  -shared -Wl,-z,origin -Wl,--enable-new-dtags -Wl,-rpath,'$ORIGIN'
)

"${common[@]}" \
  "${SRC}/qrt_ck_fmha_q8192_provider.cpp" \
  "${SRC}/fmha_fwd_api.cpp" \
  "${SRC}/fmha_fwd_gfx1151_d256_bf16_f32out.cpp" \
  -Wl,-soname,libaima-fmha-ck.so \
  -o "${OUT_DIR}/libaima-fmha-ck.so"
ln -sfn libaima-fmha-ck.so \
  "${OUT_DIR}/libqrt_ck_fmha_q8192_provider.so"

"${HIPCC}" -std=c++17 -O3 -fPIC --offload-arch=gfx1151 \
  -fno-rtlib-add-rpath -shared -pthread \
  -I "${AOTRITON_ROOT}/include" \
  "${ROOT}/native/src/native_aotriton_fmha_provider.hip.cpp" \
  -L "$(dirname "${AOTRITON_LIBRARY}")" -l:"${AOTRITON_SONAME}" \
  -Wl,-z,origin -Wl,--enable-new-dtags -Wl,-rpath,'$ORIGIN' \
  -Wl,-soname,libaima-fmha-aotriton.so \
  -o "${OUT_DIR}/libaima-fmha-aotriton.so"
ln -sfn libaima-fmha-aotriton.so \
  "${OUT_DIR}/libqrt_aotriton_fmha_provider.so"

"${common[@]}" \
  "${SRC}/qrt_ck_fmha_q8192_kv16384_bottom_right_provider.cpp" \
  "${SRC}/fmha_fwd_api.cpp" \
  "${SRC}/fmha_fwd_gfx1151_d256_bf16_f32out.cpp" \
  -o "${OUT_DIR}/libqrt_ck_fmha_q8192_kv16384_bottom_right_provider.so"

"${common[@]}" \
  -DAIMA_Q16384_PACKED_HEADS=16 \
  -DAIMA_Q16384_PACKED_FIRST_WINDOW=1 \
  "${ROOT}/native/src/native_q16384_hybrid_fmha_provider.hip.cpp" \
  "${SRC}/fmha_fwd_api.cpp" \
  "${SRC}/fmha_fwd_gfx1151_d256_bf16_f32out.cpp" \
  -x none "${PACKED_GQA_OBJECT}" \
  -Wl,-soname,libaima-fmha-q16384-hybrid.so \
  -o "${OUT_DIR}/libaima-fmha-q16384-hybrid.so"

"${HIPCC}" -O3 -std=c++17 --offload-arch=gfx1151 -shared -fPIC -pthread \
  "${SRC}/torch_owned_striped_image_loader.hip.cpp" \
  -o "${OUT_DIR}/libtorch_owned_striped_image_loader.so"

"${HIPCC}" -O3 -std=c++17 --offload-arch=gfx1151 -shared -fPIC -pthread \
  "${SRC}/torch_owned_safetensors_loader.hip.cpp" \
  -o "${OUT_DIR}/libtorch_owned_safetensors_loader.so"

"${CXX}" -O3 -std=c++17 -pthread \
  "${SRC}/striped_image_builder.cpp" \
  -o "${OUT_DIR}/striped_image_builder"

sha256sum \
  "${OUT_DIR}/libaima-fmha-ck.so" \
  "${OUT_DIR}/libaima-fmha-aotriton.so" \
  "${OUT_DIR}/libaima-fmha-q16384-hybrid.so" \
  "${OUT_DIR}/libqrt_ck_fmha_q8192_kv16384_bottom_right_provider.so" \
  "${OUT_DIR}/libtorch_owned_striped_image_loader.so" \
  "${OUT_DIR}/libtorch_owned_safetensors_loader.so" \
  "${OUT_DIR}/striped_image_builder"

for provider in \
  "${OUT_DIR}/libaima-fmha-ck.so" \
  "${OUT_DIR}/libaima-fmha-aotriton.so" \
  "${OUT_DIR}/libaima-fmha-q16384-hybrid.so"; do
  if readelf -d "${provider}" | grep -E 'Library (runpath|rpath): \[[^]]*/(opt|home|usr)/' >/dev/null; then
    echo "native FMHA provider contains an absolute RUNPATH: ${provider}" >&2
    exit 1
  fi
done
