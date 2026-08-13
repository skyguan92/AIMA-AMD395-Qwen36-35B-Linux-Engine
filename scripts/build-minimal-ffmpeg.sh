#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

FFMPEG_VERSION="6.1.1"
FFMPEG_ARCHIVE_SHA256="8684f4b00f94b85461884c3719382f1261f0d9eb3d59640a1f4ac0873616f968"
FFMPEG_URL="https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
OUTPUT="${1:?usage: build-minimal-ffmpeg.sh OUTPUT_DIR}"
CACHE_DIR="${AIMA_FFMPEG_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/aima-engine}"
ARCHIVE="${AIMA_FFMPEG_ARCHIVE:-${CACHE_DIR}/ffmpeg-${FFMPEG_VERSION}.tar.xz}"
JOBS="${AIMA_FFMPEG_JOBS:-$(getconf _NPROCESSORS_ONLN)}"

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to replace existing minimal FFmpeg output: ${OUTPUT}" >&2
  exit 1
fi
mkdir -p "${CACHE_DIR}" "$(dirname "${OUTPUT}")"
if [[ ! -f "${ARCHIVE}" ]]; then
  curl --fail --location --proto '=https' --tlsv1.2 \
    --output "${ARCHIVE}.partial" "${FFMPEG_URL}"
  mv "${ARCHIVE}.partial" "${ARCHIVE}"
fi
if [[ "$(sha256sum "${ARCHIVE}" | awk '{print $1}')" != \
      "${FFMPEG_ARCHIVE_SHA256}" ]]; then
  echo "minimal FFmpeg source SHA-256 mismatch: ${ARCHIVE}" >&2
  exit 1
fi
if ! command -v nasm >/dev/null; then
  echo "nasm is required to preserve OpenCV-compatible FFmpeg decode numerics" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d "$(dirname "${OUTPUT}")/.ffmpeg-build.XXXXXX")"
cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT
tar -xJf "${ARCHIVE}" -C "${WORK_DIR}"
SOURCE_DIR="${WORK_DIR}/ffmpeg-${FFMPEG_VERSION}"
PREFIX="${WORK_DIR}/prefix"
(
  cd "${SOURCE_DIR}"
  ./configure \
    --prefix="${PREFIX}" \
    --disable-everything \
    --disable-programs \
    --disable-doc \
    --disable-debug \
    --disable-network \
    --disable-autodetect \
    --disable-avdevice \
    --disable-avfilter \
    --disable-swresample \
    --enable-shared \
    --disable-static \
    --enable-pic \
    --enable-avcodec \
    --enable-avformat \
    --enable-avutil \
    --enable-swscale \
    --enable-demuxer=mov,avi \
    --enable-decoder=mpeg4,mjpeg \
    --enable-parser=mpeg4video,mjpeg \
    --enable-bsf=mpeg4_unpack_bframes
  make -j"${JOBS}"
  make install
)

if grep -Eq '^CONFIG_(GPL|GPLV3|NONFREE|NETWORK)=yes$' \
    "${SOURCE_DIR}/ffbuild/config.mak"; then
  echo "minimal FFmpeg unexpectedly enabled a forbidden feature" >&2
  exit 1
fi

mkdir -p "${OUTPUT}/lib" "${OUTPUT}/include" "${OUTPUT}/licenses"
for soname in libavformat.so.60 libavcodec.so.60 libavutil.so.58 \
              libswscale.so.7; do
  install -m 0755 "$(readlink -f "${PREFIX}/lib/${soname}")" \
    "${OUTPUT}/lib/${soname}"
done
cp -a "${PREFIX}/include/." "${OUTPUT}/include/"
install -m 0644 "${SOURCE_DIR}/COPYING.LGPLv2.1" \
  "${OUTPUT}/licenses/FFMPEG-LGPL-2.1-OR-LATER.txt"
install -m 0644 "${SOURCE_DIR}/LICENSE.md" \
  "${OUTPUT}/licenses/FFMPEG-LICENSE.md"
{
  printf 'version=%s\n' "${FFMPEG_VERSION}"
  printf 'source_url=%s\n' "${FFMPEG_URL}"
  printf 'source_sha256=%s\n' "${FFMPEG_ARCHIVE_SHA256}"
  printf 'license=LGPL-2.1-or-later\n'
  printf 'network=disabled\n'
  printf 'demuxers=avi,mov\n'
  printf 'decoders=mjpeg,mpeg4\n'
  printf 'libraries=avcodec,avformat,avutil,swscale\n'
} >"${OUTPUT}/BUILD-CONTRACT.txt"

for library in "${OUTPUT}"/lib/*.so.*; do
  unresolved="$(readelf -d "${library}" | awk \
    '/Shared library:/ {gsub(/[][]/, "", $5); print $5}' | \
    grep -Ev '^(lib(avcodec|avutil)\.so\.(60|58)|libm\.so\.6|libc\.so\.6)$' || true)"
  if [[ -n "${unresolved}" ]]; then
    echo "minimal FFmpeg library has unexpected dependency: ${unresolved}" >&2
    exit 1
  fi
done
(
  cd "${OUTPUT}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 sha256sum >SHA256SUMS
)
chmod -R a-w "${OUTPUT}"
trap - EXIT
rm -rf "${WORK_DIR}"
printf '%s\n' "${OUTPUT}"
