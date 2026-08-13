#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

CURL_VERSION="8.21.0"
CURL_ARCHIVE_SHA256="aa1b66a70eace83dc624508745646c08ae561de512ab403adffb93ac87fc72e6"
CURL_URL="https://curl.se/download/curl-${CURL_VERSION}.tar.xz"
CARES_VERSION="1.34.8"
CARES_ARCHIVE_SHA256="c222b6d681096f9444d2c4863d2c1174019e27cacca0a4a5c114d36dd7d7bf78"
CARES_URL="https://github.com/c-ares/c-ares/releases/download/v${CARES_VERSION}/c-ares-${CARES_VERSION}.tar.gz"
OUTPUT="${1:?usage: build-minimal-curl.sh OUTPUT_DIR}"
CACHE_DIR="${AIMA_CURL_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/aima-engine}"
CURL_ARCHIVE="${AIMA_CURL_ARCHIVE:-${CACHE_DIR}/curl-${CURL_VERSION}.tar.xz}"
CARES_ARCHIVE="${AIMA_CARES_ARCHIVE:-${CACHE_DIR}/c-ares-${CARES_VERSION}.tar.gz}"
CA_BUNDLE="${AIMA_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
CA_LICENSE="${AIMA_CA_CERTIFICATES_LICENSE:-/usr/share/doc/ca-certificates/copyright}"
JOBS="${AIMA_CURL_JOBS:-$(getconf _NPROCESSORS_ONLN)}"

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to replace existing minimal curl output: ${OUTPUT}" >&2
  exit 1
fi
for command in cmake curl make pkg-config readelf sha256sum; do
  if ! command -v "${command}" >/dev/null; then
    echo "minimal curl build requires ${command}" >&2
    exit 1
  fi
done
if [[ ! -f "${CA_BUNDLE}" || ! -f "${CA_LICENSE}" ]]; then
  echo "CA certificate bundle or distribution license is unavailable" >&2
  exit 1
fi
mkdir -p "${CACHE_DIR}" "$(dirname "${OUTPUT}")"
download() {
  local url="$1"
  local archive="$2"
  if [[ ! -f "${archive}" ]]; then
    curl --fail --location --proto '=https' --tlsv1.2 \
      --output "${archive}.partial" "${url}"
    mv "${archive}.partial" "${archive}"
  fi
}
download "${CURL_URL}" "${CURL_ARCHIVE}"
download "${CARES_URL}" "${CARES_ARCHIVE}"
if [[ "$(sha256sum "${CURL_ARCHIVE}" | awk '{print $1}')" != \
      "${CURL_ARCHIVE_SHA256}" ]]; then
  echo "curl source SHA-256 mismatch: ${CURL_ARCHIVE}" >&2
  exit 1
fi
if [[ "$(sha256sum "${CARES_ARCHIVE}" | awk '{print $1}')" != \
      "${CARES_ARCHIVE_SHA256}" ]]; then
  echo "c-ares source SHA-256 mismatch: ${CARES_ARCHIVE}" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d "$(dirname "${OUTPUT}")/.curl-build.XXXXXX")"
cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT
tar -xJf "${CURL_ARCHIVE}" -C "${WORK_DIR}"
tar -xzf "${CARES_ARCHIVE}" -C "${WORK_DIR}"
CARES_SOURCE="${WORK_DIR}/c-ares-${CARES_VERSION}"
CURL_SOURCE="${WORK_DIR}/curl-${CURL_VERSION}"
CARES_PREFIX="${WORK_DIR}/cares-prefix"
CURL_PREFIX="${WORK_DIR}/curl-prefix"

cmake -S "${CARES_SOURCE}" -B "${WORK_DIR}/cares-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${CARES_PREFIX}" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCARES_SHARED=ON \
  -DCARES_STATIC=OFF \
  -DCARES_BUILD_TOOLS=OFF \
  -DCARES_BUILD_TESTS=OFF \
  -DCARES_BUILD_CONTAINER_TESTS=OFF \
  -DCARES_SYMBOL_HIDING=ON
cmake --build "${WORK_DIR}/cares-build" --parallel "${JOBS}"
cmake --install "${WORK_DIR}/cares-build"

(
  cd "${CURL_SOURCE}"
  PKG_CONFIG_PATH="${CARES_PREFIX}/lib/pkgconfig" \
  LD_LIBRARY_PATH="${CARES_PREFIX}/lib" \
    ./configure \
      --prefix="${CURL_PREFIX}" \
      --enable-shared \
      --disable-static \
      --disable-docs \
      --disable-manual \
      --disable-debug \
      --disable-dependency-tracking \
      --disable-libcurl-option \
      --disable-verbose \
      --disable-ftp \
      --disable-file \
      --disable-ipfs \
      --disable-ldap \
      --disable-ldaps \
      --disable-rtsp \
      --disable-dict \
      --disable-telnet \
      --disable-tftp \
      --disable-pop3 \
      --disable-imap \
      --disable-smb \
      --disable-smtp \
      --disable-gopher \
      --disable-mqtt \
      --disable-proxy \
      --disable-unix-sockets \
      --disable-cookies \
      --disable-http-auth \
      --disable-doh \
      --disable-mime \
      --disable-dateparse \
      --disable-netrc \
      --disable-progress-meter \
      --disable-alt-svc \
      --disable-hsts \
      --disable-websockets \
      --disable-aws \
      --disable-tls-srp \
      --disable-ntlm \
      --disable-form-api \
      --disable-dnsshuffle \
      --disable-get-easy-options \
      --disable-headers-api \
      --disable-httpsrr \
      --disable-ech \
      --without-zlib \
      --without-brotli \
      --without-zstd \
      --without-libpsl \
      --without-libidn2 \
      --without-nghttp2 \
      --without-ngtcp2 \
      --without-nghttp3 \
      --without-quiche \
      --without-libssh2 \
      --without-libssh \
      --without-librtmp \
      --without-ca-bundle \
      --without-ca-path \
      --without-ca-fallback \
      --with-openssl \
      --enable-ares="${CARES_PREFIX}"
  make -j"${JOBS}"
  make install
)

protocols="$(LD_LIBRARY_PATH="${CURL_PREFIX}/lib:${CARES_PREFIX}/lib" \
  "${CURL_PREFIX}/bin/curl-config" --protocols | xargs)"
features="$(LD_LIBRARY_PATH="${CURL_PREFIX}/lib:${CARES_PREFIX}/lib" \
  "${CURL_PREFIX}/bin/curl-config" --features | xargs)"
if [[ "${protocols}" != "HTTP HTTPS" ]] ||
   [[ " ${features} " != *" AsynchDNS "* ]] ||
   [[ " ${features} " != *" SSL "* ]]; then
  echo "minimal curl feature closure drifted: protocols=${protocols} features=${features}" >&2
  exit 1
fi

mkdir -p "${OUTPUT}/lib" "${OUTPUT}/include" "${OUTPUT}/licenses" \
  "${OUTPUT}/share/certs"
install -m 0755 "$(readlink -f "${CURL_PREFIX}/lib/libcurl.so.4")" \
  "${OUTPUT}/lib/libcurl.so.4"
install -m 0755 "$(readlink -f "${CARES_PREFIX}/lib/libcares.so.2")" \
  "${OUTPUT}/lib/libcares.so.2"
cp -a "${CURL_PREFIX}/include/." "${OUTPUT}/include/"
install -m 0644 "${CURL_SOURCE}/COPYING" \
  "${OUTPUT}/licenses/CURL-LICENSE.txt"
install -m 0644 "${CARES_SOURCE}/LICENSE.md" \
  "${OUTPUT}/licenses/CARES-LICENSE.md"
install -m 0644 "${CA_LICENSE}" \
  "${OUTPUT}/licenses/CA-CERTIFICATES-LICENSE.txt"
install -m 0644 "${CA_BUNDLE}" \
  "${OUTPUT}/share/certs/ca-certificates.crt"
{
  printf 'curl_version=%s\n' "${CURL_VERSION}"
  printf 'curl_source_url=%s\n' "${CURL_URL}"
  printf 'curl_source_sha256=%s\n' "${CURL_ARCHIVE_SHA256}"
  printf 'cares_version=%s\n' "${CARES_VERSION}"
  printf 'cares_source_url=%s\n' "${CARES_URL}"
  printf 'cares_source_sha256=%s\n' "${CARES_ARCHIVE_SHA256}"
  printf 'licenses=curl,MIT\n'
  printf 'protocols=HTTP,HTTPS\n'
  printf 'dns=async-c-ares\n'
  printf 'tls=OpenSSL\n'
  printf 'proxy=disabled\n'
  printf 'default_ca_store=disabled;runtime_bundle_required\n'
} >"${OUTPUT}/BUILD-CONTRACT.txt"

for library in "${OUTPUT}"/lib/*.so.*; do
  unresolved="$(readelf -d "${library}" | awk \
    '/Shared library:/ {gsub(/[][]/, "", $5); print $5}' | \
    grep -Ev '^(libcares\.so\.2|libssl\.so\.3|libcrypto\.so\.3|libm\.so\.6|libc\.so\.6)$' || true)"
  if [[ -n "${unresolved}" ]]; then
    echo "minimal curl library has unexpected dependency: ${unresolved}" >&2
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
