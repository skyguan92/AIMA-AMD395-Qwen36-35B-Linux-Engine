#!/usr/bin/env bash
# Generate an ephemeral loopback-only test CA/server identity.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

OUTPUT_DIR="${1:?usage: generate-vl-test-tls-material.sh OUTPUT_DIR}"
CERTIFICATE="${OUTPUT_DIR}/loopback-test-ca.pem"
PRIVATE_KEY="${OUTPUT_DIR}/loopback-test-server.key"

if [[ -e "${CERTIFICATE}" || -e "${PRIVATE_KEY}" ]]; then
  echo "test TLS material already exists in ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"
umask 077
openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 2 \
  -subj '/CN=127.0.0.1' \
  -addext 'basicConstraints=critical,CA:TRUE' \
  -addext 'keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign' \
  -addext 'extendedKeyUsage=serverAuth' \
  -addext 'subjectAltName=IP:127.0.0.1' \
  -keyout "${PRIVATE_KEY}" \
  -out "${CERTIFICATE}" >/dev/null 2>&1
chmod 0600 "${PRIVATE_KEY}"
chmod 0644 "${CERTIFICATE}"
openssl x509 -in "${CERTIFICATE}" -noout -checkend 3600 >/dev/null
printf 'certificate=%s\nprivate_key=%s\n' "${CERTIFICATE}" "${PRIVATE_KEY}"
