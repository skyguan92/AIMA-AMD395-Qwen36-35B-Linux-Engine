#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -uo pipefail

: "${AIMA_BISECT_WORKTREE:?set the detached bisect worktree}"
: "${AIMA_MODEL_DIR:?set the frozen model directory}"
: "${AIMA_COMPLETE_FMHA_PROVIDER:?set the hash-bound FMHA provider}"
: "${FFMPEG_ROOT:?set the pinned minimal FFmpeg root}"
: "${CURL_ROOT:?set the pinned minimal curl root}"

ARTIFACT_ROOT="${AIMA_BISECT_ARTIFACT_ROOT:-/tmp/aima-native-text-bisect}"
ORACLE="${AIMA_BISECT_ORACLE:-/tmp/native-q1024-frozen-uniform1-ref-v4/oracle/0001-return-final_logits-output.bin}"
EXPECTED_ORACLE_SHA256="8b4b9cab6ba68abcdc3b36182ca7f2c9f832b990c4dbe2fc6447cd3cd0298156"
EXPECTED_PROVIDER_SHA256="98e6c47c017837ab796e3ca2e8256740d1e9cb6ec2f460af45ee586cd5fb7bd1"
JOURNAL="${ARTIFACT_ROOT}/results.jsonl"

mkdir -p "${ARTIFACT_ROOT}"
if [[ ! -f "${ORACLE}" ]] ||
   [[ "$(sha256sum "${ORACLE}" | awk '{print $1}')" != "${EXPECTED_ORACLE_SHA256}" ]]; then
  echo "frozen q1024 oracle is missing or changed" >&2
  exit 125
fi
if [[ ! -f "${AIMA_COMPLETE_FMHA_PROVIDER}" ]] ||
   [[ "$(sha256sum "${AIMA_COMPLETE_FMHA_PROVIDER}" | awk '{print $1}')" != "${EXPECTED_PROVIDER_SHA256}" ]]; then
  echo "frozen FMHA provider is missing or changed" >&2
  exit 125
fi

commit="$(git -C "${AIMA_BISECT_WORKTREE}" rev-parse HEAD)"
short="$(git -C "${AIMA_BISECT_WORKTREE}" rev-parse --short=12 HEAD)"
build_root="${ARTIFACT_ROOT}/build-${short}"
result_path="${ARTIFACT_ROOT}/result-${short}.json"
build_log="${ARTIFACT_ROOT}/build-${short}.log"
run_log="${ARTIFACT_ROOT}/run-${short}.log"
binary="${build_root}/aima-engine-native"

if ! env \
  OUT_DIR="${build_root}" \
  FFMPEG_ROOT="${FFMPEG_ROOT}" \
  CURL_ROOT="${CURL_ROOT}" \
  "${AIMA_BISECT_WORKTREE}/scripts/build-native-runtime.sh" \
  >"${build_log}" 2>&1; then
  python3 - "${commit}" "${JOURNAL}" <<'PY'
import json
from pathlib import Path
import sys

commit, journal = sys.argv[1:]
with Path(journal).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"commit": commit, "result": "skip-build"}) + "\n")
PY
  exit 125
fi

provider_parent="$(dirname "${AIMA_COMPLETE_FMHA_PROVIDER}")"
env \
  LD_LIBRARY_PATH="${provider_parent}:${FFMPEG_ROOT}/lib:${CURL_ROOT}/lib:${LD_LIBRARY_PATH:-}" \
  "${binary}" resident-session-probe \
  --model-dir "${AIMA_MODEL_DIR}" \
  --context-tokens 1024 \
  --uniform-input-token-id 1 \
  --max-new-tokens 1 \
  --requests 1 \
  --disable-prefix-cache \
  --fmha-provider "${AIMA_COMPLETE_FMHA_PROVIDER}" \
  --reference-logits "${ORACLE}" \
  --report "${build_root}/load.json" \
  >"${result_path}" 2>"${run_log}"
runtime_exit=$?

python3 - \
  "${commit}" "${result_path}" "${binary}" "${runtime_exit}" "${JOURNAL}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

commit, result_raw, binary_raw, runtime_exit, journal_raw = sys.argv[1:]
result_path = Path(result_raw)
binary = Path(binary_raw)
journal = Path(journal_raw)
record = {"commit": commit, "runtime_exit": int(runtime_exit)}
decision = 125
try:
    value = json.loads(result_path.read_text(encoding="utf-8"))
    comparison = value["reference_logits"]
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    good = bool(
        comparison["top1_match"] is True
        and float(comparison["kl_divergence"]) < 0.005
        and int(comparison["finite_elements"]) == 248320
    )
    record.update(
        {
            "result": "good" if good else "bad",
            "binary_sha256": digest.hexdigest(),
            "kld": comparison["kl_divergence"],
            "top1_match": comparison["top1_match"],
        }
    )
    decision = 0 if good else 1
except Exception as error:
    record.update(
        {"result": "skip-runtime", "error_type": type(error).__name__}
    )
with journal.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
raise SystemExit(decision)
PY
