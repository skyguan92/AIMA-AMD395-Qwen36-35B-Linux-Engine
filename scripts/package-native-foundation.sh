#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
SOURCE_DIRTY=false
if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=normal)" ]]; then
  SOURCE_DIRTY=true
fi
if [[ "${SOURCE_DIRTY}" == true && "${AIMA_ALLOW_DIRTY_PACKAGE:-0}" != 1 ]]; then
  echo "refusing to package a dirty source tree; commit or remove all non-ignored changes, or set AIMA_ALLOW_DIRTY_PACKAGE=1 for a non-release build" >&2
  exit 1
fi
ROCM_ROOT="$(readlink -f "${ROCM_ROOT:-/opt/rocm}")"
BINARY="${BINARY:-${ROOT}/build/native/aima-engine-native}"
LAUNCHER="${LAUNCHER:-${ROOT}/build/native/aima-engine-launcher}"
FMHA_AOTRITON_PROVIDER="${FMHA_AOTRITON_PROVIDER:-${ROOT}/build/native/libaima-fmha-aotriton.so}"
FMHA_CK_PROVIDER="${FMHA_CK_PROVIDER:-${ROOT}/build/native/libaima-fmha-ck.so}"
FMHA_HYBRID_PROVIDER="${FMHA_HYBRID_PROVIDER:-${ROOT}/build/native/libaima-fmha-q16384-hybrid.so}"
AOTRITON_ROOT="${AOTRITON_ROOT:?set AOTRITON_ROOT to the qualified distribution root containing lib/ and aotriton.images/}"
AOTRITON_SONAME="libaotriton_v2.so.0.11.1"
AOTRITON_LIBRARY="${AOTRITON_LIBRARY:-${AOTRITON_ROOT}/lib/${AOTRITON_SONAME}}"
AOTRITON_LIBRARY_SHA256="e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5"
AOTRITON_IMAGE_RELATIVE="amd-gfx11xx/flash/attn_fwd/FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2"
AOTRITON_IMAGE="${AOTRITON_ROOT}/lib/aotriton.images/${AOTRITON_IMAGE_RELATIVE}"
AOTRITON_IMAGE_SHA256="0f3a6a2f9dee6620443ee2145ee1f8257bde65a378589952840d99bf3d485c10"
RELEASE_VERSION="${AIMA_RELEASE_VERSION:-1.5.1}"
RELEASE_TAG="${AIMA_RELEASE_TAG:-v${RELEASE_VERSION}}"
QUALIFICATION_RECORD="${QUALIFICATION_RECORD:-${ROOT}/output/native-portable-product-v${RELEASE_VERSION}.json}"
QUALIFICATION_BASENAME="$(basename "${QUALIFICATION_RECORD}")"
PRODUCT_CONTRACT="${PRODUCT_CONTRACT:-${ROOT}/native/product-contract-v${RELEASE_VERSION}.json}"
release_metadata=(
  "${ROOT}/LICENSE"
  "${ROOT}/SECURITY.md"
  "${ROOT}/CHANGELOG.md"
  "${ROOT}/README.md"
  "${ROOT}/README.zh-CN.md"
  "${ROOT}/NOTICE"
  "${ROOT}/THIRD_PARTY_NOTICES.md"
  "${PRODUCT_CONTRACT}"
  "${ROOT}/docs/INSTALL.md"
  "${ROOT}/docs/API.md"
  "${ROOT}/docs/ARCHITECTURE.md"
  "${ROOT}/docs/MEMORY.md"
  "${ROOT}/docs/MEMORY.zh-CN.md"
  "${ROOT}/docs/PERFORMANCE.md"
  "${ROOT}/docs/RELEASE.md"
  "${ROOT}/packaging/systemd/aima-engine.service"
  "${ROOT}/packaging/systemd/aima-engine.env.example"
  "${ROOT}/benchmarks/results/v1.0.0.json"
  "${ROOT}/benchmarks/results/v1.1.0.json"
  "${ROOT}/benchmarks/results/native-foundation-v0.1.0.json"
  "${QUALIFICATION_RECORD}"
)
release_results=("${ROOT}"/benchmarks/results/*.json)
release_contracts=("${ROOT}"/native/product-contract-v*.json)
release_assets=(
  "${ROOT}/assets/demos/amd395-three-engine-comparison.gif"
  "${ROOT}/assets/demos/amd395-three-engine-comparison.mp4"
)
release_evidence_dirs=(
  "${ROOT}/benchmarks/runs/native-correctness-20260723-v130"
  "${ROOT}/benchmarks/runs/native-full-matrix-20260723-v130"
  "${ROOT}/benchmarks/runs/native-openai-features-20260723-v130"
  "${ROOT}/benchmarks/runs/native-portable-bundle-20260723-v130"
  "${ROOT}/benchmarks/runs/native-product-surfaces-20260723-v130"
  "${ROOT}/benchmarks/runs/native-correctness-20260804-v141-release"
  "${ROOT}/benchmarks/runs/native-full-matrix-20260804-v141-release"
  "${ROOT}/benchmarks/runs/native-openai-features-20260804-v141-release"
  "${ROOT}/benchmarks/runs/native-portable-bundle-20260804-v141-release-final"
  "${ROOT}/benchmarks/runs/native-product-surfaces-20260804-v141-release"
  "${ROOT}/benchmarks/runs/native-correctness-20260805-v150-release"
  "${ROOT}/benchmarks/runs/native-full-matrix-20260805-v150-release"
  "${ROOT}/benchmarks/runs/native-mmlu256-eval-20260805-v150-release"
  "${ROOT}/benchmarks/runs/native-openai-features-20260805-v150-release"
  "${ROOT}/benchmarks/runs/native-portable-baiying-compat-20260805-v150-release"
  "${ROOT}/benchmarks/runs/native-portable-bundle-20260805-v150-release"
  "${ROOT}/benchmarks/runs/native-product-surfaces-20260805-v150-release"
)
release_metadata+=(
  "${release_results[@]}"
  "${release_contracts[@]}"
  "${release_assets[@]}"
)
for evidence_dir in "${release_evidence_dirs[@]}"; do
  while IFS= read -r -d '' evidence_file; do
    release_metadata+=("${evidence_file}")
  done < <(find "${evidence_dir}" -type f -print0 | sort -z)
done
if [[ -z "${AOTRITON_LICENSE:-}" || -z "${AOTRITON_NOTICE:-}" ]]; then
  shopt -s nullglob
  aotriton_license_candidates=(
    "${AOTRITON_ROOT}"/LICENSE*
    "${AOTRITON_ROOT}"/licenses/LICENSE*
    "${AOTRITON_ROOT}"/../torch-*.dist-info/licenses/LICENSE*
  )
  aotriton_notice_candidates=(
    "${AOTRITON_ROOT}"/NOTICE*
    "${AOTRITON_ROOT}"/licenses/NOTICE*
    "${AOTRITON_ROOT}"/../torch-*.dist-info/licenses/NOTICE*
  )
  shopt -u nullglob
  AOTRITON_LICENSE="${AOTRITON_LICENSE:-${aotriton_license_candidates[0]:-}}"
  AOTRITON_NOTICE="${AOTRITON_NOTICE:-${aotriton_notice_candidates[0]:-}}"
fi

if [[ ! -x "${BINARY}" ]]; then
  echo "native executable is missing; run make build-native-runtime" >&2
  exit 1
fi
BINARY_SOURCE_COMMIT="$("${BINARY}" --build-info | python3 -c \
  'import json, sys; print(json.load(sys.stdin)["source_commit"])')"
if [[ ! "${BINARY_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "native executable does not carry a clean source commit" >&2
  exit 1
fi
if [[ ! -x "${LAUNCHER}" ]]; then
  echo "native portable launcher is missing; run make build-native-runtime" >&2
  exit 1
fi
for artifact in "${FMHA_AOTRITON_PROVIDER}" "${FMHA_CK_PROVIDER}" \
                "${FMHA_HYBRID_PROVIDER}" \
                "${AOTRITON_LIBRARY}" "${AOTRITON_IMAGE}" \
                "${AOTRITON_LICENSE}" "${AOTRITON_NOTICE}" \
                "${release_metadata[@]}"; do
  if [[ ! -f "${artifact}" ]]; then
    echo "qualified native bundle artifact is missing: ${artifact}" >&2
    exit 1
  fi
done
if [[ "$(sha256sum "${AOTRITON_LIBRARY}" | awk '{print $1}')" != \
      "${AOTRITON_LIBRARY_SHA256}" ]]; then
  echo "qualified AOTriton library SHA-256 mismatch" >&2
  exit 1
fi
if [[ "$(sha256sum "${AOTRITON_IMAGE}" | awk '{print $1}')" != \
      "${AOTRITON_IMAGE_SHA256}" ]]; then
  echo "qualified AOTriton gfx1151 image SHA-256 mismatch" >&2
  exit 1
fi
python3 "${ROOT}/scripts/verify-native-package-inputs.py" \
  --qualification "${QUALIFICATION_RECORD}" \
  --release "${RELEASE_VERSION}" \
  --release-tag "${RELEASE_TAG}" \
  --source-commit "${SOURCE_COMMIT}" \
  --native-source-commit "${BINARY_SOURCE_COMMIT}" \
  --component "native_engine=${BINARY}" \
  --component "static_launcher=${LAUNCHER}" \
  --component "aotriton_fmha_provider=${FMHA_AOTRITON_PROVIDER}" \
  --component "ck_fmha_provider=${FMHA_CK_PROVIDER}" \
  --component "q16384_hybrid_fmha_provider=${FMHA_HYBRID_PROVIDER}" \
  --component "aotriton_runtime=${AOTRITON_LIBRARY}" \
  --component "aotriton_gfx1151_image=${AOTRITON_IMAGE}"
BUNDLE_ID="$(sha256sum "${BINARY}" "${LAUNCHER}" \
  "${FMHA_AOTRITON_PROVIDER}" "${FMHA_CK_PROVIDER}" \
  "${FMHA_HYBRID_PROVIDER}" \
  "${AOTRITON_LIBRARY}" "${AOTRITON_IMAGE}" \
  "${ROOT}/scripts/package-native-foundation.sh" \
  "${ROOT}/scripts/generate-native-bundle-manifest.py" \
  "${ROOT}/scripts/native_bundle_closure.py" \
  "${ROOT}/scripts/verify-native-package-inputs.py" \
  "${release_metadata[@]}" | \
  awk '{print $1}' | sha256sum | awk '{print $1}')"
OUTPUT="${OUTPUT:-${ROOT}/dist/aima-engine-native-portable-${BUNDLE_ID:0:12}}"
ARCHIVE="${ARCHIVE:-${OUTPUT}.tar.zst}"
ARCHIVE_CHECKSUM="${ARCHIVE}.sha256"
if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to replace existing native bundle: ${OUTPUT}" >&2
  exit 1
fi
if [[ -e "${ARCHIVE}" ]]; then
  echo "refusing to replace existing native bundle archive: ${ARCHIVE}" >&2
  exit 1
fi
if [[ -e "${ARCHIVE_CHECKSUM}" ]]; then
  echo "refusing to replace existing archive checksum: ${ARCHIVE_CHECKSUM}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
STAGING="$(mktemp -d "${OUTPUT}.tmp.XXXXXX")"
cleanup() {
  rm -rf "${STAGING}"
}
trap cleanup EXIT

install -Dm755 "${LAUNCHER}" "${STAGING}/bin/aima-engine"
install -Dm755 "${BINARY}" "${STAGING}/libexec/aima-engine.real"
install -Dm755 "${FMHA_AOTRITON_PROVIDER}" \
  "${STAGING}/lib/libaima-fmha-aotriton.so"
install -Dm755 "${FMHA_CK_PROVIDER}" \
  "${STAGING}/lib/libaima-fmha-ck.so"
install -Dm755 "${FMHA_HYBRID_PROVIDER}" \
  "${STAGING}/lib/libaima-fmha-q16384-hybrid.so"
install -Dm644 "${AOTRITON_LIBRARY}" \
  "${STAGING}/lib/${AOTRITON_SONAME}"
install -Dm644 "${AOTRITON_IMAGE}" \
  "${STAGING}/lib/aotriton.images/${AOTRITON_IMAGE_RELATIVE}"

libraries=(
  libamdhip64.so.7
  libhipblaslt.so.1
  libhsa-runtime64.so.1
  librocprofiler-register.so.0
  libroctx64.so.4
  librocroller.so.1
  libamd_comgr.so.3
  libhsa-amd-aqlprofile64.so.1
)
for soname in "${libraries[@]}"; do
  source_path="${ROCM_ROOT}/lib/${soname}"
  if [[ ! -e "${source_path}" ]]; then
    echo "qualified ROCm library is missing: ${source_path}" >&2
    exit 1
  fi
  install -Dm644 "$(readlink -f "${source_path}")" "${STAGING}/lib/${soname}"
  readelf -d "${STAGING}/lib/${soname}" | grep -Fq "Library soname: [${soname}]"
done
ln -s libhsa-amd-aqlprofile64.so.1 "${STAGING}/lib/libhsa-amd-aqlprofile64.so"

system_libraries=(
  ld-linux-x86-64.so.2
  libc.so.6
  libm.so.6
  libstdc++.so.6
  libgcc_s.so.1
  libelf.so.1
  libdrm.so.2
  libdrm_amdgpu.so.1
  libnuma.so.1
  libz.so.1
  libzstd.so.1
  liblzma.so.5
  libpng16.so.16
  libjpeg.so.8
  libwebp.so.7
  libsharpyuv.so.0
)
for soname in "${system_libraries[@]}"; do
  source_path="$(ldconfig -p | awk -v soname="${soname}" \
    '$1 == soname && /x86-64/ {path=$NF} END {print path}')"
  if [[ -z "${source_path}" || ! -e "${source_path}" ]]; then
    echo "qualified system userspace library is missing: ${soname}" >&2
    exit 1
  fi
  install -Dm755 "$(readlink -f "${source_path}")" "${STAGING}/lib/${soname}"
done

hipblaslt_source="${ROCM_ROOT}/lib/hipblaslt/library"
hipblaslt_target="${STAGING}/lib/hipblaslt/library"
mkdir -p "${hipblaslt_target}"
shopt -s nullglob
hipblaslt_assets=(
  "${hipblaslt_source}"/*gfx1151*
  "${hipblaslt_source}/TensileLiteLibrary_lazy_Mapping.dat"
)
shopt -u nullglob
if [[ ${#hipblaslt_assets[@]} -lt 2 ]]; then
  echo "qualified gfx1151 hipBLASLt assets are missing" >&2
  exit 1
fi
for source_path in "${hipblaslt_assets[@]}"; do
  install -Dm644 "${source_path}" "${hipblaslt_target}/$(basename "${source_path}")"
done
install -Dm644 "${ROCM_ROOT}/share/hip/version" "${STAGING}/share/hip/version"
mkdir -p "${STAGING}/amdgcn/bitcode"
cp -a "${ROCM_ROOT}/amdgcn/bitcode/." "${STAGING}/amdgcn/bitcode/"

licenses=(
  "share/doc/hip/LICENSE.md:HIP-LICENSE.md"
  "share/doc/hsa-rocr/LICENSE.md:HSA-ROCR-LICENSE.md"
  "share/doc/rocprofiler-register/LICENSE.md:ROCPROFILER-REGISTER-LICENSE.md"
  "share/doc/amd_comgr/LICENSE.txt:AMD-COMGR-LICENSE.txt"
  "share/doc/rocm-device-libs/LICENSE.TXT:ROCM-DEVICE-LIBS-LICENSE.txt"
  "share/doc/hipblaslt/LICENSE.md:HIPBLASLT-LICENSE.md"
  "share/doc/rocprofiler-sdk/LICENSE.md:ROCPROFILER-SDK-LICENSE.md"
  "/usr/share/doc/libc6/copyright:GLIBC-LICENSE.txt"
  "/usr/share/doc/libstdc++6/copyright:LIBSTDCXX-LICENSE.txt"
  "/usr/share/doc/libgcc-s1/copyright:LIBGCC-LICENSE.txt"
  "/usr/share/doc/libelf1t64/copyright:LIBELF-LICENSE.txt"
  "/usr/share/doc/libdrm2/copyright:LIBDRM-LICENSE.txt"
  "/usr/share/doc/libdrm-amdgpu1/copyright:LIBDRM-AMDGPU-LICENSE.txt"
  "/usr/share/doc/libnuma1/copyright:LIBNUMA-LICENSE.txt"
  "/usr/share/doc/zlib1g/copyright:ZLIB-LICENSE.txt"
  "/usr/share/doc/libzstd1/copyright:LIBZSTD-LICENSE.txt"
  "/usr/share/doc/liblzma5/copyright:LIBLZMA-LICENSE.txt"
  "/usr/share/doc/libpng16-16t64/copyright:LIBPNG-LICENSE.txt"
  "/usr/share/doc/libjpeg-turbo8/copyright:LIBJPEG-TURBO-LICENSE.txt"
  "/usr/share/doc/libwebp7/copyright:LIBWEBP-LICENSE.txt"
  "/usr/share/doc/libsharpyuv0/copyright:LIBSHARPYUV-LICENSE.txt"
)
for mapping in "${licenses[@]}"; do
  source_name="${mapping%%:*}"
  target_name="${mapping#*:}"
  if [[ "${source_name}" = /* ]]; then
    license_source="${source_name}"
  else
    license_source="${ROCM_ROOT}/${source_name}"
  fi
  install -Dm644 "${license_source}" "${STAGING}/licenses/${target_name}"
done
install -Dm644 "${ROOT}/LICENSE" "${STAGING}/licenses/AIMA-APACHE-2.0.txt"
install -Dm644 "${ROOT}/third_party/licenses/AMD_COMPOSABLE_KERNEL_MIT.txt" \
  "${STAGING}/licenses/AMD-COMPOSABLE-KERNEL-MIT.txt"
install -Dm644 "${AOTRITON_LICENSE}" \
  "${STAGING}/licenses/AOTRITON-DISTRIBUTION-LICENSE.txt"
install -Dm644 "${AOTRITON_NOTICE}" \
  "${STAGING}/licenses/AOTRITON-DISTRIBUTION-NOTICE.txt"
install -Dm644 "/usr/share/doc/libicu-dev/copyright" "${STAGING}/licenses/ICU-LICENSE.txt"
install -Dm644 "${ROOT}/NOTICE" "${STAGING}/NOTICE"
install -Dm644 "${ROOT}/THIRD_PARTY_NOTICES.md" "${STAGING}/THIRD_PARTY_NOTICES.md"
install -Dm644 "${ROOT}/LICENSE" "${STAGING}/LICENSE"
install -Dm644 "${ROOT}/SECURITY.md" "${STAGING}/SECURITY.md"
install -Dm644 "${ROOT}/CHANGELOG.md" "${STAGING}/CHANGELOG.md"
install -Dm644 "${ROOT}/README.md" "${STAGING}/README.md"
install -Dm644 "${ROOT}/README.zh-CN.md" "${STAGING}/README.zh-CN.md"
for asset in "${release_assets[@]}"; do
  install -Dm644 "${asset}" \
    "${STAGING}/assets/demos/$(basename "${asset}")"
done
for document in INSTALL API ARCHITECTURE MEMORY MEMORY.zh-CN PERFORMANCE RELEASE; do
  install -Dm644 "${ROOT}/docs/${document}.md" \
    "${STAGING}/docs/${document}.md"
done
install -Dm644 "${PRODUCT_CONTRACT}" \
  "${STAGING}/share/aima/product-contract.json"
install -Dm644 "${PRODUCT_CONTRACT}" \
  "${STAGING}/native/product-contract.json"
for contract in "${release_contracts[@]}"; do
  install -Dm644 "${contract}" \
    "${STAGING}/native/$(basename "${contract}")"
done
install -Dm644 "${QUALIFICATION_RECORD}" \
  "${STAGING}/share/aima/qualification.json"
install -Dm644 "${QUALIFICATION_RECORD}" \
  "${STAGING}/benchmarks/results/${QUALIFICATION_BASENAME}"
for result in "${release_results[@]}"; do
  install -Dm644 "${result}" \
    "${STAGING}/benchmarks/results/$(basename "${result}")"
done
mkdir -p "${STAGING}/benchmarks/runs"
for evidence_dir in "${release_evidence_dirs[@]}"; do
  cp -a "${evidence_dir}" "${STAGING}/benchmarks/runs/"
done
install -Dm644 "${ROOT}/packaging/systemd/aima-engine.service" \
  "${STAGING}/share/systemd/aima-engine.service"
install -Dm644 "${ROOT}/packaging/systemd/aima-engine.env.example" \
  "${STAGING}/share/systemd/aima-engine.env.example"

python3 "${ROOT}/scripts/native_bundle_closure.py" "${STAGING}" >/dev/null
manifest_args=(
  "${STAGING}"
  --release "${RELEASE_VERSION}"
  --release-tag "${RELEASE_TAG}"
  --source-commit "${SOURCE_COMMIT}"
  --native-source-commit "${BINARY_SOURCE_COMMIT}"
)
if [[ "${SOURCE_DIRTY}" == true ]]; then
  manifest_args+=(--source-dirty)
fi
python3 "${ROOT}/scripts/generate-native-bundle-manifest.py" "${manifest_args[@]}"
chmod 0755 "${STAGING}"
mv "${STAGING}" "${OUTPUT}"
trap - EXIT
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1784678400}"
tar --sort=name --mtime="@${SOURCE_DATE_EPOCH}" --owner=0 --group=0 \
  --numeric-owner --format=posix \
  --pax-option=delete=atime,delete=ctime --zstd -cf "${ARCHIVE}" \
  -C "$(dirname "${OUTPUT}")" "$(basename "${OUTPUT}")"
(
  cd "$(dirname "${ARCHIVE}")"
  sha256sum "$(basename "${ARCHIVE}")" > \
    "$(basename "${ARCHIVE_CHECKSUM}")"
)

"${OUTPUT}/bin/aima-engine" --version
sha256sum "${OUTPUT}/bin/aima-engine"
du -sh "${OUTPUT}"
sha256sum "${ARCHIVE}"
cat "${ARCHIVE_CHECKSUM}"
du -sh "${ARCHIVE}"
echo "${OUTPUT}"
echo "${ARCHIVE}"
echo "${ARCHIVE_CHECKSUM}"
