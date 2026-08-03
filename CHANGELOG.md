# Changelog

All notable changes are documented here. This project follows Semantic
Versioning.

## Unreleased

## 1.4.0 - 2026-08-03

- Added a native, no-model-load `doctor` command for host, KFD/render, HIP
  `gfx1151`, memory-pool, portable-bundle and optional model integrity checks.
- Added permission-checked bearer-token authentication, fail-closed remote
  binding, configurable socket timeouts, optional HTTP shutdown removal and
  graceful handling of disconnected non-stream clients. Request reads use an
  absolute deadline, and address conflicts now fail before model loading.
- Hardened the packaged systemd service with a required key file, disabled
  HTTP shutdown, a bounded socket timeout, native readiness/stopping
  notification and a restrictive umask.
- Replaced credential-specific hygiene checks with a reusable public-tree
  secret/private-host scanner enforced by CI and release verification.
- Fixed the PEP 621 license metadata for the declared setuptools floor and
  added an offline wheel build to the standard check. Installed wheels now
  expose only their dependency-free HTTP client instead of broken source-only
  runtime commands, with permission-checked bearer-key file support.
- Added the compact raw v1.3 qualification reports, an immutable
  provenance erratum, exact evidence-tree inventories and deterministic
  checksummed evidence packaging.
- Bound future bundle manifests to an exact source commit and every executable
  or provider hash in the qualification, rejected dirty release packaging by
  default and separated personal-upstream release/CI ownership from the
  Approaching AI showcase fork.
- Changed the direct-checkpoint default to one O_DIRECT reader after the
  qualified target showed same-NVMe contention with two or more readers;
  `--workers` remains available for storage-specific tuning.

## 1.3.0 - 2026-07-23

- Added live HTTP/1.1 SSE token streaming with incremental UTF-8 handling,
  optional usage chunks, client-disconnect cancellation and exact
  stream/non-stream token parity.
- Added OpenAI function tools, tool-choice modes, parallel call handling,
  assistant/tool message history and schema-aware Qwen XML call parsing.
- Added a byte-exact native Qwen tool template, protocol unit tests and a
  target-host streaming/tool lifecycle qualification.

## 1.2.0 - 2026-07-22

- Added a relocatable native `gfx1151` runtime bundle with a static launcher,
  bundle-local glibc loader, pinned ROCm userspace, native tokenizer, AOT
  closures and automatic FMHA-provider selection.
- Removed Python, PyTorch, vLLM, Triton, Transformers and host ROCm userspace
  from the runtime dependency set for the complete published batch-1 envelope.
- Qualified all 19 standard and maximum-window performance cells against the
  frozen v1.1 floor, plus nine-context full-vocabulary KLD/top-1 correctness,
  exact 128-token identity, command-to-ready startup, prefix reuse and
  resident HTTP.
- Added chunked long-context prefill with recurrent/KV state carry, admitted
  tail schedules, a q16384 packed-GQA hybrid and a split long-decode softmax.
- Added a one-entry prefix-extension path that restores the cached state and
  runs only the native suffix decode without rerunning cold prefill.
- Preserved the v1.1 compatibility runtime as provenance while promoting the
  portable native package as its complete published performance replacement.

## 1.1.0 - 2026-07-21

- Made direct loading from the standard Safetensors checkpoint the default;
  generated startup images and `AIMA_IMAGE_MANIFEST` are no longer required.
- Added a native O_DIRECT/pinned-memory scatter loader that writes the 693
  active BF16 tensors into their final Torch-owned device storages and verifies
  the complete GPU payload before readiness.
- Retained the two-lane striped-image path as an explicit optional mode for the
  lowest qualified cold-start latency.
- Added direct-checkpoint diagnostics, independent native rebuild support and
  target-host startup qualification evidence.
- Documented the required 128 GB BIOS UMA split, 96 GiB AMDGPU GTT kernel
  parameters, GPU device permissions, post-reboot checks and rollback steps in
  English and Chinese.

## 1.0.0 - 2026-07-21

- First public, production-qualified AMD395 release.
- Added resident batch-1 BF16 engine with an OpenAI-compatible HTTP subset.
- Added exact-prefix cache reuse for one entry up to 32,768 tokens.
- Added portable startup-image preparation and registration commands.
- Added release integrity checks, environment diagnostics and native rebuild
  sources.
- Closed output-one, raw-token usage, EOS stopping and strict/exact prefix-cache
  conformance.
