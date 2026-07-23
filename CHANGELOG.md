# Changelog

All notable changes are documented here. This project follows Semantic
Versioning.

## Unreleased

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
