# Changelog

All notable changes are documented here. This project follows Semantic
Versioning.

## Unreleased

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
