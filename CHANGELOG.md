# Changelog

All notable changes are documented here. This project follows Semantic
Versioning.

## Unreleased

## 1.5.1-native-vl.5 - 2026-09-01

- Added opt-in Qwen reasoning through the validated top-level `thinking`
  request object, with non-stream and SSE `reasoning_content` separation,
  answer-only backward compatibility, text/VL prompt control, and a declared
  reasoning budget inside the total generation limit.
- Suppressed exact normalized tool-call duplicates, bounded retries after
  repeated empty/error history results, unified stream/non-stream admission,
  and exposed machine-readable `aima_amd395.tool_progress` guidance for caller
  strategy changes or blocked results.
- Rejected known unsupported sampling and anti-repetition request fields
  instead of silently ignoring them.
- Moved chat inference to one bounded serial executor so `/health`, model-list
  and shutdown requests stay responsive while preserving batch-1 execution and
  tokenizer/engine/cache serialization.
- Made signal shutdown interrupt a blocked zero-timeout request read, retained
  request deadlines above 600 seconds, and completed an admitted chat before
  graceful shutdown.
- Qualified the exact patch engine for thinking/tool behavior and HTTP
  concurrency. The sealed `.4` GPU correctness, performance and two-host
  portable-userspace evidence is inherited only through an exact runtime-diff
  allowlist and unchanged provider/AOT hashes; no `.4` measurement is reported
  as an exact `.5` result.
- Published the checksum-bound portable archive and additive evidence for its
  isolated AMD395 bundle run, 3600-second/360-request resident soak, exact
  v1.5.1 rollback and repository/security/evidence gates.

## 1.5.1-native-vl.4 - 2026-08-25

- Added the complete fixed `Qwen3.6-35B-A3B-BF16` native image, video,
  mixed-media, multimodal conversation, tools and OpenAI stream/non-stream
  surface in the existing single resident process.
- Added the native processor, dynamic image and sampled-video pipeline,
  27-block visual tower, merger, media-embedding injection and M-RoPE/KV
  continuation without a Python, PyTorch, vLLM, Triton or Transformers runtime.
- Loaded and verified all 693 language plus 333 visual tensors before
  readiness, and warmed both hash-locked gfx1151 vision-attention variants.
- Added content-bound media and vision-embedding caches with A/B/A correctness,
  explicit local/remote allowlists, traversal/symlink/SSRF defenses and bounded
  download, decode, frame, pixel, redirect and deadline policies.
- Qualified processor, vision/language boundaries, full-vocabulary logits,
  deterministic generation, task quality and error behavior against the
  pinned vLLM/processor reference.
- Requalified the complete v1.5.1 text product with six alternating paired
  runs per 19-cell/maximum-window cell, plus correctness, MMLU, API, cache,
  startup and memory gates.
- Qualified every fixed-reference-available VL performance cell with five or
  more alternating pairs; reference-unavailable capability cells remain an
  explicit non-passing ledger rather than being counted as candidate wins.
- Added a native-VL product contract, recursive portable bundle evidence,
  two-host qualification, one-hour resident mixed-workload soak, exact v1.5.1
  rollback and immutable release provenance.
- Added positive `temperature` and `top_p` generation for text and VL using an
  exact full-vocabulary BF16 LM-head projection, stable optional seeds and
  stream/non-stream replay qualification; the greedy path remains unchanged.
- Reduced the default direct-checkpoint read chunk from 512 MiB to 128 MiB.
  Balanced AMD395 measurements cut median language-weight load time by about
  15% while retaining the single-reader policy that avoids same-device
  contention.

## 1.5.1-native-vl.3 - 2026-08-25

- Tagged but not promoted after the static release gate found an unsanitized
  package-cache path in an otherwise successful `make check` log. The
  immutable candidate is superseded by `v1.5.1-native-vl.4`.

## 1.5.1-native-vl.2 - 2026-08-25

- Tagged but not promoted after the pre-soak audit found the same invalid
  server cache-capacity argument in the one-hour resident-soak gate. The
  immutable candidate is superseded by `v1.5.1-native-vl.3`.

## 1.5.1-native-vl.1 - 2026-08-25

- Tagged but not promoted after the isolated-bundle gate exposed an unreadable
  optional DMI identity source and an invalid server cache-capacity argument.
  The immutable candidate is superseded by `v1.5.1-native-vl.2`.

## 1.5.1 - 2026-08-10

- Replaced variable-length cold-prompt and prefix-extension serial decode
  tails with composed resident AOT prefill. Non-bucket lengths now use the
  smallest covering q1024/q2048/q4096/q8192 schedule combination; padded
  linear-attention state is repaired at the logical prompt boundary.
- Removed the unused legacy framework-runtime requirements manifest. The
  supported portable native deployment has no PyTorch, vLLM, Triton or
  Transformers runtime dependency; retaining vulnerable historical pins as an
  installable manifest created a misleading security surface.

## 1.5.0 - 2026-08-05

- Added resident q1024/q2048/q4096/q8192 cold-prefill dispatch. A q8192
  service now selects the largest fitting AOT bucket for each request and
  token-decodes only the unmatched tail, without prompt padding.
- Replaced the single request-prefix snapshot with a capacity-bounded LRU:
  four entries for normal q8192 service, two for medium long-context profiles
  and one for the largest windows so the 96 GiB GTT contract remains intact.
- Exposed resident buckets, prefix-cache capacity and selected AOT-prefill
  tokens through health, CLI and per-request HTTP metrics.
- Added a hash-checked raw-token HTTP extension and resumable, prompt-free
  answer-eval scorecard generator for exact release-binary regression tests.
- Bound release qualification to the frozen privacy-safe MMLU-256 capability
  scorecard and separated packaging from native compilation so qualified bytes
  cannot be rebuilt between measurement and archive creation.
- Pinned GitHub Actions by immutable commit, added weekly update automation and
  a release-evidence pull-request checklist.
- Pinned the qualified Hugging Face checkpoint revision and metadata hashes in
  the deployment guide.
- Published the exact 19-cell matrix, nine-context KLD gate, MMLU-256/GB10
  paired scorecard, isolated bundle check and checksum-identical second-AMD395
  reproduction as hash-bound release evidence.

## 1.4.1 - 2026-08-04

- Fixed variable-length cold prompts and ordinary multi-turn cache misses:
  every request that fits the configured context now falls back to correct
  resident cold execution, while exact/append hits remain optional latency
  optimizations. Variable-length checkpoints and per-request attention-scratch
  reset prevent stale state from crossing independent conversations.
- Made the v1.4 qualification, product contract, raw reports and evidence
  verifier the repository defaults while retaining v1.3 as a versioned
  historical record.

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
