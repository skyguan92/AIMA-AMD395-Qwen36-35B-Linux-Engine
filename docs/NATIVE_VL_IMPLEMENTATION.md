# Native VL implementation status

> Governing goal: [`NATIVE_VL_GOAL.md`](NATIVE_VL_GOAL.md)
>
> Frozen text baseline: `v1.5.1` (`6f3e669`)
>
> Current phase: Phase 1 — native media and vision implementation

This file is a live requirement-to-evidence index. A status of `in progress`
or `implemented` is not a release claim. Only evidence that satisfies every
blocking condition in the governing goal can move a gate to `passed`.

## Gate status

| Gate | Current state | Evidence required to pass | Next blocking action |
|---|---|---|---|
| G1 full VL functional parity | ordered chat media parts, bounded local/data/HTTP/HTTPS admission, image/video processors and resident visual weights are implemented; vision execution and serving remain incomplete | complete image/video/mixed/conversation/API/tools/transport/residency native conformance results | execute processor tensors through the resident vision/model path |
| G2 VL correctness parity | reference processor/boundary/logits/generation oracles frozen; native patch and position boundaries qualified; remaining native comparison and task suites pending | processor, vision/language boundary, full-vocabulary logits, deterministic generation, task quality and error results | implement vision blocks and merger, then compare the remaining frozen raw tensors before running task quality |
| G3 text product no regression | frozen baseline identified | paired 19-cell, maximum-window, correctness, MMLU, API, cache, startup and memory requalification | retain `v1.5.1` as an immutable paired binary |
| G4 native VL performance | not started | paired per-cell stage timings and memory records against the fixed VL-enabled vLLM | generate matrix cells from the capability manifest |
| G5 native release product | not started | native-only package, security, isolated bundle, second-host, soak and rollback evidence | keep Python tooling qualification-only and outside the product runtime |

## Phase 0 invariants

The reference capture is fail-closed:

- it rejects `--language-model-only` and `--skip-mm-profiling`;
- it requires explicit non-zero multi-image and multi-video limits;
- it requires an already completed, SHA-256-bound capability manifest;
- it freezes a clean, non-inherited launch environment, processor/media
  arguments, allowlists, video sampling backend and every timing boundary;
- it hashes all model identity/processor files and all Safetensors shards;
- it records the resolved processor, runtime packages, key vLLM/Transformers
  source modules, host/GPU/ROCm facts and media decoder binaries;
- it refuses dirty capture source and a model/runtime that differs from the
  frozen goal.

The capture implementation is
`scripts/capture-vl-reference-manifest.py`; its unit contract is
`tests/test_vl_reference_manifest.py`. The processor and API capability
envelopes are now frozen in
`benchmarks/results/vl-processor-capability-v0.1.0.json` and
`benchmarks/results/vl-capability-manifest.json`. The latter binds 30 API
cases, the former and the generated media corpus by SHA-256. The sole launch
contract is `benchmarks/results/vl-reference-launch.json`. No boundary,
logits or generation oracle may be accepted before
`benchmarks/results/vl-reference-manifest.json` is generated and verified on
`amd395`.

## Verified live facts (2026-08-13)

Read-only probing on the target established:

- Linux host label `amd395`, hostname `quings`, GPU architecture `gfx1151`, and
  both `/dev/kfd` and `/dev/dri/renderD128` available to the serving account;
- vLLM `0.19.1rc1.dev300+g29e5d1020.rocm721`, PyTorch
  `2.10.0+git8514f05`, and Transformers `4.57.6` in the reference virtual
  environment;
- the frozen config, checkpoint index, tokenizer and tokenizer-config hashes
  match `NATIVE_VL_GOAL.md` and `native/product-contract-v1.5.1.json`;
- the resolved processor is `Qwen3VLProcessor`, with
  `Qwen2VLImageProcessorFast` and `Qwen3VLVideoProcessor`; the video defaults
  resolve to 2 fps, 4 minimum frames and 768 maximum frames;
- the old stock vLLM script is text-only and therefore cannot be used as a VL
  reference.

The live processor and server probes additionally established:

- maximum processor output per item is 16,384 image tokens and 12,288 video
  tokens; the fixed count limits are 16 images and 21 videos, with a 16,384
  aggregate encoder-token serving budget and chunked prefill preserving the
  262,144 total window;
- a full-window encoder profiling budget is invalid on this gfx1151 runtime:
  spatial-merge expansion drives the fused visual RoPE grid beyond the HIP
  launch boundary. The exact probe passed 524,288 pre-merge tokens and failed
  at 524,296. The frozen 16,384-token serving budget covers every single
  processor-legal maximum item without weakening the model context window;
- the VL-enabled vLLM loaded all 26 checkpoint shards, reported 67.64 GiB of
  model residency, completed real multimodal profiling and served from the
  same process with language and vision attention fixed to `TRITON_ATTN`;
- all 20 required success probes returned HTTP 200, including image, video,
  mixed ordering, multi-media, prior-turn media, local/data/HTTP transport,
  forced and auto Qwen3 XML tool calls, and image/video SSE; all 10 frozen
  invalid-input probes returned HTTP 400;
- the checkpoint contains 333 `model.visual.*` tensors totaling 893,142,496
  bytes. The existing native layout contains only the 693 language/lm-head
  tensors, so vision inclusion is a bounded approximately 0.9 GB weight
  addition; the unrelated MTP tensors remain outside this goal.

The generated reference manifest has now bound the target host, runtime,
model, launch and capability hashes. It was captured from clean commit
`7d9813992d12f64bf668a04e9f25d17736dd984f`, then independently verified by a
second full read of all checkpoint shards. Its file SHA-256 is
`8f03e939a9b15f58c2e623541d48be88c328ff19adac9eb52210e171766e9a00`.

The offline oracle capture is also complete for the five blocking numerical
shapes: image, video, multi-image, multi-video and mixed image/video. Each case
starts with empty processor and encoder caches, while the 67.64 GiB model stays
resident in one process. The result contains 12 raw processor tensors and all
50 required model boundaries (patch embed, vision blocks 0/13/26, merger,
M-RoPE positions/delta, injected embeddings, language layer 0, final norm and
selected full-vocabulary teacher-forced logits), totaling 51,064,728 bytes.
All raw shape/byte/SHA-256 records were independently reread on `amd395`; the
manifest contains no target-private paths. The manifest SHA-256 is
`87dcdf76b7251f78da01a2a5f4312a9fb5c7d07a1ca2b2420566e77930f23d44` and
was captured from clean commit `09e3fac8a07d9e5884007f0afdf46fb6603ae78d`.
This closes oracle creation only; G2 remains blocked on native comparison,
task-quality suites and error parity.

## Phase 1 implementation evidence

The native request layer now parses ordered OpenAI `text`, `image_url` and
`video_url` content parts, emits the model's canonical image/video markers,
retains media/message ordering metadata and enforces the frozen 16-image and
21-video count limits. Media is restricted to user messages. The HTTP path
continues to reject these admitted requests before language execution until
native processor tensors and vision embeddings are attached, preventing a
placeholder-only false success.

The first native media boundary is also present. It:

- accepts bounded base64 data URIs and allowlisted local files opened beneath
  a validated root through descriptor-relative, no-symlink traversal;
- rejects malformed/non-canonical base64, MIME/kind mismatches, empty or
  over-limit payloads, local-root escapes and credential-bearing URLs;
- fetches HTTP/HTTPS through exact per-hop domain allowlists, manual bounded
  redirects, aggregate byte/deadline limits and checked socket addresses;
- identifies cacheable media by the SHA-256 of the decoded source bytes, so
  equivalent data/local transports have the same content identity.

Local validation and loading no longer trust a canonicalized path followed by
a second pathname open. The loader walks from an already opened allowlist root
with `openat`/`O_NOFOLLOW`, verifies the final regular-file descriptor and reads
that descriptor under a stable metadata check. Parent traversal, intermediate
or final symlinks, and a file swapped to a symlink after validation all fail
closed.

Remote fetch disables proxy inheritance, credentials, non-HTTP protocols and
automatic redirects. Every hop is reparsed and allowlisted, an HTTPS hop cannot
downgrade to HTTP, and the socket callback rejects DNS rebinding into loopback,
private, link-local, carrier-grade, multicast or unspecified space unless the
operator explicitly authorizes the private hostname; canonical literal
loopback remains available for the frozen local transport fixture. Redirect
bodies and the final payload share one byte budget, while DNS, connect,
redirect and body transfer share one deadline. Tests cover relative and
cross-host redirects, redirect loops, proxy environment poisoning, numeric-IP
aliases, over-limit/slow bodies, untrusted TLS, a dedicated trusted test CA and
TLS downgrade rejection.

The portable dependency is a 1.4 MiB self-verifying distribution built from
curl 8.21.0 and c-ares 1.34.8. It exposes only HTTP/HTTPS, uses asynchronous DNS
and OpenSSL, disables proxy and ambient CA-store discovery, and carries the
qualified CA bundle plus curl/c-ares/CA licenses. Its two libraries depend only
on bundled c-ares/OpenSSL/glibc. This is transport implementation evidence; a
final product archive and live native API conformance run remain pending.

The resident prefix cache now uses a composite identity: exact/prefix text
token comparison plus a versioned digest over the processor configuration and
every ordered media content digest, kind, placeholder token and expanded token
span. Different bytes behind the same URL or filename, changed processor
settings, changed spans and reordered media conservatively miss; equivalent
decoded data/local inputs share an identity. The pure matching regression
covers text-only preservation, same-media extension, A/B collision rejection
and A/B/A identity recovery. The HTTP path will be required to provide this
namespace before media requests are enabled.

The frozen processor's discrete contract is now implemented independently in
C++: ties-to-even image/video smart-resize geometry, spatial/temporal factors,
per-item and aggregate token budgets, 2-fps frame sampling, temporal timestamp
construction, image placeholder expansion and per-temporal-grid video prompt
expansion. Its versioned configuration identity binds both preprocessor files,
the chat template, all effective numeric parameters and the four special-token
IDs. A native regression covers every frozen resize boundary and compares the
complete frame-index vectors for all six accepted sampling cases by SHA-256;
it passes both locally and on `amd395`.

The native processor now also performs the exact fused normalization, odd-frame
repeat and Qwen temporal/spatial patch permutation into contiguous
`[patches,1536]` BF16. The frozen 256x256 image oracle matches byte-for-byte,
and a four-frame deterministic video regression closes channel, temporal,
patch and merge ordering.

The native image decoder now covers PNG, JPEG and WebP from admitted in-memory
bytes with compressed-byte and decoded-pixel/dimension bounds. It drops alpha
without compositing to match `Image.convert("RGB")`, rejects corrupt/truncated
inputs without decoder stderr, and never reopens the source path. All four
frozen image fixtures match the reference decoded RGB SHA-256 exactly,
including RGBA PNG and JPEG. The processor also reproduces torchvision v2's
separable uint8 antialiased bicubic kernel, including per-axis int16 weight
quantization and uint8 rounding. A resize-required RGBA fixture and an
independent mixed up/down-scale tensor match both resized RGB and final BF16
processor output byte-for-byte. The runtime link and portable-bundle closure
include the image codec libraries and their distribution license texts.

The native video decoder now demuxes admitted MP4 and AVI bytes through a
MIME-selected FFmpeg input surface and decodes selected frames directly to
RGB8. Compressed bytes, source/sample frame counts, duration, dimensions,
decoded pixels and decode wall time are bounded before tensors are admitted.
The serving sampler separately reproduces the frozen vLLM OpenCV backend's
2-fps floor/linspace behavior; it must not be confused with the direct
Transformers processor's ties-to-even sampling probe. Both frozen videos match
OpenCV's complete selected-frame RGB SHA-256, and their final `[128,1536]` and
`[192,1536]` BF16 tensors match the raw reference oracles byte-for-byte. The
MP4 timestamp regression also fixes the second temporal group at 1.4 seconds.
The runtime and packaging scripts are wired to a pinned FFmpeg 6.1.1 minimal
build (source SHA-256 `8684f4b00f94b85461884c3719382f1261f0d9eb3d59640a1f4ac0873616f968`):
network/GPL/nonfree features are disabled, only the AVI/MOV demuxers and
MJPEG/MPEG-4 decoder closure are enabled, and the resulting four libraries
depend only on each other plus bundled glibc/libm. This 4.1 MiB distribution
reproduces both video oracles and carries a self-verifying manifest and LGPL
texts. A final portable product archive has not yet been built or qualified,
so this is implementation evidence rather than G5 release evidence.

The visual checkpoint boundary is now frozen independently from the immutable
`v1.5.1` language layout. A capture tool reads only the pinned checkpoint index
and Safetensors headers, validates the exact model/config/index identity, and
records all 333 `model.visual.*` tensors from their two source shards. The
manifest closes the fixed 27-block architecture, exact BF16 shapes and source
offsets, 893,142,496 payload bytes, source-file SHA-256 identities and a
byte-exact uint64 payload XOR/sum. A separate deterministic generator emits the
rank-5 compile-time visual layout and fails if a block tensor, patch/position
embedding or merger tensor is added, missing or reshaped. Keeping this contract
separate lets the current 693-tensor text baseline remain directly comparable
while the same resident process gains the bounded visual store. Two independent
captures produced the same 88,646-byte manifest, SHA-256
`abc5b3a0cc0881ba2d3e815b472eebe3404a6e3bc6438a430faccfbe8093c0aa`.

The existing Safetensors scatter implementation now accepts language-only,
visual-only or combined resident layouts. Production residency uses one
combined scatter and one name registry, so the first two checkpoint shards are
not reread solely for vision. A target probe loaded all 1,026 active tensors
into 1,026 unique device pointers: 70,214,363,872 payload bytes from 26 shards,
with combined XOR `0x27b3037f0725611f` and sum `0x9a017e7d5747ae3d`
matching exactly. It read 70,214,401,080 source bytes in 151 chunks; compared
with the language layout this is one additional chunk, rather than the 2.02 s
second pass measured by the visual-only diagnostic. Resident load metrics and
the HTTP ready event expose the total, language, visual and manifest identities.
The vision kernels and processor-to-encoder execution remain the next boundary.

The first visual compute boundary is qualified. Patch projection treats the
frozen Conv3d kernel as its exact `[patches,1536] x [1152,1536]^T` BF16 linear
surface and uses the checkpoint bias as a hipBLASLt epilogue before BF16
quantization. A clean `9d29d9d` build matched the image, video, multi-image and
multi-video patch oracles bit-for-bit across 1,548,288 BF16 elements (128, 256,
320 and 640 patch rows); every actual SHA-256 equals its oracle SHA-256. The
hash-bound result is `benchmarks/results/native-vision-patch-v0.1.0.json`.
This result advances only the patch boundary of G2.

The native position plan interpolates the resident 48x48 BF16 table for a
parameterized list of image/video grids, preserves Qwen's 2x2 spatial-merge
order and repeats each spatial table over the temporal dimension. It can fuse
the BF16 addition into the patch output in place. The original v0.1
qualification was withdrawn: it captured the eager
`pos_embed_interpolate_native` helper, while the frozen serving runtime has
Triton enabled and selects `triton_pos_embed_interpolate`. The historical
measurements remain in `benchmarks/results/native-vision-position-v0.1.0.json`
with `status=withdrawn` and cannot satisfy a gate. The corrected capture binds
the actual Triton function and the HIP implementation reproduces its fused
float32 coordinate remainder plus gfx1151 BF16 dot-product lowering. Two
independent serving-path captures produced the same manifest (SHA-256
`9d316fd6904764f88cd5f25726ecaed33d95bb6cfb4bbe21454c909d66c5d9f6`).
A clean `851605b` build matched all four square/non-square image/video cases
bit-for-bit across 1,105,920 BF16 elements; duplicated-media concatenation and
zero-input in-place addition each matched another 2,211,840 elements exactly.
The corrected result is
`benchmarks/results/native-vision-position-v0.2.0.json`. The 27 vision blocks,
merger and serving integration remain incomplete, so neither G1 nor G2 passes.

Vision block development now has a serving-path internal oracle rather than a
standalone mathematical replay. A clean `23ddda5` full-model run hooked block 0
at 16 boundaries: input, both LayerNorms, QKV, BF16 rotary inputs and rotated
Q/K/V, segmented Triton attention, projection/residual, both MLP projections,
GELU and output. It captured one 256-patch image and one 128-patch/two-frame
video, including `[0,256]` and `[0,64,128]` sequence boundaries. Both newly
captured block outputs were byte-identical to the earlier independent full-model
oracle. The sealed raw manifest SHA-256 is
`443cae9b5dc0b426ec04725a7dede893c3c433d6cb910b81cbd48ecd9bfd782a`;
the public compact record is
`benchmarks/results/native-vision-block-oracle-v0.1.0.json`. This freezes the
next implementation boundaries.

The first native block compute slice is now qualified without overstating it as
a complete block. `NativeVisionBlockPrefixPlan` implements parameterized BF16
LayerNorm (`epsilon=1e-6`, FP32 Welford reduction) followed by the checkpoint
QKV projection and bias. From a clean `7349828` checkout, block 0 passed both
the 256-row image and 128-row/two-frame video serving oracles. Across the two
cases, LayerNorm had relative L2 error at most `1.37e-5` and cosine above
`0.9999999999`; QKV had relative L2 error at most `5.62e-5` and cosine above
`0.9999999984`. All 1,769,472 compared BF16 elements were finite. The hash-bound
record is `benchmarks/results/native-vision-block-prefix-v0.1.0.json`.
Attention, the projection/residual and MLP half of each block, blocks 13/26,
the merger and serving integration remain unqualified, so neither G1 nor G2
passes.

The following Q/K rotary and QKV-layout boundary is also native and qualified.
The frozen ROCm reference uses FlashAttention's Triton Neox rotary kernel with
BF16 inputs, FP32 multiply-add and 36+36 pairing inside each 72-wide head. A
clean `b35d5db` native build reproduced query, key and value tensors for both
the image and two-frame video byte-for-byte: 1,327,104 of 1,327,104 BF16
elements and every SHA-256 matched. The compact record is
`benchmarks/results/native-vision-rotary-v0.1.0.json`. Segmented attention is
implemented as a bounded online-softmax kernel rather than an `S x S` score
allocation. A clean `e86b76b` build passed the 256-token image and two
independent 64-token video-frame sequences. Maximum relative L2 error was
`4.54e-4`, minimum cosine was `0.999999897`, and all 442,368 outputs were
finite. After zeroing the entire second video segment, all 73,728 BF16 outputs
in the first segment remained bit-exact, directly proving the frame boundary.
The result is
`benchmarks/results/native-vision-segmented-attention-v0.1.0.json`.

The post-attention suffix is qualified independently at every arithmetic
boundary. A clean `8e278f8` build reproduced attention projection, attention
residual, FC1, ROCm's effective exact-GELU path, FC2 and final residual
bit-for-bit for both cases. Norm2 differed by only 4 image and 5 video BF16
elements; the complete suffix chain ended at relative L2 `3.54e-5` for image
and `1.28e-4` for video. All 5,959,680 compared values were finite and the
result is `benchmarks/results/native-vision-block-suffix-v0.1.0.json`.
The next boundary is a single end-to-end native block invocation fed only by
the frozen block input, rotary tables and sequence metadata.

`native_media_test` and `native_chat_protocol_test` both compile with strict
warnings and pass on `amd395`. This is implementation progress, not G1 or G2
acceptance evidence.
