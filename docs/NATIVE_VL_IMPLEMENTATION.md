# Native VL implementation status

> Governing goal: [`NATIVE_VL_GOAL.md`](NATIVE_VL_GOAL.md)
>
> Frozen text baseline: `v1.5.1` (`6f3e669`)
>
> Current phase: Phase 2 — resident serving and product qualification

This file is a live requirement-to-evidence index. A status of `in progress`
or `implemented` is not a release claim. Only evidence that satisfies every
blocking condition in the governing goal can move a gate to `passed`.

## Gate status

| Gate | Current state | Evidence required to pass | Next blocking action |
|---|---|---|---|
| G1 full VL functional parity | the native HTTP path passes the frozen 30-case status/finish/tool/SSE surface, all 18 media prompt vectors match the real vLLM render boundary, and named-tool decoding is schema constrained; the corrected deterministic 23-cell min/typical/max envelope is sealed but not yet executed | complete image/video/mixed/conversation/API/tools/transport/residency native conformance results | execute all remaining envelope cells without shrinking the frozen support surface |
| G2 VL correctness parity | the five private and five independently rendered real-HTTP language prompts now pass final norm and 84/84 selected full-vocabulary rows bit-exact; resident serving still preserves the five frozen 8-token outputs; task-quality suites and unexecuted envelope cells remain open | processor, vision/language boundary, full-vocabulary logits, deterministic generation, task quality and error results | run the frozen image/video task-quality suites and all remaining envelope/error cells |
| G3 text product no regression | frozen baseline identified; the resident engine and certified lm_head now have an optional mask path that is disabled for ordinary requests, but the complete paired release matrix has not run | paired 19-cell, maximum-window, correctness, MMLU, API, cache, startup and memory requalification | retain `v1.5.1` as an immutable paired binary and run the complete paired matrix after language integration stabilizes |
| G4 native VL performance | the 23-cell capability envelope is generated, but paired execution has not started | paired per-cell stage timings and memory records against the fixed VL-enabled vLLM | execute the generated cells with the frozen paired timing protocol |
| G5 native release product | the full runtime now includes all qualified vision sources and loads the 333 visual tensors in the same resident process; the external vision-attention code object is hash-checked and wired into the portable package contract, but final package qualification has not run | native-only package, security, isolated bundle, second-host, soak and rollback evidence | generate a clean product qualification containing the vision code object, then run isolated bundle, second-host, soak and rollback gates |

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

## Verified live facts (2026-08-14)

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

A deterministic projection of the frozen processor and API manifests now
seals all 29 discrete resize, sampling and media-count boundaries plus 23
min/typical/max and pairwise execution cells. The hash-bound artifact is
`benchmarks/results/vl-capability-envelope-v0.1.0.json`, SHA-256
`7f8431565404000ef1692da31d461d93c02c3322e420e35fefc6dc3dcd2976a6`.
The sampling cells project the frozen 4/20/768 sampled-frame counts through
the probe's 256x256 source metadata and video smart-resize, yielding 128, 640
and 9,600 visual tokens. They no longer borrow the unrelated two-frame resize
cell's token count. The 768-frame source boundary admits 50,331,648 selected
decoded pixels before smart-resize; the separate post-resize feature budget
remains 25,165,824 pixels. The direct `fps` plus `num_frames` processor
conflict is processor evidence only and is not mislabelled as a frozen OpenAI
HTTP content-part contract.
The native OpenCV-compatible sampler clamps the requested sample count to the
frozen 768-frame maximum before computing linspace indices, so the 18,432-frame
above-maximum source boundary remains accepted with exactly 768 selected
frames instead of being rejected by a stricter implementation limit.
It exposes a previously conflated contract: ordered media may consume the full
262,144 encoder-token budget, while each visual-tower execution batch remains
bounded to 16,384 merged tokens and 65,536 patches. The native request path now
admits that aggregate budget and partitions consecutive whole media items into
bounded batches with exact patch and embedding offsets; requests at or below
16,384 tokens retain the original single-batch path. HTTP metrics expose the
batch count and maximum batch token/patch counts so qualification can prove
the bound directly. The full 262,144-image cell is explicitly
processor/vision-only because an HTTP request also needs text and wrapper
headroom. These source and CPU boundary checks are not target execution
evidence; all 23 cells still must run before G1 or G2 can pass.

A checked-in 16-file deterministic media corpus now makes that execution plan
replayable without synthesizing inputs on the target. Its sealed manifest is
`benchmarks/fixtures/vl-envelope-v0.1.0/fixtures-manifest.json`, SHA-256
`5833acac02f6eb68d057c431e73f57434b4fc6c20c000e0ed3eec5dc00236161`,
and binds the generator plus NumPy 2.1.3, Pillow 12.2.0 and imageio-ffmpeg
0.6.0. The fail-closed qualifier maps 23 HTTP observations to the 21 HTTP
cells, runs the processor-only option-conflict cell directly, and executes the
full encoder-budget cell with a dedicated 16-batch visual-tower probe. It also
requires one resident model load, contiguous accepted-request indices, exact
media/token/patch/batch metrics, native-only runtime markers and no oracle
reads. The server retains its frozen 600-second request-read/write timeout;
the qualification client has a separate 7,200-second response wait so a legal
full-window multi-batch compute cell is not mistaken for a socket-policy
failure. The original 3,600-second client bound expired while the target was
still actively computing the 245,760-token image cell, with empty server
stderr, so it is not used as a native execution limit. Resident
vision plans retain a four-entry LRU surface but are also bounded to one
65,536-patch execution batch in aggregate. Plans at or above one fourth of
that budget are exclusive because each shape also owns fixed HIPBLASLt plan
state that a patch-only sum does not measure. On a cache miss, exclusive or
least-recently-used plans are released before constructing the replacement;
this avoids the transient double allocations exposed by both the
small/small/small/maximum-image and small/small/maximum-video target
sequences. Smaller shapes retain the four-entry LRU behavior. This paragraph
describes the qualification mechanism only; no `amd395` execution artifact
has yet been accepted.

The first full-window HTTP attempt also exposed a launch-geometry bug that the
original 81-token unified-attention capture could not reveal. Timeline-mode
synchronization localized the failure to full-attention layer 7, segment 29;
a minimum-image plus 245,000-token diagnostic then isolated the failing
substage to the unified-attention core in 427 seconds. Its two-query block
grid used `query_tokens / 2 + 1`, which is only ceil-division for odd lengths
and launched one extra block for every even 8,192-token segment. The runtime
now uses exact ceil-division. This diagnosis is not a passing full-window or
envelope artifact; both must be rerun against the corrected binary.

Exact ceil-division moved the deterministic failure from layer 7 to layer 31
but did not make the q81-captured kernel a stable long-window implementation.
Complete 8,192-token M-RoPE chunks now reuse the existing qualified
rectangular text FMHA provider after Q/K have received their M-RoPE rotation;
only padded logical tails retain the unified-attention kernel. This preserves
the short prompt path already used by the exact language evidence while
removing hundreds of unsupported full-chunk launches. The long-window and
full envelope reruns remain required.

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
The same components are now composed behind `NativeVisionBlockPlan`, with an
external temporary arena reusable across the 27 resident block plans. A clean
`e1b4680` build passed the end-to-end block 0 serving oracle for both cases:
image relative L2 `0.001729` and cosine `0.999998506`; two-frame video relative
L2 `0.001116` and cosine `0.999999377`. Every one of the 442,368 BF16 outputs
was finite and both repeated outputs were SHA-256 identical. The 256-patch
case uses 4,997,120 bytes of temporary storage (19,520 bytes per patch), with
no full attention score matrix. The hash-bound record is
`benchmarks/results/native-vision-block-v0.1.0.json`. The timings in that
record are diagnostic probe timings, not G4 serving evidence. At that
implementation checkpoint blocks 13 and 26, the complete 27-block encoder,
merger and serving integration were still unqualified.

The representative-depth inputs are now frozen from the same serving path. A
clean `41d6f6b` capture hooked blocks 0, 13 and 26 during one image and one
two-frame video request, retaining only block input/output, rotary tables and
segmented-attention metadata. All six captured outputs were byte-identical in
shape, dtype and payload to the independently frozen full-model boundaries.
A second full capture reproduced all 36 binary files and the sealed manifest
byte-for-byte (manifest SHA-256 `0e4967ca0c0ca0d54be1ab0854736b60e7b0cd57141340c9743f46aec8d69604`).
The compact record is
`benchmarks/results/native-vision-depth-oracle-v0.1.0.json`. This qualifies the
oracle inputs. A clean `3681adb` native build then ran the same parameterized
block plan at depths 0, 13 and 26 for both media cases. All 1,327,104 outputs
were finite; the worst relative L2 was `0.001729`, the minimum cosine was
`0.999998506`, and every repeated output SHA-256 was exact. The block 26 image
has a maximum absolute delta of 32 only in its large BF16 tail (reference
maximum magnitude 12,480); its relative L2 is `0.000646`, within the frozen
metric without changing thresholds. The result is
`benchmarks/results/native-vision-representative-blocks-v0.1.0.json`.
That record qualifies isolated representative blocks, not a sequential
encoder.

The cumulative error was then resolved without weakening any threshold. The
reference Triton attention launch was frozen as a standalone gfx1151 code
object (`BLOCK_M=128`, `BLOCK_N=128`, head dimension 72) and the native loader
replayed it with no Python, Torch, vLLM or Triton runtime. Both raw attention
cases are bit-exact: 294,912 image and 147,456 video BF16 elements. The
remaining cumulative delta came from using a different LayerNorm reduction
shape and expression order. A native port of the pinned PyTorch `8514f05`
ROCm vec4, 32-by-8 Welford path selected the AMD fast reciprocal using the
video discriminator and reproduced both norm1 cases bit-for-bit. The sealed
record is
`benchmarks/results/native-vision-exact-layer-norm-v0.1.0.json`.

The exact LayerNorm and shared AOT attention module are now composed in one
parameterized resident 27-block stack. On the frozen image and two-frame video
cases, cumulative outputs after blocks 0, 13 and 26 are all bit-exact: all
1,327,104 BF16 elements match, relative L2 is zero, cosine is one, and every
repeat SHA-256 is identical. Full block-stack medians were `14.817 ms` for
256 image patches and `11.741 ms` for two 64-patch video segments; these are
diagnostic kernel timings, not G4 serving evidence. The hash-bound source,
binary, AOT and six comparisons are in
`benchmarks/results/native-vision-aot-encoder-v0.1.0.json`. The representative
27-block encoder is qualified.

The native patch merger now applies the pinned exact 1152-wide LayerNorm,
uses the reference contiguous four-patch view, and executes the two biased
linear layers around exact GELU without an explicit patch reorder. It was
qualified independently on all five frozen oracle shapes. Single image,
single video, multi-image, multi-video and mixed image/video produced all
884,736 BF16 elements bit-for-bit, with zero relative L2, cosine one and
identical repeat hashes. The selected hipBLASLt plans required no library
workspace; median kernel-chain times ranged from `0.429 ms` to `0.695 ms`.
The hash-bound evidence is
`benchmarks/results/native-vision-merger-v0.1.0.json`. Full patch-to-merger
composition was qualified next.

The fixed-shape native pipeline now composes processor-native BF16 pixels,
patch projection, interpolated positions, request-shape RoPE metadata, the
resident 27-block AOT encoder and merger. Its first multi-image run exposed a
precision discriminator outside the original square-image coverage: vLLM
builds the rotary cache on the GPU, while CPU libm changed 12 of 23,040 cosine
values by one BF16 ULP for the 12-by-32 grid. The production path now uses a
pure HIP cache-initialization kernel and the pinned gfx1151 float32 inverse
frequencies. The resulting metadata hashes match independent serving hooks.

All five processor-to-merger cases now match at blocks 0 and 26 and at the
final merger: 4,866,048 of 4,866,048 BF16 boundary elements are bit-exact,
relative L2 is zero, cosine is one and repeated final hashes are identical.
Mixed image/video preserves the reference's two visual calls and concatenates
their outputs in request order. Diagnostic full-pipeline medians range from
`12.213 ms` for the two-frame video to `35.281 ms` for multi-image; they are
kernel-chain measurements, not G4 serving results. The hash-bound records are
`benchmarks/results/native-vision-multimedia-block-oracle-v0.1.0.json` and
`benchmarks/results/native-vision-pipeline-v0.1.0.json`.

Media embedding injection is now a separate fail-closed native boundary. The
host plan accepts processor-owned prompt spans and explicit visual source
offsets, derives image/video replacement positions from token IDs `248056`
and `248057`, and rejects overlapping spans, wrong modality/count, incomplete
visual-row coverage and orphan placeholders. This reproduces vLLM's video
`is_embed` behavior without carrying a second mutable mask: every frozen mask
bit was independently re-derived from the prompt token vector. The GPU path
first performs the existing resident token-embedding lookup, then scatters
BF16 merger rows only into those validated positions.

A clean `ca92be3` build loaded the 69,321,221,376-byte language layout once and
qualified image, video, multi-image, multi-video and mixed image/video in one
resident run. All 1,198,080 injected BF16 elements are bit-exact, all five
actual SHA-256 values equal their frozen oracle values, and repeated outputs
are deterministic. Per-case diagnostic injection medians were `0.0182` to
`0.0788 ms`; these include prompt/index host-to-device uploads but are not G4
serving timings. The hash-bound evidence is
`benchmarks/results/native-vl-embedding-v0.1.0.json`. At that checkpoint,
M-RoPE positions/delta, language boundaries and serving were still
unqualified.

The M-RoPE host plan now ports the pinned vLLM
`Qwen3VLForConditionalGeneration._get_mrope_input_positions` integer contract.
It emits contiguous row-major `int64[3,prompt_tokens]` positions, keeps video
timestamp and vision-boundary tokens on the text sequence, emits one spatial
grid for each temporal row, handles lumped video placeholders, and returns the
position delta used by decode continuation. External spans and grids are
validated for bounds, overlap, merge alignment, multiplication overflow,
aggregate visual budget, per-frame boundary tokens and orphan placeholders.

A clean `ad21b57` target build reproduced all five frozen position tensors and
deltas exactly: 1,755 of 1,755 integers, with every actual SHA-256 equal to its
oracle. The cases include square and non-square images, two-frame videos,
multi-image, multi-video and mixed media; deltas range from `-24` to `-136`.
The hash-bound evidence is
`benchmarks/results/native-vl-mrope-v0.1.0.json`. This closes the discrete
prompt-position boundary and implements the decode-position formula. Language
layer 0 is a linear-attention layer and does not consume rotary positions; the
first consumer is the full-attention layer 3. Its isolated table and Q/K
boundary is now qualified below, while resident integration and the complete
layer remain blocking. G1 and G2 therefore remain false.

The first resident language compute boundary is qualified at the actual
product geometry. A clean `85fa597` worktree executes the complete q1024
bucket for every case, including its zero-padded tail, and compares only the
63-to-182-token logical prefix. The q1024 closure uses the serving `BT=64` FLA
chain (401 launches, 13 code objects). Short VL requests use the qualified
four-warp recompute-W/U image at logical `T`; the resident bucket argument is
restored after that launch. Native 32-by-16 vec4 RMSNorm reproduces the pinned
PyTorch eager reduction order, and logical A/B and router projections remain
VL-scoped.

| Case | Logical tokens | Layer-0 relL2 | Cosine | Diagnostic median | Main/seeded expert sets |
|---|---:|---:|---:|---:|---:|
| Image | 81 | `0` | `1` | `15.755 ms` | `81/81`, `81/81` |
| Video | 63 | `0` | `1` | `16.037 ms` | `63/63`, `63/63` |
| Multi-image | 182 | `0` | `1` | `16.535 ms` | `182/182`, `182/182` |
| Multi-video | 128 | `0` | `1` | `16.609 ms` | `128/128`, `128/128` |
| Mixed image/video | 131 | `0` | `1` | `16.799 ms` | `131/131`, `131/131` |

All five outputs are finite, repeat deterministic and bit-exact: 1,198,080 of
1,198,080 BF16 elements. Each case also passes 24 main-chain and 9 seeded-MoE
diagnostic comparisons. Input RMSNorm and the logical B projection are
bit-exact in all five cases; the unordered top-k expert sets are exact for all
585 rows in both the main and seeded runs. The exact-commit probe binary
SHA-256 is
`77e7c7fa6847b664a24dcd883e5d43e6acfedad2f01693d37d0d545bcf49d130`;
the raw result SHA-256 is
`c4561c025f897bf53d1afebfdf17a721041ad9ba06a720ae9796f0e7d634015e`.
The superseding hash-bound record is
`benchmarks/results/native-vl-language-layer0-v0.2.0.json`; v0.1 remains as
historical evidence.

The same source identity is tied to the formal resident runtime SHA-256
`7fe6ceb07dbae924e8da5efa378b3a47ae7b0cd8e6fc023eff3c74d1298e67b2`,
which embeds the complete default 61-kernel, eleven-manifest closure and has
separate resident-serving qualification. It has no Python, Torch, vLLM or
Triton runtime dependency. The build-tree binary is still not a portable
package, second-host, soak, rollback or release qualification, so G5 remains
false.

The layer-3 reference capture uses the real pinned vLLM serving path. Two
clean `8e6f66c` runs each captured 24 components for image, video,
multi-image, multi-video and mixed image/video. All 120 component shapes,
dtypes and 57,227,480 payload bytes repeat exactly, and every captured
`int64[3,tokens]` position tensor is byte-identical to the frozen processor
oracle.

The native consumer reads those resident positions, selects the interleaved
`[11,11,10]` T/H/W pair axes and rounds the FP32 trigonometric cache through
BF16 before using the existing FP32 workspace ABI. Attribution established
that the 256-wide head RMSNorm was already bit-exact; the remaining rotary
discriminator was the pinned AMD Triton sequence of BF16 RTZ products,
FP32 add/subtract and final BF16 RNE store. This arithmetic is exposed only by
the explicit M-RoPE entry point, leaving the scalar-position text consumer
unchanged.

On clean `764fd57`, generated cos/sin, head-RMSNorm Q/K, oracle-table rotary
Q/K and generated-table rotary Q/K all pass bit-for-bit for all five cases.
The six nonduplicate comparisons total 5,428,800 of 5,428,800 exact BF16
elements; 1 warmup plus 5 measured runs per case are deterministic. A second
independent reference capture produces the same 20 native output files and
all 5,466,240 bytes exactly. Diagnostic medians are `0.042` to `0.085 ms` and
cover only table generation plus Q/K normalization/rotation. The probe binary
SHA-256 is
`fab0bd21e95f6f1f497a02e45a739b042418a0f1b031966e12d7c19e71dca8af`;
the two result SHA-256 values are
`95697e3faab79a95b00a10249599412a6065fd1f7547b713e7c5248bcd571fe6`
and
`7a27750619bf709ba5e3b892f264bf10cdab036ac8deb7aa77ed7ae2fdca0c9d`.
The curated record is
`benchmarks/results/native-vl-language-layer3-mrope-v0.1.0.json`.

This closes the isolated layer-3 position-table and Q/K-consumption boundary,
not the complete layer. The resident q1024 request path still must upload and
bind the plan, layers 0 through 3 must be composed in one request, and layer-3
causal attention, output projection, residual and MoE must pass before moving
to the remaining language layers, final norm, lm_head and logits. G3 paired
text qualification and G4 serving performance also remain blocking.

`native_media_test` and `native_chat_protocol_test` both compile with strict
warnings and pass on `amd395`. This is implementation progress, not G1 or G2
acceptance evidence.

The resident serving boundary now implements the fixed vLLM OpenAI server's
auto-resolved `string` content layout. It prepends each message's missing media
placeholders, preserves the resulting media association, joins content parts
with the same newline rules and leaves the `v1.5.1` text-only template path
unchanged. A host request object loads and processes every media part, derives
injection spans, builds the exact M-RoPE plan and seals the multimodal prefix
identity. Inside the resident engine, a bounded four-entry LRU selects the
shape-specific visual plan, processor BF16 pixels feed the patch-to-merger
pipeline, and the resulting 2048-wide rows are scattered into the normal
language prompt embedding store. Text-only requests do not enter this path.

The real fixed-vLLM prompt boundary is frozen independently through its
GPU-less `/v1/chat/completions/render` endpoint. This distinction is required:
the original numerical-oracle capture explicitly forced
`chat_template_content_format="openai"`, while the real server resolves this
model to `string`. The 20-case API render manifest stores the complete token
vectors, placeholder spans, request hashes, sampling parameters and structured
output. Clean commit `6e309d9e85c0fe79545dd0597255a514af5bc015`
produced
`benchmarks/results/vl-api-render-manifest-v0.1.0.json`, SHA-256
`a80e9977678606b0148f45008e2f389434618c8c9011d45af3415f61e71a54ca`.
All 18 non-stream cases equal the full server's prompt usage; the two stream
requests intentionally omit usage. The capture also proved that a former
one-token tool discrepancy was request-key ordering, not a vLLM accounting
rule.

Named VL `tool_choice` now follows the fixed vLLM split: thinking remains
enabled in the prompt, while generation is constrained to the selected
function's parameter schema. The currently frozen closed-object/one-required-
string schema is validated fail-closed. A byte-level incremental JSON grammar
admits only viable BPE tokens; EOS is admitted only after the object is
complete. The resident engine uploads the 248,320-byte mask before every token
selection, and both the LM-head lower-bound and exact-candidate stages exclude
disallowed rows. This remains certified top-1 selection over the allowed set,
not post-generation repair. Unsupported schemas return HTTP 400 before model
execution.

The formal runtime was most recently requalified from clean commit
`85fa597c782d28c05c51467060d8e03a8a47646e`, native binary SHA-256
`7fe6ceb07dbae924e8da5efa378b3a47ae7b0cd8e6fc023eff3c74d1298e67b2`.
In one resident run, all 20 successful API cases returned HTTP 200, all 10
invalid cases returned compatible HTTP 400, status matched `30/30`, finish
reason matched `20/20`, both tool and both SSE cases were complete, and every
one of the 18 media prompt counts and token-ID hashes matched the fixed render
manifest. Only the named forced case enabled structured decoding; its 349-token
prompt hash was
`c00ccaf4063b7a0eb5f30ca053d3484cd2658aac57d1ef7ee79d38287d940566`,
and 18 certified selections uploaded exactly 4,469,760 mask bytes. The sealed
result is `benchmarks/results/native-vl-capability-v0.1.0.json`, SHA-256
`53fe6babad27686a4b7a5eb27f800da247352b5bb3cb80a6148a76441e04defa`.

The exact usage triplet now matches fixed vLLM for `14/18` comparable
non-stream successes. The four remaining differences are explicit: the two
text-only residency probes preserve the `v1.5.1` text prompt, while forced and
auto tool requests differ only in completion length. Their media prompt counts
and hashes are already exact. These completion differences remain G2 work and
are not hidden by synthetic usage adjustments.

The five serving-oracle requests have their own real-HTTP render manifest so
the private numerical prompt and public serving prompt are no longer
conflated. Clean commit `2af339a31e6a9d982a90bd521f2558fc0f18ad5e`
captured image, video, multi-image, multi-video and mixed-media vectors at
82, 64, 186, 131 and 134 tokens respectively. All five differ from their
private-preprocessor vectors as expected. The sealed result is
`benchmarks/results/vl-serving-render-manifest-v0.1.0.json`, SHA-256
`6649347f8e9c606b4098a4b3d4fbe8e857f4acbd2518ee37f16744f6ce277336`.

The real-HTTP language attribution isolated its only material downstream drift
to short-sequence FLA recompute-W/U: the frozen q1024 schedule ran a two-warp
bucket launch, while fixed vLLM selected the four-warp kernel and applied its
boundary mask at logical `T`. The production path now embeds that exact gfx1151
image, launches `ceil(logical_T/64) x 32` with logical `T`, and restores the
resident bucket argument after the launch. On clean commit
`e402b3ac33fa942877aa3a6a820bbcc0f6af432f`, one native binary ran both
independent five-case prompt identities. All 2,420,736 final-norm BF16 elements
and all 20,858,880 selected-logit BF16 elements were bit-exact; all 84 rows had
top-1 equality and KLD zero. The deep real-HTTP multi-video run also retained
40/40 exact layer outputs and 5,240/5,240 exact router rows. This qualification
starts from frozen injected embeddings and M-RoPE positions, so it is numerical
language-boundary evidence rather than G4 serving timing. The hash-bound record
is `benchmarks/results/native-vl-language-full-v0.2.0.json`, SHA-256
`6de4f46b10a659c358350b1c42dec6e1d361f81c01926bf628b44b892fffb636`.

Clean commit `85fa597c782d28c05c51467060d8e03a8a47646e` then rebuilt the
formal resident runtime with binary SHA-256
`7fe6ceb07dbae924e8da5efa378b3a47ae7b0cd8e6fc023eff3c74d1298e67b2`.
One resident process matched all five real-HTTP prompt vectors and preserved
all five frozen private-oracle 8-token outputs, output text, finish reasons,
vision shapes and M-RoPE deltas. The same process passed all A/B/A,
same-path/same-shape invalidation, data/local equivalence, exact-prefix and
safe media/shape reuse checks. It emitted one READY and one stopped event,
loaded the model once, read no oracle tensors and reported no Python, Torch,
vLLM or Triton runtime. The sealed result is
`benchmarks/results/native-vl-serving-v0.1.0.json`, SHA-256
`01bacc552c0c93a6878efb745e976a187ff6c94458b7ef4fc8501d4ca087dc65`.

Formal binaries must retain the complete default AOT closure. A q1024-only
diagnostic build embedded 14 kernels from two manifests and correctly failed
when a longer tool decode referenced a q8192 image; the accepted builds embed
61 kernels from eleven manifests. The serving context may be q1024, but its
decode schedule still depends on the q8192 closure.

Processor results continue to use the resident 4 GiB/64-entry
content-addressed LRU. Only matching processor identity, media kind and content
digest can reuse decoded grids and BF16 pixels. Exact prefix hits skip vision
execution and pixel upload; changed text reuses only safe media/shape plans and
reruns vision/language normally.

The runtime build still installs and verifies the fixed vision-attention code
object, and the portable package contract includes it in the deterministic
bundle identity. These records close the frozen API/render and deterministic
resident-serving slices only. Execution of the sealed min/typical/max
capability envelope, image/video task quality, complete text no-regression,
paired performance, portable-package, second-host, soak and rollback
qualifications remain blocking. Therefore G1 through G5 all remain false.
