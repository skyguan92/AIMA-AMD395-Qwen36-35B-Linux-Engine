# Native VL implementation status

> Governing goal: [`NATIVE_VL_GOAL.md`](NATIVE_VL_GOAL.md)
>
> Frozen text baseline: `v1.5.1` (`6f3e669`)
>
> Current phase: Phase 2 — resident serving and product qualification
>
> Formal native candidate: `bd012874027defa528279a357609b713e9069df4`,
> binary SHA-256
> `fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`

This file is a live requirement-to-evidence index. A status of `in progress`
or `implemented` is not a release claim. Only evidence that satisfies every
blocking condition in the governing goal can move a gate to `passed`.

## Gate status

| Gate | Current state | Evidence required to pass | Next blocking action |
|---|---|---|---|
| G1 full VL functional parity | **passed** on the single formal candidate: all `14/14` requirement groups are covered, with no partial or missing group; the sealed audit binds the complete VL surface plus the passed G2 and G3 product-preservation records | complete image/video/mixed/conversation/API/tools/transport/residency native conformance results plus product preservation | none; preserve the sealed G1–G4 evidence while completing G5 |
| G2 VL correctness parity | **passed** on the formal candidate: five visual cases pass block 0/13/26 plus merger (`6,856,704/6,856,704` exact), and five private plus five independently rendered HTTP language cases pass `84/84` full-vocabulary rows with top-1 exact, KLD `0`, `20,858,880` selected logits exact and `2,420,736` final-norm elements exact | processor, vision/language boundary, full-vocabulary logits, deterministic generation, task quality and error results | none; preserve the sealed evidence while completing G5 |
| G3 text product no regression | **passed** on the exact formal `bd01287` candidate against the retained portable `v1.5.1` baseline: all 19 frozen cells pass at least six balanced pairs, with 12 pairs at q262143/output1; startup, correctness, MMLU-256, OpenAI, cache/product and doctor/memory requalification all pass | paired 19-cell, maximum-window, correctness, MMLU, API, cache, startup and memory requalification | none; preserve the sealed G1–G4 evidence while completing G5 |
| G4 native VL performance | **passed** on the formal candidate: all 20 fixed-reference-available cells pass ten alternating pairs for every applicable TTFT, total-latency, vision, prefill and decode gate; three fixed-reference-unavailable cells remain explicit non-passing capability records, and cache-enabled/disabled startup medians are `36.03 s`/`36.75 s` | sealed 23-cell coverage accounting, paired per-cell stage/memory records, same-process text decode controls and explicit reference-unavailable evidence | none; preserve the sealed G4 evidence while completing G5 |
| G5 native release product | the full runtime includes all qualified vision sources, loads the 333 visual tensors in one resident native process and survives the complete execution envelope; the general vision-attention code object remains an external hash-checked package artifact and the dense-image variant is hash-checked inside the embedded AOT registry, but final package qualification has not run | native-only package, security, isolated bundle, second-host, soak and rollback evidence | generate a clean product qualification containing both bound vision variants, then run isolated bundle, second-host, soak and rollback gates |

## Formal G1/G2/G3 requalification (2026-08-24)

Every current promotion artifact binds clean source commit
`bd012874027defa528279a357609b713e9069df4` and native binary SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`.
The consolidated G2 record is
`benchmarks/results/vl-correctness-v0.1.0.json`, SHA-256
`970fddd4eb0a91defd4398d863e38b31e4f3a8827df3ef2983d6523092207dbd`.
It passes every processor/envelope, vision, layer-0, independent layer-3,
private-language, HTTP-language, deep-router, generation, task-quality and
error-parity check without threshold widening.

The current visual record is
`benchmarks/results/native-vision-pipeline-current-head-v0.2.0.json`,
SHA-256
`49d090b58e42d372089e1374a4f6b12229f5298b2c45752f20d21e5c6afbad8e`.
Across the five frozen image/video/mixed shapes, block 0, block 13, block 26
and merger form 20 qualified boundaries. All `6,856,704` BF16 boundary
elements are finite and bit-exact, all repeats are deterministic, all 27
blocks execute, and the `884,736` merger output elements are qualified.

The language evaluator executes both five-case prompt identities. Layer 0 is
bit-exact for `1,198,080/1,198,080` output elements and passes `165/165`
diagnostic comparisons. Each independent layer-3 capture passes
`5,428,800/5,428,800` elements. The private and real-HTTP full-language runs
jointly pass `84/84` selected full-vocabulary rows, top-1 equality, KLD `0`,
`20,858,880/20,858,880` selected logits and
`2,420,736/2,420,736` final-norm elements. The deep multi-video diagnostic
passes 89 tensor comparisons, 40 router-layer sets and `5,240/5,240` router
rows.

## G1 coverage audit (2026-08-24)

The first requirement-to-evidence audit is sealed at
`benchmarks/results/native-vl-g1-coverage-audit-v0.1.0.json`, SHA-256
`47ec49546628a16cbb7a1d63962b26c595f301850bf2949dd4e1ff4caca038ee`.
It binds the governing goal, frozen reference surface, native 30-case surface,
23-cell execution result, resident-serving/cache evidence, visual pipeline,
language boundary, the new mixed/conversation reference/native pair,
the verified-HTTPS/sampling/cache reference/native pair, the image-I/O and
error/limit reference/native evidence, the long task-quality pair, the
generation/layer/prefill-state oracle closure, the current-HEAD native
generation result, cache-identity unit contract, the exact-candidate G3 text
product record and its generator.

The regenerated audit promotes G1. All fourteen requirement groups are fully
covered, no group is partial or missing, and its content-bound decisions mark
G1, G2 and G3 passed. `G1.2.3.product_preservation` is closed by the formal
G3 text/release no-regression record below.

This replaces the ambiguous instruction to “add more coverage” with named
cases and preserves every existing passing observation as evidence rather
than rerunning it without closing a gap.

## G3 paired text qualification (2026-08-24)

`scripts/qualify-native-paired-text-matrix.py` converts the G3 performance
language into an executable fail-closed protocol. It binds both executables by
SHA-256 and embedded source commit before the first model load, covers the
sixteen standard input/output cells plus all three maximum-window endpoints,
and stores each release/candidate process as a separate raw record. Odd pairs
run release then candidate; even pairs reverse the order. Fewer than five
pairs are rejected, and the default is six so both execution orders are
balanced.

Each cell is decided independently from the median paired ratio. Candidate
prefill/decode throughput must be at least `1.000x`; component and composed
request latency must be at most `1.000x`. The historical `0.97x` floor remains
an additional safety check and cannot mask a paired regression. The q8192
startup gate uses five or more candidate runs and requires their median to be
no greater than the frozen `44.90 s` ceiling. Its paired release ratio remains
an ordering/noise diagnostic, not a second blocking threshold absent from the
governing goal. In candidate text records, M-RoPE,
VL unified attention, logical projection and request-level VL workspace
metrics must all remain disabled or zero. HTTP media/vision-idle behavior and
`READY=1` vision readiness remain separate surface gates rather than being
inferred from the probe.

The runner writes a partial aggregate after every pair and resumes only raw
records whose engine hash, role, pair index, order, context and output sequence
still match. A final aggregate cannot pass when any one of the nineteen cells,
startup, identity, or text-path checks fails. The script and its threshold
semantics are covered by `tests/test_native_paired_text_matrix.py`. The runner
is itself content-bound by the sealed result, and the formal live matrix below
supplies the promotion evidence.

The candidate runtime binding covers all product-selected attention artifacts:
the short-context AOTriton provider and its exact runtime/image closure, the
long-context CK provider, the q16384 packed-GQA/CK hybrid, and the native vision
attention image. Each must occupy the path the engine would discover beside a
build binary or in a portable archive's sibling `lib/` directory. Paired runs
do not pass `--fmha-provider`; doing so would disable the frozen automatic
context routing and could benchmark AOTriton at q8192 or long context instead
of the actual product path. Each candidate raw record binds a digest of that
runtime closure and policy, so `--resume` cannot reuse an older run made with
the same engine binary but a different provider selection.

### Formal G3 result

The sealed decision is
`benchmarks/results/text-v151-nonregression-v0.1.0.json`, SHA-256
`29051b6b8d2b77b53897bfa9f9e3a47f74603d037d3855c2fcfce64c5af2b0e7`.
It binds candidate source
`bd012874027defa528279a357609b713e9069df4` and binary SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`
against the retained `v1.5.1` baseline source
`65c198415709dad6d046c247acab3dc9df2a95a0` and binary SHA-256
`a9f18771175757af080c8a1d8d7e3fb3906c9aa41b43a496686103b626f80262`.
All four cross-evidence identity/host checks and all 45 blocking checks pass.

The paired matrix covers all 19 frozen cells with at least six alternating
adjacent pairs per cell; q262143/output1 uses 12 pairs because its early
observations exposed larger order/clock noise. Every per-cell paired and
historical-floor check passes. The worst aggregate boundaries are
`1.0055095990x` minimum prefill throughput, `1.0256446969x` minimum decode
throughput and `0.9907389240x` maximum candidate/baseline total wall time. The
q8192 candidate startup median is `34,765.892 ms`, below the fixed `44,900 ms`
ceiling. All 144 unique role reports have continuous execution sequence
numbers, empty driver stderr and a valid report/load/stderr hash chain; the
retained tree contains no private machine paths.

The five companion gates also pass: nine-context correctness has top-1 exact
at every context, maximum KLD `0.0021737683` and an exact 128-token q8192
completion; MMLU-256 scores `218/256` versus the frozen reference's `216/256`
with zero invalid answers and `256/256` prompt hashes; the OpenAI feature run
serves all 14 requests in one model load with every text path idle; product
surfaces pass 12 prefix pairs, startup, memory and cache checks; and doctor
passes all `13/13` required checks. G3 is therefore passed; the subsequent
formal record closes G4 independently.

### Formal G4 result

The sealed paired result is
`benchmarks/results/vl-performance-v0.1.0.json`, SHA-256
`b3eeffb723a22a83f749e188c07c044da1bb1dfa1386e65718a6729dd4cfcf8a`.
It accounts for all 23 frozen cells and uses ten adjacent alternating
vLLM/candidate pairs for every one of the 20 fixed-reference-available cells.
Every applicable TTFT, total-latency, vision, prefill and decode gate passes.
The remaining three cells are retained as explicit
`reference_unavailable`/non-candidate-pass records rather than being removed or
counted as native wins: multi-image q128k, sampled-video q128k and multi-image
near the 262,144-token window.

Cache-enabled and cache-disabled candidate startup medians are
`36.028104836 s` and `36.7549729775 s`, both below `44.90 s`. The separately
sealed same-process text controls are
`benchmarks/results/vl-text-decode-retention-v0.1.0.json`, SHA-256
`0294e1c1ee8ed94027f80cfd417e89188b9ad0a34cb59f9ed83ae2ce2c8b745f`.
The three decode-bearing VL cells retain `1.0041781642x`, `1.0043523415x` and
`1.0043810954x` of the adjacent exact-shape text controls. G4 is therefore
passed without treating aggregate performance as a substitute for a failed
cell.

### Text/VL arithmetic isolation

The first current-head q1024 text requalification exposed a real semantic
regression even though top-1 remained unchanged: the candidate KLD was
`0.02421168131149516`, while the exact v1.5.1 binary on the same host and
oracle produced `0.000013813921765257257`. The candidate also replaced two AOT
RMSNorm launches per layer, changing the aggregate from `400 AOT / 291 native
pointwise` to `320 / 371`.

`scripts/bisect-native-text-correctness.sh` rebuilds every selected commit and
decides it with the immutable full-vocabulary oracle rather than a source-code
heuristic. The bisect identified `63c0a384bdd5429a98b1ba0ebc092e22121bc8bc`
as the first bad commit: its parent-side diagnostic commit remained below the
gate at `0.00496829703367488`, while `63c0a38` reached
`0.04396834423954292`. That change altered gated-RMSNorm arithmetic without
changing launch counts. A first isolation pass restored the release launch
mix but still produced KLD `0.011909960176010672`, proving that dispatch
topology alone is not a correctness certificate.

The resulting implementation rule is fail-safe: frozen text arithmetic is
the default, and current vLLM/VL arithmetic requires an explicit request-level
selection. Full-attention prefill uses the already-qualified M-RoPE boundary;
linear RMSNorm/gated-RMSNorm, shared-expert activation and MoE router semantics
use independent opt-in options propagated from a real multimodal request and
from dedicated VL qualification probes. Singleton decode selects the same
text/VL shared-activation split from its existing M-RoPE/current-projection
flag. Every such split must pass both text full-vocabulary requalification and
the current-head VL numerical chain before its evidence can be refreshed.

Arithmetic flags alone were not sufficient at q1024. The VL work had
recaptured the complete q1024 closure: manifest, launch schedule, FLA merge
shape and fused-MoE images all differ from the frozen text product. Trying to
compensate for that closure inside shared C++ code improved some logits while
remaining numerically wrong. The product therefore embeds two explicit q1024
owners:

- `q1024-output1` remains the current VL closure;
- `q1024-text-v151` is the checksum-identical frozen text closure, with manifest
  SHA-256 `93853b9f9837deba0a9e051bf5be4c516d74d1c5ea1a33e8e7e47ee81e914125`
  and schedule SHA-256
  `10565e59b0805ca407ef453caf72f3dfd254752d150903131e188527b910fb97`.

`generate-native-decode-registry.py --prefill-registry frozen-text` validates
the second owner against those fixed identities and its BT32 launch shape;
the current owner remains BT64. Both registries are embedded. Captured
`transient.N` names describe lifetimes inside one capture; they are not
semantic tensor identities across the frozen and current captures. The
previous name-based union changed 123 of 137 frozen binding offsets or
capacities and produced a repeatable q1024 prefill regression. The runtime now
keeps independent current and frozen view maps, preserving the exact
`674,086,144`-byte current layout and `669,875,456`-byte frozen layout. Because
requests execute serially, their common prefix can alias without merging a
single binding identity. Text requests select both frozen owners; VL requests
select both current owners. Conditions that interpret a schedule inspect the
selected closure's symbols rather than assuming that every q1024 schedule has
the same merge shape.

ROCm traces isolated the remaining first-request gap to the two q1024
hipBLASLt projection shapes. Their plans and private zero-buffer launches are
now materialized before `READY=1`; the temporary warmup buffers are released
before service readiness. On the exact `v1.5.1` release binary and candidate
SHA-256 `74f2157519df134415bb8b4353501296d45e990d58b5a1247e5ec3f2010462d2`,
the first physically isolated diagnostic used five adjacent alternating
q1024 pairs and qualified both standard cells. The
paired medians were `1.016477474553908x` prefill,
`1.0114218777931574x` output512 decode and
`1.010452851065543x` output1024 decode; composed latency ratios were
`0.9886920320042178x` and `0.9896282324783033x`. That result proves the layout
and warmup fix in isolation, but not its resident integration.

The first integrated owner made the larger current workspace the physical
allocation. Including its prompt-id upload, that owner is `674,090,240` bytes,
versus `669,879,552` bytes for the release-shaped frozen owner. The extra
`4,210,688` bytes were allocated before every later text workspace. ROCm traces
showed no extra q32768 dispatch after normalizing four semantically equivalent
renamed kernels; the request tails were exactly 1,162 kernels in the same
order. Moving vision residency behind the complete language topology recovered
most of the gap, but candidate SHA-256
`bb7db8fc6acf5dbe77f28d535ad490435b03536f6d1e00e1ebc8f3ca8941dcc6`
still failed five q32768 output1 pairs at `0.998382959085x` prefill throughput
and `1.001619659972x` latency. Its startup median was a passing
`42,399.074301 ms`; startup was not the cause.

The final layout reverses physical ownership. The frozen workspace now owns
the exact `669,879,552`-byte allocation in the original q1024 slot. The
current view aliases bindings below offset `668,730,624`; no binding crosses
that boundary. Bindings beginning with `transient.73` use a separately owned
`5,359,616`-byte tail allocated only after all text workspaces, decode/cache
state and prefill GEMM plans are resident. Physical q1024 residency is therefore
`675,239,168` bytes: only `1,148,928` bytes above the previous shared owner,
without dynamic reset or rebinding. Logical and physical byte metrics are
separate, so the shared prefix is counted exactly once.

Candidate SHA-256
`591f1f874b960d34db87029fcbcd4f0c1df04005698e5c7dd3b26b7925be9125`
(`39bf6082b42787593ca622eef20677fe660e70c1-native-vl-final-q1024-split-v33`)
passes the frozen q1024 full-vocabulary oracle with top-1 `248046`, KLD
`0.000013813921765257257`, and all 248,320 logits finite. Its five adjacent,
alternating q32768 output1 pairs retain every sample and pass at
`1.0009529894599638x` median prefill throughput and
`0.9990479178642767x` median latency. Candidate startup median is
`41,261.951616 ms`; all ten outputs match and every stderr is empty. The raw
q1024 record is under
`benchmarks/runs/native-correctness-q1024-20260817-q1024-split-v33/`; the raw
q32768 diagnostic and hash-bound summary are under
`benchmarks/runs/native-paired-q32768-20260817-q1024-split-v33/`. This closes
that diagnostic cell only. At that diagnostic point, the official final-binary
19-cell run and the other G3 gates had not yet run; the later formal G3 result
supersedes that boundary. The structural contract is covered by
`tests/test_native_text_closure_isolation.py`.

The separately owned tail must also survive serial switching between the two
q1024 views. A single q32768 resident v33 server therefore executed cold text,
cold image, different cold text, then the original exact image request without
reloading the model. Both text requests matched the frozen `OK` token hash;
the image requests matched the frozen `The` token, prompt hash, 256-patch and
64-visual-token oracle. The middle text request was a real cache miss and used
the frozen padded q1024 schedule after current-VL execution. The final image
then recovered an 82-token exact prefix and one media-cache hit with identical
output. All four requests reported `model_loads=1`, zero oracle reads and an
empty server stderr. The normalized, hash-bound 9/9-check record is under
`benchmarks/runs/native-vl-cross-request-20260817-q1024-split-v33/`.

The v33 formal matrix run cannot be promoted. Its five q16384 prefill
throughput ratios were
`0.9969017861205566`, `1.0021752243079622`, `0.9957627018956212`,
`0.9963280417366079` and `0.9983899030187419`; the blocking median was
`0.9969017861205566x`. Replacing additional q16384 hybrid-attention layers
with CK changed the full-vocabulary distribution: every tested layer set other
than the already qualified layer 39 exceeded the KLD limit. The attention
provider policy therefore remains frozen.

The remaining serial work was the v1.5.1-compatible text router. It formerly
used one thread to scan all 256 experts eight times. The v34 implementation
loads the BF16 row once and performs each maximum search with two wave32
reductions. Lower expert index wins equal-value reductions; the subsequent
source-order threshold gather, bitonic ordering, sequential `expf` denominator
and BF16 probability rounding remain unchanged. The VL/current router is not
modified.

Candidate SHA-256
`0abe1f5267d93ea26f30063dd37b71f9d08cc0165de8698503ccc60314398e6a`
(`39bf6082b42787593ca622eef20677fe660e70c1-native-vl-final-q16384-router-v34`)
reproduces the v33 q1024 certificate exactly: top-1 `248046`, 247,299 exact
logits and KLD `0.000013813921765257257`. It also reproduces the q16384
certificate exactly: top-1 `1`, one exact logit and KLD
`0.0021737683334905086`. Both have 248,320 finite logits and empty stderr.
Five adjacent alternating q16384 output1 diagnostic pairs produced throughput
ratios `1.019866231284811`, `1.022637066328316`, `1.018299641734469`,
`1.018973523525051` and `1.017964868732992`; the median is
`1.018973523525051x` and every paired wall ratio is below `1.0`. This closes
the isolated regression diagnosis only. A fresh v34 matrix also passed both
q1024 cells at `1.0481097991311443x` prefill, `1.0114559871890065x`
output512 decode and `1.0108736052011436x` output1024 decode; composed latency
ratios were `0.9869931566088613x` and `0.9883753217600323x`. That run was
intentionally stopped after its first complete q2048 pair because the embedded
v34 source identity was a diagnostic snapshot label. G5 requires a real,
immutable source commit, so no v33/v34 cell will be reused by the authoritative
exact-commit matrix.

The text correctness runner is also fail-closed before the first model load.
`--reference-correctness` must cover exactly the requested context set and
must bind every input-token period, oracle SHA-256 and the q8192 exact-output
fixture. That reference digest is copied into every resumable raw record, so a
run cannot resume across a changed frozen contract. Candidate, oracle and
qualification paths are normalized to explicit environment placeholders;
runtime provider, AOTriton runtime/image, q16384 hybrid, vision image, engine
and reference identities remain independently hash-bound.

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
That checkpoint closed oracle creation only; G2 was still blocked then on
native comparison, task-quality suites and error parity.

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
headroom. These source and CPU boundary checks were not target execution
evidence; at that checkpoint all 23 cells still had to run before G1 or G2
could pass.

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
the formal qualification client uses an explicit 2,400-second response wait so
a legal full-window multi-batch compute cell is not mistaken for a
socket-policy failure. A prior 7,200-second diagnostic established a safe upper
bound, but it is not the promoted record; release evidence is rerun and sealed
with the tighter 2,400-second value. Resident
vision plans retain a four-entry LRU surface but are also bounded to one
65,536-patch execution batch in aggregate. Plans at or above one fourth of
that budget are exclusive because each shape also owns fixed HIPBLASLt plan
state that a patch-only sum does not measure. On a cache miss, exclusive or
least-recently-used plans are released before constructing the replacement;
this avoids the transient double allocations exposed by both the
small/small/small/maximum-image and small/small/maximum-video target
sequences. Smaller shapes retain the four-entry LRU behavior.

The formal clean commit `bd012874027defa528279a357609b713e9069df4`
and native binary SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`
passed the full execution qualifier on `amd395`. All 23 HTTP
observations qualified: 17 accepted requests returned HTTP 200, six rejected
boundaries returned HTTP 400, and the accepted request indexes were
contiguous in one resident model load. The direct full-encoder probe executed
16 batches, 262,144 visual tokens and 1,048,576 patches twice with finite,
deterministic output. The near-window image and full-budget video requests
then served 245,760 and 258,048 visual tokens in 15 and 21 batches without
shrinking the 262,144-token model window. Their 310 and 330 full-attention
launches were fully accounted as FMHA launches, with zero launches through the
short unified-attention image; the ready record separately verifies the
automatic CK-primary plus terminal-AOTriton policy. Processor, vision and
server checks all passed with empty stderr. The sealed
artifact is `benchmarks/results/native-vl-envelope-v0.1.0.json`, SHA-256
`8585411f5d2178d6bd627f84143b041c3951e4e1eaf073e0294484572d7e67b3`.
That record closed the execution-envelope subgate only and did not by itself
pass G1, G2, G3, G4 or G5.

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
Unpadded M-RoPE chunks and continuation tails now reuse the existing qualified
rectangular text FMHA providers after Q/K have received their M-RoPE rotation.
For a padded continuation tail the admitted bucket shape preserves
bottom-right causal alignment for every live query row; only the initial
padded short-prompt boundary retains the q81-captured unified-attention kernel.
This preserves the path used by the exact language evidence while removing
the unsupported long-window launches. The envelope qualifier also leaves
long-context provider selection automatic and verifies the resulting CK plus
terminal AOTriton policy instead of overriding the service with the short
provider. M-RoPE continuation segments also select the generic rectangular CK
owner even when their compute workspace uses a q1024-q4096 bucket: the
standalone short AOTriton image rejects that short-query/long-prefix geometry.
The sealed full-envelope result above qualifies both the long-window dispatch
and the corrected provider policy.

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
merger and serving integration were still incomplete at that checkpoint, so
neither G1 nor G2 passed then.

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
the merger and serving integration were still unqualified at that checkpoint,
so neither G1 nor G2 passed then.

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

The formal current-candidate run supersedes that initial pipeline record. All
five processor-to-merger cases now match at blocks 0, 13 and 26 and at the
final merger: `6,856,704/6,856,704` BF16 boundary elements are finite and
bit-exact, relative L2 is zero, cosine is one and repeated outputs are
identical. Mixed image/video preserves the reference's two visual calls and
concatenates their outputs in request order. The 20-boundary record also
qualifies all `884,736` merger output elements and verifies that all 27 blocks
executed. The sealed current record is
`benchmarks/results/native-vision-pipeline-current-head-v0.2.0.json`, SHA-256
`4533048a5c6e6c078ebe5278f795b03460a663f2f22fbbf0d1753ed439602f20`;
the earlier multimedia and v0.1 pipeline files remain historical evidence.

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
layer were still blocking at that checkpoint. G1 and G2 therefore remained
false at that point; the later formal requalification supersedes that status.

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
to the remaining language layers, final norm, lm_head and logits. At that
checkpoint G3 paired text qualification and G4 serving performance were also
blocking; the formal evidence now closes G1–G4, while G5 remains current.

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
`bd012874027defa528279a357609b713e9069df4`, native binary SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`.
In one resident run, all 20 successful API cases returned HTTP 200, all 10
invalid cases returned compatible HTTP 400, status matched `30/30`, finish
reason matched `20/20`, both tool and both SSE cases were complete, and every
one of the 18 media prompt counts and token-ID hashes matched the fixed render
manifest. Only the named forced case enabled structured decoding; its 349-token
prompt hash was
`c00ccaf4063b7a0eb5f30ca053d3484cd2658aac57d1ef7ee79d38287d940566`,
and 18 certified selections uploaded exactly 4,469,760 mask bytes. The sealed
result is `benchmarks/results/native-vl-capability-v0.1.0.json`, SHA-256
`ca16fe64b912623b0528ccfcb086d4757e6c0a5d8907b18316dec39dc27b4290`.

The exact usage triplet matches fixed vLLM for all `16/16` VL successes and
for `16/18` successes when the two text-only residency diagnostics are
included. Those two diagnostics intentionally preserve the `v1.5.1` text
prompt rather than the vLLM VL server's thinking-enabled text prompt; their
product comparison is owned by G3. Finish reason remains exact for `20/20`,
and no VL usage difference remains.

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

Clean commit `bd012874027defa528279a357609b713e9069df4` rebuilt the
formal resident runtime with binary SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`.
One resident process matched all five real-HTTP prompt vectors and preserved
all five frozen private-oracle 8-token outputs, output text, finish reasons,
vision shapes and M-RoPE deltas. Before those oracle requests, the same process
ran twelve cache observations: image same-path A/B/A, same HTTP URL with live
response-byte A/B/A mutation, image and video local/data equivalence, prompt
variation, and mixed image/video cold/exact replay. All 21 fail-closed cache
predicates passed, including content misses, safe shape-plan reuse, exact
prefix/media hits and token-exact video/mixed hit/miss outputs. It served all
17 requests, emitted one READY and one stopped event, loaded the model once,
read no oracle tensors, wrote zero stderr bytes and reported no Python, Torch,
vLLM or Triton runtime. The sealed result is
`benchmarks/results/native-vl-serving-v0.1.0.json`, SHA-256
`c3fe421f72ac8f47d6860ad63ab374e45bbeb82879a20a33f57c10d196d6d554`.

Clean commit `1842c8f6d281d6c8e91563205cda3fb66908d8a1` then froze the
remaining mixed/conversation extension on the same fixed vLLM runtime. The
five reference prompts cover image-video-image and video-image-video
interleaving, prior/current video history, prior-turn mixed history and mixed
SSE. Their prompt lengths are respectively `240`, `214`, `160`, `153` and
`155`; all returned HTTP 200, the four non-stream usage triples equal the
independent render lengths and the stream returned exactly four tokens. The
sealed reference is
`benchmarks/results/vl-g1-mixed-conversation-reference-v0.1.0.json`, SHA-256
`e769446684c4f69b7a29dde163d050a541449ffddf0996ecbb48510a0c493451`.

The formal candidate binary SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`
then replayed all five cases in one resident model load. Every HTTP status,
prompt-token count and hash, ordered image/video count, M-RoPE boundary,
generated text, finish reason and usage signature matched the reference; mixed
SSE aggregated to the same `The user wants a` text. The server served `5/5`,
reported no Python/Torch/vLLM/Triton runtime and wrote zero stderr bytes. The
sealed native result is
`benchmarks/results/native-vl-g1-extension-v0.1.0.json`, SHA-256
`47cb7322a5d72e48137fa9ecee68db92edf091636151d28642bf9fb4609d284d`.
That record closed the mixed, conversation and OpenAI API coverage groups
without by itself promoting G1 or any later gate.

Clean commit `82fc48f7d4a0af1f1b30e9abfd26d78f73780715` then closed the
request-level video sampling mismatch exposed by the frozen vLLM source. A
request mapping is shallow-merged within each named modality over the launch
defaults: absent modalities and empty modality objects are no-ops, while a
provided key overrides only that key. Native accepts the vLLM-compatible
OpenCV `fps`, `num_frames` and backend fields, rejects unsupported or malformed
mappings, and samples by the same bounded floor/linspace frame-index rule.
Processor identity version 2 binds the effective per-request video policy, so
video bytes, sampling policy, media order and placeholder spans all participate
in both media and prefix cache namespaces. Its then-default identity was
`d5c32c48a557b75c8192a824de0992464bb307890cac0cc01f0890cfccd874d2`.

The fixed vLLM runtime then froze ten transport/cache cases: verified-CA
HTTPS image input; default, 1-fps and six-frame video sampling; the same
mutable HTTP video URL serving content A, content B and corrupt bytes; and
image/video mixed ordering plus mutation. All ten requests matched their
independent render token vectors, including three distinct sampling prompt
hashes and distinct A/B and mixed-order identities. The sealed reference is
`benchmarks/results/vl-transport-cache-reference-v0.1.0.json`, SHA-256
`d98433b1be1cd8264947c116073c88287d8181a0ad8a7e60cce868238b48607a`.

The loopback CA used for that capture is an ephemeral test credential, and its
private key is intentionally absent from the reference. Native replay may use
a freshly generated loopback CA: the result records both the capture and replay
certificate hashes, while a successful HTTPS fetch proves that the replay CA
was actually trusted. This keeps the reference behavior hash-bound without
making qualification depend on an expired or retained private credential.

The formal candidate binary SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`
replayed 17 cache-enabled and 11 cache-disabled observations in two resident
model loads. Every observation matched reference status, request contract,
render prompt, generated content, finish reason and usage. The enabled run
proved HTTPS cold/hit reuse, video content and sampling A/B/A recovery,
conservative mixed reorder/mutation misses and error non-pollution. The
disabled run remained all-miss while preserving 8-token outputs, usage and
the corrupt-video error status/payload. Both servers had empty stderr and
reported no Python, Torch, vLLM or Triton runtime. The sealed result is
`benchmarks/results/native-vl-transport-cache-v0.1.0.json`, SHA-256
`d7a9d9eb4109282259246fe70caafd90abee53c5cc7580d7eacb83c5655f7cfb`.
That checkpoint promoted only the transport, cache-identity and
cache-invariance audit groups; G1 and every later gate were still false then.

Clean implementation commit `5339e1d7f71960e175ce17e97012751238057675`
then completed the frozen image-I/O and merge semantics. RGBA images composite
onto white by default and accept the fixed vLLM
`image.rgba_background_color` request field. Processor identity version 3 binds
that effective background as well as video sampling; its default identity is
`9be676b2d0cefbe030d61e1d89776df6c7ba28d0d86ca752c60eca3ec60a9280`.
The old product-specific 768-second duration cap is disabled because the fixed
OpenCV reference accepts finite sparse videos beyond it; compressed-byte,
selected/source-frame, decoded-pixel and decode-wall limits remain fail-closed.
The derived 12-frame, 0.002-fps fixture therefore exercises a 6000-second
duration without weakening the remaining resource bounds.

Clean evidence commit `7642995e772fbdc8ae763bcffb90f2da852987f0`
froze the fixed-vLLM image-I/O oracle and ten request cases. The image-I/O
artifact, SHA-256
`56dff264132bdb49a470bd0af863df6ddca0e22f83da43143ad85267c0a30e98`,
proves exact white and red compositing. The API reference, SHA-256
`36ca8fe48ce785a52a8123d72be2d9271e3a29731222672bc30989250f2e56f1`,
contains five accepted requests (the two RGBA policies, video default/empty
mapping and the long sparse video) plus empty image/video, unreachable,
64-MiB-plus-one and timeout errors. It preserves the raw vLLM contracts:
empty/oversize are HTTP 400 `BadRequestError`; unreachable/timeout are HTTP
500 `InternalServerError`.

These artifacts remain historical reference identities: later native replays
validate each clean capture identity, integrity seal, and bound source-file
hash, but do not require the current implementation commit to equal the
capture commit. The native result separately binds its own exact source commit
and binary hash.

The formal candidate binary SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`
then replayed 13 observations in one resident model load. All eight accepted
requests were HTTP 200 and reference-exact for request/render/usage; all five
invalid external-media requests preserved the existing v1.5.1 fail-closed HTTP
400 `invalid_request_error`/`bad_request` product shape with an explicit
compatible-category check against the separately frozen vLLM result. The run
also proved RGBA A/B/A cache recovery, empty-video-mapping reuse, long-duration
execution and error cache non-pollution. It wrote zero stderr bytes and reported
no Python, Torch, vLLM or Triton runtime. The sealed native result is
`benchmarks/results/native-vl-error-limits-v0.1.0.json`, SHA-256
`daddd7ca67985d19744fc0409b4ae2ada5b8cd7c9f6c93b803d1e47b4dd29814`.
This closes only the error-parity audit group; it does not promote a product
gate.

Formal binaries must retain the complete default AOT closure. A q1024-only
diagnostic build embedded 14 kernels from two manifests and correctly failed
when a longer tool decode referenced a q8192 image; the accepted builds embed
70 kernels from eighteen manifests. The serving context may be q1024, but its
decode schedule still depends on the q8192 closure.

An FMHA provider path alone is not a qualification input. Every native VL
qualifier resolves the provider-adjacent AOTriton 0.11.1 runtime and the single
frozen gfx1151 code object before launching a workload, rejects a missing,
changed or expanded code-object set, records all three artifacts in the result,
and includes the closure validator in its source identity. This prevents a
provider-only staging directory from silently selecting an incomplete runtime
through host search paths.

The capability result keeps two usage views instead of conflating product
contracts. `vl_reference_usage_exact` was the blocking G1/G2 comparison for
requests carrying image/video/mixed surfaces and is now satisfied. The
all-surface diagnostic also
includes the two text-only residency sentinels. The fixed vLLM VL server leaves
thinking enabled for those sentinels (15/18 prompt tokens), while the immutable
`v1.5.1` text product disables thinking (17/20 prompt tokens). Changing native
text rendering to erase that diagnostic would violate G3. Their exactness is
therefore proved against the paired `v1.5.1` binary in G3, while the qualifier
continues to report the vLLM difference transparently.

Processor results continue to use the resident 4 GiB/64-entry
content-addressed LRU. Only matching processor identity, media kind and content
digest can reuse decoded grids and BF16 pixels. Exact prefix hits skip vision
execution and pixel upload; changed text reuses only safe media/shape plans and
reruns vision/language normally.

The runtime keeps two bit-exact vision-attention occupancy variants. The
external WPE3 image remains the general and video-safe package artifact. A WPE6
image is verified from the embedded AOT registry and selected for image-only
requests whose batch has 1024 through 4096 patches across single-frame grids.
Smaller and larger image batches, video and mixed requests remain on WPE3 even
when sampling produces temporal-depth-1 grids. Warmed execution entries bind
the attention image SHA-256 in addition to patch count and sequence boundaries,
so patch/GEMM/merger resources may be shared without allowing an attention
executable to cross the image/video policy boundary. The portable bundle
identity covers the external image
directly and the embedded variant through the native binary and registry
manifest. These records close the frozen API/render,
deterministic-resident-serving and min/typical/max execution-envelope slices.
The task-quality and deterministic-generation slices are closed by the
12-case and current-candidate results below. Complete text no-regression is now
sealed. Portable-package, second-host, soak and rollback qualifications remain
blocking. G1 through G4 are passed; G5 remains false.

Generation attribution now binds time as well as tensor identity. Each layer
capture writes the full-vocabulary FP32 rows for both its target output index
and output index 1 beside the corresponding boundary tensors. Probe cases carry
`reference_logits_output_index` and `reference_decode_output_index`; both must
equal the expected-prefix length before model execution. This rejects an
otherwise plausible but invalid comparison between first-decode boundaries and
later divergence logits. The earlier cross-index diagnostic is discarded as
evidence. A correctly aligned preliminary replay showed both output-index-1
token pairs and all 41 language boundaries bit-exact; target-index state and
logits still require exact-commit recapture and qualification, so no gate is
promoted by that observation.

The linear-attention diagnostic observer is layer-parameterized while retaining
layer 0 as the promotion-oracle default. A diagnostic manifest records the
selected layer in each attention and tail boundary set, and the native case
must bind the same validated non-full-attention layer. This permits resident
conv/recurrent state attribution at the first non-exact layer without widening
the product runtime path or comparing an isolated layer seeded from oracle
state.

Diagnostic capture can select the observer independently for each frozen case
with repeated `--diagnostic-linear-attention-layer CASE_ID=LAYER` arguments.
The mapping must cover both cases exactly, rejects full-attention layers, and is
sealed into the control plane separately from the historical
first-divergence preset. This is required when the first non-exact layer moves
across output indices: at aligned output index 3, layer 5/10 and all their
resident-state boundaries were exact, while the first non-exact whole-layer
rows occurred later at layer 20/13. Those observations are attribution
evidence only and do not promote a gate.

The full-attention observer has the symmetric
`--diagnostic-full-attention-layer CASE_ID=LAYER` control. Its exact two-case
mapping rejects non-full-attention layers, duplicates, incomplete mappings,
and simultaneous use of the historical first-divergence preset. Explicit
selection captures QKV, cache, unified-attention output, projected attention,
residual, post-attention norm, and routed-MoE boundaries at the selected layer,
and seals both the explicit and effective maps into the diagnostic manifest.

Current-vLLM singleton Gemma RMSNorm requires a different reduction topology
from the already-qualified multi-row prefill kernel. For a contiguous
`[1,2048]` FP32 `pow(2).mean(-1)`, the pinned PyTorch/ROCm implementation uses
one 512-thread vec4 block on gfx1151: local four-value accumulation, shared
combines at offsets 256, 128, 64 and 32, then an ascending-offset wave32
shuffle. The q1024 prefill path remains 32-by-16 vec4 because its many output
rows change ATen's block geometry. Reusing that prefill tree for singleton
decode produced a variance one FP32 ULP low on an observed layer-20 row; its
inverse RMS moved one ULP high and crossed a BF16 rounding midpoint in exactly
one output element.

Native decode now dispatches `token_count == 1` to the dedicated singleton
tree and uses it for both input and post-attention Gemma RMSNorm in linear and
full-attention VL layers. The historical AOT norms remain unchanged for the
text path. Aligned diagnostic replays on clean `b76653f` cover output indices
1, 2, 3 and 6 for both frozen tool cases. Every replay has exact prefix,
selected token and top-1; all compared whole-layer rows are 41/41 exact. The
selected internal sets are also exact: 13/13 linear and 15/15 tail boundaries
where present, plus 24/24 full-attention boundaries at output indices 3 and 6.
These are attribution scouts, not promotion evidence; the divergence-index
oracle and formal qualification must be recaptured on the final source commit.

## Task-quality qualification (2026-08-16)

The frozen corpus contains six image and six video tasks with deterministic
rendered media, rubric scoring, greedy decoding (`temperature=0`, up to 192
completion tokens) and per-case plus per-modality reference floors. The
fixture manifest SHA-256 is
`5ae5b4e3b0b6e67df5599efc93e2533b6f883fdccc36eaa09e78a5dca2b40e25`;
the fixed-vLLM reference SHA-256 is
`51b3d95e3ce420584d765350bfe6b73f76d5786a8d9d629cf7c3e69ac11b8bce`.

The formal native runtime built from clean commit
`bd012874027defa528279a357609b713e9069df4`, binary SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`,
served all 12 cases in one resident model load with empty stderr and no Python,
Torch, vLLM or Triton runtime. All blocking per-case checks passed. Native and
reference aggregate scores are identical: image `1.000000`, video `0.947368`.
The sealed result is
`benchmarks/results/native-vl-task-quality-v0.1.0.json`, SHA-256
`f1ffdff8e0c5f9000e2bb1fae576bf6dd3a158f2a169732c7ffb5d07bef4c8e3`.

Task quality and deterministic generation parity remain separate contracts.
Prompt vectors, finish reasons, usage, generated content and output-token
vectors are now exact for all `12/12` cases. The task-quality and long-greedy
reference-exact decisions are both qualified. This evidence feeds the sealed
coverage audit; after the formal G3 product-preservation pass, G1 is passed.

## Current-HEAD processor-to-output qualification (2026-08-16)

The two frozen tool-generation cases now have a durable attribution closure:
the generation oracle SHA-256 is
`954c6e55389cd90390cb517224df14719f2556555ce7bf44571cae1ad1812888`,
the 31-MB layer-oracle manifest SHA-256 is
`70cec2c7884b5e641884037212a34f6ffc6ac1944af08d59c19dc90353915188`,
and the 129-MB prefill-state manifest SHA-256 is
`ec4d23bef4058dd5f0189f703214caeb006159d3c4937b7e9ea14ba9bfc82782`.
All referenced binary components are committed under `benchmarks/oracles` and
revalidate without errors.

The formal native binary built from clean commit
`bd012874027defa528279a357609b713e9069df4`, SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`,
ran both cases in one resident model load. All 62 qualification checks passed:
both shared prompt prefixes; 120 prefill-state tensors; 82 whole-layer rows;
26 linear-attention, 30 routed-tail and 12 full-attention internal boundaries;
and both 248,320-element full-vocabulary rows were finite. Every compared
boundary element was bit-exact, both native top-1 and selected tokens matched,
and KLD was `0.0000657115` and `0.0000020521`, below the fixed `0.005` gate.
Stderr was empty.

The sealed result is
`benchmarks/results/native-vl-generation-current-head-v0.1.0.json`, SHA-256
`e6f79c105ba669f9a2c1d5c037bbe06d106d66235254c7ebb45282d9d5b74c8b`.
It sets `g1_generation_closed=true` for the two frozen tool divergences and
closes the audit's current-HEAD model-semantics gap. The separate long-greedy
task-quality replay is also `12/12` exact. The formal product-preservation
record closes G3 and G1, and the formal paired performance record closes G4;
G5 is the next blocking gate.

## Positive-temperature product extension (2026-08-23)

The G1-G5 promotion fixtures remain greedy exactly as frozen by
`NATIVE_VL_GOAL.md`. After that boundary, the product also admits finite
`temperature` values in `(0,2]`, `top_p` in `(0,1]`, and an optional
non-negative seed. `temperature=0` retains the certified top-1 implementation
without a sampling launch or logits transfer.

The stochastic path evaluates the raw BF16 LM-head weights against the final
normalized hidden row for all 248,320 vocabulary entries with the same native
wvSplitK arithmetic used by the exact certificate candidates. It reuses dead
candidate scratch, copies the 496,640-byte BF16 distribution to the resident
host owner, and applies temperature/nucleus selection with a specified
SplitMix64 upper-53-bit stream. The effective seed and byte/time counters are
published per request.

The dedicated qualification replays identical seeds for text, VL and SSE,
requires a changed sequence for an adjacent seed, verifies five fail-closed
parameter boundaries, rechecks the zero-temperature fast path, and requires a
single model load plus empty stderr. This extension does not change the greedy
correctness or G4 performance protocol.
