# Performance and correctness

## v1.5.0 portable native full envelope

The v1.5 portable runtime covers the complete published batch-1 matrix without
loading Python, PyTorch, vLLM, Triton, Transformers or a host ROCm userspace.
Promotion uses three same-configuration runs, or two expensive runs when the
first pair is within 3%.

For each standard context, a process ran cold output512 first and then restored
the exact cached prompt state for output1024. Cold prefill is context-only;
decode is measured at the requested output length.

| Input tokens | output512 prefill | output512 decode | output1024 prefill | output1024 decode |
|---:|---:|---:|---:|---:|
| 1,024 | 1630 | 34.00 | 1630 | 33.99 |
| 2,048 | 1685 | 33.86 | 1685 | 33.86 |
| 4,096 | 1572 | 33.26 | 1572 | 33.25 |
| 8,192 | 1656 | 32.29 | 1656 | 32.28 |
| 16,384 | 1438 | 30.79 | 1438 | 30.79 |
| 32,768 | 1365 | 28.23 | 1365 | 28.23 |
| 65,536 | 1176 | 24.68 | 1176 | 24.68 |
| 131,072 | 868.2 | 19.60 | 868.2 | 19.60 |

Units are tokens per second. Maximum-window results were:

| Input/output | Prefill tok/s | Decode tok/s | Prefill retention | Decode retention |
|---:|---:|---:|---:|---:|
| 262143/1 | 556.5 | n/a | 1.4957 | n/a |
| 261632/512 | 560.5 | 14.05 | 1.3855 | 1.0091 |
| 261120/1024 | 535.8 | 14.04 | 1.3185 | 1.0049 |

All 19 cells pass the independent `0.97x` prefill/decode floor. The minimum
prefill retention is `1.013`; the minimum decode retention is `0.9858`.
Against v1.4.1, the worst median prefill/decode changes are `-2.259%` and
`-0.1280%`; both stay inside the 3% measurement protocol band.

Full-vocabulary correctness is bound to the final engine SHA-256:

| Input | KLD | Top-1 | Gate |
|---:|---:|:---:|---:|
| 1,024 | 1.381e-5 | match | < 0.005 |
| 2,048 | 9.298e-5 | match | < 0.005 |
| 4,096 | 3.132e-5 | match | < 0.005 |
| 8,192 | 1.399e-5 | match | < 0.005 |
| 16,384 | 0.002174 | match | < 0.005 |
| 32,768 | 0.0004924 | match | < 0.005 |
| 65,536 | 4.573e-6 | match | < 0.005 |
| 131,072 | 5.104e-6 | match | < 0.005 |
| 261,632 | 1.105e-5 | match | < 0.005 |

The frozen q8192 completion fixture also matched all 128 expected token IDs,
with output-token SHA-256
`aa910692fd03ed4a8e89c04497751e3a28eee36c6148237f7e97c74a6dd68201`.

Three fresh q8192 HTTP processes reached readiness in `53.28`, `51.16` and
`46.67` seconds. The `51.16 s` median is below the frozen `51.41 s` ceiling.

At q32768/output512, exact-prefix reuse reduced TTFT from `24.08 s` to
`9.170 ms` (`2626x`) and retained `1.0001` of cold decode throughput. The
output-token hash was unchanged.

The resident HTTP run used one model load, served a cold q8192 request and an
exact repeat, reduced TTFT from `4.943 s` to `5.075 ms`, retained the output
hash, exposed health/model endpoints and exited through `POST /shutdown`.

The OpenAI feature lifecycle run used live HTTP/1.1 chunked SSE. A 16-token
cold prompt produced first content at `508.5 ms`, completed at `570.3 ms`,
and matched both text and generated-token hashes with its exact-cache
non-stream response. A normal 36-token next-user turn and an unrelated short
request after long-context work remained isolated with HTTP 200. Exact raw-token
requests at q1024/q2048/q4096/q8192 selected the matching resident AOT bucket,
and an A/B/A sequence restored the first of four LRU entries exactly. Stream
and non-stream tool requests both produced
`get_weather({"city":"Paris"})`, assistant/tool history produced the expected
final answer, an unavailable forced tool returned HTTP 400, and a reset client
connection left the one-load resident server healthy.

### Capability regression scorecard

The exact v1.5 native executable ran the frozen answer-only MMLU-256 subset in
one resident HTTP process with batch size 1, greedy decoding and identical
pretokenized prompts. It scored `216/256` (`84.375%`) with zero invalid
answers, exactly matching the frozen GB10 vLLM reference score. All 256 prompt
token hashes matched; 250/256 complete output-token hashes were identical and
251/256 parsed answers were identical. The five answer changes had zero net
score effect. The public scorecard contains item identifiers, answers, hashes
and aggregate metrics, but no prompt text or prompt token IDs.

This is a deterministic regression subset, not an official leaderboard score.
The pinned upstream model card separately reports `85.2` MMLU-Pro, `93.3`
MMLU-Redux and `92.7` AIME26 for Qwen3.6-35B-A3B under its published methods;
those benchmarks and sampling protocols are not interchangeable with this
greedy MMLU-256 gate. See the
[upstream model card at the qualified revision](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/README.md).

Exact components, per-run values, baselines, ratios and decision boundaries
are embedded as `share/aima/qualification.json` and mirrored after release to
`benchmarks/results/native-portable-product-v1.5.0.json`.
Its hash-bound summaries and raw reports are published under
[`benchmarks/runs/`](../benchmarks/runs/). `make verify-evidence` checks every
referenced report; `make package-evidence` emits a deterministic public
evidence archive and SHA-256 sidecar under `dist/`.
The v1.3 result remains available as a historical release record in
[`native-portable-product-v1.3.0.json`](../benchmarks/results/native-portable-product-v1.3.0.json).
The v1.4.0 result is retained in
[`native-portable-product-v1.4.0.json`](../benchmarks/results/native-portable-product-v1.4.0.json).
The archive packager rejects unresolved ELF dependencies and absolute RUNPATHs.
The remaining host requirements are the Linux kernel AMDGPU/KFD driver,
device nodes, x86-64 and `gfx1151`.

## Frozen v1.1 baseline

The wider matrix below comes from the qualified v1.0.0 model-math engine,
which is byte-identical in v1.1.0. Full-precision values are available in
[`benchmarks/results/v1.0.0.json`](../benchmarks/results/v1.0.0.json); direct
checkpoint-loading values are in
[`benchmarks/results/v1.1.0.json`](../benchmarks/results/v1.1.0.json).

## Cold prefill and decode

Each positive-output cell ran twice in mirrored order inside one resident
process. First pairs were within 3%, so no third replicate was required; the
median decides.

| Input tokens | output512 prefill | output512 decode | output1024 prefill | output1024 decode |
|---:|---:|---:|---:|---:|
| 1,024 | 1416 | 33.56 | 1407 | 33.55 |
| 2,048 | 1540 | 33.57 | 1530 | 33.62 |
| 4,096 | 1552 | 33.16 | 1553 | 33.17 |
| 8,192 | 1591 | 32.12 | 1587 | 32.18 |
| 16,384 | 1409 | 30.90 | 1414 | 30.93 |
| 32,768 | 1169 | 28.62 | 1168 | 28.63 |
| 65,536 | 927.1 | 24.82 | 926.0 | 24.83 |
| 131,072 | 646.6 | 19.64 | 646.6 | 19.65 |
| Maximum valid | 404.5 at 261,632 | 13.92 | 406.4 at 261,120 | 13.97 |

Units are tokens per second. The maximum valid input differs because total
context is fixed at 262,144 tokens.

The exact `262143/1` coverage request reached `372.1` prefill tok/s with
9.279 GB free at completion.

## Standard-checkpoint startup

The v1.1.0 default loads the original 26 Safetensors shards from one storage
device. Three fresh resident processes measured:

| Run | Load start to API ready | Checkpoint preload | Native payload GiB/s |
|---:|---:|---:|---:|
| 1 | 50.51 s | 44.19 s | 1.605 |
| 2 | 51.41 s | 45.03 s | 1.576 |
| 3 | 51.94 s | 45.64 s | 1.550 |

The median load-to-ready time was `51.41` seconds, with `2.782%` full spread.
All runs read 26/26 shards with O_DIRECT and used zero buffered fallbacks. The
native loader transferred 69,321,221,376 active bytes into 693 independent
device tensors, matched the qualified GPU XOR and sum, and introduced zero
extra full-weight copy bytes. Peak process allocation was 73,217,943,040 bytes.

A first public one-token request completed successfully after direct loading
with 16 prompt tokens, one completion token, `length` finish reason and content
`Here`. This is a resident-service smoke check; full model correctness remains
bounded by the unchanged-engine evidence below.

## Historical native-foundation measurements

Before the v1.2 profile was integrated, the dependency-removal path separately
moved ownership and direct loading
of all 693 tensors out of Python and PyTorch and added a statically linked native
tokenizer. The exact executable plus its colocated ROCm userspace and gfx1151
bitcode bundle is 219 MiB. Three fresh processes measured weight-ready wall
times of `44.02`, `42.46` and `46.68` seconds; the median was `44.02` seconds.
The comparable v1.1.0 Torch-owned checkpoint-preload median is `45.03` seconds,
so the current native stage remains `1.023x` faster and reduces its median wall
time by `2.240%`. The 9.574% spread is recorded as storage variance; the median
still decides the product-baseline gate.

An additional isolated-HOME run loaded the complete checkpoint with 26/26
O_DIRECT shards and the exact GPU payload checksum while recording zero
successful opens under a system `/opt/rocm`. This proves the dependency closure
for the weight-owning foundation only. It does not qualify model execution,
API-ready startup, correctness, prefix cache or any context-matrix cell. Exact
values and artifact hashes are in
[`native-foundation-v0.1.0.json`](../benchmarks/results/native-foundation-v0.1.0.json).

## Optional striped startup and first user request

The v1.0.0 two-device startup-image path remains available as an optional
v1.1.0 mode. Two fresh processes measured command launch to HTTP readiness at
`27.53` and `27.10` seconds: median `27.31` seconds, spread `1.560%`.

Their first cold q8192/output512 requests measured:

| Run | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 1 | 1567 | 32.17 |
| 2 | 1553 | 32.20 |

Striped startup includes imports, independent device allocations, both image
lanes, model setup and load-only kernel priming. It excludes startup-image
generation, which is a one-time installation operation and consumes an extra
64.56 GiB on disk.

## Prefix cache

Three isolated q32768 cycles seeded 32,767 tokens and then requested a strict
32,768-token extension.

| Cycle | TTFT speedup | Decode retention |
|---:|---:|---:|
| 1 | 74.34x | 0.9997 |
| 2 | 110.1x | 1.0011 |
| 3 | 111.1x | 1.0005 |

Median TTFT speedup was `110.1x`; minimum decode retention was `0.9997`.
Exact-prefix and one-token strict-prefix hits were separately covered by the
76-check live HTTP conformance test.

## Correctness

The selected-reference gate compares cached-decode distributions with official
AMD395 vLLM:

| Check | Result | Gate |
|---|---:|---:|
| KLD | 0.0002768 | < 0.005 |
| Top-1 agreement | 1.0 | = 1.0 |
| Completion identity | exact 128 tokens | exact |
| HTTP output-one | 8/8 | all |
| Usage/stop/prefix conformance | 76/76 | all |

The terminal stop token is counted in usage but omitted from assistant content.

## Context-decay boundary

Every output512/output1024 adjacent prefill and decode ratio passed the
project's blocking floor, defined as 90% of the corresponding sibling-D275
ratio. Both maximum-valid endpoint quartets also passed.

The stricter raw sibling-D275 decay ratios were not all reached. They remain a
published engineering target and are deliberately not represented as complete.

## Interpretation limits

- These results are specific to the hardware, runtime, model hash and fixed
  policy in this release.
- Startup is storage-topology dependent. The direct result used one storage
  device; the striped result used two physical NVMe devices and is not a claim
  that generated images improve resident inference performance.
- They are not multi-user throughput numbers.
- Prefix figures require exact token-prefix identity and the compatible runtime
  contract.
- Rebuilt native binaries or a different AOTriton/CK/ROCm stack require fresh
  correctness and performance qualification.
