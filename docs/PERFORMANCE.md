# Performance and correctness

All numbers in this document come from the qualified v1.0.0 source and fixed
batch-1 configuration. Full-precision values are available in
[`benchmarks/results/v1.0.0.json`](../benchmarks/results/v1.0.0.json).

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

## Startup and first user request

Two fresh processes measured command launch to HTTP readiness at `27.53` and
`27.10` seconds: median `27.31` seconds, spread `1.560%`.

Their first cold q8192/output512 requests measured:

| Run | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 1 | 1567 | 32.17 |
| 2 | 1553 | 32.20 |

Startup includes imports, independent device allocations, both image lanes,
model setup and load-only kernel priming. It excludes startup-image generation,
which is a one-time installation operation.

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
- They are not multi-user throughput numbers.
- Prefix figures require exact token-prefix identity and the compatible runtime
  contract.
- Rebuilt native binaries or a different AOTriton/vLLM stack require fresh
  correctness and performance qualification.
