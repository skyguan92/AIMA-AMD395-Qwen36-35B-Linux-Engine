# v1.0.0 release provenance

## Immutable engine boundary

The production engine is:

```text
benchmarks/shape-lab/four_layer_mini_engine.py
sha256 79b5f070a30176af2a7a87a473fe578a15abd5177fb39b2ab9e188f66572fe0e
```

`engine/production-runtime-config.json` pins the server, request adapters,
context policy, providers, manifests and native binaries. Run
`./aima-engine verify --json` for the complete current digest set.

## Public portability changes

The model-math engine was exported byte-for-byte. The public control surface
adds:

- parameterized runtime, model and image paths;
- hostname-independent discovery of the exact AOTriton library;
- AIMA model and response metadata branding;
- image build/registration and environment diagnostics;
- release, security and third-party licensing metadata.

These changes do not alter model math, provider selection, sampling or cache
state transitions.

## Correctness boundary

The released engine lineage passed:

- cached-decode KLD `0.0002768326 < 0.005` against the selected official AMD395
  vLLM reference;
- top-1 agreement `1.0`;
- exact 128-token completion identity;
- eight output-one HTTP cells;
- 76/76 usage, seed/mid-decode stop, strict-prefix and exact-prefix checks.

## Native source provenance

The two generated CK-Tile files come from AMD Composable Kernel commit
`6667a9021713f794a2c9aee4696c19f6cf376235` and retain AMD's MIT headers. The
fixed-shape wrappers and striped loader/builder are Apache-2.0 project code.

The shipped binaries were built for `gfx1151` with ROCm/HIP 7.2. They are part
of the qualified release. Rebuilds go to `build/native/` and require a separate
correctness/performance qualification before substitution.

## Excluded material

The release deliberately excludes:

- model weights and tokenizer assets;
- generated 69.3 GB startup images;
- private host paths, credentials and machine-specific manifests;
- internal route ledgers, rejected variants and raw experiment artifacts;
- oracle logits or licensed reference-model outputs.
