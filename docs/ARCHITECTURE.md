# Architecture

## Design objective

The engine is specialized for one model, one GPU architecture and batch size
one. The optimization boundary is the complete resident request, not a generic
framework abstraction.

```text
CLI / HTTP
    |
resident request adapter
    |
fixed context and prefix policy
    |
40-layer parameterized engine loop
    |-- Triton projection, normalization, routing and attention kernels
    |-- vLLM fused MoE and FLA primitives from the qualified runtime
    |-- CK-Tile fixed-shape full-attention providers
    `-- independent-storage striped checkpoint loader
```

## Residency

Startup allocates independent device tensors for 693 active checkpoint
objects. A native dual-lane loader reads the deterministic startup images and
scatters payloads directly into those Torch-owned storages. The process then
retains raw tensors, derived layouts, workspaces and model state.

The published startup clock includes command launch, imports, allocation,
image ingestion, model setup and load-only kernel priming until the HTTP API is
ready. Priming executes no hidden full-model user request and creates no prefix
cache entry.

## Request execution

The server supports one in-flight request. For a cold prompt, it performs
prefill, retains the resulting recurrent/KV state and runs greedy one-token
decode until a stop token or length limit. Terminal EOS is hidden from content
but included in usage.

## Prefix reuse

The exact-prefix cache retains one checkpoint for prompts up to 32,768 tokens.
An exact hit restores the full state; a strict hit restores the longest exact
token prefix and prefills only the suffix. The runtime contract is part of the
cache key, preventing state reuse across incompatible output policies.

## Context specialization

Cold request lengths 8k, 16k, 32k, 64k and 128k use only separately measured
policies. There is no threshold interpolation. Other valid lengths use the
safe fallback layout. Maximum-valid requests use persistent full-attention
providers for all complete and final partial chunks.

## Integrity model

The release config pins every production source and native binary by SHA-256.
The CLI verifies these hashes before serving. Startup-image manifests bind the
checkpoint index, exact geometry and lane paths; the native loader validates
the complete device payload checksum before exposing the model as ready.
