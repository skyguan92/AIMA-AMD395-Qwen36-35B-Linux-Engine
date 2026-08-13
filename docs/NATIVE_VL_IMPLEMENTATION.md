# Native VL implementation status

> Governing goal: [`NATIVE_VL_GOAL.md`](NATIVE_VL_GOAL.md)
>
> Frozen text baseline: `v1.5.1` (`6f3e669`)
>
> Current phase: Phase 0 — reference and capability foundation

This file is a live requirement-to-evidence index. A status of `in progress`
or `implemented` is not a release claim. Only evidence that satisfies every
blocking condition in the governing goal can move a gate to `passed`.

## Gate status

| Gate | Current state | Evidence required to pass | Next blocking action |
|---|---|---|---|
| G1 full VL functional parity | capability envelope qualified; native implementation not started | complete image/video/mixed/conversation/API/tools/transport/residency native conformance results | implement the native media and model path against the frozen capability manifest |
| G2 VL correctness parity | reference frozen; oracle capture pending | processor, vision/language boundary, full-vocabulary logits, deterministic generation, task quality and error results | capture processor, boundary, logits and generation oracles |
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
