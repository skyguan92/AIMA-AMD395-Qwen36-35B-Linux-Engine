# Native architecture

## Product boundary

The v1.5 runtime specializes one model, one GPU architecture and batch size one.
It is a native resident engine, not a Python wrapper around the v1.1 stack.

```text
static bin/aima-engine launcher
        |
bundled glibc loader + BUNDLE/lib
        |
native HTTP / tokenizer / resident engine
        |
40-layer parameterized loop
        |-- embedded captured gfx1151 AOT code objects
        |-- native HIP pointwise, state and cache kernels
        |-- native gfx1151 wvSplitK decode projections
        |-- hipBLASLt BF16 prefill plans
        |-- q1024/q2048/q4096 bundled AOTriton FMHA
        |-- q8192/q32768/long-chunk bundled CK-Tile FMHA
        `-- q16384 bundled packed-GQA/CK hybrid FMHA
        |
native Safetensors O_DIRECT scatter
        |
693 resident model tensors + derived layouts + LM head
```

No Python interpreter, PyTorch dispatcher/tensor owner, vLLM operator registry,
Triton JIT or Transformers tokenizer exists in the runtime process.

## Startup and ownership

The engine validates the checkpoint index, allocates 693 independent final HIP
device tensors, and scatters only the active language-model ranges from the 26
Safetensors shards. Reader workers use page-aligned pinned buffers, O_DIRECT
when supported, asynchronous H2D copies and a buffered fallback for filesystems
that reject direct I/O.

Before readiness it verifies the complete 69,321,221,376-byte GPU payload,
builds the shared derived weight layouts and int8 LM head, resolves every AOT
weight binding, loads code objects, prepares hipBLASLt plans, allocates
the configured endpoint plus smaller resident prefill workspaces, and creates
up to four prefix-cache owners. Long-window profiles reduce the entry count so
snapshots remain inside the 96 GiB GTT contract. Nothing performs a second
full-weight copy.

All owners live for the process lifetime. Normal request execution performs no
checkpoint or oracle reads.

## O(1) source structure

Layer source is parameterized. One linear-attention path, one full-attention
path, one MoE path and one 40-layer loop consume layer-indexed bindings. There
are no per-layer source files or generated host branches. Validation ledgers and
captured schedules may scale with layer count; executable source structure does
not.

## Static context profiles

Prefill uses captured schedules for the standard shapes. Long contexts run the
same qualified q8192 chunk schedule in a layer-major loop, carry recurrent/KV
state across chunks, and use one of the admitted tail schedules.

| Context | Prefill schedule | Full-attention provider |
|---:|---|---|
| 1024 | q1024 embedded AOT closure | bundled AOTriton 0.11.1 |
| 2048 | q2048 embedded AOT closure | bundled AOTriton 0.11.1 |
| 4096 | q4096 embedded AOT closure | bundled AOTriton 0.11.1 |
| 8192 | q8192 embedded AOT closure | bundled CK-Tile |
| 16384 | q16384 embedded AOT closure | packed-GQA/CK hybrid; CK layer 39 |
| 32768 | q32768 embedded AOT closure | bundled CK-Tile |
| 65536+ | repeated q8192 closure plus 7168/7680/8191 tail when needed | CK-Tile; AOTriton layer 39 |

Provider selection is derived from the admitted context and the executable's
own location. `--fmha-provider` is an explicit qualification override.

Fixed schedules remain the fast path for the published standard contexts and
three maximum-window endpoints. A default q8192 process also keeps q1024,
q2048, q4096 and q8192 workspaces, invocations and GEMM plans resident.
Variable cache misses start from empty recurrent/KV state and compose the
smallest resident AOT bucket total covering the real prompt. Only the final
segment is padded. Causal hidden rows remain unchanged; the runtime repairs the
linear-attention convolution window and replays the state-producing recurrent
kernel at the logical token count before decode begins.

## Correctness-sensitive arithmetic

The native q1024 full-attention input path reproduces PyTorch's vectorized
head-RMSNorm reduction order and correctly rounded FP32 reciprocal square root.
RoPE products are split at eager FP32 rounding boundaries. This removed the
rare one-ULP Q-head drift that had previously amplified through attention.

Full-vocabulary release gates are distributional: finite logits, matching top-1
and KLD below `0.005`. Exact boundary probes remain available for localization
but are not substituted for the end-to-end gate.

## Native VL vision attention

The visual tower uses two hash-locked, bit-exact gfx1151 attention images. The
external WPE3 image is the general path and remains selected for video, mixed
media and image-only batches above 4096 patches. The WPE6 image is embedded in
the executable's verified AOT registry and is selected for image-only batches
containing at most 4096 patches across one or more single-frame grids. Video
and mixed requests stay on WPE3 even when their sampled grids have a temporal
depth of one. Startup retains a standard-shape WPE3 warmup for shared resources
and executes the maximum image-count warmup on WPE6 before publishing
readiness.

Vision plan reuse separates shape resources from executable identity. Patch
embedding, GEMM and merger plans can be shared by patch count; attention reuse
also requires the selected image SHA-256 and exact sequence boundaries. A
different boundary layout may rebind metadata only from the same image, never
from the other occupancy variant.

## Request execution

A cold request starts from clean resident state. It runs one or more resident
prefill schedules layer-major, writes full-attention KV directly into the cache
at absolute positions and retains the logical linear recurrent/conv state. The
native LM head produces the first completion token, then the engine captures a
variable-length request-prefix checkpoint. Cache misses and LRU eviction change
latency, not admission.

Greedy decode stops on the model EOS or the requested length. EOS is included
in token usage and omitted from visible text.

An optional synchronous token callback sits immediately after each certified
top-1 token. Non-streaming requests leave it empty. SSE requests use it to
decode byte-level BPE fragments, retain incomplete UTF-8 sequences and emit
content without waiting for the full completion. Returning false cancels the
remaining decode loop while preserving resident ownership.

## Prefix cache

Each capacity-bounded LRU entry owns:

- the static token sequence;
- every linear-attention conv/recurrent state;
- K/V state for all ten full-attention layers;
- the terminal hidden row used for the cached first-token distribution.

The engine chooses the longest matching resident entry. An exact hit restores
the state and bypasses prefill. If a request begins with the cached tokens and
adds a suffix, the engine restores the same state,
executes each suffix token through native decode at its real position, and then
starts completion. Cache metrics distinguish miss, exact and prefix-extension
lookups and report that no prefill launches occurred on a hit.

## HTTP residency

The server binds its address, loads the native tokenizer and engine, then starts
listening. One process handles one request at a time, retaining model and cache
state. Health and model-list endpoints do not mutate the cache. Binding before
the expensive model load makes address conflicts fail immediately, while
listening only after the load preserves readiness semantics. `SIGINT`, `SIGTERM` and an
enabled `POST /shutdown` lead to normal RAII cleanup. The complete request read
has an absolute `--request-timeout-ms` deadline, each socket write is bounded,
and client send failures do not terminate the resident process.
The native process sends systemd `STATUS`, `READY=1` and `STOPPING=1`
datagrams when `NOTIFY_SOCKET` is present; no libsystemd runtime dependency is
introduced.

The tokenizer contains a parameterized implementation of the checkpoint's
Qwen chat template, including function definitions, assistant calls and
grouped tool responses. A bounded stream gate emits normal text immediately
but retains a possible `<tool_call>` suffix. Complete calls are validated
against admitted function names, converted with their JSON schemas and exposed
as OpenAI `delta.tool_calls` or non-streaming `message.tool_calls`.

Streaming uses HTTP/1.1 chunked SSE. Role, content, tool-call, terminal and
optional usage chunks share one completion id, followed by `[DONE]`. A failed
socket write returns cancellation through the token callback; the next request
can reuse the same resident model.

The optional Python distribution is a dependency-free remote client, not a
second packaged inference runtime. Its wheel exposes only health/model/chat/
shutdown operations. Repository-only compatibility commands are registered
only when the source runtime configuration is present.

## Portable userspace

The archive includes a fully static launcher and a complete x86-64 dynamic
closure for the real engine and all three FMHA providers. The launcher invokes the
colocated glibc loader with `--inhibit-cache`; all RUNPATH entries are
`$ORIGIN`-relative. The engine points HIP device libraries and hipBLASLt
assets at its own bundle.

This removes host ROCm/glibc/C++ version coupling. It cannot remove the Linux
kernel ABI, AMDGPU/KFD driver, device nodes, CPU architecture or `gfx1151`
hardware contract.

## Evidence separation

The full-envelope portable decision is embedded in each archive and mirrored
after release as `benchmarks/results/native-portable-product-v1.5.1.json`.
The v1.1 complete-context matrix remains the frozen per-cell floor. A
bundle-closure pass, a correctness pass and a performance pass are independent
gates; none is used as a proxy for another.
