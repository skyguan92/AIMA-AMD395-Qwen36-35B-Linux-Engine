# Portable native runtime

This page documents the v1.5.1 portable native runtime.

AIMA v1.5 provides a relocatable native runtime for
`Qwen3.6-35B-A3B-BF16` on AMD Ryzen AI Max+ 395 / `gfx1151`. The runtime
loads the standard 26-shard Safetensors checkpoint directly, keeps model and
cache state resident, and serves a deterministic OpenAI Chat Completions
subset without loading Python, PyTorch, vLLM, Triton, Transformers, or a
host-installed ROCm userspace.

This is the qualified native replacement for the complete published v1.1
batch-1 envelope:

| Cold prompt | output512 | output1024 | FMHA provider |
|---:|:---:|:---:|---|
| 1,024 | qualified | qualified | bundled AOTriton 0.11.1 |
| 2,048 | qualified | qualified | bundled AOTriton 0.11.1 |
| 4,096 | qualified | qualified | bundled AOTriton 0.11.1 |
| 8,192 | qualified | qualified | bundled CK-Tile |
| 16,384 | qualified | qualified | bundled packed-GQA/CK hybrid |
| 32,768 | qualified | qualified | bundled CK-Tile |
| 65,536 | qualified | qualified | chunked CK-Tile + AOTriton |
| 131,072 | qualified | qualified | chunked CK-Tile + AOTriton |

The 262,144-token window endpoints at output1/output512/output1024 are also
qualified. The machine-readable boundary is in
[product-contract.json](product-contract.json), and the measurements are in
`benchmarks/results/native-portable-product-v1.5.1.json` after the release
evidence mirror lands.
Each v1.5.1 archive also carries its exact qualification at
`share/aima/qualification.json`.

## Distribution shape

The release is one `.tar.zst` archive. It contains several ELF objects because
ROCm and glibc are dynamically linked, but the archive is a self-contained,
relocatable deployment unit:

```text
aima-engine-native-portable-*/
├── bin/aima-engine                 # fully static user-facing launcher
├── libexec/aima-engine.real        # native model engine
├── lib/                            # pinned ROCm, GNU and FMHA userspace
├── lib/aotriton.images/            # one selected gfx1151 code object
├── amdgcn/bitcode/                 # bundled ROCm device libraries
├── docs/
├── licenses/
├── share/aima/                     # contract and qualification result
└── share/systemd/                  # optional service unit
```

The static launcher invokes the bundled glibc loader with
`--inhibit-cache --library-path BUNDLE/lib`. Every ELF dependency and RUNPATH
is audited before packaging. The remaining host contract is necessarily the
Linux kernel, AMDGPU/KFD and render device nodes on a `gfx1151` machine.

## Run

Extract the archive anywhere and point it at a separately licensed model:

```bash
tar --zstd -xf aima-engine-native-portable-*.tar.zst
cd aima-engine-native-portable-*
./bin/aima-engine --version
./bin/aima-engine doctor --model-dir /srv/models/Qwen3.6-35B-A3B --json
./bin/aima-engine serve \
  --model-dir /srv/models/Qwen3.6-35B-A3B \
  --context-tokens 8192 \
  --host 127.0.0.1 \
  --port 8000
```

The model is loaded once and remains resident until the process stops.
Use `Ctrl-C`, send `SIGTERM`, call `POST /shutdown`, or use the packaged
systemd unit. The safe default is localhost. `--api-key-file` enables bearer
authentication for `/v1/models`, chat and shutdown; a non-loopback bind
requires it unless the explicit unsafe override is supplied. The packaged
systemd unit enables the token, a socket timeout and
`--disable-http-shutdown` by default.

After a request is fully read and routed, the HTTP control endpoints (health,
model-list and shutdown) do not wait for one serial worker executing chat
inference. Accept/read/routing itself is single-threaded. Up to 16 accepted
chats wait in its pending queue; engine, tokenizer and cache access is never
concurrent. During chat work a normal complete `/health` request still returns
HTTP 200 with `status: "ok"` and reports `busy: true`. Queue saturation or
shutdown returns/cancels chats with HTTP 503 OpenAI errors (`code`
`server_busy`, `type` `server_error`). Shutdown stops admission, cancels queued
chats, waits for active inference to finish and joins the worker before engine
teardown; it does not forcibly cancel active inference.

`--request-timeout-ms` defaults to 15000. Positive platform-representable
millisecond values have no 600-second cap and bound complete request reads and
blocking socket writes, not inference duration. `0` disables those socket
timeouts—the absolute request-read and per-blocking-write timeouts—but retains
the 1 MiB request-body limit; use it only when slow clients and blocked writes
are acceptable operationally. At `0`, an incomplete or slow request can occupy
the single accept/read/routing loop indefinitely, leaving every HTTP endpoint,
including `/health` and HTTP `/shutdown`, unresponsive. A blocked main-thread
control write or queued-chat cancellation write can also delay shutdown. Use a
finite positive timeout when liveness matters.

The selected context is the preferred AOT prefill specialization, not a
mandatory request length. Every positive prompt is admitted when prompt plus
requested output fits the cache capacity. A cache miss starts from clean
resident state. A q8192 service keeps q1024/q2048/q4096/q8192 specializations
resident, selects the largest bucket no longer than the request and executes
only the unmatched tail token by token. Requests shorter than the smallest
resident bucket use the qualified token path. Exact and append prefix hits
reduce latency but never determine admission.

`POST /v1/chat/completions` supports live SSE token output and OpenAI function
tools. The native tokenizer renders the checkpoint's Qwen tool template,
assistant tool-call history and grouped tool responses without Transformers.
Generated Qwen XML is converted to structured `tool_calls`; it never leaks
into content when a valid call is recognized. Client disconnects stop the
remaining decode loop without unloading the model.

## Build

Building is different from running. A qualified builder needs ROCm/HIP, Python
for deterministic code generation, AMD Composable Kernel at commit
`6667a9021713f794a2c9aee4696c19f6cf376235`, and the pinned AOTriton 0.11.1
headers/library/image. None of those build tools is required on a deployment
host.

```bash
export CK_DIR=/path/to/composable-kernel
export AOTRITON_ROOT=/path/to/distribution/root/containing/include-and-lib
export QUALIFICATION_RECORD=/path/to/qualified-product-result.json
export AIMA_RELEASE_VERSION=X.Y.Z
export AIMA_RELEASE_TAG=vX.Y.Z
make build-native-runtime
make build-native
make package-native
```

The packager verifies the exact AOTriton library and gfx1151 image hashes,
binds every executable/provider input to the complete qualification, copies all
upstream notices, closes the complete ELF graph, generates a recursive SHA-256
manifest, and writes the relocatable archive under `dist/`.

## Qualified behavior

The native process owns all 693 checkpoint tensors, derived layouts, AOT
modules, hipBLASLt plans, KV/recurrent state, scratch and a capacity-bounded
prefix LRU. A normal q8192 service retains four exact request-prefix snapshots;
long-window profiles reduce that count to preserve the 96 GiB GTT contract.
The engine implements cold prefill, resident greedy decode, exact-prefix
restore and one-token-or-longer prefix extension. On a hit it restores the
cached state and executes only the suffix through the native decode path;
cold-prefill launch count is zero for an exact hit.

The final v1.5.1 qualification established:

- KLD below `0.005` and matching top-1 at nine contexts through q261632;
- exact 128-token completion identity on the frozen q8192 fixture;
- all 19 prefill/decode cells at or above 97% of their frozen floor;
- 51.16 s median q8192 command-to-ready versus the 51.41 s ceiling;
- q32768 exact-cache TTFT speedup `2626x` with `1.0001` decode retention;
- variable 16-token cold/exact requests, an ordinary 36-token next-user turn
  and post-long short-request isolation;
- resident q1024/q2048/q4096/q8192 dispatch and an exact A/B/A four-entry LRU
  replay;
- resident HTTP with one model load, routed health/models/shutdown control
  endpoints during serial chat inference, exact cache reuse and clean shutdown;
- live chunked SSE/non-stream token parity, structured function-tool parity
  and healthy resident state after client-disconnect cancellation.

Full values and component hashes live in the qualification JSON; rounded values
in prose are not the source of truth.
