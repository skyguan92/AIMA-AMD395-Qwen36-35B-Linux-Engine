# AIMA AMD395 Qwen3.6 35B Linux Engine

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/approaching-ai/aima-amd395-qwen36-35b-linux-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/approaching-ai/aima-amd395-qwen36-35b-linux-engine/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v1.2.0-green.svg)](CHANGELOG.md)
[![Hardware](https://img.shields.io/badge/GPU-gfx1151-orange.svg)](docs/INSTALL.md)

A batch-1 BF16 inference engine specialized for
`Qwen3.6-35B-A3B` on AMD Ryzen AI Max+ 395 / Radeon 8060S Linux.

Version 1.2 provides a relocatable native package: no Python, PyTorch, vLLM,
Triton, Transformers, or host ROCm userspace is loaded at runtime. The package
contains a static launcher, the native engine, pinned ROCm/AOTriton/CK
userspace, its own glibc loader, licenses and qualification metadata. Model
weights are not redistributed.

中文说明见 [README.zh-CN.md](README.zh-CN.md).

## Read this boundary first

The portable native runtime is qualified for the complete published batch-1
envelope:

| Input tokens | Output tokens | Status |
|---:|---:|---|
| 1,024 | 512 / 1,024 | qualified |
| 2,048 | 512 / 1,024 | qualified |
| 4,096 | 512 / 1,024 | qualified |
| 8,192 | 512 / 1,024 | qualified |
| 16,384 | 512 / 1,024 | qualified |
| 32,768 | 512 / 1,024 | qualified |
| 65,536 | 512 / 1,024 | qualified |
| 131,072 | 512 / 1,024 | qualified |
| 262,143 | 1 | qualified window endpoint |
| 261,632 | 512 | qualified window endpoint |
| 261,120 | 1,024 | qualified window endpoint |

Cold HTTP prompts must encode to exactly the selected static context. Longer
requests are accepted only when they extend the cached token prefix. Input plus
generated tokens may not exceed 262,144. The native runtime now replaces the
published v1.1 performance envelope; the Python implementation remains only as
a compatibility and provenance reference. See
[native/product-contract.json](native/product-contract.json).

## Runtime contract

The deployment host needs:

- Linux x86-64 with an AMDGPU/KFD kernel driver and render nodes;
- Radeon 8060S / `gfx1151`;
- 128 GB installed memory with the documented 96 GiB GTT pool;
- the separately obtained, hash-matching 26-shard BF16 model checkpoint.

It does not need a system ROCm installation or a Python environment. The
qualified package is approximately 366 MiB unpacked and 101 MiB as a `.tar.zst`
archive, including the complete userspace ELF closure. Cross-version
compatibility comes from the bundled loader and libraries; kernel/GPU
compatibility cannot be bundled away.

Configure memory before loading the model:
[English](docs/MEMORY.md) · [中文](docs/MEMORY.zh-CN.md).

## Quick start

Extract the release archive anywhere:

```bash
tar --zstd -xf aima-engine-native-portable-*.tar.zst
cd aima-engine-native-portable-*

./bin/aima-engine --version
./bin/aima-engine serve \
  --model-dir /srv/models/Qwen3.6-35B-A3B \
  --context-tokens 8192 \
  --host 127.0.0.1 \
  --port 8000
```

The service loads the model once, verifies all 69,321,221,376 active bytes and
keeps weights, plans, KV/recurrent state and cache resident. Readiness is
reported as one JSON line.

In another shell:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

A deterministic chat request uses the OpenAI-compatible subset:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aima-amd395-qwen36-35b",
    "messages": [{"role": "user", "content": "PROMPT_WITH_EXACT_ADMITTED_TOKEN_LENGTH"}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 512
  }'
```

Stop it with `Ctrl-C` / `SIGTERM`, or:

```bash
curl -fsS -X POST http://127.0.0.1:8000/shutdown
```

For a managed resident service, install the templates under `share/systemd/`;
then `systemctl start|status|stop aima-engine` provides the lifecycle.

## Native CLI

```text
aima-engine --version
aima-engine serve --model-dir PATH --context-tokens N
aima-engine resident-session-probe --model-dir PATH [qualification options]
aima-engine tokenizer-probe --model-dir PATH --text TEXT
aima-engine chat-template-probe --model-dir PATH --user TEXT
```

`serve` runs in the foreground by design and is suitable for systemd,
containers and direct supervision. Internal qualification probes are shipped so
published correctness and performance claims are reproducible without a
framework runtime.

## Qualified native results

All values below were measured from the packaged native engine on the qualified
AMD395 host. Prefill/decode promotion uses a three-run median, or two runs
within 3%.

| Input | output512 prefill | output512 decode | output1024 prefill | output1024 decode |
|---:|---:|---:|---:|---:|
| 1,024 | 1636 | 33.99 | 1636 | 33.99 |
| 2,048 | 1695 | 33.89 | 1695 | 33.87 |
| 4,096 | 1576 | 33.27 | 1576 | 33.27 |
| 8,192 | 1657 | 32.27 | 1657 | 32.26 |
| 16,384 | 1429 | 30.69 | 1429 | 30.68 |
| 32,768 | 1357 | 28.17 | 1357 | 28.16 |
| 65,536 | 1168 | 24.63 | 1168 | 24.63 |
| 131,072 | 874.4 | 19.52 | 874.4 | 19.52 |

Window endpoints reached `537.4` prefill tok/s at 262143/output1,
`559.6 / 13.62` prefill/decode tok/s at 261632/output512, and
`554.9 / 13.60` at 261120/output1024. All 19 cells retained at least 97% of
their frozen baseline; the lowest was `0.9740x` at the output1024 endpoint.

Other gates:

- full-vocabulary KLD passed at nine contexts through q261632; the maximum was
  `0.002174`, with matching top-1 everywhere and the gate fixed at `0.005`;
- exact 128-token completion identity passed on the frozen q8192 fixture;
- q8192 command-to-ready median: `42.52 s` versus the `51.41 s` ceiling;
- q32768 exact-prefix TTFT: `2634x` speedup with `1.0006` decode retention;
- resident HTTP: one model load across cold and cached requests, with clean
  shutdown.

The auditable source of truth is
[benchmarks/results/native-portable-product-v1.2.0.json](benchmarks/results/native-portable-product-v1.2.0.json).
The frozen baseline and optional striped-startup evidence remain documented in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Build the archive

Runtime deployment has no framework dependency; building from source does.
The qualified builder needs ROCm/HIP, Python for generators, AMD Composable
Kernel at commit `6667a9021713f794a2c9aee4696c19f6cf376235`, and the pinned
AOTriton 0.11.1 development distribution:

```bash
export CK_DIR=/path/to/composable-kernel
export AOTRITON_ROOT=/path/to/distribution/root/containing/include-and-lib

make check
make package-native
```

The packager rejects absolute RUNPATHs and unresolved ELF dependencies,
includes all upstream notices, generates a recursive SHA-256 manifest, and
emits one deterministic `.tar.zst` archive under `dist/`.

Detailed instructions: [docs/INSTALL.md](docs/INSTALL.md).

## Repository map

```text
native/                      native engine, AOT closure and product contract
benchmarks/shape-lab/native/ CK-Tile sources and compatibility artifacts
benchmarks/results/          release qualification records
scripts/                     deterministic build, closure and package tools
packaging/systemd/           service lifecycle templates
docs/                        install, API, memory, architecture and evidence
aima_engine/                 retained v1.1 compatibility control plane
```

## Compatibility runtime

The Python control plane and the frozen v1.1 model-math engine remain in the
source tree for the wider context matrix and historical reproducibility. They
are not loaded by the portable native archive. Do not mix the two performance
or dependency claims.

## Security

The HTTP server has no built-in authentication and exposes `POST /shutdown`.
It binds to `127.0.0.1` by default. Use an authenticated reverse proxy or an
isolated host before exposing it to a network. See [SECURITY.md](SECURITY.md).

## License

AIMA project code is licensed under
[Apache License 2.0](LICENSE). Bundled and generated third-party components
retain their upstream terms; see [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and the archive's
`licenses/` directory. Model weights are not included.
