# AIMA AMD395 Qwen3.6 35B Linux Engine

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/approaching-ai/aima-amd395-qwen36-35b-linux-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/approaching-ai/aima-amd395-qwen36-35b-linux-engine/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v1.0.0-green.svg)](CHANGELOG.md)
[![Hardware](https://img.shields.io/badge/GPU-gfx1151-orange.svg)](docs/INSTALL.md)

A production-qualified, batch-1 inference engine specialized for
`Qwen3.6-35B-A3B-BF16` on the AMD Ryzen AI Max+ 395 / Radeon 8060S Linux
platform. It keeps model state resident, serves a deterministic subset of the
OpenAI Chat Completions API, and supports exact-prefix reuse.

This repository contains the complete engine source, native provider sources,
qualified native binaries, CLI, HTTP server and reproducibility metadata. It
does **not** contain model weights or generated startup images.

中文说明见 [README.zh-CN.md](README.zh-CN.md).

## Qualified envelope

| Item | v1.0.0 contract |
|---|---|
| Hardware | AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 96 GB unified memory |
| Operating system | Linux x86-64; qualified on Ubuntu 24.04, kernel 6.14 |
| Model | Qwen3.6-35B-A3B, BF16 checkpoint with the qualified index hash |
| Workload | Batch 1, deterministic greedy decoding (`temperature=0`, `top_p=1`) |
| Context | Up to 262,144 total tokens; the valid input limit is `262144 - max_tokens` |
| Service | Resident process, HTTP health/models/chat endpoints and CLI client |
| Prefix cache | One exact entry, up to 32,768 prompt tokens, strict or exact hit |
| Streaming/tools | Not supported in v1.0.0 |

The engine intentionally specializes before generalizing. Requests outside
this envelope fail closed instead of silently switching to an unqualified
mode.

The hash-qualified checkout is the deployment unit. Run the root
`aima-engine` launcher directly, or use `pip install -e .` if an activated
control-plane environment needs the same command on `PATH`. Detached wheels
are not a supported deployment form because the engine and native assets are
verified relative to the release checkout.

## Quick start

### 1. Verify the release

```bash
git clone https://github.com/approaching-ai/aima-amd395-qwen36-35b-linux-engine.git
cd aima-amd395-qwen36-35b-linux-engine
./aima-engine verify
```

### 2. Select the qualified runtime and model

```bash
export AIMA_RUNTIME_PYTHON=/path/to/rocm-vllm-venv/bin/python
export AIMA_MODEL_DIR=/path/to/Qwen3.6-35B-A3B
```

The exact qualified package versions are in
[`requirements-runtime.txt`](requirements-runtime.txt). They must be ROCm
builds; that file is not a generic `pip install` recipe.

### 3. Prepare startup images once

The startup format contains two approximately 32.28 GiB lanes. Two physical
NVMe devices are recommended for the qualified startup time.

```bash
./aima-engine prepare-images \
  --model-dir "$AIMA_MODEL_DIR" \
  --lane0-dir /mnt/nvme0/aima-qwen36 \
  --lane1-dir /mnt/nvme1/aima-qwen36 \
  --state-dir "$HOME/.cache/aima-qwen36" \
  --output-manifest "$HOME/.config/aima-qwen36/striped-image-manifest.json"

export AIMA_IMAGE_MANIFEST="$HOME/.config/aima-qwen36/striped-image-manifest.json"
```

If qualified lanes already exist, use `register-images` instead. See
[`docs/INSTALL.md`](docs/INSTALL.md).

### 4. Check the complete installation

```bash
./aima-engine doctor
./aima-engine doctor --deep  # also hashes both large image lanes
```

### 5. Start the resident service

```bash
./aima-engine serve --host 127.0.0.1 --port 8000
```

In another shell:

```bash
./aima-engine status
./aima-engine chat --max-tokens 128 "Explain why prefix caching reduces TTFT."
./aima-engine shutdown
```

Direct HTTP clients can use:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aima-amd395-qwen36-35b",
    "messages": [{"role": "user", "content": "Write a short hello."}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 64
  }'
```

## Validated results

On the qualified AMD395 host, v1.0.0 measured:

- cold q8192/output512: `1591` prefill tok/s and `32.12` decode tok/s;
- cold q8192/output1024: `1587` prefill tok/s and `32.18` decode tok/s;
- command-to-ready median: `27.31` seconds;
- near-complete q32768 prefix reuse: `110.1x` median TTFT speedup with
  `0.9997` minimum decode retention;
- selected-reference correctness: KLD `0.0002768`, top-1 agreement `1.0`, and
  exact 128-token completion identity;
- HTTP usage, stop and prefix-cache conformance: `76/76` checks.

The complete cold-context matrix and measurement boundaries are in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md). Raw sibling-D275 decay ratios are
reported as an unfinished engineering target; all published blocking floors
passed.

## Operations and API

- [Installation and image preparation](docs/INSTALL.md)
- [CLI and HTTP API](docs/API.md)
- [Architecture and residency model](docs/ARCHITECTURE.md)
- [Performance and correctness evidence](docs/PERFORMANCE.md)
- [Release provenance](docs/RELEASE.md)
- [Security policy](SECURITY.md)

Run `./aima-engine --help` or `./aima-engine <command> --help` for the complete
command surface.

## Repository layout

```text
aima_engine/                 portable control-plane CLI
benchmarks/shape-lab/        hash-qualified production engine and providers
benchmarks/shape-lab/native/ qualified gfx1151 binaries and rebuild sources
engine/                      runtime and striped-image contracts
tools/                       resident request and HTTP adapters
docs/                        operator, API and evidence documentation
tests/                       CPU-safe release and contract tests
```

The historical `shape-lab` path is retained because the released engine source
is hash-qualified. User-facing operations go through `aima-engine`.

## Security

The HTTP server has no built-in authentication and exposes a shutdown endpoint.
It binds to `127.0.0.1` by default. Do not expose it directly to an untrusted
network; use an authenticated reverse proxy or an isolated host. See
[`SECURITY.md`](SECURITY.md).

## License

Project code is licensed under the [Apache License 2.0](LICENSE). The generated
AMD CK-Tile files retain their MIT license; see [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are not bundled
or relicensed.
