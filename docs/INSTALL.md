# Install the portable native runtime

## 1. Qualified platform

The v1.3 native profile is qualified on:

- AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`);
- 128 GB installed unified memory;
- 512 MiB fixed BIOS VRAM plus a 96 GiB AMDGPU GTT pool;
- Linux x86-64, Ubuntu 24.04.3, kernel 6.14;
- the pinned ROCm 7.2 userspace shipped in the archive.

The deployment host does not need `/opt/rocm`, Python, PyTorch, vLLM, Triton,
Transformers, a compiler or a C++ runtime package. It does need a compatible
AMDGPU/KFD kernel driver and access to `/dev/kfd` and the render node.

Complete the memory configuration in [MEMORY.md](MEMORY.md) or
[MEMORY.zh-CN.md](MEMORY.zh-CN.md) before loading the model. Firmware, kernel
parameters and GPU memory settings are host-wide changes; use the documented
rollback procedure.

## 2. Obtain the model separately

Model weights are not included. The qualified checkpoint is standard Hugging
Face Safetensors with:

| Property | Required value |
|---|---|
| Shards | 26 |
| Active tensors | 693 |
| Active payload | 69,321,221,376 bytes |
| checkpoint index SHA-256 | `41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83` |
| model config SHA-256 | `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99` |
| tokenizer SHA-256 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |

Place the checkpoint on local storage, for example:

```text
/srv/models/Qwen3.6-35B-A3B/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-00026.safetensors
├── ...
├── model-00026-of-00026.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

The native loader validates the model/index geometry and the complete device
payload. This does not redistribute or authorize the model; its own license
continues to apply.

## 3. Install one archive

Verify the release checksum published beside the archive, then extract it to
any local path:

```bash
sha256sum -c aima-engine-native-portable-*.tar.zst.sha256
sudo mkdir -p /opt/aima-engine
sudo tar --zstd -xf aima-engine-native-portable-*.tar.zst \
  --strip-components=1 -C /opt/aima-engine

/opt/aima-engine/bin/aima-engine --version
```

The archive is relocatable. Moving the extracted directory does not invalidate
its internal library resolution. `manifest.json` records the SHA-256 and byte
size of every file.

The static `bin/aima-engine` launcher executes
`libexec/aima-engine.real` through the bundled glibc loader with an isolated
`lib/` search path. The packager has already checked that all x86-64 ELF
dependencies resolve inside the archive and that no absolute RUNPATH remains.

## 4. Start in the foreground

Choose one admitted static context:

```bash
/opt/aima-engine/bin/aima-engine serve \
  --model-dir /srv/models/Qwen3.6-35B-A3B \
  --context-tokens 8192 \
  --host 127.0.0.1 \
  --port 8000 \
  --report /var/tmp/aima-native-weight-load.json
```

Published standard cold contexts are `1024`, `2048`, `4096`, `8192`, `16384`,
`32768`, `65536` and `131072` tokens. Valid maximum-window contexts are also
accepted when prompt plus output stays within 262,144 tokens. The provider is
selected automatically:

- q1024/q2048/q4096: bundled AOTriton 0.11.1;
- q8192/q32768 and long-context chunks: bundled CK-Tile;
- q16384: bundled packed-GQA/CK-Tile hybrid;
- long-context terminal full-attention layer: bundled AOTriton 0.11.1.

`--fmha-provider PATH` exists for qualification overrides; normal deployment
should not set it.

Readiness is emitted only after tokenizer load, checkpoint ingestion, derived
layout construction, AOT module loading, plan preparation and cache allocation.
The process then stays resident. Stop it with `Ctrl-C`, `SIGTERM`, or:

```bash
curl -fsS -X POST http://127.0.0.1:8000/shutdown
```

## 5. Install the systemd service

The archive includes templates under `share/systemd/`:

```bash
getent passwd aima >/dev/null || \
  sudo useradd --system --user-group \
    --home-dir /var/lib/aima-qwen36 --shell /usr/sbin/nologin aima
sudo usermod -aG render,video aima
sudo install -d -o aima -g aima /var/lib/aima-qwen36
sudo install -Dm644 \
  /opt/aima-engine/share/systemd/aima-engine.service \
  /etc/systemd/system/aima-engine.service
sudo install -Dm640 \
  /opt/aima-engine/share/systemd/aima-engine.env.example \
  /etc/aima-qwen36/engine.env
sudo systemctl daemon-reload
```

Edit `/etc/aima-qwen36/engine.env` and grant `aima` read access to the model
directory. The commands above create a non-login service account and add it to
the GPU device groups. Then start the service:

```bash
sudo systemctl enable --now aima-engine
```

Lifecycle commands are standard:

```bash
systemctl status aima-engine
journalctl -u aima-engine -f
sudo systemctl restart aima-engine
sudo systemctl stop aima-engine
```

## 6. HTTP smoke test

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

The current native route is exact-length specialized. A cold chat prompt must
encode to precisely the selected context after the Qwen chat template is
applied. A shorter prompt returns HTTP 400. A longer prompt is admitted only if
its token sequence extends the one cached static prefix.

Use the native tokenizer probes when preparing deterministic fixtures:

```bash
/opt/aima-engine/bin/aima-engine chat-template-probe \
  --model-dir /srv/models/Qwen3.6-35B-A3B \
  --user 'prompt text' \
  --disable-thinking
```

## 7. Build from source

Build-time requirements are intentionally separate from runtime requirements:

- ROCm/HIP matching `7.2.26015-fc0010cf6a`;
- Python 3 for deterministic registry/manifest generation;
- static ICU development archives;
- GNU binutils and a static-capable C compiler;
- GNU tar with Zstandard support;
- AMD Composable Kernel checkout at
  `6667a9021713f794a2c9aee4696c19f6cf376235`;
- the qualified AOTriton 0.11.1 distribution with `include/`, `lib/`,
  `aotriton.images/`, LICENSE and NOTICE.

```bash
export CK_DIR=/src/composable-kernel
export AOTRITON_ROOT=/path/to/qualified/distribution/root

make check
make package-native
```

If the AOTriton distribution does not expose conventional `LICENSE*` and
`NOTICE*` paths, also set `AOTRITON_LICENSE` and `AOTRITON_NOTICE` to the exact
files. Packaging fails closed when either legal artifact is absent.

The AOTriton shared library and selected gfx1151 image are hash-gated. Rebuilt
providers or a different ROCm/AOTriton input require fresh correctness and
performance qualification.

The resulting directory and deterministic `.tar.zst` archive are written
under `dist/`. The archive contains no model weights.

## Compatibility runtime

The source checkout retains the v1.1 Python control plane for provenance and
compatibility testing. It is not copied into the v1.3 native archive. The
portable native runtime now covers the complete published v1.1 context/output
performance envelope.

## Troubleshooting

- **AMDGPU allocation failure:** stop other GPU workloads and recheck the GTT
  pool and kernel command line in [MEMORY.md](MEMORY.md).
- **Permission denied on KFD/render:** add the service user to `render` and
  `video`, then start a new login/session.
- **Model hash or payload mismatch:** use the qualified BF16 checkpoint; do not
  mix shards or tokenizer files from different revisions.
- **Unsupported context:** select a published standard context or a valid
  long-context specialization whose prompt plus output fits 262,144 tokens.
  The engine does not silently fall back to an unqualified length.
- **Short HTTP prompt:** pre-tokenize the full chat template and choose an
  admitted fixture length.
- **Remote clients cannot connect:** localhost is the safe default. If binding
  another interface, put authentication and TLS in a reverse proxy.
- **Different GPU or kernel:** the bundled userspace removes host software
  coupling, not the `gfx1151` and AMDGPU/KFD hardware/driver contract.
