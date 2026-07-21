# Installation and startup-image preparation

## 1. Hardware and operating system

The qualified v1.0.0 platform is:

- AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`);
- 128 GB installed unified memory, configured as a 96 GiB AMDGPU GTT pool;
- Linux x86-64, Ubuntu 24.04;
- ROCm 7.2.1 / HIP `7.2.26015`;
- two physical NVMe devices for the published startup result.

One storage device may be functionally usable, but its startup performance is
not qualified. The full model process reached approximately 73.2 GB allocated
memory in the accepted runs, while maximum-context cases peaked higher. Close
other GPU/unified-memory workloads before serving.

The memory figure above is a runtime GTT-pool requirement, not a request for a
96 GB fixed BIOS framebuffer. Before installing the runtime, follow
[`MEMORY.md`](MEMORY.md) to set 512 MiB fixed VRAM, the 96 GiB GTT kernel
parameters, and the serving account's GPU groups. A Chinese version is
available at [`MEMORY.zh-CN.md`](MEMORY.zh-CN.md).

## 2. Runtime environment

Prepare a Python environment that contains ROCm-compatible builds matching
[`requirements-runtime.txt`](../requirements-runtime.txt):

| Package | Qualified build |
|---|---|
| Python | 3.12.3 |
| PyTorch | `2.10.0+git8514f05` |
| vLLM | `0.19.1rc1.dev300+g29e5d1020.rocm721` |
| Triton | `3.6.0` |
| Transformers | `4.57.6` |
| Safetensors | `0.8.0rc0` |

The context policy also binds the exact AOTriton library
`libaotriton_v2.so.0.11.1` with SHA-256
`e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5`.
This process-private binding is used only for qualified 64k/128k cold-context
requests and is restored after each request.

Set the runtime explicitly:

```bash
export AIMA_RUNTIME_PYTHON=/path/to/venv/bin/python
```

The CLI control plane uses only the standard library; the selected runtime is
loaded only when `doctor` probes it or `serve` starts the engine.

Keep the release checkout intact because component hashes and native assets
are resolved from it. The root `./aima-engine` launcher is the supported
deployment entrypoint. An optional `python -m pip install -e .` exposes the
same launcher from an editable checkout; a detached wheel is not supported.

## 3. Model checkpoint

Obtain Qwen3.6-35B-A3B BF16 weights independently. The directory must contain
at least `config.json`, tokenizer assets, all Safetensors shards and
`model.safetensors.index.json`.

The qualified checkpoint-index SHA-256 is:

```text
41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83
```

Set:

```bash
export AIMA_MODEL_DIR=/path/to/Qwen3.6-35B-A3B
```

`doctor` and `prepare-images` fail closed if this index does not match. This
does not redistribute or authorize the model; the model's own license applies.

## 4. Build startup images

The optimized loader uses a deterministic two-lane image layout:

| Lane | Exact bytes | Approximate size |
|---|---:|---:|
| lane0 | 34,660,827,136 | 32.2804 GiB |
| lane1 | 34,660,823,040 | 32.2804 GiB |
| total | 69,321,650,176 | 64.5608 GiB |

Build once from the local licensed checkpoint:

```bash
./aima-engine prepare-images \
  --model-dir "$AIMA_MODEL_DIR" \
  --lane0-dir /mnt/nvme0/aima-qwen36 \
  --lane1-dir /mnt/nvme1/aima-qwen36 \
  --state-dir "$HOME/.cache/aima-qwen36" \
  --output-manifest "$HOME/.config/aima-qwen36/striped-image-manifest.json"
```

The command:

1. validates the checkpoint index;
2. checks available disk space;
3. compiles the CPU image builder;
4. creates both lanes concurrently;
5. validates content checksums and full SHA-256 hashes;
6. writes a portable manifest containing the selected absolute paths.

The command refuses to overwrite images unless `--force` is explicitly passed.

### Register existing qualified images

```bash
./aima-engine register-images \
  --model-dir "$AIMA_MODEL_DIR" \
  --lane0 /mnt/nvme0/aima-qwen36/lane0.bin \
  --lane1 /mnt/nvme1/aima-qwen36/lane1.bin \
  --output-manifest "$HOME/.config/aima-qwen36/striped-image-manifest.json"
```

Registration performs full SHA-256 verification and does not modify the lane
files.

Set the resulting manifest:

```bash
export AIMA_IMAGE_MANIFEST="$HOME/.config/aima-qwen36/striped-image-manifest.json"
```

## 5. Validate and serve

```bash
./aima-engine verify
./aima-engine doctor
./aima-engine doctor --deep
./aima-engine serve --host 127.0.0.1 --port 8000
```

`doctor --deep` reads and hashes both large lanes. Normal service startup uses
size checks and the native aggregate payload checksum, avoiding a redundant
full disk pass.

## 6. Native binaries

The repository ships the exact qualified gfx1151 binaries. To rebuild them,
clone AMD Composable Kernel at commit
`6667a9021713f794a2c9aee4696c19f6cf376235`, set `CK_DIR`, and run:

```bash
CK_DIR=/path/to/composable-kernel make build-native
```

Rebuilt binaries are written under `build/native/`; they do not silently
replace the qualified release binaries. See [`docs/RELEASE.md`](RELEASE.md).

## 7. Optional systemd service

Examples are provided in [`packaging/systemd`](../packaging/systemd). Install
the repository under `/opt/aima-amd395-qwen36-35b-linux-engine`, copy and edit
the environment file, then install the unit:

```bash
sudo useradd --system --user-group --home-dir /nonexistent \
  --shell /usr/sbin/nologin aima
sudo usermod -aG render,video aima
sudo install -d -m 0750 /etc/aima-qwen36 /var/lib/aima-qwen36
sudo chown aima:aima /var/lib/aima-qwen36
sudo install -m 0640 packaging/systemd/aima-engine.env.example \
  /etc/aima-qwen36/engine.env
sudo install -m 0644 packaging/systemd/aima-engine.service \
  /etc/systemd/system/aima-engine.service
sudo systemctl daemon-reload
sudo systemctl enable --now aima-engine
```

Create the dedicated `aima` user, grant it access to the GPU/render device,
model, image lanes and state directory, and edit `engine.env` before starting.
The unit binds localhost and creates a fresh timestamped artifact directory on
every start.

## Troubleshooting

- **`gfx1151` check fails:** confirm `rocminfo` can see the Radeon 8060S and
  that the process has device permissions.
- **AOTriton SHA mismatch:** use the exact qualified runtime build; do not patch
  the disk library.
- **Checkpoint index mismatch:** use the qualified BF16 model revision.
- **Image SHA mismatch:** rebuild both lanes from the matching model; do not mix
  lanes from different builds.
- **Out of memory:** stop other GPU workloads, then verify the fixed VRAM, GTT
  pool and kernel command line against [`MEMORY.md`](MEMORY.md).
- **Remote clients cannot connect:** the safe default binds only localhost. If
  changing the host, put authentication and TLS in front of the engine.
