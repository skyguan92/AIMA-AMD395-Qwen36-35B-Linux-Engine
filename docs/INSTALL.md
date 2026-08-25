# Install the portable native runtime

This page documents the v1.5.1-native-vl.2 portable package. Archives before
v1.4.0 do not contain the deployment doctor, bearer authentication, socket
timeouts or the hardened systemd template; the exact v1.5.1 baseline contains
those controls but not the native vision runtime. Use the documentation bundled
with the version you deploy.

## 1. Qualified platform

The v1.5.1-native-vl.2 profile is qualified on:

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
Face Safetensors from
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), pinned
to revision `995ad96eacd98c81ed38be0c5b274b04031597b0`:

| Property | Required value |
|---|---|
| Shards | 26 |
| Language tensors | 693 |
| Language payload | 69,321,221,376 bytes |
| Visual tensors | 333 |
| Visual payload | 893,142,496 bytes |
| Total tensors | 1,026 |
| Total payload | 70,214,363,872 bytes |
| checkpoint index SHA-256 | `41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83` |
| model config SHA-256 | `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99` |
| tokenizer SHA-256 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| tokenizer config SHA-256 | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |

With the Hugging Face CLI installed on a download machine:

```bash
hf download Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --local-dir /srv/models/Qwen3.6-35B-A3B
```

Keep the revision argument: a moving branch may no longer match the release's
hash-gated product contract. Review and accept the model repository's own
license and usage terms independently from this engine's Apache-2.0 license.

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
/opt/aima-engine/bin/aima-engine doctor \
  --model-dir /srv/models/Qwen3.6-35B-A3B --json
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
  --allowed-local-media-path /srv/aima-media \
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

Readiness is emitted only after tokenizer load, all 693 language and 333 visual
tensors are loaded and verified, derived layout construction, language and
vision AOT module loading, both vision-attention warmups, plan preparation and
cache allocation.
For the default q8192 service, the q1024/q2048/q4096/q8192 prefill buckets and
four exact-token request-prefix snapshots stay resident. The process then stays
resident. Stop it with `Ctrl-C`, `SIGTERM`, or:

```bash
curl -fsS -X POST http://127.0.0.1:8000/shutdown
```

For a bearer-protected foreground service, create a non-world-readable key
file and add `--api-key-file PATH`. Binding a non-loopback address without a
key fails closed. `--disable-http-shutdown` removes the lifecycle endpoint;
`--request-timeout-ms` applies an absolute deadline to the complete request
read and bounds each socket write.

## 5. Install the systemd service

The archive includes templates under `share/systemd/`:

```bash
getent passwd aima >/dev/null || \
  sudo useradd --system --user-group \
    --home-dir /var/lib/aima-qwen36 --shell /usr/sbin/nologin aima
sudo usermod -aG render,video aima
sudo install -d -o aima -g aima /var/lib/aima-qwen36
sudo install -d -m 0750 -o root -g aima /srv/aima-media
sudo install -d -m 0750 -o root -g aima /etc/aima-qwen36
sudo install -Dm644 \
  /opt/aima-engine/share/systemd/aima-engine.service \
  /etc/systemd/system/aima-engine.service
sudo install -Dm640 \
  /opt/aima-engine/share/systemd/aima-engine.env.example \
  /etc/aima-qwen36/engine.env
openssl rand -hex 32 | sudo tee /etc/aima-qwen36/api-key >/dev/null
sudo chown root:aima /etc/aima-qwen36/api-key
sudo chmod 0640 /etc/aima-qwen36/api-key
sudo systemctl daemon-reload
```

Edit `/etc/aima-qwen36/engine.env` and grant `aima` read access to the model
directory and to every file placed under `AIMA_ALLOWED_LOCAL_MEDIA_PATH`. The
commands above create a non-login service account and add it to the GPU device
groups. The systemd template requires the bearer key, admits local media only
under the explicit `/srv/aima-media` root, sets a 15-second socket-operation
timeout and disables HTTP shutdown. It uses native systemd readiness
notification, so `systemctl start` completes only after both language and
vision paths are ready and the socket is listening. Then start the service:

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
API_KEY="$(sudo cat /etc/aima-qwen36/api-key)"
curl -fsS http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer ${API_KEY}"
```

The selected context is the preferred AOT prefill specialization, not a
mandatory request length. Any positive chat prompt is admitted when prompt plus
requested output fits the configured cache capacity. A cache miss starts from
empty resident state. A q8192 process keeps q1024/q2048/q4096/q8192 AOT
buckets resident, selects the largest one no longer than the prompt and uses
the token path only below q1024 or for a remaining tail. Exact and append cache
hits reduce latency but do not affect correctness or admission.

Use the native tokenizer probes when preparing deterministic fixtures:

```bash
/opt/aima-engine/bin/aima-engine chat-template-probe \
  --model-dir /srv/models/Qwen3.6-35B-A3B \
  --user 'prompt text' \
  --disable-thinking
```

To verify the native image path, first copy a test image under the allowlisted
root, then send an ordered OpenAI content-part request:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer ${API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"aima-amd395-qwen36-35b",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Describe the image briefly."},
      {"type":"image_url","image_url":{"url":"file:///srv/aima-media/example.jpg"}}
    ]}],
    "temperature":0,
    "max_tokens":128
  }'
```

Multiple `image_url` and `video_url` parts may be interleaved with text in one
request. Data URIs are accepted within the fixed byte limits. Remote HTTP/HTTPS
media remains disabled by the packaged systemd template until an administrator
adds exact `--allowed-media-domain` entries; private-address resolution needs
the separate `--allowed-private-media-domain` opt-in. See [API.md](API.md) for
the full media and cache contract.

## 7. Roll back to the exact v1.5.1 baseline

Keep the v1.5.1 archive and sidecar until the native VL deployment has passed
your own acceptance window. The qualified rollback target archive has SHA-256
`4e38f90fce3feb7bccf1965d87a3ec2bebddc439ce62e75fe1bc797c6ce1a5bc`.
Stop the resident process before changing files; never replace a running
bundle in place.

```bash
sha256sum -c aima-engine-native-portable-c12eb036ad77.tar.zst.sha256
sudo systemctl stop aima-engine
sudo mv /opt/aima-engine /opt/aima-engine-native-vl.1-stopped
sudo install -d /opt/aima-engine
sudo tar --zstd \
  -xf aima-engine-native-portable-c12eb036ad77.tar.zst \
  --strip-components=1 -C /opt/aima-engine
sudo install -Dm644 \
  /opt/aima-engine/share/systemd/aima-engine.service \
  /etc/systemd/system/aima-engine.service
sudo systemctl daemon-reload
/opt/aima-engine/bin/aima-engine doctor \
  --model-dir /srv/models/Qwen3.6-35B-A3B --json
sudo systemctl start aima-engine
curl -fsS http://127.0.0.1:8000/health
```

The move preserves the stopped native VL bundle for inspection or recovery.
The v1.5.1 target restores its exact text engine, launcher, providers and
product contract; image and video requests are intentionally unavailable after
rollback. Formal release evidence additionally runs the frozen q8192
128-token identity probe after the one-hour candidate soak and clean shutdown.

## 8. Build from source

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
export QUALIFICATION_RECORD=/path/to/qualified-product-result.json
export AIMA_RELEASE_VERSION=X.Y.Z
export AIMA_RELEASE_TAG=vX.Y.Z

make check
make build-native build-native-runtime
# Run the full qualification against these exact files, then:
make package-native
```

Release packaging rejects a dirty source tree, including non-ignored untracked
files, and records the exact release commit and the engine's embedded native
source commit in `manifest.json`. A release-only packaging or documentation
commit may follow the qualified native source; both roles remain explicit and
hash-bound. The packager also requires the native engine, launcher, three
providers, AOTriton runtime and selected GPU image to match the complete
qualification byte-for-byte. `AIMA_ALLOW_DIRTY_PACKAGE=1` exists only for
clearly marked local development bundles; do not publish such a bundle.
`make package-native` packages existing qualified artifacts and does not
rebuild them.

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
compatibility testing. It is not copied into the v1.5 native archive. The
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
- **Unexpected short-prompt latency:** inspect `aima_amd395.aot_prefill_tokens`.
  q8192 services keep q1024/q2048/q4096/q8192 AOT buckets resident; prompts
  below q1024 are padded into q1024, and non-bucket lengths compose the smallest
  resident bucket total that covers the prompt. Inspect
  `aot_prefill_bucket_tokens`, `aot_prefill_segments`, and
  `padded_prefill_tokens` to distinguish useful tokens from fixed-shape work;
  prompt ingestion never falls through to serial decode.
- **Remote bind is rejected:** configure `--api-key-file`; the unsafe override
  exists only for isolated diagnostics. Put TLS and network policy in a reverse
  proxy even when the built-in bearer token is enabled.
- **Different GPU or kernel:** the bundled userspace removes host software
  coupling, not the `gfx1151` and AMDGPU/KFD hardware/driver contract.
