# Release provenance and procedure

## v1.4.0 boundary

The personal upstream owns the immutable `v1.4.0` tag and release assets; the
Approaching AI repository mirrors the product commit as the public showcase.
The native archive carries
[`product-contract-v1.4.0.json`](../native/product-contract-v1.4.0.json) as
`share/aima/product-contract.json`, together with the exact generated
qualification as `share/aima/qualification.json`. The qualification binds the
release/tag, clean source commit, native engine, launcher, all three FMHA
providers, AOTriton runtime and selected `gfx1151` image by byte size and
SHA-256. The recursive bundle manifest independently binds every packaged
file.

The v1.4.0 qualification reruns the complete 19-cell performance matrix,
nine-context full-vocabulary correctness gate, exact 128-token completion,
startup, prefix cache, resident HTTP, SSE, tools and isolated portable-bundle
smokes against the exact release binary before the tag is published. The
machine-readable qualification and raw reports are release assets; an
additive evidence commit may mirror them on `main` without moving the tag.

## v1.3.0 provenance

The immutable release tag is `v1.3.0` at commit
`032dc137992365649a47353910b76f93acb86d75`. The native source and generated
qualification record correspond to commit
`745930457f06629542ea996c8771ab38382fce98`; that change was developed from
v1.2 commit `e430e50dcb41af04465386287d696caa0ff22b10`. The already-published
qualification JSON and product contract remain byte-for-byte immutable. These
roles are recorded in the additive machine-readable
[`native-release-provenance-v1.3.0.json`](../benchmarks/results/native-release-provenance-v1.3.0.json)
erratum so a development base cannot be mistaken for the shipped release
commit.

## Fail-closed release flow

1. Commit the release source and make the checkout clean.
2. Build that commit, run the complete correctness/performance/surface matrix,
   and generate the candidate product result under ignored `output/`; both its
   release and native source commits must equal the clean checkout commit.
3. Set `QUALIFICATION_RECORD`, `AIMA_RELEASE_VERSION` and `AIMA_RELEASE_TAG`,
   then run `make package-native`. Packaging rejects a dirty checkout, a stale
   binary commit, an incomplete result, or any executable/provider byte that
   differs from the qualification.
4. Tag that same commit in the personal upstream and publish the archive,
   checksum and deterministic evidence asset there. Never move the tag.
5. Synchronize the product commit to the Approaching AI showcase fork. Raw
   evidence may be added to `main` in a later commit without changing the
   immutable release tag; use an additive erratum for any later clarification.

## Portable native release boundary

The v1.3 deployment unit is the deterministic
`aima-engine-native-portable-*.tar.zst` archive produced by
`make package-native`. The archive includes a recursive `manifest.json`, the
machine product contract and the qualification record.

Qualified executable components:

| Component | SHA-256 |
|---|---|
| native engine | `7b8ae090a80b90a066c39539313ebed92c899d370fc2ccf7021cda03cec0a81d` |
| static launcher | `7d945e0306299b4878b33b26f82d9b2d5c463bdca2422019e986abe1edd8e960` |
| AOTriton adapter | `8f42d7b17a778168a1bb66b34eff282e13955541ededfa838355ffbc176b43a5` |
| CK-Tile adapter | `77f6f6429ed7ef2e34a33372f6096a6d62957ba46f1866e7f40c39da9add25b4` |
| q16384 packed-GQA/CK hybrid | `ab48a7d605d92aaaf9dc17a10f217538e57974f0fdce9f06ddc5536cae601858` |
| AOTriton 0.11.1 library | `e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5` |
| selected gfx1151 AOTriton image | `0f3a6a2f9dee6620443ee2145ee1f8257bde65a378589952840d99bf3d485c10` |

The engine and provider RUNPATHs are `$ORIGIN`-relative. A static launcher
starts the engine with the archive's glibc loader, disables the host loader
cache, and supplies only `BUNDLE/lib`. The generated closure audit includes
all dynamically loaded FMHA providers and rejects unresolved dependencies or
non-relocatable RUNPATHs.

`scripts/qualify-native-portable-bundle.py` verifies the archive checksum and
recursive manifest, extracts it to a new path, audits the ELF closure, starts
the public CLI with only `HOME`, `LANG` and `PATH`, and runs full-model smokes
through the AOTriton, q16384 hybrid and long-context CK/AOTriton provider
policies.

Runtime dependency flags are all false for Python, PyTorch, vLLM, Triton and
Transformers; a host ROCm userspace is not required. The immutable host
contract remains Linux x86-64, AMDGPU/KFD/render nodes and `gfx1151`.

The admitted release profile contains all eight standard contexts from q1024
through q131072 plus the three valid 262144-token-window endpoints. All 19
performance cells, nine full-vocabulary correctness contexts, exact 128-token
identity, startup, prefix cache, resident HTTP, live SSE and function tools
passed their independent frozen gates.
The exact decision is in
[`native-portable-product-v1.3.0.json`](../benchmarks/results/native-portable-product-v1.3.0.json).
The result records that the published v1.1 long-context envelope is replaced
by the native package.

The five compact raw qualification directories referenced by the product and
portable-bundle results are published under `benchmarks/runs/`. Run
`make verify-evidence` to verify every summary and recursively referenced raw
report. The provenance erratum also binds the exact 92-file inventory, byte
count and sorted-path tree hash of all five directories, so extra, missing or
modified raw files fail verification. Run `make package-evidence` to create a deterministic checksummed
release asset containing that public evidence. Licensed oracle logits, model
weights and prompt content remain excluded.

## Native build provenance

The CK-Tile instance comes from AMD Composable Kernel commit
`6667a9021713f794a2c9aee4696c19f6cf376235`. The AOTriton adapter is built
against AOTriton 0.11.1; the q16384 hybrid embeds its packed-GQA code object.
The packager admits only the hashes above.
The engine embeds the captured gfx1151 prefill/decode code-object closure;
Python is used only at build time to generate static registries and the file
manifest.

All AIMA adapters and the native engine are Apache-2.0 project code except
where files preserve another SPDX header. The bundle copies the complete
upstream license/notice material for ROCm, AOTriton distribution assets,
Composable Kernel, ICU, glibc and other GNU/system libraries.

## Retained v1.1 compatibility provenance

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

- parameterized runtime and model paths;
- hostname-independent discovery of the exact AOTriton library;
- AIMA model and response metadata branding;
- direct standard-Safetensors loading and environment diagnostics;
- optional startup-image build and registration;
- release, security and third-party licensing metadata.

These changes do not alter model math, provider selection, sampling or cache
state transitions.

## v1.1.0 checkpoint-loading boundary

The default loader is new in v1.1.0, but the model-math engine hash above is
unchanged from v1.0.0. The native path allocates the same 693 independent BF16
Torch storages, reads the qualified byte ranges from the original Safetensors
shards and registers them under the existing raw-cache keys. It performs no
weight transformation, quantization or extra full-device copy.

Before readiness, the loader verifies all destination pointer types and the
complete 69,321,221,376-byte GPU payload against the qualified XOR and sum.
Three target-host starts passed that gate, used O_DIRECT for all 26 shards and
completed a live resident HTTP request. The optional striped loader remains
hash-qualified and uses the same destination/payload integrity contract.

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
fixed-shape wrappers, direct Safetensors loader and striped loader/builder are
Apache-2.0 project code.

The shipped binaries were built for `gfx1151` with ROCm/HIP 7.2. They are part
of the qualified release. Rebuilds go to `build/native/` and require a separate
correctness/performance qualification before substitution.

## Excluded material

The release deliberately excludes:

- model weights and tokenizer assets;
- optional generated 69.3 GB startup images;
- private host paths, credentials and machine-specific manifests;
- internal route ledgers, rejected variants and raw experiment artifacts;
- oracle logits or licensed reference-model outputs.
