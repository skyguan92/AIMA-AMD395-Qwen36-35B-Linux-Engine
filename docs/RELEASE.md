# Release provenance and procedure

## v1.5.1-native-vl.4 boundary

The immutable `v1.5.1-native-vl.4` tag is the first portable native release
that includes the fixed model's complete image, video and mixed-media product
surface. It retains batch size one and the 262,144-token total window, preserves
the certified greedy path, and adds seeded positive-temperature full-vocabulary
BF16/top-p generation. One resident native process performs media admission
and decode, processor transforms, the 27-block vision stack, merger, M-RoPE
media injection, language prefill/decode, SSE and tools. Model weights remain
an external pinned Hugging Face Safetensors input.

The qualified engine embeds native source commit
`bd012874027defa528279a357609b713e9069df4` and has SHA-256
`fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9`.
The release-only source/tag commit is recorded by
`native-release-provenance-v1.5.1-native-vl.4.json`; it does not rebuild or
substitute the qualified executable. The product contract is
[`product-contract-v1.5.1-native-vl.4.json`](../native/product-contract-v1.5.1-native-vl.4.json).

Promotion requires all five gates in `NATIVE_VL_GOAL.md`: complete frozen VL
capability, processor/boundary/logits/generation/task/error correctness, strict
v1.5.1 text no-regression, strict paired VL performance against the pinned
vLLM reference, and the native product/release boundary. The final G5 record
binds the exact archive, recursive manifest, package-input qualification,
primary and independent-host isolated bundle runs, at least one hour and 240
requests of resident text/image/video/mixed traffic, exact v1.5.1 rollback,
`make check`, `make security-scan` and `make verify-evidence`.

Raw evidence is added after the immutable source tag without moving it. The
additive provenance record binds G1-G5 summaries, every raw evidence tree and
checksum sidecar so an extra, missing or modified file fails verification.

## v1.5.1 boundary

The personal upstream owns the immutable `v1.5.1` patch tag and downloadable
assets; the Approaching AI repository mirrors the product commit and remains
the public issue surface. This release resolves
[`Approaching-AI#5`](https://github.com/Approaching-AI/AIMA-AMD395-Qwen36-35B-Linux-Engine/issues/5):
variable-length cold prompts and prefix extensions now compose resident
q1024/q2048/q4096/q8192 AOT schedules instead of sending unmatched prompt
tokens through serial decode. Only the final segment is padded when necessary,
and recurrent linear-attention state is repaired at the logical prompt tail.

The qualified native executable embeds source commit
`65c198415709dad6d046c247acab3dc9df2a95a0`. Release-only contract,
qualification and packaging commits do not rebuild it. Qualification reruns
the 19-cell performance matrix, nine full-vocabulary correctness contexts,
exact 128-token output, startup/prefix/HTTP surfaces, variable-length AOT
execution, SSE, tools, frozen MMLU-256/GB10 nonregression, isolated portable
bundle and second-host AMD395 compatibility. The target, `KLD<0.005`, top-1,
batch-size-one and maximum-window contracts are unchanged.

## v1.5.0 boundary

The personal upstream owns the immutable `v1.5.0` feature tag and release
assets; the Approaching AI repository mirrors the product commit and remains
the public issue surface. The release adds resident q1024/q2048/q4096/q8192
prefill dispatch for q8192 service and a capacity-bounded multi-entry exact-token
request-prefix LRU. Neither feature changes the model arithmetic, correctness
gate, batch-size-one contract or maximum context window.

Release qualification must bind the exact tagged binary and rerun the complete
19-cell performance matrix, nine-context full-vocabulary correctness gate,
exact completion, startup/prefix/HTTP surfaces, variable-length bucket cases,
SSE, tools, the frozen MMLU-256 score/nonregression pair against GB10 vLLM,
portable-bundle isolation and a second-host AMD395 compatibility smoke. The
public eval scorecard excludes prompt text and prompt token IDs. The immutable
commit, component hashes and raw-evidence links are recorded in
[`native-release-provenance-v1.5.0.json`](../benchmarks/results/native-release-provenance-v1.5.0.json).

The immutable tag is `v1.5.0` at release commit
`d82e6943bc50d821011ce79e95afee06f6b12a36`. The native executable embeds
source commit `2c4178ad95845b9a8ee00536f52671c77390c4b9`; release-only metadata
was added without rebuilding it. The exact archive was checksum-verified and
qualified in an isolated userspace, then reproduced on a second AMD395 running
Ubuntu 24.04 and kernel 7.0.0-28. That host reached `1722` cold-prefill tok/s
and `32.36` decode tok/s at q8192/output512, retaining `1.040x` and `1.002x`
of the published medians. Its exact 128-token output hash matched, and the
portable doctor, HTTP residency and exact-prefix replay all passed without
host ROCm userspace or a framework runtime.

## v1.4.1 boundary

The personal upstream owns the immutable `v1.4.1` patch tag and release
assets; the Approaching AI repository mirrors the product commit and remains
the public issue surface. This release removes the exact-static-length request
admission restriction without weakening the existing correctness gate. Every
positive prompt that fits the configured cache capacity is accepted, ordinary
multi-turn cache misses restart from clean resident state, and exact/append
prefix hits remain latency optimizations.

The fast published AOT contexts and provider policies are unchanged. A cold
prompt shorter than the selected specialization, or a tail beyond it, uses the
qualified token path and therefore runs at decode rather than AOT-prefill
throughput for that portion. Release qualification reruns the complete
19-cell performance matrix, nine-context full-vocabulary correctness gate,
startup/prefix/HTTP surfaces, variable-prompt isolation, SSE, tools and
portable-bundle isolation against the exact tagged binary.

The immutable tag is `v1.4.1` at release commit
`ba45639c178061f9bdadd22c86744f6924f5bf44`. The qualified native executable
embeds source commit `4536dbaeb6d1d013232db8150fbb6f7c3100b20a`; the later
release-only commits bind the generated qualification and preserve historical
contracts without rebuilding that executable. The archive carries
[`product-contract-v1.4.1.json`](../native/product-contract-v1.4.1.json) as
`share/aima/product-contract.json` and the exact generated product result as
`share/aima/qualification.json`.

## v1.4.0 boundary

The personal upstream owns the immutable `v1.4.0` tag and release assets; the
Approaching AI repository mirrors the product commit as the public showcase.
The native archive carries
[`product-contract-v1.4.0.json`](../native/product-contract-v1.4.0.json) as
`share/aima/product-contract.json`, together with the exact generated
qualification as `share/aima/qualification.json`. The qualification binds the
release/tag, clean release commit, embedded native source commit, native
engine, launcher, all three FMHA providers, AOTriton runtime and selected
`gfx1151` image by byte size and SHA-256. The recursive bundle manifest
independently binds every packaged file.

The v1.4.0 qualification reruns the complete 19-cell performance matrix,
nine-context full-vocabulary correctness gate, exact 128-token completion,
startup, prefix cache, resident HTTP, SSE, tools and isolated portable-bundle
smokes against the exact release binary before the tag is published. The
machine-readable qualification and raw reports are release assets and are
mirrored by an additive evidence commit on `main` without moving the tag.

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

1. Commit the native release source and make the checkout clean.
2. Build that commit and run the complete correctness/performance/surface
   matrix against its exact binary.
3. If packaging or documentation needs a release-only fix, commit it without
   rebuilding the qualified executable. Generate the candidate product result
   under ignored `output/` with the tag commit as `release_commit` and the
   executable's embedded commit as `native_source_commit`; the component
   hashes must remain unchanged.
4. Set `QUALIFICATION_RECORD`, `AIMA_RELEASE_VERSION` and `AIMA_RELEASE_TAG`,
   then run `make package-native`. This target packages the already-qualified
   artifacts; it deliberately does not rebuild them. Packaging rejects a dirty
   checkout, a stale or unbound binary commit, an incomplete result, or any
   executable/provider byte that differs from the qualification. Run
   `make build-native-runtime build-native` explicitly before qualification,
   never between qualification and packaging.
5. Tag the release commit in the personal upstream and publish the archive,
   checksum and deterministic evidence asset there. Never move the tag.
6. Synchronize the product commit to the Approaching AI showcase fork. Raw
   evidence may be added to `main` in a later commit without changing the
   immutable release tag; use an additive erratum for any later clarification.

For `v1.5.1-native-vl.4`, steps 3-5 are stricter: generate the package-input
qualification from a clean checkout of the intended tag, package those exact
bytes, run the same archive in isolated environments on two distinct AMD395
hosts, complete the primary-host one-hour resident soak, shut it down cleanly,
and prove rollback with the checksum-identical v1.5.1 archive. Run the static
release gates and final G5 aggregator from the same clean tagged checkout.
Only then copy the sealed outputs into an additive evidence commit and switch
the default evidence record to the new release.

## v1.5.1-native-vl.4 portable native release boundary

The archive contains the static launcher, qualified engine, all three language
attention providers, AOTriton runtime/image, both general and dense
vision-attention variants, minimal pinned FFmpeg and curl/c-ares stacks,
complete ROCm/system userspace closure, licenses, product contract,
qualification and public evidence. It loads all 693 language and 333 visual
tensors and completes both vision warmups before `READY=1`.

| Component | SHA-256 |
|---|---|
| native engine | `fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9` |
| static launcher | `d913b44ff33ad3903470817793e5bf095bc3cc6fe5eda00fc1562ed818323a43` |
| AOTriton adapter | `e5336b2d66b36c5f17aeb07ab780fa8f60a6092910f9b01b3ebf4bc31f766bb4` |
| CK-Tile adapter | `0145e819869d3ea5b25661f8f11279f5e6bd3484b29e8c7910a8b30c927baa93` |
| q16384 packed-GQA/CK hybrid | `e6b8c50e76c3c7d49b8c208275234d7f4607faff250019826866f86e37fedd29` |
| AOTriton 0.11.1 library | `e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5` |
| selected gfx1151 AOTriton image | `0f3a6a2f9dee6620443ee2145ee1f8257bde65a378589952840d99bf3d485c10` |
| general vision-attention image | `8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e` |
| embedded dense vision-attention image | `e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c` |

The package runs with only the Linux AMDGPU/KFD kernel interface from the
host. Python, PyTorch, vLLM, Triton, Transformers and host ROCm userspace remain
absent. Local and remote media are fail-closed behind explicit allowlists and
bounded I/O/decode policies. The recursive manifest and final archive hash are
recorded only after packaging; they are never predicted in this source
document.

## v1.5.1 portable native release boundary

The current v1.5.1 deployment unit is the deterministic
`aima-engine-native-portable-*.tar.zst` archive produced by
`make package-native`. The archive includes a recursive `manifest.json`, the
machine product contract and the qualification record.

Qualified executable components:

| Component | SHA-256 |
|---|---|
| native engine | `a9f18771175757af080c8a1d8d7e3fb3906c9aa41b43a496686103b626f80262` |
| static launcher | `ac43fb95a8bad8f9fb4e0f4eac9cadc4fb92f22189f4f35ce21a81f1d56fcf98` |
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
passed their independent frozen gates. Variable-prompt qualification also
passed a 16-token cold request and exact replay, an ordinary 36-token next-user
turn and an unrelated short request after a long-context request; both cache
misses restarted from clean state and returned HTTP 200.
The exact decision is embedded as `share/aima/qualification.json` and mirrored
after release as `benchmarks/results/native-portable-product-v1.5.1.json`.
The result records that the published v1.1 long-context envelope is replaced
by the native package.

The seven compact qualification directories referenced by the product,
portable-bundle and independent-host results are published under
[`benchmarks/runs/`](../benchmarks/runs/). Run
`make verify-evidence` to verify every summary and recursively referenced raw
report. The v1.5.0 provenance binds the exact 114-file inventory, byte count and
sorted-path tree hash of all seven directories, so extra, missing or modified
raw files fail verification. Run `make package-evidence` to create a
deterministic checksummed release asset containing that public evidence.
Licensed oracle logits, model weights and prompt content remain excluded.

The published portable archive is
`aima-engine-native-portable-86e806e8bc5d.tar.zst` (105,761,430 bytes), with
SHA-256
`5ca97a234c1132ec0e715f463107faa7bdce6dab6ab053918cf1465b8d2ea62b`.
The companion public-evidence asset is
`aima-engine-v1.5.0-public-evidence.tar.zst` (82,339 bytes), with SHA-256
`d47c49b874d8c421fdd270a0eaa698f6938348726576a20b2c5f0931eaac5e36`.

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
