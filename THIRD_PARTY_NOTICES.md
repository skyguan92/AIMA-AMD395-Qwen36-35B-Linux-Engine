# Third-party notices

The repository is licensed under Apache License 2.0 except where a file carries
a different SPDX identifier. The following third-party material is included.

## AMD Composable Kernel

The generated CK-Tile sources below are distributed under the MIT License:

- `benchmarks/shape-lab/native/src/fmha_fwd_api.cpp`
- `benchmarks/shape-lab/native/src/fmha_fwd_gfx1151_d256_bf16_f32out.cpp`

The `libaima-fmha-ck.so` release binary links the generated CK-Tile instance
with Apache-2.0 wrapper code. The AMD MIT text is preserved in
`third_party/licenses/AMD_COMPOSABLE_KERNEL_MIT.txt`.

## AOTriton

The portable native bundle contains `libaotriton_v2.so.0.11.1` and one
shape-selected gfx1151 forward-attention image from the qualified PyTorch
distribution. AIMA's `libaima-fmha-aotriton.so` adapter is Apache-2.0 project
code. The generated bundle preserves the distributing wheel's complete
LICENSE and NOTICE files; AOTriton and its image are not relicensed by AIMA.

## vLLM ROCm skinny GEMM

`native/src/bf16_wvsplitk.hip.cpp` contains a modified, BF16 batch-1 gfx1151
specialization of `csrc/rocm/skinny_gemms.cu` from vLLM commit
`29e5d102050669d03992a2eb863ad364ea50fab2`. The upstream source and this
derived file are licensed under Apache License 2.0. The native version removes
Torch/c10 dispatch and allocation; it retains the qualified device-resident
projection algorithm.

## Compatibility runtime dependencies

The retained v1.1 compatibility runtime uses ROCm, PyTorch, Triton, vLLM,
Transformers and Safetensors. They are not copied into this repository and
their upstream licenses continue to apply. The v1.4 portable native runtime
does not load them.

## Portable native bundle

The `package-native` output copies the qualified HIP,
hipBLASLt, RocRoller, ROCTx, HSA Runtime, AQL Profile, rocprofiler-register, AMD
COMGR and ROCm device-library assets from the builder's ROCm installation. The
hipBLASLt payload is restricted to its gfx1151 library data and code objects.
Those components are not
relicensed by this project. The generated bundle includes their corresponding
upstream license texts. The distributing PyTorch wheel is used only as a
build-time source for the pinned AOTriton library, image, LICENSE and NOTICE;
Python, PyTorch, Triton, vLLM, Transformers and Safetensors are not included
as runtimes in that bundle.

The native tokenizer statically links ICU. The generated bundle includes the
ICU copyright and license text. ICU is not relicensed by this project.

The portable native bundle also contains the GNU C Library dynamic loader and
runtime, libstdc++, libgcc, libelf, libdrm, libnuma, zlib, zstd, liblzma,
libpng, libjpeg-turbo, libwebp and libsharpyuv copied from
the qualified builder. These libraries retain their respective upstream
licenses; the generated bundle includes the corresponding distribution
copyright and license notices. They are included to remove host userspace
version coupling, not relicensed under the project's Apache-2.0 license.

## Model weights not bundled

Qwen3.6-35B-A3B model weights and tokenizer assets are not part of this
repository. Users must obtain them independently and comply with the model's
license and acceptable-use terms.
