# Third-party notices

The repository is licensed under Apache License 2.0 except where a file carries
a different SPDX identifier. The following third-party material is included.

## AMD Composable Kernel

The generated CK-Tile sources below are distributed under the MIT License:

- `benchmarks/shape-lab/native/src/fmha_fwd_api.cpp`
- `benchmarks/shape-lab/native/src/fmha_fwd_gfx1151_d256_bf16_f32out.cpp`

The two `libqrt_ck_fmha_*.so` release binaries link the generated CK-Tile
instances with Apache-2.0 wrapper code. The AMD MIT text is preserved in
`third_party/licenses/AMD_COMPOSABLE_KERNEL_MIT.txt`.

## Runtime dependencies not bundled

ROCm, PyTorch, Triton, vLLM, Transformers and Safetensors are runtime
dependencies but are not copied into this repository. Their upstream licenses
continue to apply. See `docs/INSTALL.md` for the exact qualified versions.

## Model weights not bundled

Qwen3.6-35B-A3B model weights and tokenizer assets are not part of this
repository. Users must obtain them independently and comply with the model's
license and acceptable-use terms.
