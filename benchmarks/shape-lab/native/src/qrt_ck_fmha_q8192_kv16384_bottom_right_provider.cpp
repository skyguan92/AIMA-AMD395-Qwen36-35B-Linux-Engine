// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

// Fixed q8192-query / kv16384 bottom-right causal wrapper around the exact
// generated CK-Tile gfx1151 BF16 D256 instance.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "fmha_fwd.hpp"

#define QRT_CK_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

constexpr unsigned int kQueryTokens = 8192u;
constexpr unsigned int kKvTokens = 16384u;
constexpr unsigned int kQueryHeads = 16u;
constexpr unsigned int kKvHeads = 2u;
constexpr unsigned int kHeadDim = 256u;
constexpr unsigned int kQueryFeatures = kQueryHeads * kHeadDim;
constexpr unsigned int kKvFeatures = kKvHeads * kHeadDim;

int launch_bf16_qkv(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    hipStream_t stream) {
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    fmha_fwd_traits traits{};
    traits.hdim_q = kHeadDim;
    traits.hdim_v = kHeadDim;
    traits.data_type = "bf16";
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.has_logits_soft_cap = false;
    traits.mask_type = mask_enum::mask_bottom_right;
    traits.bias_type = bias_enum::no_bias;
    traits.has_lse = false;
    traits.has_dropout = false;
    traits.qscale_type = quant_scale_enum::no_scale;
    traits.skip_min_seqlen_q = false;
    traits.has_sink = false;

    fmha_fwd_args args{};
    args.q_ptr = const_cast<uint16_t *>(q);
    args.k_ptr = const_cast<uint16_t *>(k);
    args.v_ptr = const_cast<uint16_t *>(v);
    args.o_ptr = output;
    args.seqlen_q = kQueryTokens;
    args.seqlen_k = kKvTokens;
    args.batch = 1;
    args.max_seqlen_q = kQueryTokens;
    args.hdim_q = kHeadDim;
    args.hdim_v = kHeadDim;
    args.nhead_q = kQueryHeads;
    args.nhead_k = kKvHeads;
    args.num_head_q_total = kQueryHeads;
    args.head_start = 0;
    args.scale_s = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    args.logits_soft_cap = 0.0f;
    args.stride_q = kQueryFeatures;
    args.stride_k = kKvFeatures;
    args.stride_v = kKvFeatures;
    args.stride_o = kQueryFeatures;
    args.nhead_stride_q = kHeadDim;
    args.nhead_stride_k = kHeadDim;
    args.nhead_stride_v = kHeadDim;
    args.nhead_stride_o = kHeadDim;
    args.batch_stride_q = kQueryTokens * kQueryFeatures;
    args.batch_stride_k = kKvTokens * kKvFeatures;
    args.batch_stride_v = kKvTokens * kKvFeatures;
    args.batch_stride_o = kQueryTokens * kQueryFeatures;
    args.window_size_left = -1;
    args.window_size_right = 0;
    args.sink_size = 0;
    args.mask_type = static_cast<int>(mask_enum::mask_bottom_right);
    args.min_seqlen_q = kQueryTokens;
    args.p_drop = 0.0f;
    args.s_randval = false;
    args.drop_seed_offset = std::make_pair(uint64_t{0}, uint64_t{0});

    const ck_tile::stream_config config{stream};
    const float launch_result = fmha_fwd(traits, args, config);
    if (launch_result < 0.0f) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    return static_cast<int>(hipGetLastError());
}

}  // namespace

QRT_CK_EXPORT int qrt_ck_fmha_q8192_kv16384_bottom_right_prepare() {
    return static_cast<int>(hipSuccess);
}

QRT_CK_EXPORT int qrt_ck_fmha_q8192_kv16384_bottom_right_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_qkv(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle));
}

QRT_CK_EXPORT int qrt_ck_fmha_q8192_kv16384_bottom_right_release() {
    return static_cast<int>(hipSuccess);
}
