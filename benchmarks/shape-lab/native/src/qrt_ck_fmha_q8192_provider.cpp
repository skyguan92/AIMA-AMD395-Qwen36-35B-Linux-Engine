// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

// Fixed-shape wrapper around the generated CK-Tile q8192 FMHA instance.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "fmha_fwd.hpp"

#if defined(_WIN32)
#define QRT_CK_EXPORT extern "C" __declspec(dllexport)
#else
#define QRT_CK_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr unsigned int kTokens = 8192u;
constexpr unsigned int kQueryHeads = 16u;
constexpr unsigned int kKvHeads = 2u;
constexpr unsigned int kHeadDim = 256u;
constexpr unsigned int kQueryFeatures = kQueryHeads * kHeadDim;
constexpr unsigned int kKvFeatures = kKvHeads * kHeadDim;
constexpr unsigned int kPackedRows = 2u * kQueryFeatures + 2u * kKvFeatures;
constexpr unsigned int kThreads = 256u;

struct ProviderState {
    uint16_t *q = nullptr;
    uint16_t *k = nullptr;
    uint16_t *v = nullptr;
};

ProviderState g_state;
std::mutex g_state_mutex;

__device__ uint16_t f32_to_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        uint16_t upper = static_cast<uint16_t>(bits >> 16);
        if ((bits & 0x007fffffu) != 0u) {
            upper |= 0x0040u;
        }
        return upper;
    }
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

__global__ void pack_qkv_kernel(
    const float *__restrict__ packed,
    uint16_t *__restrict__ q,
    uint16_t *__restrict__ k,
    uint16_t *__restrict__ v) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t q_elements =
        static_cast<size_t>(kTokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(kTokens) * kKvFeatures;
    if (index < q_elements) {
        const size_t token = index / kQueryFeatures;
        const size_t feature = index - token * kQueryFeatures;
        q[index] = f32_to_bf16(
            packed[token * kPackedRows + feature]);
        return;
    }
    const size_t kv_index = index - q_elements;
    if (kv_index >= 2u * kv_elements) {
        return;
    }
    const size_t token = (kv_index % kv_elements) / kKvFeatures;
    const size_t feature = (kv_index % kv_elements) - token * kKvFeatures;
    if (kv_index < kv_elements) {
        k[kv_index] = f32_to_bf16(
            packed[token * kPackedRows +
                   2u * kQueryFeatures + feature]);
    } else {
        const size_t value_index = kv_index - kv_elements;
        v[value_index] = f32_to_bf16(
            packed[token * kPackedRows +
                   2u * kQueryFeatures + kKvFeatures + feature]);
    }
}

int prepare_locked() {
    if (g_state.q != nullptr && g_state.k != nullptr &&
        g_state.v != nullptr) {
        return static_cast<int>(hipSuccess);
    }
    const size_t q_bytes =
        static_cast<size_t>(kTokens) * kQueryFeatures * sizeof(uint16_t);
    const size_t kv_bytes =
        static_cast<size_t>(kTokens) * kKvFeatures * sizeof(uint16_t);
    hipError_t status = hipMalloc(
        reinterpret_cast<void **>(&g_state.q), q_bytes);
    if (status == hipSuccess) {
        status = hipMalloc(reinterpret_cast<void **>(&g_state.k), kv_bytes);
    }
    if (status == hipSuccess) {
        status = hipMalloc(reinterpret_cast<void **>(&g_state.v), kv_bytes);
    }
    if (status != hipSuccess) {
        (void)hipFree(g_state.v);
        (void)hipFree(g_state.k);
        (void)hipFree(g_state.q);
        g_state = ProviderState{};
    }
    return static_cast<int>(status);
}

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
    traits.mask_type = mask_enum::mask_top_left;
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
    args.seqlen_q = kTokens;
    args.seqlen_k = kTokens;
    args.batch = 1;
    args.max_seqlen_q = kTokens;
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
    args.batch_stride_q = kTokens * kQueryFeatures;
    args.batch_stride_k = kTokens * kKvFeatures;
    args.batch_stride_v = kTokens * kKvFeatures;
    args.batch_stride_o = kTokens * kQueryFeatures;
    args.window_size_left = -1;
    args.window_size_right = 0;
    args.sink_size = 0;
    args.mask_type = static_cast<int>(mask_enum::mask_top_left);
    args.min_seqlen_q = kTokens;
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

QRT_CK_EXPORT int qrt_ck_fmha_q8192_prepare() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return prepare_locked();
}

QRT_CK_EXPORT int qrt_ck_fmha_q8192_f32_launch(
    const float *packed_qkv,
    float *output,
    void *stream_handle) {
    if (packed_qkv == nullptr || output == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        const int status = prepare_locked();
        if (status != static_cast<int>(hipSuccess)) {
            return status;
        }
    }
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_handle);
    const size_t q_elements =
        static_cast<size_t>(kTokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(kTokens) * kKvFeatures;
    const size_t packed_elements = q_elements + 2u * kv_elements;
    hipLaunchKernelGGL(
        pack_qkv_kernel,
        dim3(static_cast<unsigned int>(
            (packed_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads),
        0u,
        stream,
        packed_qkv,
        g_state.q,
        g_state.k,
        g_state.v);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        return static_cast<int>(status);
    }
    return launch_bf16_qkv(
        g_state.q,
        g_state.k,
        g_state.v,
        output,
        stream);
}

QRT_CK_EXPORT int qrt_ck_fmha_q8192_bf16_launch(
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

QRT_CK_EXPORT int qrt_ck_fmha_q8192_release() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    (void)hipFree(g_state.v);
    (void)hipFree(g_state.k);
    (void)hipFree(g_state.q);
    g_state = ProviderState{};
    return static_cast<int>(hipSuccess);
}
