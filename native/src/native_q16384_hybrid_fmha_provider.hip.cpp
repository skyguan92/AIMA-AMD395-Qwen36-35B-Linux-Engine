// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

// Fixed q16384 prefill provider.  It evaluates the two causal 8k query
// windows with embedded packed-GQA images and retains optional compile-time
// CK correction arms for qualification experiments.  The released profile
// uses all 16 packed heads in both windows; layer 39 is independently selected
// to the bundled CK provider by the resident-engine profile.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

#include "fmha_fwd.hpp"

#if defined(_WIN32)
#define AIMA_FMHA_EXPORT extern "C" __declspec(dllexport)
#else
#define AIMA_FMHA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

extern "C" const unsigned char aima_q16384_packed_gqa_hsaco_start[];
extern "C" const unsigned char aima_q16384_packed_gqa_hsaco_end[];
extern "C" const unsigned char aima_q8192_packed_gqa_hsaco_start[];
extern "C" const unsigned char aima_q8192_packed_gqa_hsaco_end[];

namespace {

constexpr unsigned int kTokens = 16384;
constexpr unsigned int kWindowTokens = 8192;
constexpr unsigned int kQueryHeads = 16;
constexpr unsigned int kKvHeads = 2;
constexpr unsigned int kHeadDim = 256;
constexpr unsigned int kQueryFeatures = kQueryHeads * kHeadDim;
constexpr unsigned int kKvFeatures = kKvHeads * kHeadDim;
constexpr unsigned int kThreads = 256;
#ifndef AIMA_Q16384_PACKED_HEADS
#define AIMA_Q16384_PACKED_HEADS 16
#endif
#ifndef AIMA_Q16384_PACKED_FIRST_WINDOW
#define AIMA_Q16384_PACKED_FIRST_WINDOW 1
#endif
constexpr unsigned int kPackedHeads = AIMA_Q16384_PACKED_HEADS;
constexpr bool kPackedFirstWindow = AIMA_Q16384_PACKED_FIRST_WINDOW != 0;
static_assert(kPackedHeads <= kQueryHeads);

struct ProviderState {
  float* continuation_ck_f32 = nullptr;
  hip_bfloat16* first_packed_bf16 = nullptr;
  hip_bfloat16* continuation_packed_bf16 = nullptr;
  hipModule_t first_packed_module = nullptr;
  hipFunction_t first_packed_function = nullptr;
  hipModule_t packed_module = nullptr;
  hipFunction_t packed_function = nullptr;
  bool prepared = false;
};

ProviderState g_state;
std::mutex g_mutex;

void release_locked() {
  if (g_state.first_packed_module != nullptr) {
    (void)hipModuleUnload(g_state.first_packed_module);
  }
  if (g_state.packed_module != nullptr) {
    (void)hipModuleUnload(g_state.packed_module);
  }
  (void)hipFree(g_state.continuation_packed_bf16);
  (void)hipFree(g_state.first_packed_bf16);
  (void)hipFree(g_state.continuation_ck_f32);
  g_state = {};
}

int prepare_locked() {
  if (g_state.prepared) return static_cast<int>(hipSuccess);
  const std::size_t continuation_elements =
      static_cast<std::size_t>(kWindowTokens) * kQueryFeatures;
  hipError_t status = hipMalloc(
      reinterpret_cast<void**>(&g_state.continuation_ck_f32),
      continuation_elements * sizeof(float));
  if (status == hipSuccess) {
    status = hipMalloc(
        reinterpret_cast<void**>(&g_state.first_packed_bf16),
        continuation_elements * sizeof(hip_bfloat16));
  }
  if (status == hipSuccess) {
    status = hipMalloc(
        reinterpret_cast<void**>(&g_state.continuation_packed_bf16),
        continuation_elements * sizeof(hip_bfloat16));
  }
  if (status == hipSuccess) {
    status = hipModuleLoadData(
        &g_state.first_packed_module,
        static_cast<const void*>(aima_q8192_packed_gqa_hsaco_start));
  }
  if (status == hipSuccess) {
    status = hipModuleGetFunction(
        &g_state.first_packed_function, g_state.first_packed_module,
        "_packed_gqa_mha_fwd");
  }
  if (status == hipSuccess) {
    status = hipModuleLoadData(
        &g_state.packed_module,
        static_cast<const void*>(aima_q16384_packed_gqa_hsaco_start));
  }
  if (status == hipSuccess) {
    status = hipModuleGetFunction(
        &g_state.packed_function, g_state.packed_module,
        "_packed_gqa_mha_fwd");
  }
  if (status != hipSuccess) {
    release_locked();
    return static_cast<int>(status);
  }
  g_state.prepared = true;
  return static_cast<int>(hipSuccess);
}

int launch_ck(const hip_bfloat16* q,
              const hip_bfloat16* k,
              const hip_bfloat16* v,
              float* output,
              unsigned int query_tokens,
              unsigned int kv_tokens,
              mask_enum mask,
              hipStream_t stream) {
  fmha_fwd_traits traits{};
  traits.hdim_q = kHeadDim;
  traits.hdim_v = kHeadDim;
  traits.data_type = "bf16";
  traits.is_group_mode = false;
  traits.is_v_rowmajor = true;
  traits.has_logits_soft_cap = false;
  traits.mask_type = mask;
  traits.bias_type = bias_enum::no_bias;
  traits.has_lse = false;
  traits.has_dropout = false;
  traits.qscale_type = quant_scale_enum::no_scale;
  traits.skip_min_seqlen_q = false;
  traits.has_sink = false;

  fmha_fwd_args args{};
  args.q_ptr = const_cast<hip_bfloat16*>(q);
  args.k_ptr = const_cast<hip_bfloat16*>(k);
  args.v_ptr = const_cast<hip_bfloat16*>(v);
  args.o_ptr = output;
  args.seqlen_q = query_tokens;
  args.seqlen_k = kv_tokens;
  args.batch = 1;
  args.max_seqlen_q = query_tokens;
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
  args.batch_stride_q =
      static_cast<std::uint64_t>(query_tokens) * kQueryFeatures;
  args.batch_stride_k =
      static_cast<std::uint64_t>(kv_tokens) * kKvFeatures;
  args.batch_stride_v =
      static_cast<std::uint64_t>(kv_tokens) * kKvFeatures;
  args.batch_stride_o =
      static_cast<std::uint64_t>(query_tokens) * kQueryFeatures;
  args.window_size_left = -1;
  args.window_size_right = 0;
  args.sink_size = 0;
  args.mask_type = static_cast<int>(mask);
  args.min_seqlen_q = query_tokens;
  args.p_drop = 0.0f;
  args.s_randval = false;
  args.drop_seed_offset = std::make_pair(std::uint64_t{0}, std::uint64_t{0});

  const ck_tile::stream_config config{stream};
  const float result = fmha_fwd(traits, args, config);
  if (result < 0.0f) return static_cast<int>(hipErrorInvalidValue);
  return static_cast<int>(hipGetLastError());
}

int launch_packed(hipFunction_t function,
                  const hip_bfloat16* q,
                  const hip_bfloat16* k,
                  const hip_bfloat16* v,
                  hip_bfloat16* output,
                  hipStream_t stream) {
  void* q_argument = const_cast<hip_bfloat16*>(q);
  void* k_argument = const_cast<hip_bfloat16*>(k);
  void* v_argument = const_cast<hip_bfloat16*>(v);
  void* output_argument = output;
  std::int32_t stride_qs = kQueryFeatures;
  std::int32_t stride_qh = kHeadDim;
  std::int32_t stride_ks = kKvFeatures;
  std::int32_t stride_kh = kHeadDim;
  std::int32_t stride_vs = kKvFeatures;
  std::int32_t stride_vh = kHeadDim;
  std::int32_t stride_os = kQueryFeatures;
  std::int32_t stride_oh = kHeadDim;
  hipDeviceptr_t global_scratch = 0;
  hipDeviceptr_t profile_scratch = 0;
  void* arguments[] = {
      &q_argument, &k_argument, &v_argument, &output_argument,
      &stride_qs, &stride_qh, &stride_ks, &stride_kh,
      &stride_vs, &stride_vh, &stride_os, &stride_oh,
      &global_scratch, &profile_scratch,
  };
  return static_cast<int>(hipModuleLaunchKernel(
      function, 512, 2, 1, 256, 1, 1, 65536, stream,
      arguments, nullptr));
}

__global__ void bf16_to_f32_kernel(const hip_bfloat16* input,
                                   float* output,
                                   std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = static_cast<float>(input[index]);
}

__global__ void merge_continuation_kernel(
    const hip_bfloat16* packed,
    const float* ck,
    float* output,
    std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::size_t feature = index % kQueryFeatures;
  const std::size_t head = feature / kHeadDim;
  if (head < kPackedHeads) {
    output[index] = static_cast<float>(packed[index]);
  } else {
    output[index] = static_cast<float>(hip_bfloat16(ck[index]));
  }
}

int launch(const void* q_pointer,
           const void* k_pointer,
           const void* v_pointer,
           void* output_pointer,
           unsigned int tokens,
           hipStream_t stream) {
  if (q_pointer == nullptr || k_pointer == nullptr || v_pointer == nullptr ||
      output_pointer == nullptr || tokens != kTokens) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    const int status = prepare_locked();
    if (status != static_cast<int>(hipSuccess)) return status;
  }
  const auto* q = static_cast<const hip_bfloat16*>(q_pointer);
  const auto* k = static_cast<const hip_bfloat16*>(k_pointer);
  const auto* v = static_cast<const hip_bfloat16*>(v_pointer);
  auto* output = static_cast<float*>(output_pointer);
  const std::size_t q_window_elements =
      static_cast<std::size_t>(kWindowTokens) * kQueryFeatures;
  const hip_bfloat16* continuation_q = q + q_window_elements;

  int status = static_cast<int>(hipSuccess);
  if constexpr (kPackedFirstWindow) {
    status = launch_packed(g_state.first_packed_function, q, k, v,
                           g_state.first_packed_bf16, stream);
    if (status == static_cast<int>(hipSuccess)) {
      hipLaunchKernelGGL(
          bf16_to_f32_kernel,
          dim3(static_cast<unsigned int>(
              (q_window_elements + kThreads - 1) / kThreads)),
          dim3(kThreads), 0, stream, g_state.first_packed_bf16, output,
          q_window_elements);
      status = static_cast<int>(hipGetLastError());
    }
  } else {
    status = launch_ck(q, k, v, output, kWindowTokens, kWindowTokens,
                       mask_enum::mask_top_left, stream);
  }
  if (status != static_cast<int>(hipSuccess)) return status;
  if constexpr (kPackedHeads < kQueryHeads) {
    status = launch_ck(continuation_q, k, v, g_state.continuation_ck_f32,
                       kWindowTokens, kTokens,
                       mask_enum::mask_bottom_right, stream);
    if (status != static_cast<int>(hipSuccess)) return status;
  }
  status = launch_packed(g_state.packed_function, continuation_q, k, v,
                         g_state.continuation_packed_bf16, stream);
  if (status != static_cast<int>(hipSuccess)) return status;

  const std::size_t continuation_elements = q_window_elements;
  hipLaunchKernelGGL(
      merge_continuation_kernel,
      dim3(static_cast<unsigned int>(
          (continuation_elements + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, stream,
      g_state.continuation_packed_bf16, g_state.continuation_ck_f32,
      output + q_window_elements, continuation_elements);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

AIMA_FMHA_EXPORT int qrt_ck_fmha_prepare(unsigned int tokens) {
  if (tokens != kTokens) return static_cast<int>(hipErrorInvalidValue);
  std::lock_guard<std::mutex> lock(g_mutex);
  return prepare_locked();
}

AIMA_FMHA_EXPORT int qrt_ck_fmha_bf16_launch(
    const void* q, const void* k, const void* v, void* output,
    unsigned int tokens, void* stream) {
  return launch(q, k, v, output, tokens,
                reinterpret_cast<hipStream_t>(stream));
}

AIMA_FMHA_EXPORT int qrt_ck_fmha_release() {
  std::lock_guard<std::mutex> lock(g_mutex);
  release_locked();
  return static_cast<int>(hipSuccess);
}
