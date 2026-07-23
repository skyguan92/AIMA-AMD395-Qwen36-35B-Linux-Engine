// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <aotriton/flash.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>

#if defined(_WIN32)
#define QRT_EXPORT extern "C" __declspec(dllexport)
#else
#define QRT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr std::size_t kQueryHeads = 16;
constexpr std::size_t kKvHeads = 2;
constexpr std::size_t kHeadDimension = 256;
constexpr std::size_t kQueryFeatures = kQueryHeads * kHeadDimension;
constexpr std::size_t kKvFeatures = kKvHeads * kHeadDimension;

struct State {
  void* output_bf16 = nullptr;
  void* softmax_lse = nullptr;
  void* persistent_atomic_counter = nullptr;
  void* philox_seed = nullptr;
  void* philox_offset = nullptr;
  std::size_t query_capacity = 0;
};

State g_state;
std::mutex g_mutex;

void release_locked() {
  (void)hipFree(g_state.philox_offset);
  (void)hipFree(g_state.philox_seed);
  (void)hipFree(g_state.persistent_atomic_counter);
  (void)hipFree(g_state.softmax_lse);
  (void)hipFree(g_state.output_bf16);
  g_state = {};
}

int prepare_locked(std::size_t query_capacity) {
  if (g_state.query_capacity >= query_capacity &&
      g_state.output_bf16 != nullptr &&
      g_state.softmax_lse != nullptr &&
      g_state.persistent_atomic_counter != nullptr &&
      g_state.philox_seed != nullptr && g_state.philox_offset != nullptr) {
    return static_cast<int>(hipSuccess);
  }
  release_locked();
  hipError_t status = hipMalloc(
      &g_state.output_bf16,
      query_capacity * kQueryFeatures * sizeof(hip_bfloat16));
  if (status == hipSuccess) {
    status = hipMalloc(
        &g_state.softmax_lse,
        query_capacity * kQueryHeads * sizeof(float));
  }
  if (status == hipSuccess) {
    status = hipMalloc(&g_state.persistent_atomic_counter,
                       sizeof(std::int32_t));
  }
  if (status == hipSuccess) {
    status = hipMalloc(&g_state.philox_seed, sizeof(std::uint64_t));
  }
  if (status == hipSuccess) {
    status = hipMalloc(&g_state.philox_offset, sizeof(std::uint64_t));
  }
  if (status != hipSuccess) {
    release_locked();
    return static_cast<int>(status);
  }
  g_state.query_capacity = query_capacity;
  return static_cast<int>(hipSuccess);
}

__global__ void bf16_to_f32_kernel(const hip_bfloat16* input,
                                   float* output,
                                   std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = static_cast<float>(input[index]);
}

int launch(const void* q, const void* k, const void* v, void* output,
           std::size_t query_tokens, std::size_t kv_tokens,
           hipStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      query_tokens == 0 || query_tokens > 262144 ||
      kv_tokens < query_tokens || kv_tokens > 262144) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    const int status = prepare_locked(query_tokens);
    if (status != static_cast<int>(hipSuccess)) return status;
  }

  using aotriton::DType;
  using aotriton::TensorView;
  const std::array<std::uint64_t, 4> q_sizes = {
      1, kQueryHeads, query_tokens, kHeadDimension};
  const std::array<std::uint64_t, 4> q_strides = {
      query_tokens * kQueryFeatures, kHeadDimension, kQueryFeatures, 1};
  const std::array<std::uint64_t, 4> kv_sizes = {
      1, kKvHeads, kv_tokens, kHeadDimension};
  const std::array<std::uint64_t, 4> kv_strides = {
      kv_tokens * kKvFeatures, kHeadDimension, kKvFeatures, 1};
  const std::array<std::uint64_t, 2> lse_sizes = {
      kQueryHeads, query_tokens};
  const std::array<std::uint64_t, 2> lse_strides = {query_tokens, 1};

  const TensorView<4> q_view{
      reinterpret_cast<std::intptr_t>(q), q_sizes, q_strides,
      DType::kBFloat16};
  const TensorView<4> k_view{
      reinterpret_cast<std::intptr_t>(k), kv_sizes, kv_strides,
      DType::kBFloat16};
  const TensorView<4> v_view{
      reinterpret_cast<std::intptr_t>(v), kv_sizes, kv_strides,
      DType::kBFloat16};
  const TensorView<4> output_view{
      reinterpret_cast<std::intptr_t>(g_state.output_bf16), q_sizes,
      q_strides, DType::kBFloat16};
  const TensorView<2> lse_view{
      reinterpret_cast<std::intptr_t>(g_state.softmax_lse), lse_sizes,
      lse_strides, DType::kFloat32};
  const TensorView<4> null_bf16 =
      TensorView<4>::get_null_tensor(DType::kBFloat16);
  const TensorView<0> seed_view{
      reinterpret_cast<std::intptr_t>(g_state.philox_seed), DType::kUInt64};
  const TensorView<0> offset_view{
      reinterpret_cast<std::intptr_t>(g_state.philox_offset),
      DType::kUInt64};
  const TensorView<0> null_u64 =
      TensorView<0>::get_null_tensor(DType::kUInt64);
  const TensorView<1> null_i32_1 =
      TensorView<1>::get_null_tensor(DType::kInt32);
  const TensorView<2> null_f32_2 =
      TensorView<2>::get_null_tensor(DType::kFloat32);

  aotriton::v3::flash::attn_fwd_params params;
  params.Q = q_view;
  params.K = k_view;
  params.V = v_view;
  params.B = null_bf16;
  params.A = null_f32_2;
  params.Sm_scale =
      1.0f / std::sqrt(static_cast<float>(kHeadDimension));
  params.L = lse_view;
  params.Out = output_view;
  params.cu_seqlens_q = null_i32_1;
  params.cu_seqlens_k = null_i32_1;
  params.Max_seqlen_q = static_cast<std::int32_t>(query_tokens);
  params.Max_seqlen_k = static_cast<std::int32_t>(kv_tokens);
  params.dropout_p = 0.0f;
  params.philox_seed_ptr = seed_view;
  params.philox_offset1 = offset_view;
  params.philox_offset2 = 0;
  params.philox_seed_output = null_u64;
  params.philox_offset_output = null_u64;
  params.encoded_softmax = null_bf16;
  params.persistent_atomic_counter = TensorView<0>{
      reinterpret_cast<std::intptr_t>(g_state.persistent_atomic_counter),
      DType::kInt32};
  params.causal_type = aotriton::v3::flash::CausalType::WindowedAttention;
  params.varlen_type = aotriton::v3::flash::VarlenType::None;
  params.window_left =
      aotriton::v3::flash::WindowValue::BottomRightAligned;
  params.window_right =
      aotriton::v3::flash::WindowValue::BottomRightAligned;
  hipError_t status = hipMemsetAsync(
      g_state.persistent_atomic_counter, 0, sizeof(std::int32_t), stream);
  if (status != hipSuccess) return static_cast<int>(status);
  status = aotriton::v3::flash::attn_fwd(
      params, aotriton::v3::flash::attn_fwd_params::kVersion,
      aotriton::Stream{stream}, nullptr);
  if (status != hipSuccess) return static_cast<int>(status);

  const std::size_t elements = query_tokens * kQueryFeatures;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(
      bf16_to_f32_kernel,
      dim3(static_cast<unsigned int>((elements + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, stream,
      static_cast<const hip_bfloat16*>(g_state.output_bf16),
      static_cast<float*>(output), elements);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

QRT_EXPORT int qrt_ck_fmha_prepare(unsigned int tokens) {
  if (tokens == 0 || tokens > 262144) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  return prepare_locked(tokens);
}

QRT_EXPORT int qrt_ck_fmha_bf16_launch(
    const void* q, const void* k, const void* v, void* output,
    unsigned int tokens, void* stream) {
  return launch(q, k, v, output, tokens, tokens,
                reinterpret_cast<hipStream_t>(stream));
}

QRT_EXPORT int qrt_ck_fmha_bf16_launch_ex(
    const void* q, const void* k, const void* v, void* output,
    unsigned int query_tokens, unsigned int kv_tokens, void* stream) {
  return launch(q, k, v, output, query_tokens, kv_tokens,
                reinterpret_cast<hipStream_t>(stream));
}

QRT_EXPORT int qrt_ck_fmha_release() {
  std::lock_guard<std::mutex> lock(g_mutex);
  release_locked();
  return static_cast<int>(hipSuccess);
}
