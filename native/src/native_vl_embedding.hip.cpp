// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vl_embedding.h"

#include "aima/native_pointwise.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kLanguageHidden = 2048;
constexpr unsigned kThreads = 256;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

__global__ void native_vl_embedding_scatter_kernel(
    const __hip_bfloat16* visual_embeddings,
    const std::uint32_t* prompt_positions,
    const std::uint32_t* visual_rows,
    __hip_bfloat16* output,
    std::size_t visual_embedding_count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t elements = visual_embedding_count * kLanguageHidden;
  if (index >= elements) return;
  const std::size_t plan_row = index / kLanguageHidden;
  const std::size_t hidden = index - plan_row * kLanguageHidden;
  output[static_cast<std::size_t>(prompt_positions[plan_row]) *
             kLanguageHidden +
         hidden] =
      visual_embeddings[static_cast<std::size_t>(visual_rows[plan_row]) *
                            kLanguageHidden +
                        hidden];
}

}  // namespace

void launch_native_vl_embeddings(
    const void* token_embedding_bf16,
    const std::uint32_t* host_prompt_token_ids,
    const NativeVlEmbeddingPlan& plan,
    const void* visual_embeddings_bf16,
    void* device_prompt_token_ids,
    void* device_scatter_indices,
    void* output_bf16,
    void* stream_value) {
  if (token_embedding_bf16 == nullptr || host_prompt_token_ids == nullptr ||
      visual_embeddings_bf16 == nullptr ||
      device_prompt_token_ids == nullptr ||
      device_scatter_indices == nullptr || output_bf16 == nullptr ||
      plan.prompt_token_count() == 0 ||
      plan.visual_embedding_count() == 0 ||
      plan.prompt_positions().size() != plan.visual_embedding_count() ||
      plan.visual_rows().size() != plan.visual_embedding_count()) {
    throw std::invalid_argument(
        "native VL embeddings require a complete validated plan");
  }
  hipStream_t stream = static_cast<hipStream_t>(stream_value);
  launch_prompt_embeddings(token_embedding_bf16, host_prompt_token_ids,
                           device_prompt_token_ids, output_bf16,
                           plan.prompt_token_count(), stream);

  auto* prompt_positions =
      static_cast<std::uint32_t*>(device_scatter_indices);
  auto* visual_rows = prompt_positions + plan.visual_embedding_count();
  const std::size_t one_index_bytes =
      plan.visual_embedding_count() * sizeof(std::uint32_t);
  check_hip(hipMemcpyAsync(prompt_positions, plan.prompt_positions().data(),
                           one_index_bytes, hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync native VL prompt positions");
  check_hip(hipMemcpyAsync(visual_rows, plan.visual_rows().data(),
                           one_index_bytes, hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync native VL visual rows");
  const std::size_t elements =
      plan.visual_embedding_count() * kLanguageHidden;
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      native_vl_embedding_scatter_kernel, dim3(blocks), dim3(kThreads), 0,
      stream, static_cast<const __hip_bfloat16*>(visual_embeddings_bf16),
      prompt_positions, visual_rows,
      static_cast<__hip_bfloat16*>(output_bf16),
      plan.visual_embedding_count());
  check_hip(hipGetLastError(), "native_vl_embedding_scatter_kernel");
}

}  // namespace aima
