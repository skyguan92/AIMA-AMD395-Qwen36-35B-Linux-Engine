// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_segmented_attention.h"

#include "aima/native_vl_processor.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kVisionHeads = 16;
constexpr std::size_t kVisionHeadDimension = 72;
constexpr std::size_t kAttentionThreads = 32;
constexpr std::size_t kAttentionKeyBlock = 64;
// The serving Triton kernel multiplies 1/sqrt(72) by its pinned RCP_LN2
// constant and performs exp2 softmax. This is the resulting float32 value.
constexpr float kSoftmaxScaleLog2 = 0.17002323269844055f;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

__device__ float warp_sum_32(float value) {
  for (int offset = 16; offset != 0; offset /= 2) {
    value += __shfl_down(value, offset, 32);
  }
  return value;
}

__global__ void vision_segmented_attention_kernel(
    const __hip_bfloat16* query, const __hip_bfloat16* key,
    const __hip_bfloat16* value, const std::uint32_t* segment_starts,
    const std::uint32_t* segment_lengths, __hip_bfloat16* output) {
  const std::size_t query_patch = blockIdx.x / kVisionHeads;
  const std::size_t head = blockIdx.x % kVisionHeads;
  const std::size_t lane = threadIdx.x;
  const std::size_t segment_start = segment_starts[query_patch];
  const std::size_t segment_length = segment_lengths[query_patch];
  const std::size_t query_base =
      (query_patch * kVisionHeads + head) * kVisionHeadDimension;

  float query_values[3] = {0.0f, 0.0f, 0.0f};
  float accumulators[3] = {0.0f, 0.0f, 0.0f};
  bool active_dimensions[3] = {false, false, false};
  for (std::size_t part = 0; part < 3; ++part) {
    const std::size_t dimension = lane + part * kAttentionThreads;
    active_dimensions[part] = dimension < kVisionHeadDimension;
    if (active_dimensions[part]) {
      query_values[part] =
          __bfloat162float(query[query_base + dimension]);
    }
  }

  __shared__ float scores[kAttentionKeyBlock];
  __shared__ __hip_bfloat16 probabilities[kAttentionKeyBlock];
  float running_maximum = -std::numeric_limits<float>::infinity();
  float running_sum = 0.0f;
  for (std::size_t key_block = 0; key_block < segment_length;
       key_block += kAttentionKeyBlock) {
    const std::size_t block_keys =
        segment_length - key_block < kAttentionKeyBlock
            ? segment_length - key_block
            : kAttentionKeyBlock;
    for (std::size_t key_offset = 0; key_offset < block_keys; ++key_offset) {
      const std::size_t key_patch = segment_start + key_block + key_offset;
      const std::size_t key_base =
          (key_patch * kVisionHeads + head) * kVisionHeadDimension;
      float partial_dot = 0.0f;
      for (std::size_t part = 0; part < 3; ++part) {
        const std::size_t dimension = lane + part * kAttentionThreads;
        if (active_dimensions[part]) {
          partial_dot = fmaf(
              query_values[part],
              __bfloat162float(key[key_base + dimension]), partial_dot);
        }
      }
      const float dot = warp_sum_32(partial_dot);
      if (lane == 0) scores[key_offset] = dot * kSoftmaxScaleLog2;
    }

    float old_accumulator_scale = 0.0f;
    if (lane == 0) {
      float block_maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t key_offset = 0; key_offset < block_keys; ++key_offset) {
        block_maximum = fmaxf(block_maximum, scores[key_offset]);
      }
      const float updated_maximum = fmaxf(running_maximum, block_maximum);
      old_accumulator_scale = exp2f(running_maximum - updated_maximum);
      float block_sum = 0.0f;
      for (std::size_t key_offset = 0; key_offset < block_keys; ++key_offset) {
        const float probability = exp2f(scores[key_offset] - updated_maximum);
        block_sum += probability;
        probabilities[key_offset] = __float2bfloat16(probability);
      }
      running_sum = running_sum * old_accumulator_scale + block_sum;
      running_maximum = updated_maximum;
    }
    old_accumulator_scale = __shfl(old_accumulator_scale, 0, 32);
    for (std::size_t part = 0; part < 3; ++part) {
      if (active_dimensions[part]) {
        accumulators[part] *= old_accumulator_scale;
      }
    }
    __syncthreads();
    for (std::size_t key_offset = 0; key_offset < block_keys; ++key_offset) {
      const std::size_t key_patch = segment_start + key_block + key_offset;
      const std::size_t value_base =
          (key_patch * kVisionHeads + head) * kVisionHeadDimension;
      const float probability =
          __bfloat162float(probabilities[key_offset]);
      for (std::size_t part = 0; part < 3; ++part) {
        const std::size_t dimension = lane + part * kAttentionThreads;
        if (active_dimensions[part]) {
          accumulators[part] = fmaf(
              probability,
              __bfloat162float(value[value_base + dimension]),
              accumulators[part]);
        }
      }
    }
    __syncthreads();
  }

  running_sum = __shfl(running_sum, 0, 32);
  for (std::size_t part = 0; part < 3; ++part) {
    const std::size_t dimension = lane + part * kAttentionThreads;
    if (active_dimensions[part]) {
      output[query_base + dimension] =
          __float2bfloat16(accumulators[part] / running_sum);
    }
  }
}

}  // namespace

struct NativeVisionSegmentedAttentionPlan::Impl {
  Impl(std::size_t requested_patch_count,
       const std::vector<std::uint32_t>& requested_cu_seqlens)
      : patch_count_value(requested_patch_count),
        workspace_bytes_value(2 * requested_patch_count *
                              sizeof(std::uint32_t)) {
    if (requested_patch_count == 0 ||
        requested_patch_count > 4 * kNativeVlAggregateTokenLimit) {
      throw std::invalid_argument(
          "native vision attention patch count is outside the serving budget");
    }
    if (requested_cu_seqlens.size() < 2 ||
        requested_cu_seqlens.front() != 0 ||
        requested_cu_seqlens.back() != requested_patch_count) {
      throw std::invalid_argument(
          "native vision attention sequence boundaries are invalid");
    }
    segment_count_value = requested_cu_seqlens.size() - 1;
    std::vector<std::uint32_t> starts(requested_patch_count);
    std::vector<std::uint32_t> lengths(requested_patch_count);
    for (std::size_t segment = 0; segment < segment_count_value; ++segment) {
      const std::uint32_t start = requested_cu_seqlens[segment];
      const std::uint32_t end = requested_cu_seqlens[segment + 1];
      if (end <= start || end > requested_patch_count) {
        throw std::invalid_argument(
            "native vision attention sequences must be non-empty");
      }
      const std::uint32_t length = end - start;
      for (std::uint32_t patch = start; patch < end; ++patch) {
        starts[patch] = start;
        lengths[patch] = length;
      }
    }
    try {
      check_hip(hipMalloc(&metadata_device, workspace_bytes_value),
                "hipMalloc vision attention metadata");
      segment_starts_device =
          static_cast<std::uint32_t*>(metadata_device);
      segment_lengths_device = segment_starts_device + requested_patch_count;
      check_hip(hipMemcpy(segment_starts_device, starts.data(),
                          starts.size() * sizeof(std::uint32_t),
                          hipMemcpyHostToDevice),
                "hipMemcpy vision attention starts");
      check_hip(hipMemcpy(segment_lengths_device, lengths.data(),
                          lengths.size() * sizeof(std::uint32_t),
                          hipMemcpyHostToDevice),
                "hipMemcpy vision attention lengths");
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() { release(); }

  void release() noexcept {
    if (metadata_device != nullptr) {
      const hipError_t ignored = hipFree(metadata_device);
      static_cast<void>(ignored);
      metadata_device = nullptr;
    }
    segment_starts_device = nullptr;
    segment_lengths_device = nullptr;
  }

  std::size_t patch_count_value = 0;
  std::size_t segment_count_value = 0;
  std::size_t workspace_bytes_value = 0;
  void* metadata_device = nullptr;
  std::uint32_t* segment_starts_device = nullptr;
  std::uint32_t* segment_lengths_device = nullptr;
};

NativeVisionSegmentedAttentionPlan::NativeVisionSegmentedAttentionPlan(
    std::size_t patch_count,
    const std::vector<std::uint32_t>& cu_seqlens)
    : impl_(std::make_unique<Impl>(patch_count, cu_seqlens)) {}
NativeVisionSegmentedAttentionPlan::~NativeVisionSegmentedAttentionPlan() =
    default;
NativeVisionSegmentedAttentionPlan::NativeVisionSegmentedAttentionPlan(
    NativeVisionSegmentedAttentionPlan&&) noexcept = default;
NativeVisionSegmentedAttentionPlan&
NativeVisionSegmentedAttentionPlan::operator=(
    NativeVisionSegmentedAttentionPlan&&) noexcept = default;

void NativeVisionSegmentedAttentionPlan::launch(
    const void* query_device, const void* key_device, const void* value_device,
    void* output_device, void* stream_pointer) const {
  if (!impl_ || query_device == nullptr || key_device == nullptr ||
      value_device == nullptr || output_device == nullptr ||
      query_device == output_device || key_device == output_device ||
      value_device == output_device) {
    throw std::invalid_argument(
        "native vision segmented attention launch is invalid");
  }
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  const std::size_t blocks = impl_->patch_count_value * kVisionHeads;
  hipLaunchKernelGGL(
      vision_segmented_attention_kernel, dim3(blocks),
      dim3(kAttentionThreads), 0, stream,
      static_cast<const __hip_bfloat16*>(query_device),
      static_cast<const __hip_bfloat16*>(key_device),
      static_cast<const __hip_bfloat16*>(value_device),
      impl_->segment_starts_device, impl_->segment_lengths_device,
      static_cast<__hip_bfloat16*>(output_device));
  check_hip(hipGetLastError(),
            "native vision segmented attention kernel launch");
}

std::size_t NativeVisionSegmentedAttentionPlan::patch_count() const {
  return impl_->patch_count_value;
}

std::size_t NativeVisionSegmentedAttentionPlan::segment_count() const {
  return impl_->segment_count_value;
}

std::size_t NativeVisionSegmentedAttentionPlan::workspace_bytes() const {
  return impl_->workspace_bytes_value;
}

}  // namespace aima
