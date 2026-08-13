// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_rotary.h"

#include "aima/native_vl_processor.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kVisionHidden = 1152;
constexpr std::size_t kVisionHeads = 16;
constexpr std::size_t kVisionHeadDimension = 72;
constexpr std::size_t kVisionRotaryHalfDimension = 36;
constexpr std::size_t kVisionQkvHidden = 3 * kVisionHidden;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

__global__ void vision_rotary_kernel(const __hip_bfloat16* qkv,
                                     const __hip_bfloat16* cos,
                                     const __hip_bfloat16* sin,
                                     __hip_bfloat16* query,
                                     __hip_bfloat16* key,
                                     __hip_bfloat16* value,
                                     std::size_t element_count) {
  const std::size_t output_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= element_count) return;

  const std::size_t patch = output_index / kVisionHidden;
  const std::size_t hidden = output_index % kVisionHidden;
  const std::size_t head_dimension = hidden % kVisionHeadDimension;
  const std::size_t rotary_dimension =
      head_dimension % kVisionRotaryHalfDimension;
  const std::size_t paired_hidden =
      hidden - head_dimension + rotary_dimension;
  const std::size_t qkv_row = patch * kVisionQkvHidden;
  const float cosine = __bfloat162float(
      cos[patch * kVisionRotaryHalfDimension + rotary_dimension]);
  const float sine = __bfloat162float(
      sin[patch * kVisionRotaryHalfDimension + rotary_dimension]);

  for (std::size_t qk = 0; qk < 2; ++qk) {
    const std::size_t qk_offset = qkv_row + qk * kVisionHidden;
    const float first = __bfloat162float(qkv[qk_offset + paired_hidden]);
    const float second = __bfloat162float(
        qkv[qk_offset + paired_hidden + kVisionRotaryHalfDimension]);
    // The serving Triton kernel promotes BF16 operands to FP32 and lets the
    // gfx1151 backend fuse the multiply-add/subtract before BF16 storage.
    const float rotated =
        head_dimension < kVisionRotaryHalfDimension
            ? fmaf(-second, sine, first * cosine)
            : fmaf(first, sine, second * cosine);
    (qk == 0 ? query : key)[output_index] = __float2bfloat16(rotated);
  }
  value[output_index] =
      qkv[qkv_row + 2 * kVisionHidden + hidden];
}

}  // namespace

NativeVisionRotaryPlan::NativeVisionRotaryPlan(std::size_t patch_count)
    : patch_count_(patch_count) {
  if (patch_count == 0 ||
      patch_count > 4 * kNativeVlAggregateTokenLimit) {
    throw std::invalid_argument(
        "native vision rotary patch count is outside the serving budget");
  }
}

void NativeVisionRotaryPlan::launch(
    const void* qkv_device, const void* cos_device, const void* sin_device,
    void* query_device, void* key_device, void* value_device,
    void* stream_pointer) const {
  if (qkv_device == nullptr || cos_device == nullptr || sin_device == nullptr ||
      query_device == nullptr || key_device == nullptr ||
      value_device == nullptr || query_device == key_device ||
      query_device == value_device || key_device == value_device ||
      qkv_device == query_device || qkv_device == key_device ||
      qkv_device == value_device || cos_device == query_device ||
      cos_device == key_device || cos_device == value_device ||
      sin_device == query_device || sin_device == key_device ||
      sin_device == value_device) {
    throw std::invalid_argument("native vision rotary launch is invalid");
  }
  constexpr std::size_t kThreads = 256;
  const std::size_t elements = patch_count_ * kVisionHeads * kVisionHeadDimension;
  const std::size_t blocks = (elements + kThreads - 1) / kThreads;
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
  hipLaunchKernelGGL(
      vision_rotary_kernel, dim3(blocks), dim3(kThreads), 0, stream,
      static_cast<const __hip_bfloat16*>(qkv_device),
      static_cast<const __hip_bfloat16*>(cos_device),
      static_cast<const __hip_bfloat16*>(sin_device),
      static_cast<__hip_bfloat16*>(query_device),
      static_cast<__hip_bfloat16*>(key_device),
      static_cast<__hip_bfloat16*>(value_device), elements);
  check_hip(hipGetLastError(), "native vision rotary kernel launch");
}

std::size_t NativeVisionRotaryPlan::patch_count() const {
  return patch_count_;
}

}  // namespace aima
