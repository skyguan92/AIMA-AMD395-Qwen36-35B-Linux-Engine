// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_encoder.h"

#include "aima/bf16_gemm.h"
#include "aima/native_weight_store.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kPatchFeatures = 3 * 2 * 16 * 16;
constexpr std::size_t kVisionHidden = 1152;
constexpr std::size_t kPositionGridSide = 48;
constexpr std::size_t kWorkspaceLimit = 128ULL * 1024ULL * 1024ULL;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

const NativeTensorView& require_tensor(const NativeWeightStore& weights,
                                       const char* name, std::uint8_t rank,
                                       std::uint64_t payload_bytes) {
  const NativeTensorView* tensor = weights.find(name);
  if (tensor == nullptr || tensor->device_pointer == nullptr ||
      tensor->rank != rank || tensor->payload_bytes != payload_bytes) {
    throw std::runtime_error(std::string("native vision weight mismatch: ") +
                             name);
  }
  return *tensor;
}

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char* description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(std::string(description) + " overflows");
  }
  return left * right;
}

__device__ float triton_position_scale(std::size_t count) {
  return count <= 1 ? 0.0f
                    : static_cast<float>(kPositionGridSide - 1) /
                          static_cast<float>(count - 1);
}

__device__ float triton_position_coordinate(std::size_t index, float scale) {
  return static_cast<float>(index) * scale;
}

__device__ float triton_position_fraction(std::size_t index, float scale,
                                          std::size_t floor) {
  // Triton's gfx1151 lowering fuses index * scale - floor even though the
  // separately consumed coordinate is rounded to float32 first.
  return fmaf(static_cast<float>(index), scale,
              -static_cast<float>(floor));
}

__device__ __hip_bfloat16 triton_bf16_product(__hip_bfloat16 left,
                                              __hip_bfloat16 right) {
  const __hip_bfloat16_raw left_raw = left;
  const __hip_bfloat16_raw right_raw = right;
  const std::uint32_t left_pair = left_raw.x;
  const std::uint32_t right_pair = right_raw.x;
  std::uint32_t result = 0;
  // Triton lowers a scalar BF16 multiply on gfx1151 to a packed dot product
  // whose unused upper lanes and BF16 accumulator are zero.
  asm volatile("v_dot2_bf16_bf16 %0, %1, %2, 0"
               : "=v"(result)
               : "v"(left_pair), "v"(right_pair));
  return __hip_bfloat16(__hip_bfloat16_raw{
      static_cast<unsigned short>(result & 0xffffU)});
}

__global__ void vision_position_kernel(
    const __hip_bfloat16* table, const __hip_bfloat16* patch_embeddings,
    __hip_bfloat16* output, std::size_t output_row_offset,
    std::size_t height, std::size_t width, std::size_t element_count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) return;

  const std::size_t local_row = index / kVisionHidden;
  const std::size_t hidden = index % kVisionHidden;
  const std::size_t spatial_row = local_row % (height * width);
  const std::size_t merged_width = width / kNativeVlMergeSize;
  const std::size_t merged_cell = spatial_row / 4;
  const std::size_t inner = spatial_row % 4;
  const std::size_t source_y =
      (merged_cell / merged_width) * kNativeVlMergeSize + inner / 2;
  const std::size_t source_x =
      (merged_cell % merged_width) * kNativeVlMergeSize + inner % 2;

  const float y_scale = triton_position_scale(height);
  const float x_scale = triton_position_scale(width);
  const float y_position = triton_position_coordinate(source_y, y_scale);
  const float x_position = triton_position_coordinate(source_x, x_scale);
  const std::size_t y_floor = static_cast<std::size_t>(y_position);
  const std::size_t x_floor = static_cast<std::size_t>(x_position);
  const std::size_t y_ceil =
      y_floor + 1 < kPositionGridSide ? y_floor + 1 : y_floor;
  const std::size_t x_ceil =
      x_floor + 1 < kPositionGridSide ? x_floor + 1 : x_floor;
  const float dy = triton_position_fraction(source_y, y_scale, y_floor);
  const float dx = triton_position_fraction(source_x, x_scale, x_floor);
  const float w11 = dy * dx;
  const float w10 = dy - w11;
  const float w01 = dx - w11;
  const float w00 = 1.0f - dy - w01;

  const __hip_bfloat16 weight00 = __float2bfloat16(w00);
  const __hip_bfloat16 weight01 = __float2bfloat16(w01);
  const __hip_bfloat16 weight10 = __float2bfloat16(w10);
  const __hip_bfloat16 weight11 = __float2bfloat16(w11);
  const std::size_t table00 =
      (y_floor * kPositionGridSide + x_floor) * kVisionHidden + hidden;
  const std::size_t table01 =
      (y_floor * kPositionGridSide + x_ceil) * kVisionHidden + hidden;
  const std::size_t table10 =
      (y_ceil * kPositionGridSide + x_floor) * kVisionHidden + hidden;
  const std::size_t table11 =
      (y_ceil * kPositionGridSide + x_ceil) * kVisionHidden + hidden;
  // Triton's semantic layer keeps BF16 x BF16 arithmetic in BF16 for this
  // expression. Preserve its left-associative multiply/add rounding points.
  const __hip_bfloat16 product00 =
      triton_bf16_product(weight00, table[table00]);
  const __hip_bfloat16 product01 =
      triton_bf16_product(weight01, table[table01]);
  const __hip_bfloat16 product10 =
      triton_bf16_product(weight10, table[table10]);
  const __hip_bfloat16 product11 =
      triton_bf16_product(weight11, table[table11]);
  const __hip_bfloat16 sum01 = __float2bfloat16(
      __bfloat162float(product00) + __bfloat162float(product01));
  const __hip_bfloat16 sum012 = __float2bfloat16(
      __bfloat162float(sum01) + __bfloat162float(product10));
  const __hip_bfloat16 position = __float2bfloat16(
      __bfloat162float(sum012) + __bfloat162float(product11));
  const std::size_t output_index = output_row_offset * kVisionHidden + index;
  output[output_index] =
      patch_embeddings == nullptr
          ? position
          : __float2bfloat16(__bfloat162float(patch_embeddings[output_index]) +
                             __bfloat162float(position));
}

}  // namespace

struct NativeVisionPatchEmbedPlan::Impl {
  Impl(const NativeWeightStore& weights, std::size_t patches)
      : patch_count(patches),
        projection(
            require_tensor(weights,
                           "model.visual.patch_embed.proj.weight", 5,
                           kVisionHidden * kPatchFeatures *
                               sizeof(std::uint16_t))),
        bias(require_tensor(weights, "model.visual.patch_embed.proj.bias", 1,
                            kVisionHidden * sizeof(std::uint16_t))),
        gemm(patches, kVisionHidden, kPatchFeatures, kWorkspaceLimit, true,
             true) {
    if (patches == 0 || patches > 4 * 16384) {
      throw std::invalid_argument(
          "native vision patch count is outside the serving budget");
    }
    if (projection.shape !=
            std::array<std::uint32_t, 5>{1152, 3, 2, 16, 16} ||
        bias.shape[0] != kVisionHidden) {
      throw std::runtime_error("native vision patch weight shape is invalid");
    }
  }

  std::size_t patch_count = 0;
  const NativeTensorView& projection;
  const NativeTensorView& bias;
  Bf16GemmPlan gemm;
};

NativeVisionPatchEmbedPlan::NativeVisionPatchEmbedPlan(
    const NativeWeightStore& weights, std::size_t patch_count)
    : impl_(std::make_unique<Impl>(weights, patch_count)) {}
NativeVisionPatchEmbedPlan::~NativeVisionPatchEmbedPlan() = default;
NativeVisionPatchEmbedPlan::NativeVisionPatchEmbedPlan(
    NativeVisionPatchEmbedPlan&&) noexcept = default;
NativeVisionPatchEmbedPlan& NativeVisionPatchEmbedPlan::operator=(
    NativeVisionPatchEmbedPlan&&) noexcept = default;

void NativeVisionPatchEmbedPlan::launch(const void* pixel_values_device,
                                        void* output_device,
                                        void* stream) const {
  if (!impl_ || pixel_values_device == nullptr || output_device == nullptr) {
    throw std::invalid_argument("native vision patch launch is incomplete");
  }
  impl_->gemm.launch_with_bias(pixel_values_device,
                               impl_->projection.device_pointer,
                               impl_->bias.device_pointer, output_device,
                               stream);
}

std::size_t NativeVisionPatchEmbedPlan::patch_count() const {
  return impl_->patch_count;
}

std::size_t NativeVisionPatchEmbedPlan::workspace_bytes() const {
  return impl_->gemm.workspace_bytes();
}

struct NativeVisionPositionPlan::Impl {
  Impl(const NativeWeightStore& weights,
       const std::vector<NativeVlGrid>& requested_grids)
      : position_table(require_tensor(weights, "model.visual.pos_embed.weight",
                                      2, kPositionGridSide *
                                             kPositionGridSide * kVisionHidden *
                                             sizeof(std::uint16_t))),
        grids(requested_grids) {
    if (position_table.shape !=
        std::array<std::uint32_t, 5>{2304, 1152, 1, 1, 1}) {
      throw std::runtime_error("native vision position weight shape is invalid");
    }
    if (grids.empty()) {
      throw std::invalid_argument("native vision position grids are empty");
    }
    for (const NativeVlGrid& grid : grids) {
      if (grid.temporal == 0 || grid.height == 0 || grid.width == 0 ||
          grid.height % kNativeVlMergeSize != 0 ||
          grid.width % kNativeVlMergeSize != 0) {
        throw std::invalid_argument("native vision position grid is invalid");
      }
      const std::size_t spatial =
          checked_product(grid.height, grid.width, "vision spatial grid");
      const std::size_t patches =
          checked_product(grid.temporal, spatial, "vision patch grid");
      if (patches > 4 * kNativeVlAggregateTokenLimit ||
          patch_count_value > 4 * kNativeVlAggregateTokenLimit - patches) {
        throw std::invalid_argument(
            "native vision position patches exceed the serving budget");
      }
      patch_count_value += patches;
    }
  }

  void launch(const void* patch_embeddings_device, void* output_device,
              void* stream_pointer) const {
    if (output_device == nullptr) {
      throw std::invalid_argument("native vision position launch is incomplete");
    }
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
    std::size_t row_offset = 0;
    for (const NativeVlGrid& grid : grids) {
      const std::size_t rows = grid.temporal * grid.height * grid.width;
      const std::size_t elements = rows * kVisionHidden;
      constexpr std::size_t kThreads = 256;
      const std::size_t blocks = (elements + kThreads - 1) / kThreads;
      hipLaunchKernelGGL(vision_position_kernel, dim3(blocks), dim3(kThreads),
                         0, stream,
                         static_cast<const __hip_bfloat16*>(
                             position_table.device_pointer),
                         static_cast<const __hip_bfloat16*>(
                             patch_embeddings_device),
                         static_cast<__hip_bfloat16*>(output_device),
                         row_offset, grid.height, grid.width, elements);
      check_hip(hipGetLastError(), "native vision position kernel launch");
      row_offset += rows;
    }
  }

  const NativeTensorView& position_table;
  std::vector<NativeVlGrid> grids;
  std::size_t patch_count_value = 0;
};

NativeVisionPositionPlan::NativeVisionPositionPlan(
    const NativeWeightStore& weights, const std::vector<NativeVlGrid>& grids)
    : impl_(std::make_unique<Impl>(weights, grids)) {}
NativeVisionPositionPlan::~NativeVisionPositionPlan() = default;
NativeVisionPositionPlan::NativeVisionPositionPlan(
    NativeVisionPositionPlan&&) noexcept = default;
NativeVisionPositionPlan& NativeVisionPositionPlan::operator=(
    NativeVisionPositionPlan&&) noexcept = default;

void NativeVisionPositionPlan::launch(void* output_device, void* stream) const {
  if (!impl_) {
    throw std::invalid_argument("native vision position plan is empty");
  }
  impl_->launch(nullptr, output_device, stream);
}

void NativeVisionPositionPlan::launch_add(const void* patch_embeddings_device,
                                          void* output_device,
                                          void* stream) const {
  if (!impl_ || patch_embeddings_device == nullptr) {
    throw std::invalid_argument("native vision position add is incomplete");
  }
  impl_->launch(patch_embeddings_device, output_device, stream);
}

std::size_t NativeVisionPositionPlan::patch_count() const {
  return impl_->patch_count_value;
}

}  // namespace aima
