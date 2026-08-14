// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_pipeline.h"

#include "aima/native_vision_aot_block_stack.h"
#include "aima/native_vision_encoder.h"
#include "aima/native_vision_merger.h"
#include "aima/native_weight_store.h"
#include "aima/sha256.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kVisionHidden = 1152;
constexpr std::size_t kSpatialMergeArea = 4;
constexpr std::size_t kVisionHeadDimension = 72;
constexpr std::size_t kVisionRotaryDimension = 36;
constexpr std::size_t kVisionRotaryFrequencyCount = 18;
constexpr std::size_t kVisionRotaryAxes = 2;
constexpr std::size_t kVisionRotaryMaximumPosition = 8192;
static_assert(kVisionRotaryAxes * kVisionRotaryFrequencyCount ==
              kVisionRotaryDimension);
static_assert(kVisionHeadDimension / 2 == kVisionRotaryDimension);

// Exact float32 output of vLLM's gfx1151
//   1 / (10000 ** (arange(0, 36, 2) / 36))
// cache initialization. PyTorch deliberately builds this cache on the GPU;
// CPU libm differs by one BF16 ULP for a small set of positions above the
// original square-image qualification range.
__device__ __constant__ std::int32_t kVisionInverseFrequencyBits[
    kVisionRotaryFrequencyCount] = {
    0x3f800000, 0x3f1977cc, 0x3eb800d6, 0x3e5c9d37, 0x3e044133,
    0x3d9e91b6, 0x3d3e1e95, 0x3ce3f27e, 0x3c88a69b, 0x3c23d70a,
    0x3bc47060, 0x3b6b8631, 0x3b0d3169, 0x3aa94938, 0x3a4af7f3,
    0x39f35a5c, 0x3991e2e1, 0x392ee9b8};

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(std::string(description) + " overflows");
  }
  return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char* description) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::invalid_argument(std::string(description) + " overflows");
  }
  return left + right;
}

struct HostEncoderMetadata {
  std::size_t patch_count = 0;
  std::vector<std::uint32_t> cu_seqlens;
  std::vector<std::uint16_t> position_ids;
};

__global__ void vision_rotary_metadata_kernel(
    const std::uint16_t* position_ids, __hip_bfloat16* cos,
    __hip_bfloat16* sin, std::size_t element_count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) return;
  const std::size_t patch = index / kVisionRotaryDimension;
  const std::size_t dimension = index % kVisionRotaryDimension;
  const std::size_t axis = dimension / kVisionRotaryFrequencyCount;
  const std::size_t frequency = dimension % kVisionRotaryFrequencyCount;
  const float inverse_frequency =
      __int_as_float(kVisionInverseFrequencyBits[frequency]);
  const float angle = static_cast<float>(
                          position_ids[patch * kVisionRotaryAxes + axis]) *
                      inverse_frequency;
  cos[index] = __float2bfloat16(cosf(angle));
  sin[index] = __float2bfloat16(sinf(angle));
}

HostEncoderMetadata build_host_metadata(
    const std::vector<NativeVlGrid>& grids) {
  if (grids.empty()) {
    throw std::invalid_argument("native vision encoder grids are empty");
  }

  HostEncoderMetadata result;
  result.cu_seqlens.push_back(0);
  for (const NativeVlGrid& grid : grids) {
    if (grid.temporal == 0 || grid.height == 0 || grid.width == 0 ||
        grid.height % kNativeVlMergeSize != 0 ||
        grid.width % kNativeVlMergeSize != 0 ||
        grid.height > kVisionRotaryMaximumPosition ||
        grid.width > kVisionRotaryMaximumPosition) {
      throw std::invalid_argument("native vision encoder grid is invalid");
    }
    const std::size_t patches_per_frame =
        checked_multiply(grid.height, grid.width, "vision frame patches");
    const std::size_t grid_patches = checked_multiply(
        grid.temporal, patches_per_frame, "vision grid patches");
    if (grid_patches > kNativeVlVisionBatchPatchLimit ||
        result.patch_count >
            kNativeVlVisionBatchPatchLimit - grid_patches) {
      throw std::invalid_argument(
          "native vision encoder grids exceed the serving budget");
    }
    for (std::size_t frame = 0; frame < grid.temporal; ++frame) {
      const std::size_t end = result.patch_count +
                              (frame + 1) * patches_per_frame;
      if (end > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "native vision encoder sequence boundary overflows");
      }
      result.cu_seqlens.push_back(static_cast<std::uint32_t>(end));
    }
    result.patch_count += grid_patches;
  }

  const std::size_t position_elements = checked_multiply(
      result.patch_count, kVisionRotaryAxes, "vision rotary positions");
  result.position_ids.reserve(position_elements);
  for (const NativeVlGrid& grid : grids) {
    const std::size_t merged_height = grid.height / kNativeVlMergeSize;
    const std::size_t merged_width = grid.width / kNativeVlMergeSize;
    for (std::size_t frame = 0; frame < grid.temporal; ++frame) {
      static_cast<void>(frame);
      for (std::size_t merged_y = 0; merged_y < merged_height; ++merged_y) {
        for (std::size_t merged_x = 0; merged_x < merged_width; ++merged_x) {
          for (std::size_t inner_y = 0; inner_y < kNativeVlMergeSize;
               ++inner_y) {
            for (std::size_t inner_x = 0; inner_x < kNativeVlMergeSize;
                 ++inner_x) {
              const std::array<std::size_t, kVisionRotaryAxes> positions{
                  merged_y * kNativeVlMergeSize + inner_y,
                  merged_x * kNativeVlMergeSize + inner_x};
              for (const std::size_t position : positions) {
                result.position_ids.push_back(
                    static_cast<std::uint16_t>(position));
              }
            }
          }
        }
      }
    }
  }
  if (result.position_ids.size() != position_elements ||
      result.cu_seqlens.back() != result.patch_count) {
    throw std::runtime_error("native vision encoder metadata is inconsistent");
  }
  return result;
}

}  // namespace

struct NativeVisionEncoderMetadataPlan::Impl {
  explicit Impl(const std::vector<NativeVlGrid>& grids) {
    HostEncoderMetadata host = build_host_metadata(grids);
    patch_count_value = host.patch_count;
    cu_seqlens_value = std::move(host.cu_seqlens);
    const std::size_t one_table_bytes = checked_multiply(
        checked_multiply(patch_count_value, kVisionRotaryDimension,
                         "vision rotary rows"),
        sizeof(std::uint16_t), "vision rotary bytes");
    resident_bytes_value =
        checked_multiply(2, one_table_bytes, "vision metadata residency");
    const std::size_t position_bytes = checked_multiply(
        host.position_ids.size(), sizeof(std::uint16_t),
        "vision position metadata bytes");
    void* position_device = nullptr;
    try {
      check_hip(hipMalloc(&metadata_device, resident_bytes_value),
                "hipMalloc native vision encoder metadata");
      cos_device = metadata_device;
      sin_device = static_cast<unsigned char*>(metadata_device) +
                   one_table_bytes;
      check_hip(hipMalloc(&position_device, position_bytes),
                "hipMalloc native vision rotary positions");
      check_hip(hipMemcpy(position_device, host.position_ids.data(),
                          position_bytes, hipMemcpyHostToDevice),
                "hipMemcpy native vision rotary positions");
      const std::size_t output_elements =
          patch_count_value * kVisionRotaryDimension;
      constexpr std::size_t kThreads = 256;
      const std::size_t blocks =
          (output_elements + kThreads - 1) / kThreads;
      hipLaunchKernelGGL(
          vision_rotary_metadata_kernel, dim3(blocks), dim3(kThreads), 0,
          nullptr, static_cast<const std::uint16_t*>(position_device),
          static_cast<__hip_bfloat16*>(cos_device),
          static_cast<__hip_bfloat16*>(sin_device), output_elements);
      check_hip(hipGetLastError(),
                "native vision rotary metadata kernel launch");
      std::vector<std::uint16_t> cos_host(output_elements);
      std::vector<std::uint16_t> sin_host(output_elements);
      check_hip(hipMemcpy(cos_host.data(), cos_device, one_table_bytes,
                          hipMemcpyDeviceToHost),
                "hipMemcpy native vision rotary cosine hash");
      check_hip(hipMemcpy(sin_host.data(), sin_device, one_table_bytes,
                          hipMemcpyDeviceToHost),
                "hipMemcpy native vision rotary sine hash");
      cos_sha256_value = sha256_bytes(cos_host.data(), one_table_bytes);
      sin_sha256_value = sha256_bytes(sin_host.data(), one_table_bytes);
      check_hip(hipFree(position_device),
                "hipFree native vision rotary positions");
      position_device = nullptr;
    } catch (...) {
      if (position_device != nullptr) {
        const hipError_t ignored = hipFree(position_device);
        static_cast<void>(ignored);
      }
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
    cos_device = nullptr;
    sin_device = nullptr;
  }

  std::size_t patch_count_value = 0;
  std::size_t resident_bytes_value = 0;
  std::vector<std::uint32_t> cu_seqlens_value;
  std::string cos_sha256_value;
  std::string sin_sha256_value;
  void* metadata_device = nullptr;
  void* cos_device = nullptr;
  void* sin_device = nullptr;
};

NativeVisionEncoderMetadataPlan::NativeVisionEncoderMetadataPlan(
    const std::vector<NativeVlGrid>& grids)
    : impl_(std::make_unique<Impl>(grids)) {}
NativeVisionEncoderMetadataPlan::~NativeVisionEncoderMetadataPlan() = default;
NativeVisionEncoderMetadataPlan::NativeVisionEncoderMetadataPlan(
    NativeVisionEncoderMetadataPlan&&) noexcept = default;
NativeVisionEncoderMetadataPlan& NativeVisionEncoderMetadataPlan::operator=(
    NativeVisionEncoderMetadataPlan&&) noexcept = default;

const void* NativeVisionEncoderMetadataPlan::rotary_cos_device() const {
  return impl_->cos_device;
}

const void* NativeVisionEncoderMetadataPlan::rotary_sin_device() const {
  return impl_->sin_device;
}

const std::vector<std::uint32_t>&
NativeVisionEncoderMetadataPlan::cu_seqlens() const {
  return impl_->cu_seqlens_value;
}

const std::string& NativeVisionEncoderMetadataPlan::rotary_cos_sha256() const {
  return impl_->cos_sha256_value;
}

const std::string& NativeVisionEncoderMetadataPlan::rotary_sin_sha256() const {
  return impl_->sin_sha256_value;
}

std::size_t NativeVisionEncoderMetadataPlan::patch_count() const {
  return impl_->patch_count_value;
}

std::size_t NativeVisionEncoderMetadataPlan::resident_bytes() const {
  return impl_->resident_bytes_value;
}

struct NativeVisionPipelinePlan::Impl {
  Impl(const NativeWeightStore& weights,
       const std::filesystem::path& attention_image_path,
       const std::vector<NativeVlGrid>& grids)
      : metadata(grids),
        patch(weights, metadata.patch_count()),
        position(weights, grids),
        blocks(weights, attention_image_path, metadata.patch_count(),
               metadata.cu_seqlens()),
        merger(weights, metadata.patch_count()),
        hidden_bytes(checked_multiply(
            checked_multiply(metadata.patch_count(), kVisionHidden,
                             "vision hidden rows"),
            sizeof(std::uint16_t), "vision hidden bytes")),
        shared_temporary_bytes(
            std::max(blocks.temporary_bytes(), merger.temporary_bytes())),
        temporary_bytes_value(checked_add(
            checked_multiply(2, hidden_bytes, "vision hidden arenas"),
            shared_temporary_bytes, "vision pipeline workspace")),
        library_workspace_bytes_value(checked_add(
            checked_add(patch.workspace_bytes(),
                        blocks.library_workspace_bytes(),
                        "vision library workspace"),
            merger.library_workspace_bytes(), "vision library workspace")) {
    if (metadata.patch_count() % kSpatialMergeArea != 0 ||
        merger.merged_token_count() !=
            metadata.patch_count() / kSpatialMergeArea ||
        position.patch_count() != metadata.patch_count() ||
        blocks.patch_count() != metadata.patch_count() ||
        merger.patch_count() != metadata.patch_count()) {
      throw std::runtime_error("native vision pipeline subplans disagree");
    }
  }

  NativeVisionEncoderMetadataPlan metadata;
  NativeVisionPatchEmbedPlan patch;
  NativeVisionPositionPlan position;
  NativeVisionAotBlockStackPlan blocks;
  NativeVisionMergerPlan merger;
  std::size_t hidden_bytes = 0;
  std::size_t shared_temporary_bytes = 0;
  std::size_t temporary_bytes_value = 0;
  std::size_t library_workspace_bytes_value = 0;
};

NativeVisionPipelinePlan::NativeVisionPipelinePlan(
    const NativeWeightStore& weights,
    const std::filesystem::path& attention_image_path,
    const std::vector<NativeVlGrid>& grids)
    : impl_(
          std::make_unique<Impl>(weights, attention_image_path, grids)) {}
NativeVisionPipelinePlan::~NativeVisionPipelinePlan() = default;
NativeVisionPipelinePlan::NativeVisionPipelinePlan(
    NativeVisionPipelinePlan&&) noexcept = default;
NativeVisionPipelinePlan& NativeVisionPipelinePlan::operator=(
    NativeVisionPipelinePlan&&) noexcept = default;

void NativeVisionPipelinePlan::launch(
    const void* pixel_values_device, void* output_device,
    void* temporary_device, std::size_t supplied_temporary_bytes,
    void* stream) const {
  if (!impl_ || pixel_values_device == nullptr || output_device == nullptr ||
      temporary_device == nullptr || pixel_values_device == output_device ||
      pixel_values_device == temporary_device ||
      output_device == temporary_device ||
      supplied_temporary_bytes < impl_->temporary_bytes_value) {
    throw std::invalid_argument("native vision pipeline launch is invalid");
  }
  auto* hidden_a = static_cast<unsigned char*>(temporary_device);
  auto* hidden_b = hidden_a + impl_->hidden_bytes;
  auto* shared_temporary = hidden_b + impl_->hidden_bytes;
  impl_->patch.launch(pixel_values_device, hidden_a, stream);
  impl_->position.launch_add(hidden_a, hidden_b, stream);
  impl_->blocks.launch(
      hidden_b, impl_->metadata.rotary_cos_device(),
      impl_->metadata.rotary_sin_device(), hidden_a, shared_temporary,
      impl_->shared_temporary_bytes, stream);
  impl_->merger.launch(hidden_a, output_device, shared_temporary,
                       impl_->shared_temporary_bytes, stream);
}

void NativeVisionPipelinePlan::launch_encoder_through(
    std::size_t last_block_index, const void* pixel_values_device,
    void* output_device, void* temporary_device,
    std::size_t supplied_temporary_bytes, void* stream) const {
  if (!impl_ || pixel_values_device == nullptr || output_device == nullptr ||
      temporary_device == nullptr || pixel_values_device == output_device ||
      pixel_values_device == temporary_device ||
      output_device == temporary_device ||
      supplied_temporary_bytes < impl_->temporary_bytes_value) {
    throw std::invalid_argument(
        "native vision pipeline encoder launch is invalid");
  }
  auto* hidden_a = static_cast<unsigned char*>(temporary_device);
  auto* hidden_b = hidden_a + impl_->hidden_bytes;
  auto* shared_temporary = hidden_b + impl_->hidden_bytes;
  impl_->patch.launch(pixel_values_device, hidden_a, stream);
  impl_->position.launch_add(hidden_a, hidden_b, stream);
  impl_->blocks.launch_through(
      last_block_index, hidden_b, impl_->metadata.rotary_cos_device(),
      impl_->metadata.rotary_sin_device(), output_device, shared_temporary,
      impl_->shared_temporary_bytes, stream);
}

std::size_t NativeVisionPipelinePlan::patch_count() const {
  return impl_->metadata.patch_count();
}

std::size_t NativeVisionPipelinePlan::merged_token_count() const {
  return impl_->merger.merged_token_count();
}

std::size_t NativeVisionPipelinePlan::temporary_bytes() const {
  return impl_->temporary_bytes_value;
}

std::size_t NativeVisionPipelinePlan::metadata_resident_bytes() const {
  return impl_->metadata.resident_bytes();
}

std::size_t NativeVisionPipelinePlan::library_workspace_bytes() const {
  return impl_->library_workspace_bytes_value;
}

const std::vector<std::uint32_t>& NativeVisionPipelinePlan::cu_seqlens() const {
  return impl_->metadata.cu_seqlens();
}

const std::string& NativeVisionPipelinePlan::rotary_cos_sha256() const {
  return impl_->metadata.rotary_cos_sha256();
}

const std::string& NativeVisionPipelinePlan::rotary_sin_sha256() const {
  return impl_->metadata.rotary_sin_sha256();
}

const std::string& NativeVisionPipelinePlan::attention_image_sha256() const {
  return impl_->blocks.attention_image_sha256();
}

}  // namespace aima
