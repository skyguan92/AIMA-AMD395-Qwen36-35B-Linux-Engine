// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_pipeline.h"

#include "aima/native_vision_aot_block_stack.h"
#include "aima/native_vision_encoder.h"
#include "aima/native_vision_merger.h"
#include "aima/native_weight_store.h"

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
constexpr float kVisionRotaryBase = 10000.0f;
static_assert(kVisionRotaryAxes * kVisionRotaryFrequencyCount ==
              kVisionRotaryDimension);
static_assert(kVisionHeadDimension / 2 == kVisionRotaryDimension);

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

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t round_to_nearest_even =
      0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + round_to_nearest_even) >> 16U);
}

struct HostEncoderMetadata {
  std::size_t patch_count = 0;
  std::vector<std::uint32_t> cu_seqlens;
  std::vector<std::uint16_t> cos;
  std::vector<std::uint16_t> sin;
};

HostEncoderMetadata build_host_metadata(
    const std::vector<NativeVlGrid>& grids) {
  if (grids.empty()) {
    throw std::invalid_argument("native vision encoder grids are empty");
  }

  HostEncoderMetadata result;
  result.cu_seqlens.push_back(0);
  std::size_t maximum_grid_size = 0;
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
    if (grid_patches > kSpatialMergeArea * kNativeVlAggregateTokenLimit ||
        result.patch_count >
            kSpatialMergeArea * kNativeVlAggregateTokenLimit - grid_patches) {
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
    maximum_grid_size =
        std::max(maximum_grid_size, std::max(grid.height, grid.width));
  }

  std::array<float, kVisionRotaryFrequencyCount> inverse_frequencies{};
  for (std::size_t frequency = 0;
       frequency < kVisionRotaryFrequencyCount; ++frequency) {
    const float exponent =
        static_cast<float>(2 * frequency) /
        static_cast<float>(kVisionRotaryDimension);
    inverse_frequencies[frequency] =
        1.0f / std::pow(kVisionRotaryBase, exponent);
  }

  const std::size_t cache_elements = checked_multiply(
      maximum_grid_size, kVisionRotaryFrequencyCount,
      "vision rotary cache");
  std::vector<std::uint16_t> cos_cache(cache_elements);
  std::vector<std::uint16_t> sin_cache(cache_elements);
  for (std::size_t position = 0; position < maximum_grid_size; ++position) {
    for (std::size_t frequency = 0;
         frequency < kVisionRotaryFrequencyCount; ++frequency) {
      const float angle =
          static_cast<float>(position) * inverse_frequencies[frequency];
      const std::size_t index =
          position * kVisionRotaryFrequencyCount + frequency;
      cos_cache[index] = float_to_bf16(std::cos(angle));
      sin_cache[index] = float_to_bf16(std::sin(angle));
    }
  }

  const std::size_t output_elements = checked_multiply(
      result.patch_count, kVisionRotaryDimension,
      "vision rotary metadata");
  result.cos.reserve(output_elements);
  result.sin.reserve(output_elements);
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
                const std::size_t cache_offset =
                    position * kVisionRotaryFrequencyCount;
                result.cos.insert(
                    result.cos.end(), cos_cache.begin() + cache_offset,
                    cos_cache.begin() + cache_offset +
                        kVisionRotaryFrequencyCount);
                result.sin.insert(
                    result.sin.end(), sin_cache.begin() + cache_offset,
                    sin_cache.begin() + cache_offset +
                        kVisionRotaryFrequencyCount);
              }
            }
          }
        }
      }
    }
  }
  if (result.cos.size() != output_elements ||
      result.sin.size() != output_elements ||
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
    try {
      check_hip(hipMalloc(&metadata_device, resident_bytes_value),
                "hipMalloc native vision encoder metadata");
      cos_device = metadata_device;
      sin_device = static_cast<unsigned char*>(metadata_device) +
                   one_table_bytes;
      check_hip(hipMemcpy(cos_device, host.cos.data(), one_table_bytes,
                          hipMemcpyHostToDevice),
                "hipMemcpy native vision rotary cosine");
      check_hip(hipMemcpy(sin_device, host.sin.data(), one_table_bytes,
                          hipMemcpyHostToDevice),
                "hipMemcpy native vision rotary sine");
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
    cos_device = nullptr;
    sin_device = nullptr;
  }

  std::size_t patch_count_value = 0;
  std::size_t resident_bytes_value = 0;
  std::vector<std::uint32_t> cu_seqlens_value;
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
    if (position.patch_count() != metadata.patch_count() ||
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

const std::string& NativeVisionPipelinePlan::attention_image_sha256() const {
  return impl_->blocks.attention_image_sha256();
}

}  // namespace aima
