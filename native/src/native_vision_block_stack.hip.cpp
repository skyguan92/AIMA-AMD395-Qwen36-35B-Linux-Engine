// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_block_stack.h"

#include "aima/native_vision_block.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kVisionBlockCount = 27;
constexpr std::size_t kVisionHidden = 1152;
static_assert(kVisionBlockCount % 2 == 1);

std::size_t checked_multiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > static_cast<std::size_t>(-1) / left) {
    throw std::invalid_argument("native vision block stack workspace overflows");
  }
  return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right) {
  if (right > static_cast<std::size_t>(-1) - left) {
    throw std::invalid_argument("native vision block stack workspace overflows");
  }
  return left + right;
}

}  // namespace

struct NativeVisionBlockStackPlan::Impl {
  Impl(const NativeWeightStore& weights, std::size_t patches,
       const std::vector<std::uint32_t>& cu_seqlens)
      : patch_count_value(patches),
        intermediate_bytes(checked_multiply(
            checked_multiply(patches, kVisionHidden),
            sizeof(std::uint16_t))) {
    if (patches == 0) {
      throw std::invalid_argument("native vision block stack is empty");
    }
    blocks.reserve(kVisionBlockCount);
    for (std::size_t block_index = 0; block_index < kVisionBlockCount;
         ++block_index) {
      blocks.emplace_back(weights, block_index, patches, cu_seqlens);
      block_temporary_bytes =
          std::max(block_temporary_bytes, blocks.back().temporary_bytes());
      library_workspace_bytes_value = checked_add(
          library_workspace_bytes_value,
          blocks.back().library_workspace_bytes());
    }
    temporary_bytes_value =
        checked_add(intermediate_bytes, block_temporary_bytes);
  }

  std::size_t patch_count_value = 0;
  std::size_t intermediate_bytes = 0;
  std::size_t block_temporary_bytes = 0;
  std::size_t temporary_bytes_value = 0;
  std::size_t library_workspace_bytes_value = 0;
  std::vector<NativeVisionBlockPlan> blocks;
};

NativeVisionBlockStackPlan::NativeVisionBlockStackPlan(
    const NativeWeightStore& weights, std::size_t patch_count,
    const std::vector<std::uint32_t>& cu_seqlens)
    : impl_(std::make_unique<Impl>(weights, patch_count, cu_seqlens)) {}
NativeVisionBlockStackPlan::~NativeVisionBlockStackPlan() = default;
NativeVisionBlockStackPlan::NativeVisionBlockStackPlan(
    NativeVisionBlockStackPlan&&) noexcept = default;
NativeVisionBlockStackPlan& NativeVisionBlockStackPlan::operator=(
    NativeVisionBlockStackPlan&&) noexcept = default;

void NativeVisionBlockStackPlan::launch(
    const void* input_device, const void* cos_device, const void* sin_device,
    void* output_device, void* temporary_device,
    std::size_t supplied_temporary_bytes, void* stream) const {
  if (!impl_ || input_device == nullptr || cos_device == nullptr ||
      sin_device == nullptr || output_device == nullptr ||
      temporary_device == nullptr || input_device == output_device ||
      input_device == temporary_device || output_device == temporary_device ||
      cos_device == temporary_device || sin_device == temporary_device ||
      supplied_temporary_bytes < impl_->temporary_bytes_value) {
    throw std::invalid_argument("native vision block stack launch is invalid");
  }
  auto* intermediate = static_cast<unsigned char*>(temporary_device);
  auto* block_temporary = intermediate + impl_->intermediate_bytes;
  const void* current = input_device;
  for (std::size_t block_index = 0; block_index < impl_->blocks.size();
       ++block_index) {
    void* next = block_index % 2 == 0 ? output_device : intermediate;
    impl_->blocks[block_index].launch(
        current, cos_device, sin_device, next, block_temporary,
        impl_->block_temporary_bytes, stream);
    current = next;
  }
}

std::size_t NativeVisionBlockStackPlan::patch_count() const {
  return impl_->patch_count_value;
}

std::size_t NativeVisionBlockStackPlan::block_count() const {
  return impl_->blocks.size();
}

std::size_t NativeVisionBlockStackPlan::temporary_bytes() const {
  return impl_->temporary_bytes_value;
}

std::size_t NativeVisionBlockStackPlan::library_workspace_bytes() const {
  return impl_->library_workspace_bytes_value;
}

}  // namespace aima
