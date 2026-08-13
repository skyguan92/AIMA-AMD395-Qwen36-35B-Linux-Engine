// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_aot_block.h"

#include "aima/native_vision_aot_attention.h"
#include "aima/native_vision_block_suffix.h"
#include "aima/native_vision_encoder.h"
#include "aima/native_vision_rotary.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace aima {
namespace {

constexpr std::size_t kVisionHidden = 1152;
constexpr std::size_t kVisionIntermediate = 4304;
static_assert(3 * kVisionHidden <= kVisionIntermediate);

std::size_t checked_multiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > static_cast<std::size_t>(-1) / left) {
    throw std::invalid_argument("native AOT vision block workspace overflows");
  }
  return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right) {
  if (right > static_cast<std::size_t>(-1) - left) {
    throw std::invalid_argument("native AOT vision block workspace overflows");
  }
  return left + right;
}

}  // namespace

struct NativeVisionAotBlockPlan::Impl {
  Impl(const NativeWeightStore& weights, std::size_t requested_block_index,
       std::size_t patches,
       std::shared_ptr<const NativeVisionAotAttentionPlan> shared_attention)
      : block_index_value(requested_block_index),
        patch_count_value(patches),
        arena_large_bytes(checked_multiply(
            checked_multiply(patches, kVisionIntermediate),
            sizeof(std::uint16_t))),
        arena_hidden_bytes(checked_multiply(
            checked_multiply(patches, kVisionHidden),
            sizeof(std::uint16_t))),
        temporary_bytes_value(checked_add(
            checked_multiply(2, arena_large_bytes), arena_hidden_bytes)),
        prefix(weights, requested_block_index, patches),
        rotary(patches),
        attention(std::move(shared_attention)),
        suffix(weights, requested_block_index, patches) {
    if (!attention || prefix.block_index() != suffix.block_index() ||
        prefix.patch_count() != suffix.patch_count() ||
        prefix.patch_count() != rotary.patch_count() ||
        prefix.patch_count() != attention->patch_count()) {
      throw std::runtime_error("native AOT vision block subplans disagree");
    }
  }

  std::size_t block_index_value = 0;
  std::size_t patch_count_value = 0;
  std::size_t arena_large_bytes = 0;
  std::size_t arena_hidden_bytes = 0;
  std::size_t temporary_bytes_value = 0;
  NativeVisionBlockPrefixPlan prefix;
  NativeVisionRotaryPlan rotary;
  std::shared_ptr<const NativeVisionAotAttentionPlan> attention;
  NativeVisionBlockSuffixPlan suffix;
};

NativeVisionAotBlockPlan::NativeVisionAotBlockPlan(
    const NativeWeightStore& weights, std::size_t block_index,
    std::size_t patch_count,
    std::shared_ptr<const NativeVisionAotAttentionPlan> attention)
    : impl_(std::make_unique<Impl>(weights, block_index, patch_count,
                                  std::move(attention))) {}
NativeVisionAotBlockPlan::~NativeVisionAotBlockPlan() = default;
NativeVisionAotBlockPlan::NativeVisionAotBlockPlan(
    NativeVisionAotBlockPlan&&) noexcept = default;
NativeVisionAotBlockPlan& NativeVisionAotBlockPlan::operator=(
    NativeVisionAotBlockPlan&&) noexcept = default;

void NativeVisionAotBlockPlan::launch(
    const void* input_device, const void* cos_device, const void* sin_device,
    void* output_device, void* temporary_device,
    std::size_t supplied_temporary_bytes, void* stream) const {
  if (!impl_ || input_device == nullptr || cos_device == nullptr ||
      sin_device == nullptr || output_device == nullptr ||
      temporary_device == nullptr || input_device == output_device ||
      input_device == temporary_device || output_device == temporary_device ||
      cos_device == temporary_device || sin_device == temporary_device ||
      supplied_temporary_bytes < impl_->temporary_bytes_value) {
    throw std::invalid_argument("native AOT vision block launch is invalid");
  }
  auto* arena_a = static_cast<unsigned char*>(temporary_device);
  auto* arena_b = arena_a + impl_->arena_large_bytes;
  auto* arena_c = arena_b + impl_->arena_large_bytes;
  auto* query = arena_b;
  auto* key = query + impl_->arena_hidden_bytes;
  auto* value = key + impl_->arena_hidden_bytes;

  impl_->prefix.launch(input_device, arena_b, arena_a, stream);
  impl_->rotary.launch(arena_a, cos_device, sin_device, query, key, value,
                       stream);
  impl_->attention->launch(query, key, value, arena_a, stream);
  impl_->suffix.launch_attention_projection(arena_a, arena_b, stream);
  impl_->suffix.launch_residual(input_device, arena_b, arena_c, stream);
  impl_->suffix.launch_norm2(arena_c, arena_a, stream);
  impl_->suffix.launch_mlp_fc1(arena_a, arena_b, stream);
  impl_->suffix.launch_gelu(arena_b, arena_a, stream);
  impl_->suffix.launch_mlp_fc2(arena_a, arena_b, stream);
  impl_->suffix.launch_residual(arena_c, arena_b, output_device, stream);
}

std::size_t NativeVisionAotBlockPlan::block_index() const {
  return impl_->block_index_value;
}

std::size_t NativeVisionAotBlockPlan::patch_count() const {
  return impl_->patch_count_value;
}

std::size_t NativeVisionAotBlockPlan::temporary_bytes() const {
  return impl_->temporary_bytes_value;
}

std::size_t NativeVisionAotBlockPlan::library_workspace_bytes() const {
  return impl_->prefix.workspace_bytes() + impl_->suffix.workspace_bytes();
}

}  // namespace aima
