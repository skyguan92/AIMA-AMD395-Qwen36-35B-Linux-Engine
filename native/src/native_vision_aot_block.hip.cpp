// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_aot_block.h"

#include "aima/bf16_gemm.h"
#include "aima/native_vision_aot_attention.h"
#include "aima/native_vision_block_suffix.h"
#include "aima/native_vision_exact_layer_norm.h"
#include "aima/native_vision_rotary.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace aima {
namespace {

constexpr std::size_t kVisionHidden = 1152;
constexpr std::size_t kVisionQkvHidden = 3 * kVisionHidden;
constexpr std::size_t kVisionIntermediate = 4304;
constexpr std::size_t kVisionBlockCount = 27;
constexpr std::size_t kWorkspaceLimit = 128ULL * 1024ULL * 1024ULL;
static_assert(3 * kVisionHidden <= kVisionIntermediate);

std::size_t validate_block_index(std::size_t block_index) {
  if (block_index >= kVisionBlockCount) {
    throw std::invalid_argument("native AOT vision block index is invalid");
  }
  return block_index;
}

const NativeTensorView& require_tensor(const NativeWeightStore& weights,
                                       std::string_view name,
                                       std::uint8_t rank,
                                       std::uint64_t payload_bytes) {
  const NativeTensorView* tensor = weights.find(name);
  if (tensor == nullptr || tensor->device_pointer == nullptr ||
      tensor->rank != rank || tensor->payload_bytes != payload_bytes) {
    throw std::runtime_error(std::string("native AOT vision weight mismatch: ") +
                             std::string(name));
  }
  return *tensor;
}

std::string block_tensor_name(std::size_t block_index,
                              std::string_view suffix) {
  return "model.visual.blocks." + std::to_string(block_index) + "." +
         std::string(suffix);
}

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
      : block_index_value(validate_block_index(requested_block_index)),
        patch_count_value(patches),
        arena_large_bytes(checked_multiply(
            checked_multiply(patches, kVisionIntermediate),
            sizeof(std::uint16_t))),
        arena_hidden_bytes(checked_multiply(
            checked_multiply(patches, kVisionHidden),
            sizeof(std::uint16_t))),
        temporary_bytes_value(checked_add(
            checked_multiply(2, arena_large_bytes), arena_hidden_bytes)),
        norm1_weight(require_tensor(
            weights, block_tensor_name(block_index_value, "norm1.weight"), 1,
            kVisionHidden * sizeof(std::uint16_t))),
        norm1_bias(require_tensor(
            weights, block_tensor_name(block_index_value, "norm1.bias"), 1,
            kVisionHidden * sizeof(std::uint16_t))),
        qkv_weight(require_tensor(
            weights, block_tensor_name(block_index_value, "attn.qkv.weight"),
            2, kVisionQkvHidden * kVisionHidden * sizeof(std::uint16_t))),
        qkv_bias(require_tensor(
            weights, block_tensor_name(block_index_value, "attn.qkv.bias"), 1,
            kVisionQkvHidden * sizeof(std::uint16_t))),
        norm2_weight(require_tensor(
            weights, block_tensor_name(block_index_value, "norm2.weight"), 1,
            kVisionHidden * sizeof(std::uint16_t))),
        norm2_bias(require_tensor(
            weights, block_tensor_name(block_index_value, "norm2.bias"), 1,
            kVisionHidden * sizeof(std::uint16_t))),
        norm1(patches,
              NativeVisionLayerNormReciprocal::kFastAmdReciprocal),
        qkv_gemm(patches, kVisionQkvHidden, kVisionHidden, kWorkspaceLimit,
                 true, true),
        rotary(patches),
        attention(std::move(shared_attention)),
        suffix(weights, block_index_value, patches),
        norm2(patches,
              NativeVisionLayerNormReciprocal::kFastAmdReciprocal) {
    if (!attention || suffix.block_index() != block_index_value ||
        suffix.patch_count() != patch_count_value ||
        patch_count_value != rotary.patch_count() ||
        patch_count_value != attention->patch_count()) {
      throw std::runtime_error("native AOT vision block subplans disagree");
    }
  }

  std::size_t block_index_value = 0;
  std::size_t patch_count_value = 0;
  std::size_t arena_large_bytes = 0;
  std::size_t arena_hidden_bytes = 0;
  std::size_t temporary_bytes_value = 0;
  const NativeTensorView& norm1_weight;
  const NativeTensorView& norm1_bias;
  const NativeTensorView& qkv_weight;
  const NativeTensorView& qkv_bias;
  const NativeTensorView& norm2_weight;
  const NativeTensorView& norm2_bias;
  NativeVisionExactLayerNormPlan norm1;
  Bf16GemmPlan qkv_gemm;
  NativeVisionRotaryPlan rotary;
  std::shared_ptr<const NativeVisionAotAttentionPlan> attention;
  NativeVisionBlockSuffixPlan suffix;
  NativeVisionExactLayerNormPlan norm2;
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

  impl_->norm1.launch(input_device, impl_->norm1_weight.device_pointer,
                      impl_->norm1_bias.device_pointer, arena_b, stream);
  impl_->qkv_gemm.launch_with_bias(
      arena_b, impl_->qkv_weight.device_pointer,
      impl_->qkv_bias.device_pointer, arena_a, stream);
  impl_->rotary.launch(arena_a, cos_device, sin_device, query, key, value,
                       stream);
  impl_->attention->launch(query, key, value, arena_a, stream);
  impl_->suffix.launch_attention_projection(arena_a, arena_b, stream);
  impl_->suffix.launch_residual(input_device, arena_b, arena_c, stream);
  impl_->norm2.launch(arena_c, impl_->norm2_weight.device_pointer,
                      impl_->norm2_bias.device_pointer, arena_a, stream);
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
  return impl_->qkv_gemm.workspace_bytes() + impl_->suffix.workspace_bytes();
}

}  // namespace aima
