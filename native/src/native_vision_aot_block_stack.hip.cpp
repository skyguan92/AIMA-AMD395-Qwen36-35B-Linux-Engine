// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_aot_block_stack.h"

#include "aima/native_vision_aot_attention.h"
#include "aima/native_vision_aot_block.h"

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
    throw std::invalid_argument(
        "native AOT vision block stack workspace overflows");
  }
  return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right) {
  if (right > static_cast<std::size_t>(-1) - left) {
    throw std::invalid_argument(
        "native AOT vision block stack workspace overflows");
  }
  return left + right;
}

}  // namespace

struct NativeVisionAotBlockStackPlan::Impl {
  Impl(const NativeWeightStore& weights,
       const std::filesystem::path& attention_image_path,
       std::size_t patches,
       const std::vector<std::uint32_t>& cu_seqlens,
       std::shared_ptr<const NativeVisionAotAttentionPlan>
           requested_attention,
       std::shared_ptr<NativeVisionAotBlockGemmPlans> requested_gemm_plans)
      : patch_count_value(patches),
        intermediate_bytes(checked_multiply(
            checked_multiply(patches, kVisionHidden),
            sizeof(std::uint16_t))),
        attention(requested_attention
                      ? std::move(requested_attention)
                      : std::make_shared<NativeVisionAotAttentionPlan>(
                            attention_image_path, patches, cu_seqlens)),
        gemm_plans(requested_gemm_plans
                       ? std::move(requested_gemm_plans)
                       : std::make_shared<NativeVisionAotBlockGemmPlans>(
                             patches)) {
    if (patches == 0 || !attention || !gemm_plans ||
        attention->patch_count() != patches ||
        attention->segment_count() + 1 != cu_seqlens.size() ||
        gemm_plans->patch_count() != patches) {
      throw std::invalid_argument(
          "native AOT vision block stack plans disagree");
    }
    library_workspace_bytes_value =
        checked_add(attention->workspace_bytes(),
                    gemm_plans->workspace_bytes());
    blocks.reserve(kVisionBlockCount);
    for (std::size_t block_index = 0; block_index < kVisionBlockCount;
         ++block_index) {
      blocks.emplace_back(weights, block_index, patches, attention,
                          gemm_plans);
      block_temporary_bytes =
          std::max(block_temporary_bytes, blocks.back().temporary_bytes());
      if (blocks.back().library_workspace_bytes() !=
          gemm_plans->workspace_bytes()) {
        throw std::runtime_error(
            "native AOT vision block stack GEMM plans disagree");
      }
    }
    temporary_bytes_value =
        checked_add(intermediate_bytes, block_temporary_bytes);
  }

  std::size_t patch_count_value = 0;
  std::size_t intermediate_bytes = 0;
  std::size_t block_temporary_bytes = 0;
  std::size_t temporary_bytes_value = 0;
  std::size_t library_workspace_bytes_value = 0;
  std::shared_ptr<const NativeVisionAotAttentionPlan> attention;
  std::shared_ptr<NativeVisionAotBlockGemmPlans> gemm_plans;
  std::vector<NativeVisionAotBlockPlan> blocks;
};

NativeVisionAotBlockStackPlan::NativeVisionAotBlockStackPlan(
    const NativeWeightStore& weights,
    const std::filesystem::path& attention_image_path,
    std::size_t patch_count,
    const std::vector<std::uint32_t>& cu_seqlens)
    : NativeVisionAotBlockStackPlan(
          weights, attention_image_path, patch_count, cu_seqlens, nullptr,
          nullptr) {}

NativeVisionAotBlockStackPlan::NativeVisionAotBlockStackPlan(
    const NativeWeightStore& weights,
    const std::filesystem::path& attention_image_path,
    std::size_t patch_count,
    const std::vector<std::uint32_t>& cu_seqlens,
    std::shared_ptr<const NativeVisionAotAttentionPlan> attention,
    std::shared_ptr<NativeVisionAotBlockGemmPlans> gemm_plans)
    : impl_(std::make_unique<Impl>(weights, attention_image_path, patch_count,
                                  cu_seqlens, std::move(attention),
                                  std::move(gemm_plans))) {}
NativeVisionAotBlockStackPlan::~NativeVisionAotBlockStackPlan() = default;
NativeVisionAotBlockStackPlan::NativeVisionAotBlockStackPlan(
    NativeVisionAotBlockStackPlan&&) noexcept = default;
NativeVisionAotBlockStackPlan& NativeVisionAotBlockStackPlan::operator=(
    NativeVisionAotBlockStackPlan&&) noexcept = default;

void NativeVisionAotBlockStackPlan::launch(
    const void* input_device, const void* cos_device, const void* sin_device,
    void* output_device, void* temporary_device,
    std::size_t supplied_temporary_bytes, void* stream) const {
  launch_through(kVisionBlockCount - 1, input_device, cos_device, sin_device,
                 output_device, temporary_device, supplied_temporary_bytes,
                 stream);
}

void NativeVisionAotBlockStackPlan::launch_through(
    std::size_t last_block_index, const void* input_device,
    const void* cos_device, const void* sin_device, void* output_device,
    void* temporary_device, std::size_t supplied_temporary_bytes,
    void* stream) const {
  if (!impl_ || input_device == nullptr || cos_device == nullptr ||
      sin_device == nullptr || output_device == nullptr ||
      temporary_device == nullptr || input_device == output_device ||
      input_device == temporary_device || output_device == temporary_device ||
      cos_device == temporary_device || sin_device == temporary_device ||
      supplied_temporary_bytes < impl_->temporary_bytes_value ||
      last_block_index >= impl_->blocks.size()) {
    throw std::invalid_argument(
        "native AOT vision block stack launch is invalid");
  }
  auto* intermediate = static_cast<unsigned char*>(temporary_device);
  auto* block_temporary = intermediate + impl_->intermediate_bytes;
  const void* current = input_device;
  for (std::size_t block_index = 0; block_index <= last_block_index;
       ++block_index) {
    void* next = (last_block_index - block_index) % 2 == 0
                     ? output_device
                     : intermediate;
    impl_->blocks[block_index].launch(
        current, cos_device, sin_device, next, block_temporary,
        impl_->block_temporary_bytes, stream);
    current = next;
  }
}

std::size_t NativeVisionAotBlockStackPlan::patch_count() const {
  return impl_->patch_count_value;
}

std::size_t NativeVisionAotBlockStackPlan::block_count() const {
  return impl_->blocks.size();
}

std::size_t NativeVisionAotBlockStackPlan::temporary_bytes() const {
  return impl_->temporary_bytes_value;
}

std::size_t NativeVisionAotBlockStackPlan::library_workspace_bytes() const {
  return impl_->library_workspace_bytes_value;
}

std::shared_ptr<const NativeVisionAotAttentionPlan>
NativeVisionAotBlockStackPlan::attention_plan() const {
  return impl_->attention;
}

std::shared_ptr<NativeVisionAotBlockGemmPlans>
NativeVisionAotBlockStackPlan::gemm_plans() const {
  return impl_->gemm_plans;
}

const std::string&
NativeVisionAotBlockStackPlan::attention_image_sha256() const {
  return impl_->attention->image_sha256();
}

}  // namespace aima
