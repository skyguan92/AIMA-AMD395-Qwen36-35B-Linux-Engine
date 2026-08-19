// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <memory>

namespace aima {

class Bf16GemmPlan;
class NativeWeightStore;

// Executes the suffix of one Qwen3.6 vision block after segmented attention:
// attention projection/residual, norm2, FC1, exact GELU, FC2 and the final
// residual. Every boundary remains externally visible for qualification.
class NativeVisionBlockSuffixPlan {
 public:
  NativeVisionBlockSuffixPlan(const NativeWeightStore& weights,
                              std::size_t block_index,
                              std::size_t patch_count);
  NativeVisionBlockSuffixPlan(
      const NativeWeightStore& weights, std::size_t block_index,
      std::size_t patch_count,
      std::shared_ptr<Bf16GemmPlan> attention_projection_gemm,
      std::shared_ptr<Bf16GemmPlan> mlp_fc1_gemm,
      std::shared_ptr<Bf16GemmPlan> mlp_fc2_gemm);
  ~NativeVisionBlockSuffixPlan();
  NativeVisionBlockSuffixPlan(const NativeVisionBlockSuffixPlan&) = delete;
  NativeVisionBlockSuffixPlan& operator=(
      const NativeVisionBlockSuffixPlan&) = delete;
  NativeVisionBlockSuffixPlan(NativeVisionBlockSuffixPlan&&) noexcept;
  NativeVisionBlockSuffixPlan& operator=(
      NativeVisionBlockSuffixPlan&&) noexcept;

  // block_input and attention are BF16 [patch_count,1152]. FC1 and activation
  // are [patch_count,4304]; all remaining outputs are [patch_count,1152].
  // Inputs and outputs must be pairwise distinct.
  void launch(const void* block_input_device, const void* attention_device,
              void* attention_projection_device,
              void* attention_residual_device, void* norm2_device,
              void* mlp_fc1_device, void* mlp_activation_device,
              void* mlp_fc2_device, void* block_output_device,
              void* stream = nullptr) const;
  void launch_attention_projection(const void* attention_device,
                                   void* output_device,
                                   void* stream = nullptr) const;
  void launch_residual(const void* left_device, const void* right_device,
                       void* output_device, void* stream = nullptr) const;
  void launch_norm2(const void* input_device, void* output_device,
                    void* stream = nullptr) const;
  void launch_mlp_fc1(const void* input_device, void* output_device,
                      void* stream = nullptr) const;
  void launch_gelu(const void* input_device, void* output_device,
                   void* stream = nullptr) const;
  void launch_mlp_fc2(const void* input_device, void* output_device,
                      void* stream = nullptr) const;
  std::size_t block_index() const;
  std::size_t patch_count() const;
  std::size_t workspace_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
