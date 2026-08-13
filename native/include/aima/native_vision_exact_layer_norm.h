// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <memory>

namespace aima {

enum class NativeVisionLayerNormReciprocal {
  kDivision,
  kFastAmdReciprocal,
};

// Reproduces the frozen PyTorch ROCm vectorized LayerNorm reduction and
// expression ordering for BF16 [rows,1152]. The mode is explicit because the
// upstream ROCm build can select a fast reciprocal at compile time.
class NativeVisionExactLayerNormPlan {
 public:
  NativeVisionExactLayerNormPlan(
      std::size_t row_count,
      NativeVisionLayerNormReciprocal reciprocal_mode);
  ~NativeVisionExactLayerNormPlan();
  NativeVisionExactLayerNormPlan(
      const NativeVisionExactLayerNormPlan&) = delete;
  NativeVisionExactLayerNormPlan& operator=(
      const NativeVisionExactLayerNormPlan&) = delete;
  NativeVisionExactLayerNormPlan(
      NativeVisionExactLayerNormPlan&&) noexcept;
  NativeVisionExactLayerNormPlan& operator=(
      NativeVisionExactLayerNormPlan&&) noexcept;

  void launch(const void* input_device, const void* weight_device,
              const void* bias_device, void* output_device,
              void* stream = nullptr) const;
  std::size_t row_count() const;
  NativeVisionLayerNormReciprocal reciprocal_mode() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
