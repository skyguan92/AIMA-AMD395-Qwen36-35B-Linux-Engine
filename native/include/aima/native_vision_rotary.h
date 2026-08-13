// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace aima {

// Converts biased QKV output into the exact contiguous tensors consumed by
// Qwen3.6 vision attention. Q and K receive Neox-style 2D rotary embedding;
// V is extracted without arithmetic. All inputs and outputs are BF16.
class NativeVisionRotaryPlan {
 public:
  explicit NativeVisionRotaryPlan(std::size_t patch_count);

  // qkv is [patch_count,3456], cos/sin are [patch_count,36], and each output
  // is [patch_count,16,72]. Outputs must be distinct from every input.
  void launch(const void* qkv_device, const void* cos_device,
              const void* sin_device, void* query_device, void* key_device,
              void* value_device, void* stream = nullptr) const;
  std::size_t patch_count() const;

 private:
  std::size_t patch_count_ = 0;
};

}  // namespace aima
