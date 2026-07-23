#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/decode_schedule.h"
#include "aima/native_derived_weights.h"
#include "aima/native_lm_head.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aima {

struct NativeDecodeBindingView {
  std::string name;
  void* device_pointer = nullptr;
  std::uint64_t payload_bytes = 0;
  DecodeTensorDtype dtype = DecodeTensorDtype::kNone;
};

struct NativeDecodeBindingMetrics {
  std::size_t schedule_weight_arguments = 0;
  std::size_t unique_bindings = 0;
  std::size_t raw_weight_bindings = 0;
  std::size_t layer_derived_bindings = 0;
  std::size_t lm_head_derived_bindings = 0;
  std::size_t device_pointer_checks = 0;
  std::size_t exact_payload_byte_checks = 0;
};

class NativeDecodeBindings {
 public:
  NativeDecodeBindingMetrics build(
      const NativeWeightStore& weights,
      const NativeDerivedWeightStore& derived,
      const NativeLmHeadStore& lm_head);
  const NativeDecodeBindingView* find(std::string_view name) const;
  const std::vector<NativeDecodeBindingView>& views() const { return views_; }

 private:
  std::vector<NativeDecodeBindingView> views_;
  std::unordered_map<std::string, std::size_t> name_to_index_;
};

}  // namespace aima
