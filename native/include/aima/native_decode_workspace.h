#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/decode_schedule.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aima {

struct NativeDecodeWorkspaceView {
  std::string name;
  void* device_pointer = nullptr;
  std::uint64_t payload_bytes = 0;
  DecodeTensorDtype dtype = DecodeTensorDtype::kNone;
  DecodeBindingKind binding_kind = DecodeBindingKind::kNone;
};

struct NativeDecodeWorkspaceMetrics {
  std::size_t schedule_tensor_arguments = 0;
  std::size_t unique_bindings = 0;
  std::size_t resident_bindings = 0;
  std::size_t transient_bindings = 0;
  std::size_t runtime_scratch_bindings = 0;
  std::size_t total_bindings = 0;
  std::uint64_t logical_payload_bytes = 0;
  std::uint64_t runtime_scratch_payload_bytes = 0;
  std::uint64_t total_logical_payload_bytes = 0;
  std::uint64_t allocation_bytes = 0;
  double allocation_and_zero_ms = 0.0;
};

class NativeDecodeWorkspace {
 public:
  NativeDecodeWorkspace() = default;
  ~NativeDecodeWorkspace();
  NativeDecodeWorkspace(const NativeDecodeWorkspace&) = delete;
  NativeDecodeWorkspace& operator=(const NativeDecodeWorkspace&) = delete;

  NativeDecodeWorkspaceMetrics build(int device = 0);
  const NativeDecodeWorkspaceView* find(std::string_view name) const;
  const std::vector<NativeDecodeWorkspaceView>& views() const { return views_; }
  bool built() const { return allocation_ != nullptr; }
  // Restore the empty recurrent/conv state used when an unrelated prompt has
  // no admitted AOT prefill prefix.  The allocation also contains transient
  // tensors, so clearing it is both simpler and safer than maintaining a
  // second per-binding reset list.
  std::uint64_t clear(void* stream = nullptr);
  void reset() noexcept;

 private:
  int device_ = 0;
  void* allocation_ = nullptr;
  std::uint64_t allocation_bytes_ = 0;
  std::vector<NativeDecodeWorkspaceView> views_;
  std::unordered_map<std::string, std::size_t> name_to_index_;
};

}  // namespace aima
