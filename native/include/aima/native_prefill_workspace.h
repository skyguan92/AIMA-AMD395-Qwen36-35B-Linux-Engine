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

// Prefill reuses a small number of large byte storages across unrelated
// tensor views.  In particular, two captured slots are observed as both BF16
// and FP32 at different points in the schedule.  The owner therefore records
// byte capacity and binding lifetime, while each launch keeps the authoritative
// argument dtype and offset.
struct NativePrefillWorkspaceView {
  std::string name;
  void* device_pointer = nullptr;
  std::uint64_t payload_bytes = 0;
  DecodeBindingKind binding_kind = DecodeBindingKind::kNone;
};

struct NativePrefillWorkspaceMetrics {
  std::size_t schedule_tensor_arguments = 0;
  std::size_t unique_bindings = 0;
  std::size_t resident_bindings = 0;
  std::size_t transient_bindings = 0;
  std::size_t mixed_dtype_bindings = 0;
  std::size_t runtime_scratch_bindings = 0;
  std::size_t total_bindings = 0;
  std::uint64_t logical_payload_bytes = 0;
  std::uint64_t runtime_scratch_payload_bytes = 0;
  std::uint64_t total_logical_payload_bytes = 0;
  std::uint64_t allocation_bytes = 0;
  double allocation_and_zero_ms = 0.0;
};

class NativePrefillWorkspace {
 public:
  NativePrefillWorkspace() = default;
  ~NativePrefillWorkspace();
  NativePrefillWorkspace(const NativePrefillWorkspace&) = delete;
  NativePrefillWorkspace& operator=(const NativePrefillWorkspace&) = delete;

  NativePrefillWorkspaceMetrics build(int device = 0,
                                      std::size_t context_tokens = 8192);
  const NativePrefillWorkspaceView* find(std::string_view name) const;
  const std::vector<NativePrefillWorkspaceView>& views() const {
    return views_;
  }
  bool built() const { return allocation_ != nullptr; }
  std::size_t context_tokens() const { return context_tokens_; }
  void reset() noexcept;

 private:
  int device_ = 0;
  void* allocation_ = nullptr;
  std::uint64_t allocation_bytes_ = 0;
  std::size_t context_tokens_ = 0;
  std::vector<NativePrefillWorkspaceView> views_;
  std::unordered_map<std::string, std::size_t> name_to_index_;
};

}  // namespace aima
