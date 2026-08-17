#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/prefill_schedule.h"

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
  // Logical byte extent of this schedule's view map.  A workspace may map a
  // shared prefix and own only its non-overlapping tail.
  std::uint64_t allocation_bytes = 0;
  // Bytes physically owned by this instance; shared bytes are never counted.
  std::uint64_t physical_allocation_bytes = 0;
  double allocation_and_zero_ms = 0.0;
};

class NativePrefillWorkspace {
 public:
  NativePrefillWorkspace() = default;
  ~NativePrefillWorkspace();
  NativePrefillWorkspace(const NativePrefillWorkspace&) = delete;
  NativePrefillWorkspace& operator=(const NativePrefillWorkspace&) = delete;

  NativePrefillWorkspaceMetrics build(
      int device = 0, std::size_t context_tokens = 8192,
      bool frozen_text_schedule = false,
      // With a non-zero split offset, bindings below the offset alias this
      // allocation and bindings at or above it use a separately owned tail.
      void* shared_allocation = nullptr,
      std::uint64_t shared_allocation_bytes = 0,
      std::uint64_t split_allocation_offset = 0);
  const NativePrefillWorkspaceView* find(std::string_view name) const;
  const std::vector<NativePrefillWorkspaceView>& views() const {
    return views_;
  }
  bool built() const { return allocation_ != nullptr; }
  void* allocation() const { return allocation_; }
  std::uint64_t allocation_bytes() const { return allocation_bytes_; }
  std::uint64_t physical_allocation_bytes() const {
    return physical_allocation_bytes_;
  }
  bool owns_allocation() const {
    return owns_allocation_ || tail_allocation_ != nullptr;
  }
  bool owns_primary_allocation() const { return owns_allocation_; }
  bool has_split_allocation() const { return tail_allocation_ != nullptr; }
  std::uint64_t split_allocation_offset() const {
    return split_allocation_offset_;
  }
  std::size_t context_tokens() const { return context_tokens_; }
  bool includes_frozen_text() const { return includes_frozen_text_; }
  void reset() noexcept;

 private:
  int device_ = 0;
  void* allocation_ = nullptr;
  std::uint64_t allocation_bytes_ = 0;
  std::uint64_t physical_allocation_bytes_ = 0;
  bool owns_allocation_ = false;
  void* tail_allocation_ = nullptr;
  std::uint64_t tail_allocation_bytes_ = 0;
  std::uint64_t split_allocation_offset_ = 0;
  std::size_t context_tokens_ = 0;
  bool includes_frozen_text_ = false;
  std::vector<NativePrefillWorkspaceView> views_;
  std::unordered_map<std::string, std::size_t> name_to_index_;
};

}  // namespace aima
