#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace aima {

struct NativeDerivedWeightView {
  std::string name;
  void* device_pointer = nullptr;
  std::uint32_t rows = 0;
  std::uint32_t columns = 0;
  std::uint64_t payload_bytes = 0;
};

struct NativeDerivedWeightMetrics {
  std::size_t view_count = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t free_bytes_before = 0;
  std::uint64_t free_bytes_after = 0;
  double allocation_ms = 0.0;
  double pack_ms = 0.0;
  double checksum_ms = 0.0;
  double build_wall_ms = 0.0;
  std::uint64_t source_u16_xor = 0;
  std::uint64_t source_u16_sum = 0;
  std::uint64_t derived_u16_xor = 0;
  std::uint64_t derived_u16_sum = 0;
  bool full_payload_checksum_equal = false;
  std::size_t exact_sample_elements = 0;
  std::size_t expected_sample_elements = 0;
};

struct NativeDerivedProjectionResult {
  std::size_t elements = 0;
  std::size_t exact_elements = 0;
  double maximum_absolute_error = 0.0;
  double relative_l2_error = 0.0;
};

class NativeDerivedWeightStore {
 public:
  NativeDerivedWeightStore() = default;
  ~NativeDerivedWeightStore();
  NativeDerivedWeightStore(const NativeDerivedWeightStore&) = delete;
  NativeDerivedWeightStore& operator=(const NativeDerivedWeightStore&) = delete;

  NativeDerivedWeightMetrics build(const NativeWeightStore& weights,
                                   int device = 0);
  const NativeDerivedWeightView* find(const std::string& name) const;
  const std::vector<NativeDerivedWeightView>& views() const { return views_; }
  bool built() const { return allocation_ != nullptr; }
  void reset() noexcept;

 private:
  int device_ = 0;
  void* allocation_ = nullptr;
  std::uint64_t allocation_bytes_ = 0;
  std::vector<NativeDerivedWeightView> views_;
  std::unordered_map<std::string, std::size_t> name_to_index_;
};

NativeDerivedProjectionResult probe_layer0_derived_projection(
    const NativeWeightStore& weights,
    const NativeDerivedWeightStore& derived);

NativeDerivedProjectionResult validate_layer0_router_transpose(
    const NativeWeightStore& weights,
    const NativeDerivedWeightStore& derived);

}  // namespace aima
