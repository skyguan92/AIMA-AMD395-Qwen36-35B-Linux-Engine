#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <cstddef>
#include <cstdint>
#include <memory>

namespace aima {

class Bf16GemmPlan;
class NativeQ8192PrefillGemmPlans;
class NativeWeightStore;

struct NativeVlLogicalProjectionLoadMetrics {
  std::uint64_t weight_bytes = 0;
  std::uint64_t output_scratch_bytes = 0;
  std::size_t linear_layer_count = 0;
  std::size_t maximum_tokens = 0;
  double build_wall_ms = 0.0;
  bool loaded = false;
};

struct NativeVlLogicalProjectionPrepareMetrics {
  std::size_t tokens = 0;
  std::size_t plan_count = 0;
  std::uint64_t workspace_bytes = 0;
  double build_wall_ms = 0.0;
  bool reused = false;
  bool prepared = false;
};

// Resident owner for the logical-M projection surfaces of a padded q1024 VL
// request. A/B weights and compact output scratch are ready-time residents;
// only the hipBLASLt shape descriptors change when a multimodal prompt has a
// new logical token count. The fixed AOT storage/grid capacity remains q1024.
class NativeVlLogicalProjectionState {
 public:
  NativeVlLogicalProjectionState();
  ~NativeVlLogicalProjectionState();
  NativeVlLogicalProjectionState(
      const NativeVlLogicalProjectionState&) = delete;
  NativeVlLogicalProjectionState& operator=(
      const NativeVlLogicalProjectionState&) = delete;

  NativeVlLogicalProjectionLoadMetrics build(
      const NativeWeightStore& weights, std::size_t maximum_tokens = 1024,
      int device = 0);
  NativeVlLogicalProjectionPrepareMetrics prepare(std::size_t tokens);
  void reset() noexcept;

  bool loaded() const;
  bool prepared() const;
  std::size_t prepared_tokens() const;
  const void* ab_weight(std::size_t layer_index) const;
  void* ab_output() const;
  Bf16GemmPlan& ab_plan() const;
  NativeQ8192PrefillGemmPlans& router_gemm_plans() const;
  const NativeVlLogicalProjectionLoadMetrics& load_metrics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
