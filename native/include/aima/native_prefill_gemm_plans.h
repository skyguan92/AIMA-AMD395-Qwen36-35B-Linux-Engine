#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <cstddef>
#include <memory>

namespace aima {

class Bf16GemmPlan;

// Process-resident owner for the fixed-shape hipBLASLt plans used by one
// admitted prefill context. Accessors build lazily; q32768 uses the checkpoint
// derived fused input weights to avoid four independent projection launches.
class NativeQ8192PrefillGemmPlans {
 public:
  explicit NativeQ8192PrefillGemmPlans(std::size_t token_count = 8192);
  ~NativeQ8192PrefillGemmPlans();
  NativeQ8192PrefillGemmPlans(const NativeQ8192PrefillGemmPlans&) = delete;
  NativeQ8192PrefillGemmPlans& operator=(
      const NativeQ8192PrefillGemmPlans&) = delete;

  Bf16GemmPlan& linear_qkv();
  Bf16GemmPlan& linear_z();
  Bf16GemmPlan& linear_ab();
  Bf16GemmPlan& linear_fused_input();
  Bf16GemmPlan& linear_output();

  Bf16GemmPlan& full_q();
  Bf16GemmPlan& full_kv();
  Bf16GemmPlan& full_qkv();
  Bf16GemmPlan& full_output();

  Bf16GemmPlan& moe_shared_gate();
  Bf16GemmPlan& moe_shared_projection();
  Bf16GemmPlan& moe_shared_down();
  Bf16GemmPlan& moe_router();

  void prepare_all();
  // Materializes only the linear-attention and MoE plans consumed by the
  // active-token VL path. Full-attention stays on the resident q1024 owner.
  void prepare_logical_linear_and_moe();
  // Materializes the two hipBLASLt kernels that otherwise pay a one-time
  // module/solution setup gap inside the first frozen q1024 text request.
  // The warmup uses private zero buffers and is complete before READY.
  void warm_up_q1024_text();
  std::size_t built_plan_count() const;
  std::size_t workspace_bytes() const;
  std::size_t token_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
