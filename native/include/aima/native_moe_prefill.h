#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_executor.h"
#include "aima/native_layer_oracle.h"
#include "aima/native_prefill_invocation.h"
#include "aima/native_prefill_workspace.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace aima {

class NativeQ8192PrefillGemmPlans;

struct NativeMoePrefillMetrics {
  std::size_t layer_index = 0;
  std::size_t tokens = 0;
  std::size_t dense_gemm_launches = 0;
  std::size_t native_router_launches = 0;
  std::size_t native_dispatch_launches = 0;
  std::size_t native_pointwise_launches = 0;
  std::size_t aot_launches = 0;
  std::size_t diagnostic_aot_launches = 0;
  std::size_t diagnostic_pointwise_launches = 0;
  std::size_t gemm_workspace_bytes = 0;
  std::int32_t padded_routed_rows = 0;
  double wall_ms = 0.0;
};

struct NativeMoePrefillOracleOptions {
  std::size_t layer_index = 0;
  // A padded fixed-shape schedule may execute only its causal active prefix.
  // Zero preserves the full workspace context.  The caller supplies GEMM
  // plans for this active shape; AOT storage/grid capacity remains unchanged.
  std::size_t active_tokens = 0;
  // Qualification may execute the product bucket while comparing/seeding
  // only its logical, unpadded prefix. Zero uses all executed rows.
  std::size_t comparison_tokens = 0;
  // Standalone MoE qualification seeds the two post-attention inputs.  A
  // complete-layer qualification disables this and consumes the tensors
  // produced by the native attention half in the same resident workspace.
  bool seed_post_attention = true;
  std::string post_attention_h2_oracle_label = "return-layer_body-h2";
  std::string post_attention_residual_oracle_label =
      "return-layer_body-after_attn";
  // The exact-routing rerun mutates the layer output.  Disable it when the
  // output feeds another native layer in the same production chain.
  bool run_routing_diagnostic = true;
  bool collect_oracle_comparisons = true;
  bool synchronize_substages = false;
  // The VL chain follows current vLLM's 256-way softmax and wave32 top-k
  // rounding. False preserves the frozen text product's serial selection and
  // selected-logit softmax order while honoring the captured weight ABI.
  bool use_vl_router_semantics = false;
  NativeQ8192PrefillGemmPlans* gemm_plans = nullptr;
  // A padded q1024 VL request may use a logical-M router plan for its active
  // prefix while retaining the bucket plans for every other MoE projection.
  // The plan token count must equal comparison_tokens. A null pointer keeps
  // the fixed bucket router used by text and wider-context paths.
  NativeQ8192PrefillGemmPlans* logical_router_gemm_plans = nullptr;
  std::string oracle_label_prefix;
  std::filesystem::path boundary_oracle_dir;
  std::string boundary_oracle_label_prefix;
  // Optional same-request layer-output ledger.  This separates native drift
  // from variation between independently captured qualification fixtures.
  std::filesystem::path chain_output_oracle_dir;
  std::string chain_output_oracle_label;
  // Long-context diagnosis can compare only the final token row at each
  // layer boundary, avoiding a 128 MiB oracle file per q32768 layer.
  bool chain_output_last_token_only = false;
};

struct NativeMoePrefillOracleResult {
  NativeMoePrefillMetrics layer;
  std::size_t seed_tensors = 0;
  std::size_t seed_bytes = 0;
  std::vector<NativeOracleComparison> comparisons;
  NativeOracleComparison chain_output_comparison;
  bool chain_output_comparison_provided = false;
  bool post_attention_seeded = false;
  bool routing_diagnostic_ran = false;
  bool all_finite = false;
  bool router_ids_exact = false;
  std::size_t router_expert_set_rows = 0;
  std::size_t router_expert_set_rows_exact = 0;
  bool router_expert_sets_exact = false;
  bool router_weights_gate_passed = false;
  bool dispatch_count_exact = false;
  bool shared_expert_gate_passed = false;
  bool combined_moe_gate_passed = false;
  bool expert_boundaries_provided = false;
  bool expert_boundaries_gate_passed = true;
  bool oracle_seeded_combined_moe_gate_passed = false;
  bool final_hidden_gate_passed = false;
};

// Executes the complete MoE half of one q8192 layer.  Standalone qualification
// may seed the post-attention boundary; complete-layer qualification consumes
// the tensors already produced by the native attention half.  The production
// operations are native HIP, hipBLASLt, and the two embedded fused-MoE code
// objects.  Oracle reads are qualification-only and are not part of the
// product request path.
NativeMoePrefillOracleResult probe_native_q8192_moe_prefill_layer0_oracle(
    const std::filesystem::path& oracle_dir,
    const NativeWeightStore& weights,
    const NativePrefillWorkspace& workspace,
    NativePrefillInvocations& invocations,
    NativeDecodeExecutor& executor,
    const NativeMoePrefillOracleOptions& options = {});

// Returns the full-sequence terminal hidden buffer selected by the same
// schedule-driven lifetime plan as the final MoE layer.
void* native_prefill_terminal_hidden_pointer(
    const NativePrefillWorkspace& workspace,
    const NativePrefillInvocations& invocations);

// Stable layer-boundary accessors used by the layer-major long-context loop.
// The scratch schedule remains O(1): every chunk is copied into the same layer
// input owner and its output is copied back to the full-prompt hidden store.
void* native_prefill_layer_input_pointer(
    const NativePrefillWorkspace& workspace,
    const NativePrefillInvocations& invocations,
    std::size_t layer_index);
void* native_prefill_layer_output_pointer(
    const NativePrefillWorkspace& workspace,
    const NativePrefillInvocations& invocations,
    std::size_t layer_index);

}  // namespace aima
