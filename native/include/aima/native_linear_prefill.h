#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_executor.h"
#include "aima/native_decode_workspace.h"
#include "aima/native_layer_oracle.h"
#include "aima/native_prefill_invocation.h"
#include "aima/native_prefill_workspace.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace aima {

class NativeQ8192PrefillGemmPlans;
class NativeDecodeBindings;

struct NativeLinearPrefillMetrics {
  std::size_t layer_index = 0;
  std::size_t tokens = 0;
  std::size_t dense_gemm_launches = 0;
  std::size_t diagnostic_gemm_launches = 0;
  std::size_t native_pointwise_launches = 0;
  std::size_t state_scratch_zero_operations = 0;
  std::size_t state_scratch_zero_bytes = 0;
  std::size_t semantic_alias_rebindings = 0;
  std::size_t resident_state_direct_bindings = 0;
  std::size_t resident_state_payload_bytes = 0;
  std::size_t aot_launches = 0;
  std::size_t gemm_workspace_bytes = 0;
  double wall_ms = 0.0;
};

struct NativeLinearPrefillOracleOptions {
  std::size_t layer_index = 0;
  bool seed_layer_input = true;
  // Keep the production layer output intact when composing multiple layers.
  // Single-layer qualification enables the seeded projection isolation pass.
  bool run_output_projection_diagnostic = true;
  // Product execution disables fixture reads while retaining the identical
  // parameterized GPU operation sequence.
  bool collect_oracle_comparisons = true;
  // When present, the prefill convolution kernel and FLA final-state kernel
  // write directly into the decode owner's resident input buffers.  This is
  // the product handoff: no host tensor owner and no post-prefill D2D copy.
  const NativeDecodeWorkspace* decode_state_workspace = nullptr;
  // Layer-major long-context execution carries the resident convolution and
  // recurrent states from the preceding chunk of the same layer.
  bool has_initial_state = false;
  // Product owners retain all fixed-shape hipBLASLt plans across layers and
  // requests. A null pointer preserves focused-probe ownership semantics.
  NativeQ8192PrefillGemmPlans* gemm_plans = nullptr;
  // Required by q32768 to consume the resident derived fused input weight.
  const NativeDecodeBindings* bindings = nullptr;
  // Multi-layer fixtures prefix local AOT labels with layer-XXX-.
  std::string oracle_label_prefix;
  // Optional focused fixture for qualification-only FLA launch boundaries.
  std::filesystem::path boundary_oracle_dir;
  std::string boundary_oracle_label_prefix;
  // Qualification-only last-token layer boundaries.  This is the bounded
  // long-context diagnostic counterpart to the full q8192 fixtures.
  std::filesystem::path tail_oracle_dir;
  std::string tail_oracle_label_prefix;
  // Qualification-only full-sequence layer boundaries. Sparse manifests are
  // allowed so a single 128 MiB boundary can answer an attribution question.
  std::filesystem::path sequence_oracle_dir;
  std::string sequence_oracle_label_prefix;
};

struct NativeLinearPrefillOracleResult {
  NativeLinearPrefillMetrics layer;
  std::size_t seed_tensors = 0;
  std::size_t seed_bytes = 0;
  std::vector<NativeOracleComparison> comparisons;
  std::vector<NativeOracleComparison> boundary_comparisons;
  bool layer_input_seeded = false;
  bool output_projection_diagnostic_ran = false;
  bool all_finite = false;
  bool final_state_gate_passed = false;
  bool attention_output_gate_passed = false;
  bool post_attention_gate_passed = false;
};

// Executes the complete attention half of one linear-attention layer for the
// qualified q8192
// shape: RMSNorm, four input projections, causal convolution, FLA chunk GDN,
// gated norm, output projection, residual add, and the post-attention RMSNorm.
// The oracle directory is a qualification fixture and is not a runtime input.
NativeLinearPrefillOracleResult
probe_native_q8192_linear_prefill_layer0_oracle(
    const std::filesystem::path& oracle_dir,
    const NativeWeightStore& weights,
    const NativePrefillWorkspace& workspace,
    NativePrefillInvocations& invocations,
    NativeDecodeExecutor& executor,
    const NativeLinearPrefillOracleOptions& options = {});

}  // namespace aima
