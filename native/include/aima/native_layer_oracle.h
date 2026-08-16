#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_executor.h"
#include "aima/native_decode_invocation.h"
#include "aima/native_decode_runner.h"
#include "aima/native_decode_workspace.h"
#include "aima/native_full_attention.h"
#include "aima/native_full_layer.h"
#include "aima/native_linear_layer.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace aima {

struct NativeOracleComparison {
  std::string label;
  std::string dtype;
  std::size_t elements = 0;
  std::size_t exact_elements = 0;
  std::size_t finite_elements = 0;
  bool first_mismatch_provided = false;
  std::size_t first_mismatch_index = 0;
  double first_mismatch_expected = 0.0;
  double first_mismatch_actual = 0.0;
  double maximum_absolute_error = 0.0;
  double relative_l2_error = 0.0;
  double cosine_similarity = 0.0;
  std::string expected_sha256;
  std::string actual_sha256;
};

struct NativeLogitsComparison {
  std::size_t elements = 0;
  std::size_t exact_elements = 0;
  std::size_t finite_elements = 0;
  std::uint32_t reference_top1_token_id = 0;
  std::uint32_t actual_top1_token_id = 0;
  bool top1_match = false;
  double maximum_absolute_error = 0.0;
  double relative_l2_error = 0.0;
  double kl_divergence = 0.0;
  std::string expected_sha256;
  std::string actual_sha256;
};

// Small public helpers used by native prefill qualification.  Oracle files are
// development/test inputs only; the product request path never reads them.
std::size_t seed_native_oracle_tensor(
    const std::filesystem::path& expected_path, void* actual_device,
    std::size_t expected_bytes);

NativeOracleComparison compare_native_oracle_tensor(
    const std::string& label, const std::string& dtype,
    const void* actual_device, std::size_t bytes,
    const std::filesystem::path& expected_path);

// Compares `bytes` against the leading bytes of a larger oracle tensor.  This
// is used to qualify a prefill-produced KV prefix against the subsequent
// decode fixture without materializing a sliced development artifact.
NativeOracleComparison compare_native_oracle_tensor_prefix(
    const std::string& label, const std::string& dtype,
    const void* actual_device, std::size_t bytes,
    const std::filesystem::path& expected_path);

// Compares a resident tensor against one byte range within a larger oracle.
// This lets decode qualification bind the current KV row to the final live
// row of a block-padded current-vLLM cache without creating derived fixtures.
NativeOracleComparison compare_native_oracle_tensor_slice(
    const std::string& label, const std::string& dtype,
    const void* actual_device, std::size_t bytes,
    const std::filesystem::path& expected_path,
    std::size_t expected_offset_bytes);

// Compares a complete resident FP32 vocabulary distribution to an oracle.
// KLD is reference||actual after a full-vocabulary softmax, not a truncated
// top-k proxy.  This helper is qualification-only and performs one D2H copy.
NativeLogitsComparison compare_native_logits_fp32(
    const void* actual_device, std::size_t elements,
    const std::filesystem::path& expected_path);

std::filesystem::path find_native_oracle_tensor_file(
    const std::filesystem::path& oracle_dir, const std::string& label);

// Optional counterpart for sparse diagnostic fixtures.  An empty path means
// the manifest does not carry the requested tensor; malformed/missing
// manifests still fail loudly.  Product execution never calls this helper.
std::filesystem::path find_native_oracle_tensor_file_if_present(
    const std::filesystem::path& oracle_dir, const std::string& label);

struct NativeLinearLayerOracleResult {
  NativeLinearLayerMetrics layer;
  std::size_t seed_tensors = 0;
  std::size_t seed_bytes = 0;
  std::vector<NativeOracleComparison> comparisons;
  bool all_finite = false;
  bool router_ids_exact = false;
  bool aot_boundaries_exact = false;
  double final_relative_l2_error = 0.0;
  double final_cosine_similarity = 0.0;
};

struct NativeFullAttentionCoreOracleResult {
  NativeFullAttentionCoreMetrics core;
  std::size_t seed_tensors = 0;
  std::size_t seed_bytes = 0;
  std::vector<NativeOracleComparison> comparisons;
  bool all_finite = false;
  bool kv_cache_exact = false;
  double scores_relative_l2_error = 0.0;
  double probabilities_relative_l2_error = 0.0;
  double attention_relative_l2_error = 0.0;
  double attention_cosine_similarity = 0.0;
};

NativeFullAttentionCoreOracleResult probe_native_full_attention_core_oracle(
    const std::filesystem::path& oracle_dir, std::size_t layer_index,
    std::size_t cache_end, NativeFullAttentionState& state);

struct NativeFullLayerOracleResult {
  NativeFullLayerMetrics layer;
  std::size_t seed_tensors = 0;
  std::size_t seed_bytes = 0;
  std::vector<NativeOracleComparison> comparisons;
  bool all_finite = false;
  bool kv_cache_exact = false;
  bool router_ids_exact = false;
  bool pre_attention_aot_exact = false;
  double attention_relative_l2_error = 0.0;
  double projected_attention_relative_l2_error = 0.0;
  double final_relative_l2_error = 0.0;
  double final_cosine_similarity = 0.0;
};

NativeFullLayerOracleResult probe_native_full_layer_oracle(
    const std::filesystem::path& oracle_dir, std::size_t layer_index,
    std::size_t cache_end, const NativeWeightStore& weights,
    const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, NativeFullAttentionState& attention_state,
    int cu_count);

struct NativeDecodeOracleResult {
  NativeDecodeRunMetrics decode;
  std::vector<NativeDecodeRunMetrics> warmup_decodes;
  std::vector<NativeDecodeRunMetrics> measured_decodes;
  std::size_t seed_tensors = 0;
  std::size_t seed_bytes = 0;
  std::vector<NativeOracleComparison> comparisons;
  std::size_t router_layers_exact = 0;
  std::size_t recurrent_states_exact = 0;
  bool all_finite = false;
  double final_hidden_relative_l2_error = 0.0;
  double final_hidden_cosine_similarity = 0.0;
};

struct NativeDecodeOracleOptions {
  std::size_t warmup_runs = 0;
  std::size_t measured_runs = 1;
};

NativeDecodeOracleResult probe_native_decode_oracle(
    const std::filesystem::path& oracle_dir, std::size_t cache_end,
    const NativeWeightStore& weights, const NativeLmHeadStore& lm_head,
    const NativeDecodeWorkspace& workspace,
    NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, NativeFullAttentionState& attention_state,
    int cu_count, const NativeDecodeOracleOptions& options = {});

NativeLinearLayerOracleResult probe_native_linear_layer_oracle(
    const std::filesystem::path& oracle_dir,
    std::size_t layer_index,
    const NativeWeightStore& weights,
    const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor,
    int cu_count);

}  // namespace aima
