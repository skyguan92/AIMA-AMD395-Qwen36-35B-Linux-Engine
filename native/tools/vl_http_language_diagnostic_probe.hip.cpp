// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

// Reuse the hash-bound full-language qualification driver without changing
// it.  This diagnostic translation unit intercepts the attention/MoE calls
// only long enough to attach sparse HTTP-rendered attention boundaries to the
// existing per-layer comparison ledger.

#include "aima/native_linear_prefill.h"
#include "aima/native_moe_prefill.h"

#include <filesystem>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aima {
namespace {

std::filesystem::path http_language_diagnostic_case_dir;
std::vector<NativeOracleComparison> pending_attention_comparisons;

}  // namespace

NativeLinearPrefillOracleResult
probe_native_q8192_linear_prefill_http_diagnostic(
    const std::filesystem::path& oracle_dir,
    const NativeWeightStore& weights,
    const NativePrefillWorkspace& workspace,
    NativePrefillInvocations& invocations,
    NativeDecodeExecutor& executor,
    const NativeLinearPrefillOracleOptions& options) {
  NativeLinearPrefillOracleOptions diagnostic_options = options;
  const bool focused =
      !http_language_diagnostic_case_dir.empty() &&
      !options.collect_oracle_comparisons && options.layer_index == 1;
  if (focused) {
    std::ostringstream prefix;
    prefix << "layer-" << std::setw(3) << std::setfill('0')
           << options.layer_index << "-";
    diagnostic_options.sequence_oracle_dir =
        http_language_diagnostic_case_dir;
    diagnostic_options.sequence_oracle_label_prefix = prefix.str();
  }
  NativeLinearPrefillOracleResult result =
      probe_native_q8192_linear_prefill_layer0_oracle(
          oracle_dir, weights, workspace, invocations, executor,
          diagnostic_options);
  pending_attention_comparisons =
      focused ? result.boundary_comparisons
              : std::vector<NativeOracleComparison>{};
  return result;
}

NativeMoePrefillOracleResult probe_native_q8192_moe_prefill_http_diagnostic(
    const std::filesystem::path& oracle_dir,
    const NativeWeightStore& weights,
    const NativePrefillWorkspace& workspace,
    NativePrefillInvocations& invocations,
    NativeDecodeExecutor& executor,
    const NativeMoePrefillOracleOptions& options) {
  NativeMoePrefillOracleResult result =
      probe_native_q8192_moe_prefill_layer0_oracle(
          oracle_dir, weights, workspace, invocations, executor, options);
  if (!pending_attention_comparisons.empty()) {
    result.comparisons.insert(
        result.comparisons.begin(),
        std::make_move_iterator(pending_attention_comparisons.begin()),
        std::make_move_iterator(pending_attention_comparisons.end()));
    pending_attention_comparisons.clear();
  }
  return result;
}

}  // namespace aima

#define probe_native_q8192_linear_prefill_layer0_oracle \
  probe_native_q8192_linear_prefill_http_diagnostic
#define probe_native_q8192_moe_prefill_layer0_oracle \
  probe_native_q8192_moe_prefill_http_diagnostic
#define main aima_http_language_diagnostic_original_main
#include "vl_language_layer3_composed_oracle_probe.hip.cpp"
#undef main
#undef probe_native_q8192_moe_prefill_layer0_oracle
#undef probe_native_q8192_linear_prefill_layer0_oracle

int main(int argc, char** argv) {
  if (argc == 14) {
    const std::string selector = argv[9];
    if (selector == "all") {
      throw std::runtime_error(
          "HTTP language diagnostic requires one explicit case");
    }
    aima::http_language_diagnostic_case_dir =
        std::filesystem::absolute(argv[13]) / selector;
  }
  return aima_http_language_diagnostic_original_main(argc, argv);
}
