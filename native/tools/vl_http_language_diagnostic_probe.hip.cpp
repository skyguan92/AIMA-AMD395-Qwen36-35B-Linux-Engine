// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

// Reuse the hash-bound full-language qualification driver without changing
// it.  This diagnostic translation unit intercepts the attention/MoE calls
// only long enough to attach sparse HTTP-rendered attention boundaries to the
// existing per-layer comparison ledger.

#include "aima/native_linear_prefill.h"
#include "aima/native_moe_prefill.h"

#include <hip/hip_runtime.h>

#include <cstring>
#include <filesystem>
#include <fstream>
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
std::filesystem::path http_language_diagnostic_actual_dir;
bool http_language_diagnostic_capture_enabled = true;
std::vector<NativeOracleComparison> pending_attention_comparisons;

void write_device_tensor(const std::string& name, const void* pointer,
                         std::size_t bytes) {
  if (http_language_diagnostic_actual_dir.empty()) return;
  std::vector<unsigned char> payload(bytes);
  const hipError_t copied = hipMemcpy(
      payload.data(), pointer, bytes, hipMemcpyDeviceToHost);
  if (copied != hipSuccess) {
    throw std::runtime_error(
        "HTTP language diagnostic device copy failed: " + name);
  }
  const std::filesystem::path path =
      http_language_diagnostic_actual_dir / name;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream ||
      !stream.write(reinterpret_cast<const char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size()))) {
    throw std::runtime_error(
        "HTTP language diagnostic tensor write failed: " + path.string());
  }
}

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
      http_language_diagnostic_capture_enabled &&
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
  if (http_language_diagnostic_capture_enabled &&
      !http_language_diagnostic_actual_dir.empty() &&
      (options.layer_index == 21 || options.layer_index == 30 ||
       options.layer_index == 31)) {
    const auto& launches = invocations.launches();
    std::size_t scratch_base = launches.size();
    std::size_t moe_first = launches.size();
    for (std::size_t index = 0; index < launches.size(); ++index) {
      const auto* launch = launches[index].launch;
      if (launch == nullptr) continue;
      if (scratch_base == launches.size() && launch->layer_index == 0 &&
          std::strcmp(launch->symbol, "triton_rmsnorm_kernel") == 0) {
        scratch_base = index;
      }
      if (moe_first == launches.size() &&
          static_cast<std::size_t>(launch->layer_index) ==
              options.layer_index &&
          std::strcmp(launch->symbol, "fused_moe_kernel") == 0) {
        moe_first = index;
      }
    }
    if (scratch_base == launches.size() || moe_first == launches.size()) {
      throw std::runtime_error(
          "HTTP language diagnostic router bindings are missing");
    }
    const std::size_t tokens =
        options.comparison_tokens == 0
            ? (options.active_tokens == 0 ? workspace.context_tokens()
                                          : options.active_tokens)
            : options.comparison_tokens;
    const std::string layer =
        "layer-" + (options.layer_index < 10 ? std::string("00")
                                             : std::string("0")) +
        std::to_string(options.layer_index) + "-";
    write_device_tensor(
        layer + "router_logits.bf16.bin",
        invocations.tensor_pointer(scratch_base + 4, "A"),
        tokens * 256 * sizeof(std::uint16_t));
    write_device_tensor(
        layer + "router_scores.f32.bin",
        invocations.tensor_pointer(scratch_base + 3, "o"),
        tokens * 8 * sizeof(float));
    write_device_tensor(
        layer + "router_weights.f32.bin",
        invocations.tensor_pointer(moe_first, "topk_weights_ptr"),
        tokens * 8 * sizeof(float));
    write_device_tensor(
        layer + "router_indices.i64.bin",
        invocations.tensor_pointer(scratch_base + 2, "a_ptr"),
        tokens * 8 * sizeof(std::int64_t));
  }
  if (!pending_attention_comparisons.empty()) {
    result.comparisons.insert(
        result.comparisons.begin(),
        std::make_move_iterator(pending_attention_comparisons.begin()),
        std::make_move_iterator(pending_attention_comparisons.end()));
    pending_attention_comparisons.clear();
  }
  if (!http_language_diagnostic_case_dir.empty() &&
      options.layer_index == 39) {
    http_language_diagnostic_capture_enabled = false;
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
    aima::http_language_diagnostic_actual_dir =
        std::filesystem::absolute(argv[11]) / "diagnostic-actual";
    std::filesystem::create_directories(
        aima::http_language_diagnostic_actual_dir);
  }
  return aima_http_language_diagnostic_original_main(argc, argv);
}
