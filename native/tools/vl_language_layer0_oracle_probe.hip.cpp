// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_bindings.h"
#include "aima/native_decode_executor.h"
#include "aima/native_derived_weights.h"
#include "aima/native_linear_prefill.h"
#include "aima/native_lm_head.h"
#include "aima/native_moe_prefill.h"
#include "aima/native_prefill_gemm_plans.h"
#include "aima/native_prefill_invocation.h"
#include "aima/native_prefill_workspace.h"
#include "aima/native_weight_store.h"
#include "aima/native_vl_logical_projections.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef AIMA_SOURCE_COMMIT
#define AIMA_SOURCE_COMMIT "unknown"
#endif

namespace {

using json = nlohmann::json;

constexpr std::size_t kBucketTokens = 1024;
constexpr std::size_t kLanguageHidden = 2048;
constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);
constexpr std::size_t kMeasuredRuns = 5;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorName(status) + " (" +
                             hipGetErrorString(status) + ")");
  }
}

class Event {
 public:
  Event() { check_hip(hipEventCreate(&event_), "hipEventCreate layer 0"); }
  ~Event() {
    if (event_ != nullptr) {
      const hipError_t ignored = hipEventDestroy(event_);
      static_cast<void>(ignored);
    }
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  operator hipEvent_t() const { return event_; }

 private:
  hipEvent_t event_ = nullptr;
};

json read_json(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("JSON file is unavailable: " + path.string());
  }
  json value;
  stream >> value;
  if (!value.is_object()) {
    throw std::runtime_error("JSON root is not an object: " + path.string());
  }
  return value;
}

std::filesystem::path checked_oracle_path(
    const std::filesystem::path& root, std::string_view relative_text) {
  const std::filesystem::path relative(relative_text);
  if (relative.empty() || relative.is_absolute()) {
    throw std::runtime_error("oracle tensor path is not relative");
  }
  for (const auto& component : relative) {
    if (component == "..") {
      throw std::runtime_error("oracle tensor path escapes its root");
    }
  }
  return root / relative;
}

std::vector<unsigned char> read_tensor_record(
    const std::filesystem::path& root, const json& record) {
  const std::size_t expected_bytes = record.at("bytes").get<std::size_t>();
  const std::filesystem::path path = checked_oracle_path(
      root, record.at("path").get<std::string>());
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() < 0 ||
      static_cast<std::size_t>(stream.tellg()) != expected_bytes) {
    throw std::runtime_error("oracle tensor size mismatch: " + path.string());
  }
  std::vector<unsigned char> bytes(expected_bytes);
  stream.seekg(0);
  if (expected_bytes != 0 &&
      !stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("oracle tensor read failed: " + path.string());
  }
  if (aima::sha256_bytes(bytes.data(), bytes.size()) !=
      record.at("sha256").get<std::string>()) {
    throw std::runtime_error("oracle tensor SHA-256 mismatch: " +
                             path.string());
  }
  return bytes;
}

void write_file(const std::filesystem::path& path,
                const std::vector<unsigned char>& bytes) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream ||
      !stream.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("language layer-0 output write failed: " +
                             path.string());
  }
}

float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0f;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

struct Comparison {
  std::size_t elements = 0;
  std::size_t exact_elements = 0;
  std::size_t finite_elements = 0;
  std::size_t first_mismatch_index = std::numeric_limits<std::size_t>::max();
  std::uint16_t first_expected_bits = 0;
  std::uint16_t first_actual_bits = 0;
  double maximum_absolute_error = 0.0;
  double relative_l2_error = 0.0;
  double cosine_similarity = 0.0;
  std::string expected_sha256;
  std::string actual_sha256;

  bool passed() const {
    return finite_elements == elements && relative_l2_error <= 0.002 &&
           cosine_similarity >= 0.999;
  }
};

Comparison compare_bf16(const std::vector<unsigned char>& actual,
                        const std::vector<unsigned char>& expected) {
  if (actual.size() != expected.size() || actual.size() % kBf16Bytes != 0) {
    throw std::invalid_argument("language layer-0 comparison sizes differ");
  }
  Comparison result;
  result.elements = actual.size() / kBf16Bytes;
  result.expected_sha256 = aima::sha256_bytes(expected.data(), expected.size());
  result.actual_sha256 = aima::sha256_bytes(actual.data(), actual.size());
  double squared_error = 0.0;
  double squared_expected = 0.0;
  double squared_actual = 0.0;
  double dot = 0.0;
  for (std::size_t index = 0; index < result.elements; ++index) {
    std::uint16_t expected_bits = 0;
    std::uint16_t actual_bits = 0;
    std::memcpy(&expected_bits,
                expected.data() + index * sizeof(expected_bits),
                sizeof(expected_bits));
    std::memcpy(&actual_bits, actual.data() + index * sizeof(actual_bits),
                sizeof(actual_bits));
    if (expected_bits == actual_bits) {
      ++result.exact_elements;
    } else if (result.first_mismatch_index ==
               std::numeric_limits<std::size_t>::max()) {
      result.first_mismatch_index = index;
      result.first_expected_bits = expected_bits;
      result.first_actual_bits = actual_bits;
    }
    const double expected_value = bf16_to_float(expected_bits);
    const double actual_value = bf16_to_float(actual_bits);
    if (std::isfinite(actual_value)) ++result.finite_elements;
    const double error = actual_value - expected_value;
    result.maximum_absolute_error =
        std::max(result.maximum_absolute_error, std::abs(error));
    squared_error += error * error;
    squared_expected += expected_value * expected_value;
    squared_actual += actual_value * actual_value;
    dot += expected_value * actual_value;
  }
  result.relative_l2_error =
      std::sqrt(squared_error / std::max(squared_expected, 1.0e-30));
  result.cosine_similarity =
      dot / std::sqrt(std::max(squared_expected * squared_actual, 1.0e-30));
  return result;
}

struct Execution {
  std::vector<unsigned char> output;
  std::vector<unsigned char> residual_output;
  std::vector<aima::NativeOracleComparison> diagnostic_comparisons;
  std::vector<aima::NativeOracleComparison>
      seeded_moe_diagnostic_comparisons;
  std::size_t router_expert_set_rows = 0;
  std::size_t router_expert_set_rows_exact = 0;
  bool router_expert_sets_exact = false;
  std::size_t seeded_router_expert_set_rows = 0;
  std::size_t seeded_router_expert_set_rows_exact = 0;
  bool seeded_router_expert_sets_exact = false;
  float measured_ms = 0.0f;
  aima::NativePrefillWorkspaceMetrics workspace;
  aima::NativePrefillInvocationMetrics invocations;
  aima::NativeLinearPrefillMetrics linear;
  aima::NativeMoePrefillMetrics moe;
};

void* reference_layer0_first_tensor_pointer(
    const aima::NativePrefillInvocations& invocations) {
  // Qwen3_5DecoderLayer returns (hidden_states, residual).  The frozen hook's
  // recursive first-tensor rule therefore captures the MoE branch before it
  // is added to the residual stream.  The direct q1024 lifetime plan reuses
  // the now-dead FLA v_new storage for that combined routed/shared MoE value.
  const auto& launches = invocations.launches();
  for (std::size_t sequence = 0; sequence < launches.size(); ++sequence) {
    const auto* launch = launches[sequence].launch;
    if (launch != nullptr && launch->symbol != nullptr &&
        launch->layer_index == 0 &&
        std::string_view(launch->symbol) ==
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64") {
      return invocations.tensor_pointer(sequence, "v_new");
    }
  }
  throw std::runtime_error(
      "q1024 layer-0 MoE branch lifetime owner is missing");
}

Execution execute_layer0(
    const std::vector<unsigned char>& injected_embeddings,
    std::size_t prompt_tokens, int device,
    const aima::NativeWeightStore& weights,
    const aima::NativeDecodeBindings& bindings,
    aima::NativeDecodeExecutor& executor,
    aima::NativeQ8192PrefillGemmPlans& active_gemm_plans,
    aima::NativeVlLogicalProjectionState& logical_projections,
    const std::filesystem::path& diagnostic_oracle_dir = {}) {
  if (prompt_tokens == 0 || prompt_tokens > kBucketTokens ||
      injected_embeddings.size() !=
          prompt_tokens * kLanguageHidden * kBf16Bytes) {
    throw std::invalid_argument("language layer-0 input geometry is invalid");
  }

  aima::NativePrefillWorkspace workspace;
  Execution result;
  result.workspace = workspace.build(device, kBucketTokens);
  aima::NativePrefillInvocations invocations;
  result.invocations = invocations.build(bindings, workspace, kBucketTokens);
  void* layer_input =
      aima::native_prefill_layer_input_pointer(workspace, invocations, 0);
  check_hip(hipMemset(layer_input, 0,
                      kBucketTokens * kLanguageHidden * kBf16Bytes),
            "hipMemset padded language layer-0 input");
  check_hip(hipMemcpy(layer_input, injected_embeddings.data(),
                      injected_embeddings.size(), hipMemcpyHostToDevice),
            "hipMemcpy injected embeddings to language layer 0");
  check_hip(hipDeviceSynchronize(),
            "hipDeviceSynchronize language layer-0 input");

  aima::NativeLinearPrefillOracleOptions linear_options;
  linear_options.layer_index = 0;
  linear_options.use_vl_rmsnorm_semantics = true;
  linear_options.active_tokens = 0;
  linear_options.comparison_tokens = prompt_tokens;
  linear_options.exact_b_projection_tokens =
      prompt_tokens <= 64 ? prompt_tokens : 0;
  linear_options.seed_layer_input = false;
  linear_options.run_output_projection_diagnostic = false;
  linear_options.collect_oracle_comparisons = false;
  linear_options.has_initial_state = false;
  linear_options.gemm_plans = &active_gemm_plans;
  linear_options.logical_ab_gemm_plan = &logical_projections.ab_plan();
  linear_options.logical_ab_weight = logical_projections.ab_weight(0);
  linear_options.logical_ab_output = logical_projections.ab_output();
  linear_options.bindings = &bindings;
  linear_options.sequence_oracle_dir = diagnostic_oracle_dir;
  aima::NativeMoePrefillOracleOptions moe_options;
  moe_options.layer_index = 0;
  moe_options.use_vl_router_semantics = true;
  moe_options.active_tokens = 0;
  moe_options.comparison_tokens = prompt_tokens;
  moe_options.seed_post_attention = false;
  moe_options.run_routing_diagnostic = false;
  moe_options.collect_oracle_comparisons = false;
  moe_options.gemm_plans = &active_gemm_plans;
  moe_options.logical_router_gemm_plans =
      &logical_projections.router_gemm_plans();
  if (!diagnostic_oracle_dir.empty()) {
    moe_options.chain_output_oracle_dir = diagnostic_oracle_dir;
    moe_options.chain_output_oracle_label = "diagnostic-output";
  }

  Event start;
  Event stop;
  check_hip(hipEventRecord(start), "hipEventRecord language layer-0 start");
  const aima::NativeLinearPrefillOracleResult linear =
      aima::probe_native_q8192_linear_prefill_layer0_oracle(
          {}, weights, workspace, invocations, executor, linear_options);
  const aima::NativeMoePrefillOracleResult moe =
      aima::probe_native_q8192_moe_prefill_layer0_oracle(
          {}, weights, workspace, invocations, executor, moe_options);
  check_hip(hipEventRecord(stop), "hipEventRecord language layer-0 stop");
  check_hip(hipEventSynchronize(stop),
            "hipEventSynchronize language layer-0 stop");
  check_hip(hipEventElapsedTime(&result.measured_ms, start, stop),
            "hipEventElapsedTime language layer 0");
  result.linear = linear.layer;
  result.moe = moe.layer;
  result.diagnostic_comparisons = linear.boundary_comparisons;
  result.diagnostic_comparisons.insert(
      result.diagnostic_comparisons.end(), moe.comparisons.begin(),
      moe.comparisons.end());
  result.router_expert_set_rows = moe.router_expert_set_rows;
  result.router_expert_set_rows_exact = moe.router_expert_set_rows_exact;
  result.router_expert_sets_exact = moe.router_expert_sets_exact;
  if (moe.chain_output_comparison_provided) {
    result.diagnostic_comparisons.push_back(moe.chain_output_comparison);
  }

  const void* layer_output =
      reference_layer0_first_tensor_pointer(invocations);
  const void* residual_output =
      aima::native_prefill_layer_output_pointer(workspace, invocations, 0);
  result.output.resize(injected_embeddings.size());
  result.residual_output.resize(injected_embeddings.size());
  check_hip(hipMemcpy(result.output.data(), layer_output, result.output.size(),
                      hipMemcpyDeviceToHost),
            "hipMemcpy vLLM first-tensor language layer-0 output");
  check_hip(hipMemcpy(result.residual_output.data(), residual_output,
                      result.residual_output.size(), hipMemcpyDeviceToHost),
            "hipMemcpy native language layer-0 residual output");
  if (!diagnostic_oracle_dir.empty()) {
    aima::NativeMoePrefillOracleOptions seeded_moe_options;
    seeded_moe_options.layer_index = 0;
    seeded_moe_options.use_vl_router_semantics = true;
    seeded_moe_options.active_tokens = 0;
    seeded_moe_options.comparison_tokens = prompt_tokens;
    seeded_moe_options.seed_post_attention = true;
    seeded_moe_options.post_attention_h2_oracle_label = "diagnostic-h2";
    seeded_moe_options.post_attention_residual_oracle_label =
        "launch-009-residual_out";
    seeded_moe_options.run_routing_diagnostic = false;
    seeded_moe_options.collect_oracle_comparisons = false;
    seeded_moe_options.gemm_plans = &active_gemm_plans;
    seeded_moe_options.logical_router_gemm_plans =
        &logical_projections.router_gemm_plans();
    seeded_moe_options.chain_output_oracle_dir = diagnostic_oracle_dir;
    seeded_moe_options.chain_output_oracle_label = "diagnostic-output";
    const aima::NativeMoePrefillOracleResult seeded_moe =
        aima::probe_native_q8192_moe_prefill_layer0_oracle(
            diagnostic_oracle_dir, weights, workspace, invocations, executor,
            seeded_moe_options);
    result.seeded_moe_diagnostic_comparisons = seeded_moe.comparisons;
    result.seeded_router_expert_set_rows =
        seeded_moe.router_expert_set_rows;
    result.seeded_router_expert_set_rows_exact =
        seeded_moe.router_expert_set_rows_exact;
    result.seeded_router_expert_sets_exact =
        seeded_moe.router_expert_sets_exact;
    if (seeded_moe.chain_output_comparison_provided) {
      result.seeded_moe_diagnostic_comparisons.push_back(
          seeded_moe.chain_output_comparison);
    }
  }
  return result;
}

json qualify_case(
    const json& case_record, const std::filesystem::path& oracle_root,
    const std::filesystem::path& actual_output,
    const std::filesystem::path& diagnostic_oracle_root, int device,
    const aima::NativeWeightStore& weights,
    const aima::NativeDecodeBindings& bindings,
    aima::NativeVlLogicalProjectionState& logical_projections,
    aima::NativeDecodeExecutor& executor) {
  const std::string case_id = case_record.at("case_id").get<std::string>();
  const json& injected_record =
      case_record.at("boundaries").at("injected_embeddings");
  const json& expected_record =
      case_record.at("boundaries").at("language_layer_0");
  if (injected_record.value("dtype", "") != "torch.bfloat16" ||
      expected_record.value("dtype", "") != "torch.bfloat16" ||
      !injected_record.at("shape").is_array() ||
      injected_record.at("shape").size() != 2 ||
      expected_record.at("shape") != injected_record.at("shape") ||
      injected_record.at("shape").at(1).get<std::size_t>() !=
          kLanguageHidden) {
    throw std::runtime_error("language layer-0 oracle shape/dtype is invalid");
  }
  const std::size_t prompt_tokens =
      injected_record.at("shape").at(0).get<std::size_t>();
  if (prompt_tokens == 0 || prompt_tokens > kBucketTokens ||
      case_record.at("processor").at("prompt_token_ids").size() !=
          prompt_tokens) {
    throw std::runtime_error("language layer-0 prompt length is invalid");
  }
  const std::vector<unsigned char> injected =
      read_tensor_record(oracle_root, injected_record);
  const std::vector<unsigned char> expected =
      read_tensor_record(oracle_root, expected_record);
  const std::size_t expected_bytes =
      prompt_tokens * kLanguageHidden * kBf16Bytes;
  if (injected.size() != expected_bytes || expected.size() != expected_bytes) {
    throw std::runtime_error("language layer-0 oracle byte count is invalid");
  }

  aima::NativeQ8192PrefillGemmPlans active_gemm_plans(kBucketTokens);
  (void)active_gemm_plans.linear_fused_input();
  (void)active_gemm_plans.linear_output();
  (void)active_gemm_plans.moe_shared_gate();
  (void)active_gemm_plans.moe_shared_projection();
  (void)active_gemm_plans.moe_shared_down();
  (void)active_gemm_plans.moe_router();
  const aima::NativeVlLogicalProjectionPrepareMetrics logical_metrics =
      logical_projections.prepare(prompt_tokens);
  if (!logical_metrics.prepared || logical_metrics.plan_count != 2) {
    throw std::runtime_error("logical VL projection plans are incomplete");
  }

  const auto case_started = std::chrono::steady_clock::now();
  Execution diagnostic;
  if (!diagnostic_oracle_root.empty()) {
    diagnostic = execute_layer0(
        injected, prompt_tokens, device, weights, bindings, executor,
        active_gemm_plans, logical_projections,
        diagnostic_oracle_root / case_id);
  }
  const Execution warmup = execute_layer0(
      injected, prompt_tokens, device, weights, bindings, executor,
      active_gemm_plans, logical_projections);
  std::vector<Execution> measured;
  measured.reserve(kMeasuredRuns);
  for (std::size_t run = 0; run < kMeasuredRuns; ++run) {
    measured.push_back(execute_layer0(
        injected, prompt_tokens, device, weights, bindings, executor,
        active_gemm_plans, logical_projections));
  }
  write_file(actual_output, measured.front().output);
  const Comparison comparison = compare_bf16(measured.front().output, expected);
  const std::string repeat_sha256 =
      aima::sha256_bytes(warmup.output.data(), warmup.output.size());
  bool deterministic = warmup.output == measured.front().output;
  bool residual_deterministic =
      warmup.residual_output == measured.front().residual_output;
  std::vector<float> measured_ms;
  measured_ms.reserve(measured.size());
  for (const Execution& execution : measured) {
    deterministic = deterministic &&
                    execution.output == measured.front().output;
    residual_deterministic =
        residual_deterministic &&
        execution.residual_output == measured.front().residual_output;
    measured_ms.push_back(execution.measured_ms);
  }
  std::vector<float> sorted_ms = measured_ms;
  std::sort(sorted_ms.begin(), sorted_ms.end());
  const Execution& representative = measured.front();
  const bool passed = comparison.passed() && deterministic;
  const double case_wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - case_started)
          .count();

  json diagnostic_comparisons = json::array();
  json seeded_moe_diagnostic_comparisons = json::array();
  bool diagnostic_complete = diagnostic_oracle_root.empty();
  bool seeded_moe_diagnostic_complete = diagnostic_oracle_root.empty();
  std::string first_failed_diagnostic_stage;
  std::string first_failed_seeded_moe_stage;
  if (!diagnostic_oracle_root.empty()) {
    const std::set<std::string> expected_labels = {
        "input_norm_full_sequence",
        "linear_projection_fused_full_sequence",
        "linear_projection_a_full_sequence",
        "linear_projection_b_full_sequence",
        "linear_convolution_full_sequence",
        "fla_q_full_sequence",
        "fla_k_full_sequence",
        "fla_v_full_sequence",
        "fla_g_full_sequence",
        "fla_beta_full_sequence",
        "fla_core_full_sequence",
        "linear_gated_output_full_sequence",
        "attention_output_full_sequence",
        "post_attention_full_sequence",
        "post_attention_norm_full_sequence",
        "diagnostic-h2",
        "diagnostic-router_logits",
        "diagnostic-router_scores",
        "diagnostic-router_weights",
        "diagnostic-router_indices",
        "diagnostic-shared_out",
        "diagnostic-routed_moe",
        "diagnostic-moe_out",
        "same_request_layer_output",
    };
    std::set<std::string> actual_labels;
    for (const aima::NativeOracleComparison& value :
         diagnostic.diagnostic_comparisons) {
      actual_labels.insert(value.label);
      const bool stage_passed =
          value.finite_elements == value.elements &&
          (value.label == "diagnostic-router_indices"
               ? diagnostic.router_expert_sets_exact
               : value.relative_l2_error <= 0.002 &&
                     value.cosine_similarity >= 0.999);
      if (!stage_passed && first_failed_diagnostic_stage.empty()) {
        first_failed_diagnostic_stage = value.label;
      }
      diagnostic_comparisons.push_back({
          {"label", value.label},
          {"dtype", value.dtype},
          {"elements", value.elements},
          {"exact_elements", value.exact_elements},
          {"finite_elements", value.finite_elements},
          {"first_mismatch_provided", value.first_mismatch_provided},
          {"first_mismatch_index", value.first_mismatch_index},
          {"first_mismatch_expected", value.first_mismatch_expected},
          {"first_mismatch_actual", value.first_mismatch_actual},
          {"maximum_absolute_error", value.maximum_absolute_error},
          {"relative_l2_error", value.relative_l2_error},
          {"cosine_similarity", value.cosine_similarity},
          {"expected_sha256", value.expected_sha256},
          {"actual_sha256", value.actual_sha256},
          {"passed", stage_passed},
      });
    }
    diagnostic_complete = actual_labels == expected_labels;
    if (!diagnostic_complete) {
      std::string detail;
      for (const std::string& label : expected_labels) {
        if (actual_labels.count(label) == 0) detail += " missing=" + label;
      }
      for (const std::string& label : actual_labels) {
        if (expected_labels.count(label) == 0) detail += " extra=" + label;
      }
      throw std::runtime_error(
          "language layer-0 diagnostic comparison set is incomplete:" +
          detail);
    }

    const std::set<std::string> expected_seeded_moe_labels = {
        "diagnostic-h2",
        "diagnostic-router_logits",
        "diagnostic-router_scores",
        "diagnostic-router_weights",
        "diagnostic-router_indices",
        "diagnostic-shared_out",
        "diagnostic-routed_moe",
        "diagnostic-moe_out",
        "same_request_layer_output",
    };
    std::set<std::string> actual_seeded_moe_labels;
    for (const aima::NativeOracleComparison& value :
         diagnostic.seeded_moe_diagnostic_comparisons) {
      actual_seeded_moe_labels.insert(value.label);
      const bool stage_passed =
          value.finite_elements == value.elements &&
          (value.label == "diagnostic-router_indices"
               ? diagnostic.seeded_router_expert_sets_exact
               : value.relative_l2_error <= 0.002 &&
                     value.cosine_similarity >= 0.999);
      if (!stage_passed && first_failed_seeded_moe_stage.empty()) {
        first_failed_seeded_moe_stage = value.label;
      }
      seeded_moe_diagnostic_comparisons.push_back({
          {"label", value.label},
          {"dtype", value.dtype},
          {"elements", value.elements},
          {"exact_elements", value.exact_elements},
          {"finite_elements", value.finite_elements},
          {"first_mismatch_provided", value.first_mismatch_provided},
          {"first_mismatch_index", value.first_mismatch_index},
          {"first_mismatch_expected", value.first_mismatch_expected},
          {"first_mismatch_actual", value.first_mismatch_actual},
          {"maximum_absolute_error", value.maximum_absolute_error},
          {"relative_l2_error", value.relative_l2_error},
          {"cosine_similarity", value.cosine_similarity},
          {"expected_sha256", value.expected_sha256},
          {"actual_sha256", value.actual_sha256},
          {"passed", stage_passed},
      });
    }
    seeded_moe_diagnostic_complete =
        actual_seeded_moe_labels == expected_seeded_moe_labels;
    if (!seeded_moe_diagnostic_complete) {
      std::string detail;
      for (const std::string& label : expected_seeded_moe_labels) {
        if (actual_seeded_moe_labels.count(label) == 0) {
          detail += " missing=" + label;
        }
      }
      for (const std::string& label : actual_seeded_moe_labels) {
        if (expected_seeded_moe_labels.count(label) == 0) {
          detail += " extra=" + label;
        }
      }
      throw std::runtime_error(
          "language layer-0 seeded MoE diagnostic comparison set is "
          "incomplete:" + detail);
    }
  }

  return {
      {"schema", "aima-amd395-qwen36/native-vl-language-layer0-case/v1"},
      {"complete", passed},
      {"case_id", case_id},
      {"prompt_tokens", prompt_tokens},
      {"bucket_tokens", kBucketTokens},
      {"padding_tokens", kBucketTokens - prompt_tokens},
      {"elements", comparison.elements},
      {"exact_elements", comparison.exact_elements},
      {"finite_elements", comparison.finite_elements},
      {"first_mismatch_index",
       comparison.first_mismatch_index ==
               std::numeric_limits<std::size_t>::max()
           ? -1LL
           : static_cast<long long>(comparison.first_mismatch_index)},
      {"first_expected_bits", comparison.first_expected_bits},
      {"first_actual_bits", comparison.first_actual_bits},
      {"maximum_absolute_error", comparison.maximum_absolute_error},
      {"relative_l2_error", comparison.relative_l2_error},
      {"cosine_similarity", comparison.cosine_similarity},
      {"expected_sha256", comparison.expected_sha256},
      {"actual_sha256", comparison.actual_sha256},
      {"warmup_actual_sha256", repeat_sha256},
      {"repeat_deterministic", deterministic},
      {"native_residual_output_sha256",
       aima::sha256_bytes(representative.residual_output.data(),
                          representative.residual_output.size())},
      {"native_residual_repeat_deterministic", residual_deterministic},
      {"bit_exact", comparison.exact_elements == comparison.elements},
      {"diagnostic_oracle_provided", !diagnostic_oracle_root.empty()},
      {"diagnostic_complete", diagnostic_complete},
      {"first_failed_diagnostic_stage", first_failed_diagnostic_stage},
      {"router_expert_set_rows", diagnostic.router_expert_set_rows},
      {"router_expert_set_rows_exact",
       diagnostic.router_expert_set_rows_exact},
      {"router_expert_sets_exact", diagnostic.router_expert_sets_exact},
      {"diagnostic_comparisons", std::move(diagnostic_comparisons)},
      {"seeded_moe_diagnostic_complete", seeded_moe_diagnostic_complete},
      {"first_failed_seeded_moe_stage", first_failed_seeded_moe_stage},
      {"seeded_router_expert_set_rows",
       diagnostic.seeded_router_expert_set_rows},
      {"seeded_router_expert_set_rows_exact",
       diagnostic.seeded_router_expert_set_rows_exact},
      {"seeded_router_expert_sets_exact",
       diagnostic.seeded_router_expert_sets_exact},
      {"seeded_moe_diagnostic_comparisons",
       std::move(seeded_moe_diagnostic_comparisons)},
      {"measured_ms", measured_ms},
      {"median_ms", sorted_ms[sorted_ms.size() / 2]},
      {"case_wall_ms", case_wall_ms},
      {"workspace_allocation_bytes",
       representative.workspace.allocation_bytes},
      {"prepared_launches", representative.invocations.launch_count},
      {"active_gemm_plan_count", active_gemm_plans.built_plan_count()},
      {"active_gemm_workspace_bytes", active_gemm_plans.workspace_bytes()},
      {"logical_projection_plan_count", logical_metrics.plan_count},
      {"logical_projection_workspace_bytes", logical_metrics.workspace_bytes},
      {"logical_projection_plan_build_wall_ms", logical_metrics.build_wall_ms},
      {"logical_projection_plan_reused", logical_metrics.reused},
      {"linear_dense_gemm_launches",
       representative.linear.dense_gemm_launches},
      {"linear_native_pointwise_launches",
       representative.linear.native_pointwise_launches},
      {"linear_aot_launches", representative.linear.aot_launches},
      {"moe_dense_gemm_launches", representative.moe.dense_gemm_launches},
      {"moe_native_router_launches",
       representative.moe.native_router_launches},
      {"moe_native_dispatch_launches",
       representative.moe.native_dispatch_launches},
      {"moe_native_pointwise_launches",
       representative.moe.native_pointwise_launches},
      {"moe_aot_launches", representative.moe.aot_launches},
      {"injected_embeddings_sha256", injected_record.at("sha256")},
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7 && argc != 8) {
    std::cerr
        << "usage: native-vl-language-layer0-probe MODEL_DIR ORACLE_MANIFEST "
           "ORACLE_ROOT CASE_ID_OR_ALL LOAD_REPORT ACTUAL_OUTPUT_OR_DIR "
           "[DIAGNOSTIC_ORACLE_ROOT]\n";
    return 2;
  }
  try {
    const std::filesystem::path manifest_path =
        std::filesystem::absolute(argv[2]);
    const std::filesystem::path oracle_root =
        std::filesystem::absolute(argv[3]);
    const std::string selector = argv[4];
    const bool all_cases = selector == "all";
    const std::filesystem::path output = std::filesystem::absolute(argv[6]);
    const std::filesystem::path diagnostic_oracle_root =
        argc == 8 ? std::filesystem::absolute(argv[7])
                  : std::filesystem::path{};
    const json manifest = read_json(manifest_path);
    if (manifest.value("schema", "") !=
            "aima-amd395-qwen36/vl-oracle-manifest/v1" ||
        !manifest.value("complete", false) ||
        !manifest.contains("cases") || !manifest.at("cases").is_array()) {
      throw std::runtime_error("VL oracle manifest is incomplete");
    }
    std::vector<const json*> selected;
    std::vector<std::string> case_ids;
    for (const json& value : manifest.at("cases")) {
      const std::string case_id = value.value("case_id", "");
      if (case_id.empty() ||
          std::find(case_ids.begin(), case_ids.end(), case_id) !=
              case_ids.end()) {
        throw std::runtime_error("VL oracle case id is empty or duplicated");
      }
      case_ids.push_back(case_id);
      if (all_cases || case_id == selector) selected.push_back(&value);
    }
    if (selected.empty()) {
      throw std::runtime_error("VL oracle case id was not found");
    }
    if (all_cases) {
      if (std::filesystem::exists(output) &&
          !std::filesystem::is_directory(output)) {
        throw std::runtime_error("all-case output path is not a directory");
      }
      std::filesystem::create_directories(output);
    }

    const auto started = std::chrono::steady_clock::now();
    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[5]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load(options);
    aima::NativeDerivedWeightStore derived;
    const aima::NativeDerivedWeightMetrics derived_metrics =
        derived.build(weights, options.device);
    aima::NativeLmHeadStore lm_head;
    const aima::NativeLmHeadMetrics lm_head_metrics =
        lm_head.build(weights, options.device);
    aima::NativeDecodeBindings bindings;
    const aima::NativeDecodeBindingMetrics binding_metrics =
        bindings.build(weights, derived, lm_head);
    aima::NativeVlLogicalProjectionState logical_projections;
    const aima::NativeVlLogicalProjectionLoadMetrics logical_load_metrics =
        logical_projections.build(weights, kBucketTokens, options.device);
    aima::NativeDecodeExecutor executor;
    const aima::NativeDecodeExecutorMetrics executor_metrics = executor.load();
    json case_results = json::array();
    bool complete = true;
    std::size_t total_elements = 0;
    std::size_t total_exact_elements = 0;
    for (const json* case_record : selected) {
      const std::string case_id =
          case_record->at("case_id").get<std::string>();
      const std::filesystem::path actual_output =
          all_cases ? output / (case_id + ".bin") : output;
      json result = qualify_case(
          *case_record, oracle_root, actual_output, diagnostic_oracle_root,
          options.device, weights, bindings, logical_projections, executor);
      complete = complete && result.at("complete").get<bool>();
      total_elements += result.at("elements").get<std::size_t>();
      total_exact_elements +=
          result.at("exact_elements").get<std::size_t>();
      case_results.push_back(std::move(result));
    }
    const double total_wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();
    const json result = {
        {"schema",
         "aima-amd395-qwen36/native-vl-language-layer0-qualification-run/v1"},
        {"complete", complete},
        {"source_commit", AIMA_SOURCE_COMMIT},
        {"case_selector", selector},
        {"case_count", case_results.size()},
        {"single_resident_weight_load", true},
        {"schedule_context_tokens", kBucketTokens},
        {"warmup_runs_per_case", 1},
        {"measured_runs_per_case", kMeasuredRuns},
        {"oracle_manifest_sha256", aima::sha256_file(manifest_path)},
        {"language_weight_payload_bytes", load.payload_bytes},
        {"language_weight_load_wall_ms", load.load_wall_ms},
        {"derived_weight_payload_bytes", derived_metrics.payload_bytes},
        {"derived_weight_build_wall_ms", derived_metrics.build_wall_ms},
        {"lm_head_payload_bytes", lm_head_metrics.payload_bytes},
        {"lm_head_build_wall_ms", lm_head_metrics.build_wall_ms},
        {"decode_weight_bindings", binding_metrics.unique_bindings},
        {"logical_projection_weights_loaded", logical_load_metrics.loaded},
        {"logical_projection_weight_bytes", logical_load_metrics.weight_bytes},
        {"logical_projection_output_scratch_bytes",
         logical_load_metrics.output_scratch_bytes},
        {"logical_projection_weight_build_wall_ms",
         logical_load_metrics.build_wall_ms},
        {"aot_loaded_modules", executor_metrics.loaded_modules},
        {"total_elements", total_elements},
        {"total_exact_elements", total_exact_elements},
        {"all_bit_exact", total_elements == total_exact_elements},
        {"runtime_python", false},
        {"runtime_numpy", false},
        {"runtime_torch", false},
        {"runtime_vllm", false},
        {"runtime_triton", false},
        {"total_wall_ms", total_wall_ms},
        {"cases", std::move(case_results)},
    };
    std::cout << result.dump() << '\n';
    return complete ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native VL language layer-0 probe: " << error.what() << '\n';
    return 1;
  }
}
