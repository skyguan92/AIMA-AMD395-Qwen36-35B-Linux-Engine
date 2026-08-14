// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/bf16_gemm.h"
#include "aima/native_decode_bindings.h"
#include "aima/native_decode_executor.h"
#include "aima/native_derived_weights.h"
#include "aima/native_full_prefill.h"
#include "aima/native_linear_prefill.h"
#include "aima/native_lm_head.h"
#include "aima/native_moe_prefill.h"
#include "aima/native_pointwise.h"
#include "aima/native_prefill_gemm_plans.h"
#include "aima/native_prefill_invocation.h"
#include "aima/native_prefill_workspace.h"
#include "aima/native_vl_unified_attention.h"
#include "aima/native_vl_logical_projections.h"
#include "aima/native_weight_store.h"
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
#include <utility>
#include <vector>

#ifndef AIMA_SOURCE_COMMIT
#define AIMA_SOURCE_COMMIT "unknown"
#endif

namespace {

using json = nlohmann::json;

constexpr std::size_t kBucketTokens = 1024;
constexpr std::size_t kLanguageHidden = 2048;
constexpr std::size_t kVocabulary = 248320;
constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);
constexpr std::size_t kMeasuredRuns = 5;

std::string layer_output_oracle_label(std::size_t layer_index) {
  return "layer-" + std::string(layer_index < 10 ? "00" : "0") +
         std::to_string(layer_index) + "-return-layer_body-output";
}

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorName(status) + " (" +
                             hipGetErrorString(status) + ")");
  }
}

class Event {
 public:
  Event() { check_hip(hipEventCreate(&event_), "hipEventCreate layer 0-3"); }
  ~Event() {
    if (event_ != nullptr) (void)hipEventDestroy(event_);
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  operator hipEvent_t() const { return event_; }

 private:
  hipEvent_t event_ = nullptr;
};

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ == 0) throw std::invalid_argument("zero device allocation");
    check_hip(hipMalloc(&pointer_, bytes_), "hipMalloc layer 0-3 probe");
  }
  ~DeviceAllocation() {
    if (pointer_ != nullptr) (void)hipFree(pointer_);
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  void* get() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
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

std::filesystem::path checked_path(
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

std::vector<unsigned char> read_tensor(
    const std::filesystem::path& root, const json& record) {
  const std::size_t bytes = record.at("bytes").get<std::size_t>();
  const std::filesystem::path path =
      checked_path(root, record.at("path").get<std::string>());
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() < 0 ||
      static_cast<std::size_t>(stream.tellg()) != bytes) {
    throw std::runtime_error("oracle tensor size mismatch: " + path.string());
  }
  std::vector<unsigned char> result(bytes);
  stream.seekg(0);
  if (bytes != 0 &&
      !stream.read(reinterpret_cast<char*>(result.data()),
                   static_cast<std::streamsize>(result.size()))) {
    throw std::runtime_error("oracle tensor read failed: " + path.string());
  }
  if (aima::sha256_bytes(result.data(), result.size()) !=
      record.at("sha256").get<std::string>()) {
    throw std::runtime_error("oracle tensor SHA-256 mismatch: " +
                             path.string());
  }
  return result;
}

void write_file(const std::filesystem::path& path,
                const std::vector<unsigned char>& bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream ||
      !stream.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("layer 0-3 output write failed: " + path.string());
  }
}

bool comparison_passed(const aima::NativeOracleComparison& comparison) {
  return comparison.finite_elements == comparison.elements &&
         comparison.relative_l2_error <= 0.002 &&
         comparison.cosine_similarity >= 0.999;
}

json comparison_json(const aima::NativeOracleComparison& value) {
  return {
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
      {"passed", comparison_passed(value)},
  };
}

float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0f;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

struct Bf16Comparison {
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

Bf16Comparison compare_bf16(const std::vector<unsigned char>& actual,
                            const std::vector<unsigned char>& expected) {
  if (actual.size() != expected.size() || actual.size() % kBf16Bytes != 0) {
    throw std::invalid_argument("full-language comparison sizes differ");
  }
  Bf16Comparison result;
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

json bf16_comparison_json(const Bf16Comparison& value) {
  const bool first_mismatch_provided =
      value.first_mismatch_index != std::numeric_limits<std::size_t>::max();
  json result = {
      {"elements", value.elements},
      {"exact_elements", value.exact_elements},
      {"finite_elements", value.finite_elements},
      {"first_mismatch_provided", first_mismatch_provided},
      {"first_expected_bits", value.first_expected_bits},
      {"first_actual_bits", value.first_actual_bits},
      {"maximum_absolute_error", value.maximum_absolute_error},
      {"relative_l2_error", value.relative_l2_error},
      {"cosine_similarity", value.cosine_similarity},
      {"expected_sha256", value.expected_sha256},
      {"actual_sha256", value.actual_sha256},
      {"passed", value.passed()},
  };
  result["first_mismatch_index"] =
      first_mismatch_provided ? json(value.first_mismatch_index) : json(nullptr);
  return result;
}

struct LogitsRowComparison {
  std::size_t selected_index = 0;
  std::size_t prompt_row = 0;
  std::uint32_t target_token_id = 0;
  std::size_t reference_top1_token_id = 0;
  std::size_t actual_top1_token_id = 0;
  double kl_divergence = 0.0;
  Bf16Comparison tensor;

  bool passed() const {
    return tensor.finite_elements == tensor.elements &&
           reference_top1_token_id == actual_top1_token_id &&
           kl_divergence < 0.005;
  }
};

LogitsRowComparison compare_logits_row(
    const std::vector<unsigned char>& actual,
    const std::vector<unsigned char>& expected, std::size_t selected_index,
    std::size_t prompt_row, std::uint32_t target_token_id) {
  if (actual.size() != kVocabulary * kBf16Bytes ||
      expected.size() != actual.size()) {
    throw std::invalid_argument("full-vocabulary row geometry differs");
  }
  LogitsRowComparison result;
  result.selected_index = selected_index;
  result.prompt_row = prompt_row;
  result.target_token_id = target_token_id;
  result.tensor = compare_bf16(actual, expected);

  double reference_max = -std::numeric_limits<double>::infinity();
  double actual_max = -std::numeric_limits<double>::infinity();
  std::vector<double> reference_values(kVocabulary);
  std::vector<double> actual_values(kVocabulary);
  for (std::size_t index = 0; index < kVocabulary; ++index) {
    std::uint16_t reference_bits = 0;
    std::uint16_t actual_bits = 0;
    std::memcpy(&reference_bits,
                expected.data() + index * sizeof(reference_bits),
                sizeof(reference_bits));
    std::memcpy(&actual_bits, actual.data() + index * sizeof(actual_bits),
                sizeof(actual_bits));
    reference_values[index] = bf16_to_float(reference_bits);
    actual_values[index] = bf16_to_float(actual_bits);
    if (reference_values[index] > reference_max) {
      reference_max = reference_values[index];
      result.reference_top1_token_id = index;
    }
    if (actual_values[index] > actual_max) {
      actual_max = actual_values[index];
      result.actual_top1_token_id = index;
    }
  }
  double reference_sum = 0.0;
  double actual_sum = 0.0;
  for (std::size_t index = 0; index < kVocabulary; ++index) {
    reference_sum += std::exp(reference_values[index] - reference_max);
    actual_sum += std::exp(actual_values[index] - actual_max);
  }
  const double reference_logsum = reference_max + std::log(reference_sum);
  const double actual_logsum = actual_max + std::log(actual_sum);
  for (std::size_t index = 0; index < kVocabulary; ++index) {
    const double log_reference = reference_values[index] - reference_logsum;
    const double probability = std::exp(log_reference);
    result.kl_divergence +=
        probability *
        (log_reference - (actual_values[index] - actual_logsum));
  }
  if (result.kl_divergence < 0.0 && result.kl_divergence > -1.0e-12) {
    result.kl_divergence = 0.0;
  }
  return result;
}

struct Execution {
  struct RouterExpertSetComparison {
    std::string label;
    std::size_t rows = 0;
    std::size_t exact_rows = 0;
  };

  std::vector<unsigned char> output;
  std::vector<aima::NativeOracleComparison> comparisons;
  std::vector<RouterExpertSetComparison> router_expert_sets;
  std::size_t aot_launches = 0;
  std::size_t dense_gemm_launches = 0;
  std::size_t native_pointwise_launches = 0;
  std::size_t native_ck_fmha_launches = 0;
  std::size_t native_vl_unified_attention_launches = 0;
  float measured_ms = 0.0f;
};

Execution execute_layers_0_through_3(
    const std::vector<unsigned char>& injected,
    const std::vector<unsigned char>& positions, std::size_t prompt_tokens,
    const std::filesystem::path& prefix_oracle_dir,
    const std::filesystem::path& layer3_oracle_dir,
    const aima::NativeWeightStore& weights,
    const aima::NativeDecodeBindings& bindings,
    const aima::NativePrefillWorkspace& workspace,
    aima::NativePrefillInvocations& invocations,
    aima::NativeQ8192PrefillGemmPlans& gemm_plans,
    aima::NativeVlLogicalProjectionState& logical_projections,
    aima::NativeDecodeExecutor& executor,
    aima::NativeQ8192CkProvider& provider,
    aima::NativeVlUnifiedAttentionPlan& vl_unified_attention,
    void* positions_device) {
  const std::size_t logical_hidden_bytes =
      prompt_tokens * kLanguageHidden * sizeof(std::uint16_t);
  if (prompt_tokens == 0 || prompt_tokens > kBucketTokens ||
      injected.size() != logical_hidden_bytes ||
      positions.size() !=
          3 * prompt_tokens * sizeof(std::int64_t) ||
      positions_device == nullptr) {
    throw std::invalid_argument("layer 0-3 input geometry is invalid");
  }

  void* layer_input =
      aima::native_prefill_layer_input_pointer(workspace, invocations, 0);
  check_hip(hipMemset(layer_input, 0,
                      kBucketTokens * kLanguageHidden *
                          sizeof(std::uint16_t)),
            "hipMemset layer 0-3 padded input");
  check_hip(hipMemcpy(layer_input, injected.data(), injected.size(),
                      hipMemcpyHostToDevice),
            "hipMemcpy layer 0-3 injected embeddings");
  check_hip(hipMemset(positions_device, 0,
                      3 * kBucketTokens * sizeof(std::int64_t)),
            "hipMemset layer 0-3 M-RoPE positions");
  for (std::size_t axis = 0; axis < 3; ++axis) {
    check_hip(
        hipMemcpy(
            static_cast<std::int64_t*>(positions_device) +
                axis * kBucketTokens,
            positions.data() + axis * prompt_tokens * sizeof(std::int64_t),
            prompt_tokens * sizeof(std::int64_t), hipMemcpyHostToDevice),
        "hipMemcpy layer 0-3 M-RoPE row");
  }

  Execution result;
  Event start;
  Event stop;
  check_hip(hipEventRecord(start), "hipEventRecord layer 0-3 start");
  for (std::size_t layer_index = 0; layer_index < 3; ++layer_index) {
    aima::NativeLinearPrefillOracleOptions attention_options;
    attention_options.layer_index = layer_index;
    attention_options.comparison_tokens = prompt_tokens;
    attention_options.exact_b_projection_tokens =
        layer_index == 0 && prompt_tokens <= 64 ? prompt_tokens : 0;
    attention_options.seed_layer_input = false;
    attention_options.run_output_projection_diagnostic = false;
    attention_options.collect_oracle_comparisons = false;
    attention_options.gemm_plans = &gemm_plans;
    attention_options.logical_ab_gemm_plan = &logical_projections.ab_plan();
    attention_options.logical_ab_weight =
        logical_projections.ab_weight(layer_index);
    attention_options.logical_ab_output = logical_projections.ab_output();
    attention_options.logical_output_gemm_plan =
        &logical_projections.linear_output_plan();
    attention_options.bindings = &bindings;
    if (!prefix_oracle_dir.empty()) {
      attention_options.sequence_oracle_dir = prefix_oracle_dir;
      attention_options.sequence_oracle_label_prefix =
          "layer-00" + std::to_string(layer_index) + "-";
    }
    const aima::NativeLinearPrefillOracleResult attention =
        aima::probe_native_q8192_linear_prefill_layer0_oracle(
            {}, weights, workspace, invocations, executor,
            attention_options);

    aima::NativeMoePrefillOracleOptions moe_options;
    moe_options.layer_index = layer_index;
    moe_options.comparison_tokens = prompt_tokens;
    moe_options.seed_post_attention = false;
    moe_options.run_routing_diagnostic = false;
    moe_options.collect_oracle_comparisons = false;
    moe_options.gemm_plans = &gemm_plans;
    moe_options.logical_router_gemm_plans =
        &logical_projections.router_gemm_plans();
    if (!prefix_oracle_dir.empty()) {
      moe_options.chain_output_oracle_dir = prefix_oracle_dir;
      moe_options.chain_output_oracle_label =
          "layer-00" + std::to_string(layer_index) +
          "-return-layer_body-output";
    }
    const aima::NativeMoePrefillOracleResult moe =
        aima::probe_native_q8192_moe_prefill_layer0_oracle(
            {}, weights, workspace, invocations, executor, moe_options);
    result.aot_launches +=
        attention.layer.aot_launches + moe.layer.aot_launches;
    result.dense_gemm_launches +=
        attention.layer.dense_gemm_launches + moe.layer.dense_gemm_launches;
    result.native_pointwise_launches +=
        attention.layer.native_pointwise_launches +
        moe.layer.native_pointwise_launches;
    result.comparisons.insert(
        result.comparisons.end(), attention.boundary_comparisons.begin(),
        attention.boundary_comparisons.end());
    result.comparisons.insert(result.comparisons.end(),
                              moe.comparisons.begin(),
                              moe.comparisons.end());
    if (moe.router_expert_set_rows != 0) {
      result.router_expert_sets.push_back(
          {"layer-00" + std::to_string(layer_index) +
               "-return-layer_body-router_indices",
           moe.router_expert_set_rows,
           moe.router_expert_set_rows_exact});
    }
    if (moe.chain_output_comparison_provided) {
      aima::NativeOracleComparison output = moe.chain_output_comparison;
      output.label = "layer-00" + std::to_string(layer_index) +
                     "-same_request_layer_output";
      result.comparisons.push_back(std::move(output));
    }
  }

  aima::NativeFullPrefillOracleOptions full_options;
  full_options.layer_index = 3;
  full_options.active_tokens = prompt_tokens;
  full_options.comparison_tokens = prompt_tokens;
  full_options.seed_layer_input = false;
  full_options.prepare_rotary_table = true;
  full_options.collect_oracle_comparisons = false;
  full_options.gemm_plans = &gemm_plans;
  full_options.bindings = &bindings;
  full_options.vl_unified_attention = &vl_unified_attention;
  full_options.mrope_positions_i64 = positions_device;
  full_options.mrope_position_row_stride = kBucketTokens;
  if (!layer3_oracle_dir.empty()) {
    full_options.sequence_oracle_dir = layer3_oracle_dir;
    full_options.sequence_oracle_label_prefix = "layer-003-";
    full_options.attention_core_oracle_dir = prefix_oracle_dir;
    full_options.attention_core_oracle_label_prefix = "layer-003-";
  }
  const aima::NativeFullPrefillOracleResult full =
      aima::probe_native_q8192_full_prefill_oracle(
          {}, weights, workspace, invocations, executor, provider,
          full_options);

  aima::NativeMoePrefillOracleOptions moe_options;
  moe_options.layer_index = 3;
  moe_options.comparison_tokens = prompt_tokens;
  moe_options.seed_post_attention = false;
  moe_options.run_routing_diagnostic = false;
  moe_options.collect_oracle_comparisons = false;
  moe_options.gemm_plans = &gemm_plans;
  moe_options.logical_router_gemm_plans =
      &logical_projections.router_gemm_plans();
  if (!prefix_oracle_dir.empty()) {
    moe_options.chain_output_oracle_dir = prefix_oracle_dir;
    moe_options.chain_output_oracle_label =
        "layer-003-return-layer_body-output";
  }
  const aima::NativeMoePrefillOracleResult moe =
      aima::probe_native_q8192_moe_prefill_layer0_oracle(
          {}, weights, workspace, invocations, executor, moe_options);
  check_hip(hipEventRecord(stop), "hipEventRecord layer 0-3 stop");
  check_hip(hipEventSynchronize(stop),
            "hipEventSynchronize layer 0-3 stop");
  check_hip(hipEventElapsedTime(&result.measured_ms, start, stop),
            "hipEventElapsedTime layer 0-3");

  result.aot_launches += full.layer.aot_launches + moe.layer.aot_launches;
  result.dense_gemm_launches +=
      full.layer.dense_gemm_launches + moe.layer.dense_gemm_launches;
  result.native_pointwise_launches +=
      full.layer.native_pointwise_launches +
      moe.layer.native_pointwise_launches;
  result.native_ck_fmha_launches += full.layer.native_ck_fmha_launches;
  result.native_vl_unified_attention_launches +=
      full.layer.native_vl_unified_attention_launches;
  result.comparisons.insert(result.comparisons.end(),
                            full.boundary_comparisons.begin(),
                            full.boundary_comparisons.end());
  result.comparisons.insert(result.comparisons.end(),
                            moe.comparisons.begin(), moe.comparisons.end());
  if (moe.router_expert_set_rows != 0) {
    result.router_expert_sets.push_back(
        {"layer-003-return-layer_body-router_indices",
         moe.router_expert_set_rows,
         moe.router_expert_set_rows_exact});
  }
  if (moe.chain_output_comparison_provided) {
    aima::NativeOracleComparison output = moe.chain_output_comparison;
    output.label = "layer-003-same_request_layer_output";
    result.comparisons.push_back(std::move(output));
  }

  result.output.resize(logical_hidden_bytes);
  check_hip(
      hipMemcpy(result.output.data(),
                aima::native_prefill_layer_output_pointer(
                    workspace, invocations, 3),
                result.output.size(), hipMemcpyDeviceToHost),
      "hipMemcpy layer 0-3 output");
  return result;
}

struct FullLanguageExecution {
  std::vector<unsigned char> final_norm;
  std::vector<unsigned char> selected_logits;
  std::vector<unsigned char> layer_outputs;
  std::vector<aima::NativeOracleComparison> comparisons;
  std::vector<Execution::RouterExpertSetComparison> router_expert_sets;
  std::size_t aot_launches = 0;
  std::size_t dense_gemm_launches = 0;
  std::size_t native_pointwise_launches = 0;
  std::size_t native_ck_fmha_launches = 0;
  std::size_t native_vl_unified_attention_launches = 0;
  float measured_ms = 0.0f;
};

FullLanguageExecution execute_full_language(
    const std::vector<unsigned char>& injected,
    const std::vector<unsigned char>& positions, std::size_t prompt_tokens,
    const std::vector<std::size_t>& selected_rows,
    const aima::NativeWeightStore& weights,
    const aima::NativeDecodeBindings& bindings,
    const aima::NativePrefillWorkspace& workspace,
    aima::NativePrefillInvocations& invocations,
    aima::NativeQ8192PrefillGemmPlans& gemm_plans,
    aima::NativeVlLogicalProjectionState& logical_projections,
    aima::NativeDecodeExecutor& executor,
    aima::NativeQ8192CkProvider& provider,
    aima::NativeVlUnifiedAttentionPlan& vl_unified_attention,
    aima::Bf16GemmPlan& logits_plan, void* positions_device,
    void* final_norm_device, void* logits_device,
    bool capture_layer_outputs,
    const std::filesystem::path& diagnostic_oracle_dir = {}) {
  const std::size_t logical_hidden_bytes =
      prompt_tokens * kLanguageHidden * kBf16Bytes;
  if (prompt_tokens <= 1 || prompt_tokens > kBucketTokens ||
      injected.size() != logical_hidden_bytes ||
      positions.size() != 3 * prompt_tokens * sizeof(std::int64_t) ||
      selected_rows.empty() || positions_device == nullptr ||
      final_norm_device == nullptr || logits_device == nullptr ||
      logits_plan.m() != prompt_tokens - 1 ||
      logits_plan.n() != kVocabulary || logits_plan.k() != kLanguageHidden) {
    throw std::invalid_argument("full-language input geometry is invalid");
  }
  if (std::any_of(selected_rows.begin(), selected_rows.end(),
                  [prompt_tokens](std::size_t row) {
                    return row >= prompt_tokens - 1;
                  })) {
    throw std::invalid_argument("selected full-vocabulary row is invalid");
  }
  const aima::NativeTensorView* final_norm_weight =
      weights.find("model.language_model.norm.weight");
  const aima::NativeTensorView* lm_head_weight = weights.find("lm_head.weight");
  if (final_norm_weight == nullptr ||
      final_norm_weight->device_pointer == nullptr ||
      final_norm_weight->payload_bytes != kLanguageHidden * kBf16Bytes ||
      lm_head_weight == nullptr || lm_head_weight->device_pointer == nullptr ||
      lm_head_weight->payload_bytes !=
          kVocabulary * kLanguageHidden * kBf16Bytes) {
    throw std::runtime_error("full-language terminal weights differ");
  }

  void* layer_input =
      aima::native_prefill_layer_input_pointer(workspace, invocations, 0);
  check_hip(hipMemset(layer_input, 0,
                      kBucketTokens * kLanguageHidden * kBf16Bytes),
            "hipMemset full-language padded input");
  check_hip(hipMemcpy(layer_input, injected.data(), injected.size(),
                      hipMemcpyHostToDevice),
            "hipMemcpy full-language injected embeddings");
  check_hip(hipMemset(positions_device, 0,
                      3 * kBucketTokens * sizeof(std::int64_t)),
            "hipMemset full-language M-RoPE positions");
  for (std::size_t axis = 0; axis < 3; ++axis) {
    check_hip(
        hipMemcpy(static_cast<std::int64_t*>(positions_device) +
                      axis * kBucketTokens,
                  positions.data() +
                      axis * prompt_tokens * sizeof(std::int64_t),
                  prompt_tokens * sizeof(std::int64_t),
                  hipMemcpyHostToDevice),
        "hipMemcpy full-language M-RoPE row");
  }

  FullLanguageExecution result;
  if (capture_layer_outputs) {
    result.layer_outputs.resize(40 * logical_hidden_bytes);
  }
  Event start;
  Event stop;
  check_hip(hipEventRecord(start), "hipEventRecord full-language start");
  for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
    if (layer_index % 4 == 3) {
      aima::NativeFullPrefillOracleOptions attention_options;
      attention_options.layer_index = layer_index;
      attention_options.active_tokens = prompt_tokens;
      attention_options.comparison_tokens = prompt_tokens;
      attention_options.seed_layer_input = false;
      attention_options.prepare_rotary_table = true;
      attention_options.collect_oracle_comparisons = false;
      attention_options.gemm_plans = &gemm_plans;
      attention_options.bindings = &bindings;
      attention_options.vl_unified_attention = &vl_unified_attention;
      attention_options.mrope_positions_i64 = positions_device;
      attention_options.mrope_position_row_stride = kBucketTokens;
      const aima::NativeFullPrefillOracleResult attention =
          aima::probe_native_q8192_full_prefill_oracle(
              {}, weights, workspace, invocations, executor, provider,
              attention_options);
      result.aot_launches += attention.layer.aot_launches;
      result.dense_gemm_launches += attention.layer.dense_gemm_launches;
      result.native_pointwise_launches +=
          attention.layer.native_pointwise_launches;
      result.native_ck_fmha_launches +=
          attention.layer.native_ck_fmha_launches;
      result.native_vl_unified_attention_launches +=
          attention.layer.native_vl_unified_attention_launches;
    } else {
      aima::NativeLinearPrefillOracleOptions attention_options;
      attention_options.layer_index = layer_index;
      attention_options.comparison_tokens = prompt_tokens;
      attention_options.exact_b_projection_tokens =
          layer_index == 0 && prompt_tokens <= 64 ? prompt_tokens : 0;
      attention_options.seed_layer_input = false;
      attention_options.run_output_projection_diagnostic = false;
      attention_options.collect_oracle_comparisons = false;
      attention_options.gemm_plans = &gemm_plans;
      attention_options.logical_ab_gemm_plan =
          &logical_projections.ab_plan();
      attention_options.logical_ab_weight =
          logical_projections.ab_weight(layer_index);
      attention_options.logical_ab_output = logical_projections.ab_output();
      attention_options.logical_output_gemm_plan =
          &logical_projections.linear_output_plan();
      attention_options.bindings = &bindings;
      const aima::NativeLinearPrefillOracleResult attention =
          aima::probe_native_q8192_linear_prefill_layer0_oracle(
              {}, weights, workspace, invocations, executor,
              attention_options);
      result.aot_launches += attention.layer.aot_launches;
      result.dense_gemm_launches += attention.layer.dense_gemm_launches;
      result.native_pointwise_launches +=
          attention.layer.native_pointwise_launches;
    }

    aima::NativeMoePrefillOracleOptions moe_options;
    moe_options.layer_index = layer_index;
    moe_options.comparison_tokens = prompt_tokens;
    moe_options.seed_post_attention = false;
    moe_options.run_routing_diagnostic = false;
    moe_options.collect_oracle_comparisons = false;
    moe_options.gemm_plans = &gemm_plans;
    moe_options.logical_router_gemm_plans =
        &logical_projections.router_gemm_plans();
    if (!diagnostic_oracle_dir.empty()) {
      moe_options.chain_output_oracle_dir = diagnostic_oracle_dir;
      moe_options.chain_output_oracle_label =
          layer_output_oracle_label(layer_index);
    }
    const aima::NativeMoePrefillOracleResult moe =
        aima::probe_native_q8192_moe_prefill_layer0_oracle(
            {}, weights, workspace, invocations, executor, moe_options);
    result.aot_launches += moe.layer.aot_launches;
    result.dense_gemm_launches += moe.layer.dense_gemm_launches;
    result.native_pointwise_launches += moe.layer.native_pointwise_launches;
    result.comparisons.insert(result.comparisons.end(),
                              moe.comparisons.begin(),
                              moe.comparisons.end());
    if (moe.router_expert_set_rows != 0) {
      result.router_expert_sets.push_back(
          {layer_output_oracle_label(layer_index) + ":router_indices",
           moe.router_expert_set_rows,
           moe.router_expert_set_rows_exact});
    }
    if (moe.chain_output_comparison_provided) {
      aima::NativeOracleComparison output = moe.chain_output_comparison;
      output.label = layer_output_oracle_label(layer_index);
      result.comparisons.push_back(std::move(output));
    }
    if (capture_layer_outputs) {
      check_hip(
          hipMemcpy(
              result.layer_outputs.data() +
                  layer_index * logical_hidden_bytes,
              aima::native_prefill_layer_output_pointer(
                  workspace, invocations, layer_index),
              logical_hidden_bytes, hipMemcpyDeviceToHost),
          "hipMemcpy full-language layer output");
    }
  }

  const void* terminal_hidden = aima::native_prefill_layer_output_pointer(
      workspace, invocations, 39);
  aima::launch_prefill_rmsnorm_2048(
      terminal_hidden, final_norm_weight->device_pointer, final_norm_device,
      kBucketTokens);
  ++result.native_pointwise_launches;
  logits_plan.launch(final_norm_device, lm_head_weight->device_pointer,
                     logits_device);
  ++result.dense_gemm_launches;
  check_hip(hipEventRecord(stop), "hipEventRecord full-language stop");
  check_hip(hipEventSynchronize(stop),
            "hipEventSynchronize full-language stop");
  check_hip(hipEventElapsedTime(&result.measured_ms, start, stop),
            "hipEventElapsedTime full-language");

  result.final_norm.resize(logical_hidden_bytes);
  check_hip(hipMemcpy(result.final_norm.data(), final_norm_device,
                      result.final_norm.size(), hipMemcpyDeviceToHost),
            "hipMemcpy full-language final norm");
  const std::size_t logits_row_bytes = kVocabulary * kBf16Bytes;
  result.selected_logits.resize(selected_rows.size() * logits_row_bytes);
  for (std::size_t index = 0; index < selected_rows.size(); ++index) {
    check_hip(
        hipMemcpy(result.selected_logits.data() + index * logits_row_bytes,
                  static_cast<const unsigned char*>(logits_device) +
                      selected_rows[index] * logits_row_bytes,
                  logits_row_bytes, hipMemcpyDeviceToHost),
        "hipMemcpy full-language selected logits");
  }
  return result;
}

const json& find_case(const json& manifest, const std::string& case_id) {
  for (const json& value : manifest.at("cases")) {
    if (value.value("case_id", "") == case_id) return value;
  }
  throw std::runtime_error("oracle case is missing: " + case_id);
}

json qualify_case(
    const json& vl_case, const std::filesystem::path& vl_root,
    const json& prefix_case, const std::filesystem::path& prefix_root,
    const json& layer3_case, const std::filesystem::path& layer3_root,
    const std::filesystem::path& output_dir,
    const aima::NativeWeightStore& weights,
    const aima::NativeDecodeBindings& bindings,
    const aima::NativePrefillWorkspace& workspace,
    aima::NativePrefillInvocations& invocations,
    aima::NativeQ8192PrefillGemmPlans& gemm_plans,
    aima::NativeVlLogicalProjectionState& logical_projections,
    aima::NativeDecodeExecutor& executor,
    aima::NativeQ8192CkProvider& provider,
    aima::NativeVlUnifiedAttentionPlan& vl_unified_attention,
    void* positions_device) {
  const std::string case_id = vl_case.at("case_id").get<std::string>();
  if (prefix_case.value("case_id", "") != case_id ||
      layer3_case.value("case_id", "") != case_id ||
      prefix_case.at("prompt_token_ids_sha256") !=
          vl_case.at("processor").at("prompt_token_ids_sha256") ||
      layer3_case.at("prompt_token_ids_sha256") !=
          vl_case.at("processor").at("prompt_token_ids_sha256") ||
      !layer3_case.at("frozen_mrope_positions_comparison")
           .at("exact")
           .get<bool>()) {
    throw std::runtime_error("layer 0-3 oracle case identities differ");
  }
  const json& injected_record =
      vl_case.at("boundaries").at("injected_embeddings");
  const json& positions_record =
      layer3_case.at("components").at("positions");
  const std::size_t prompt_tokens =
      layer3_case.at("prompt_tokens").get<std::size_t>();
  if (prompt_tokens == 0 || prompt_tokens > kBucketTokens ||
      injected_record.at("shape") !=
          json::array({prompt_tokens, kLanguageHidden}) ||
      injected_record.value("dtype", "") != "torch.bfloat16" ||
      positions_record.at("shape") != json::array({3, prompt_tokens}) ||
      positions_record.value("dtype", "") != "torch.int64") {
    throw std::runtime_error("layer 0-3 oracle geometry differs");
  }
  const std::vector<unsigned char> injected =
      read_tensor(vl_root, injected_record);
  const std::vector<unsigned char> positions =
      read_tensor(layer3_root, positions_record);
  const aima::NativeVlLogicalProjectionPrepareMetrics logical_metrics =
      logical_projections.prepare(prompt_tokens);
  if (!logical_metrics.prepared || logical_metrics.plan_count != 3) {
    throw std::runtime_error("logical VL projection plans are incomplete");
  }
  const std::filesystem::path case_oracle_dir = layer3_root / case_id;
  const std::filesystem::path prefix_case_oracle_dir = prefix_root / case_id;
  if (aima::sha256_file(prefix_case_oracle_dir / "oracle.jsonl") !=
      prefix_case.at("oracle_jsonl_sha256").get<std::string>()) {
    throw std::runtime_error("language prefix oracle ledger SHA-256 differs");
  }
  if (aima::sha256_file(case_oracle_dir / "oracle.jsonl") !=
      layer3_case.at("oracle_jsonl_sha256").get<std::string>()) {
    throw std::runtime_error("layer 3 oracle ledger SHA-256 differs");
  }

  const auto started = std::chrono::steady_clock::now();
  const Execution warmup = execute_layers_0_through_3(
      injected, positions, prompt_tokens, prefix_case_oracle_dir,
      case_oracle_dir, weights, bindings, workspace, invocations,
      gemm_plans, logical_projections, executor, provider,
      vl_unified_attention,
      positions_device);
  std::set<std::string> expected_labels = {
      "layer-003-attention_input_full_sequence",
      "layer-003-normalized_rotary_q_full_sequence",
      "layer-003-normalized_rotary_k_full_sequence",
      "layer-003-raw_v_full_sequence",
      "layer-003-attention_pre_gate_full_sequence",
      "layer-003-projected_attention_full_sequence",
      "layer-003-post_attention_residual_full_sequence",
      "layer-003-post_attention_norm_full_sequence",
      "layer-003-return-layer_body-h2",
      "layer-003-return-layer_body-router_logits",
      "layer-003-return-layer_body-router_scores",
      "layer-003-return-layer_body-router_weights",
      "layer-003-return-layer_body-router_indices",
      "layer-003-return-layer_body-shared_out",
      "layer-003-return-layer_body-routed_moe",
      "layer-003-return-layer_body-moe_out",
      "layer-003-same_request_layer_output",
  };
  for (std::size_t layer_index = 0; layer_index < 3; ++layer_index) {
    const std::string prefix =
        "layer-00" + std::to_string(layer_index) + "-";
    expected_labels.insert(prefix + "input_norm_full_sequence");
    expected_labels.insert(prefix + "attention_output_full_sequence");
    expected_labels.insert(prefix + "post_attention_full_sequence");
    expected_labels.insert(prefix + "post_attention_norm_full_sequence");
    expected_labels.insert(prefix + "return-layer_body-moe_out");
    expected_labels.insert(prefix + "same_request_layer_output");
    for (const char* suffix : {
             "return-layer_body-h2",
             "return-layer_body-router_logits",
             "return-layer_body-router_scores",
             "return-layer_body-router_weights",
             "return-layer_body-router_indices",
             "return-layer_body-shared_out",
             "return-layer_body-routed_moe",
             "return-layer_body-moe_out",
         }) {
      expected_labels.insert(prefix + suffix);
    }
  }
  std::set<std::string> actual_labels;
  json comparisons = json::array();
  const auto find_router_set = [&warmup](const std::string& label) {
    for (const auto& comparison : warmup.router_expert_sets) {
      if (comparison.label == label) return &comparison;
    }
    return static_cast<const Execution::RouterExpertSetComparison*>(nullptr);
  };
  bool boundaries_passed = true;
  std::string first_failed_boundary;
  for (const aima::NativeOracleComparison& comparison : warmup.comparisons) {
    actual_labels.insert(comparison.label);
    const Execution::RouterExpertSetComparison* router_set =
        find_router_set(comparison.label);
    const bool passed =
        router_set == nullptr
            ? comparison_passed(comparison)
            : router_set->exact_rows == router_set->rows;
    boundaries_passed = boundaries_passed && passed;
    if (!passed && first_failed_boundary.empty()) {
      first_failed_boundary = comparison.label;
    }
    json record = comparison_json(comparison);
    record["passed"] = passed;
    if (router_set != nullptr) {
      record["comparison_semantics"] = "unordered_expert_set_per_token";
      record["expert_set_rows"] = router_set->rows;
      record["exact_expert_set_rows"] = router_set->exact_rows;
    }
    comparisons.push_back(std::move(record));
  }
  if (actual_labels != expected_labels) {
    throw std::runtime_error(
        "layer 0-3 comparison set is incomplete for " + case_id);
  }

  // Qualification-only attribution: rerun Layer 1 attention from the exact
  // preceding-layer output while preserving the same q1024 kernels and GEMM
  // plans. This separates intrinsic Layer 1 arithmetic drift from Layer 0
  // propagation before the MoE amplification check below.
  aima::NativeLinearPrefillOracleOptions attention_diagnostic_options;
  attention_diagnostic_options.layer_index = 1;
  attention_diagnostic_options.comparison_tokens = prompt_tokens;
  attention_diagnostic_options.seed_layer_input = true;
  attention_diagnostic_options.layer_input_oracle_label =
      "return-layer_body-output";
  attention_diagnostic_options.run_output_projection_diagnostic = false;
  attention_diagnostic_options.collect_oracle_comparisons = false;
  attention_diagnostic_options.gemm_plans = &gemm_plans;
  attention_diagnostic_options.logical_ab_gemm_plan =
      &logical_projections.ab_plan();
  attention_diagnostic_options.logical_ab_weight =
      logical_projections.ab_weight(1);
  attention_diagnostic_options.logical_ab_output =
      logical_projections.ab_output();
  attention_diagnostic_options.logical_output_gemm_plan =
      &logical_projections.linear_output_plan();
  attention_diagnostic_options.bindings = &bindings;
  attention_diagnostic_options.oracle_label_prefix = "layer-000-";
  attention_diagnostic_options.sequence_oracle_dir =
      prefix_case_oracle_dir;
  attention_diagnostic_options.sequence_oracle_label_prefix = "layer-001-";
  const aima::NativeLinearPrefillOracleResult layer1_exact_input_attention =
      aima::probe_native_q8192_linear_prefill_layer0_oracle(
          prefix_case_oracle_dir, weights, workspace, invocations, executor,
          attention_diagnostic_options);
  json layer1_attention_diagnostic_comparisons = json::array();
  bool layer1_attention_diagnostic_passed = true;
  std::string layer1_attention_diagnostic_first_failed;
  std::set<std::string> layer1_attention_diagnostic_labels;
  for (const aima::NativeOracleComparison& comparison :
       layer1_exact_input_attention.boundary_comparisons) {
    layer1_attention_diagnostic_labels.insert(comparison.label);
    const bool passed = comparison_passed(comparison);
    layer1_attention_diagnostic_passed =
        layer1_attention_diagnostic_passed && passed;
    if (!passed && layer1_attention_diagnostic_first_failed.empty()) {
      layer1_attention_diagnostic_first_failed = comparison.label;
    }
    layer1_attention_diagnostic_comparisons.push_back(
        comparison_json(comparison));
  }
  const std::set<std::string> expected_layer1_attention_labels = {
      "layer-001-input_norm_full_sequence",
      "layer-001-attention_output_full_sequence",
      "layer-001-post_attention_full_sequence",
      "layer-001-post_attention_norm_full_sequence",
  };
  if (layer1_attention_diagnostic_labels !=
      expected_layer1_attention_labels) {
    throw std::runtime_error(
        "layer 1 exact-input attention set is incomplete for " + case_id);
  }

  // Qualification-only attribution: rerun Layer 1 MoE with the exact vLLM
  // post-attention norm and residual.  This distinguishes intrinsic MoE
  // arithmetic drift from discontinuous router amplification of the small
  // upstream attention error.  The following measured product runs rebuild
  // Layer 0-3 from injected embeddings, so this seeded rerun cannot affect
  // operation counts, timing, or output determinism.
  aima::NativeMoePrefillOracleOptions diagnostic_options;
  diagnostic_options.layer_index = 1;
  diagnostic_options.comparison_tokens = prompt_tokens;
  diagnostic_options.seed_post_attention = true;
  diagnostic_options.post_attention_h2_oracle_label =
      "return-layer_body-h2";
  diagnostic_options.post_attention_residual_oracle_label =
      "return-layer_body-after_attn";
  diagnostic_options.run_routing_diagnostic = false;
  diagnostic_options.collect_oracle_comparisons = false;
  diagnostic_options.gemm_plans = &gemm_plans;
  diagnostic_options.logical_router_gemm_plans =
      &logical_projections.router_gemm_plans();
  diagnostic_options.oracle_label_prefix = "layer-001-";
  diagnostic_options.chain_output_oracle_dir = prefix_case_oracle_dir;
  diagnostic_options.chain_output_oracle_label =
      "layer-001-return-layer_body-output";
  const aima::NativeMoePrefillOracleResult layer1_exact_input =
      aima::probe_native_q8192_moe_prefill_layer0_oracle(
          prefix_case_oracle_dir, weights, workspace, invocations, executor,
          diagnostic_options);
  json layer1_diagnostic_comparisons = json::array();
  bool layer1_diagnostic_passed = true;
  std::string layer1_diagnostic_first_failed;
  std::set<std::string> layer1_diagnostic_labels;
  for (const aima::NativeOracleComparison& comparison :
       layer1_exact_input.comparisons) {
    layer1_diagnostic_labels.insert(comparison.label);
    const bool router_indices =
        comparison.label == "layer-001-return-layer_body-router_indices";
    const bool passed =
        router_indices
            ? layer1_exact_input.router_expert_sets_exact
            : comparison_passed(comparison);
    layer1_diagnostic_passed = layer1_diagnostic_passed && passed;
    if (!passed && layer1_diagnostic_first_failed.empty()) {
      layer1_diagnostic_first_failed = comparison.label;
    }
    json record = comparison_json(comparison);
    record["passed"] = passed;
    if (router_indices) {
      record["comparison_semantics"] = "unordered_expert_set_per_token";
      record["expert_set_rows"] = layer1_exact_input.router_expert_set_rows;
      record["exact_expert_set_rows"] =
          layer1_exact_input.router_expert_set_rows_exact;
    }
    layer1_diagnostic_comparisons.push_back(std::move(record));
  }
  if (!layer1_exact_input.chain_output_comparison_provided) {
    throw std::runtime_error(
        "layer 1 exact-input output comparison is absent for " + case_id);
  }
  aima::NativeOracleComparison layer1_diagnostic_output =
      layer1_exact_input.chain_output_comparison;
  layer1_diagnostic_output.label =
      "layer-001-exact_input-same_request_layer_output";
  const bool layer1_output_passed =
      comparison_passed(layer1_diagnostic_output);
  layer1_diagnostic_passed =
      layer1_diagnostic_passed && layer1_output_passed;
  if (!layer1_output_passed && layer1_diagnostic_first_failed.empty()) {
    layer1_diagnostic_first_failed = layer1_diagnostic_output.label;
  }
  layer1_diagnostic_comparisons.push_back(
      comparison_json(layer1_diagnostic_output));
  const std::set<std::string> expected_layer1_diagnostic_labels = {
      "layer-001-return-layer_body-h2",
      "layer-001-return-layer_body-router_logits",
      "layer-001-return-layer_body-router_scores",
      "layer-001-return-layer_body-router_weights",
      "layer-001-return-layer_body-router_indices",
      "layer-001-return-layer_body-shared_out",
      "layer-001-return-layer_body-routed_moe",
      "layer-001-return-layer_body-moe_out",
  };
  if (layer1_diagnostic_labels != expected_layer1_diagnostic_labels) {
    throw std::runtime_error(
        "layer 1 exact-input diagnostic set is incomplete for " + case_id);
  }

  std::vector<Execution> measured;
  measured.reserve(kMeasuredRuns);
  for (std::size_t run = 0; run < kMeasuredRuns; ++run) {
    measured.push_back(execute_layers_0_through_3(
        injected, positions, prompt_tokens, {}, {}, weights, bindings,
        workspace, invocations, gemm_plans, logical_projections, executor,
        provider,
        vl_unified_attention,
        positions_device));
  }
  bool deterministic = true;
  std::vector<float> measured_ms;
  for (const Execution& execution : measured) {
    deterministic = deterministic && execution.output == warmup.output;
    measured_ms.push_back(execution.measured_ms);
  }
  std::sort(measured_ms.begin(), measured_ms.end());
  const std::filesystem::path output = output_dir / (case_id + ".bin");
  write_file(output, measured.front().output);
  const bool production_shape =
      warmup.aot_launches == 33 &&
      warmup.dense_gemm_launches == 31 &&
      warmup.native_ck_fmha_launches == 0 &&
      warmup.native_vl_unified_attention_launches == 1 &&
      (warmup.native_pointwise_launches == 40 ||
       warmup.native_pointwise_launches == 41);
  const bool complete = boundaries_passed && deterministic && production_shape;
  const double wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  return {
      {"schema",
       "aima-amd395-qwen36/native-vl-language-layer3-composed-case/v1"},
      {"complete", complete},
      {"case_id", case_id},
      {"prompt_tokens", prompt_tokens},
      {"bucket_tokens", kBucketTokens},
      {"padding_tokens", kBucketTokens - prompt_tokens},
      {"comparison_count", comparisons.size()},
      {"boundaries_passed", boundaries_passed},
      {"first_failed_boundary", first_failed_boundary},
      {"comparisons", std::move(comparisons)},
      {"layer1_exact_input_attention_diagnostic",
       {
           {"complete", layer1_attention_diagnostic_passed},
           {"comparison_count",
            layer1_attention_diagnostic_comparisons.size()},
           {"first_failed_boundary",
            layer1_attention_diagnostic_first_failed},
           {"comparisons",
            std::move(layer1_attention_diagnostic_comparisons)},
           {"seed_tensors", layer1_exact_input_attention.seed_tensors},
           {"seed_bytes", layer1_exact_input_attention.seed_bytes},
       }},
      {"layer1_exact_input_moe_diagnostic",
       {
           {"complete", layer1_diagnostic_passed},
           {"comparison_count", layer1_diagnostic_comparisons.size()},
           {"first_failed_boundary", layer1_diagnostic_first_failed},
           {"comparisons", std::move(layer1_diagnostic_comparisons)},
           {"seed_tensors", layer1_exact_input.seed_tensors},
           {"seed_bytes", layer1_exact_input.seed_bytes},
       }},
      {"repeat_deterministic", deterministic},
      {"output_sha256",
       aima::sha256_bytes(measured.front().output.data(),
                          measured.front().output.size())},
      {"warmup_output_sha256",
       aima::sha256_bytes(warmup.output.data(), warmup.output.size())},
      {"measured_runs", kMeasuredRuns},
      {"median_ms", measured_ms[measured_ms.size() / 2]},
      {"aot_launches", warmup.aot_launches},
      {"dense_gemm_launches", warmup.dense_gemm_launches},
      {"native_pointwise_launches", warmup.native_pointwise_launches},
      {"native_ck_fmha_launches", warmup.native_ck_fmha_launches},
      {"native_vl_unified_attention_launches",
       warmup.native_vl_unified_attention_launches},
      {"logical_projection_plan_count", logical_metrics.plan_count},
      {"logical_projection_workspace_bytes", logical_metrics.workspace_bytes},
      {"logical_projection_plan_build_wall_ms", logical_metrics.build_wall_ms},
      {"logical_projection_plan_reused", logical_metrics.reused},
      {"production_operation_shape", production_shape},
      {"case_wall_ms", wall_ms},
      {"injected_embeddings_sha256", injected_record.at("sha256")},
      {"mrope_positions_sha256", positions_record.at("sha256")},
  };
}

json qualify_full_language_case(
    const json& vl_case, const std::filesystem::path& vl_root,
    const std::filesystem::path& output_dir,
    const std::filesystem::path& diagnostic_oracle_root,
    const aima::NativeWeightStore& weights,
    const aima::NativeDecodeBindings& bindings,
    const aima::NativePrefillWorkspace& workspace,
    aima::NativePrefillInvocations& invocations,
    aima::NativeQ8192PrefillGemmPlans& gemm_plans,
    aima::NativeVlLogicalProjectionState& logical_projections,
    aima::NativeDecodeExecutor& executor,
    aima::NativeQ8192CkProvider& provider,
    aima::NativeVlUnifiedAttentionPlan& vl_unified_attention,
    void* positions_device) {
  const std::string case_id = vl_case.at("case_id").get<std::string>();
  const json& boundaries = vl_case.at("boundaries");
  const json& injected_record = boundaries.at("injected_embeddings");
  const json& positions_record = boundaries.at("mrope_positions");
  const json& final_norm_record = boundaries.at("language_final_norm");
  const json& logits_record = boundaries.at("full_vocabulary_logits");
  const std::size_t prompt_tokens =
      injected_record.at("shape").at(0).get<std::size_t>();
  const std::vector<std::size_t> selected_rows =
      logits_record.at("selected_rows").get<std::vector<std::size_t>>();
  const std::vector<std::uint32_t> target_token_ids =
      logits_record.at("teacher_forced_target_token_ids")
          .get<std::vector<std::uint32_t>>();
  if (prompt_tokens <= 1 || prompt_tokens > kBucketTokens ||
      injected_record.at("shape") !=
          json::array({prompt_tokens, kLanguageHidden}) ||
      injected_record.value("dtype", "") != "torch.bfloat16" ||
      positions_record.at("shape") != json::array({3, prompt_tokens}) ||
      positions_record.value("dtype", "") != "torch.int64" ||
      final_norm_record.at("shape") !=
          json::array({prompt_tokens, kLanguageHidden}) ||
      final_norm_record.value("dtype", "") != "torch.bfloat16" ||
      logits_record.at("shape") !=
          json::array({selected_rows.size(), kVocabulary}) ||
      logits_record.at("original_shape") !=
          json::array({prompt_tokens - 1, kVocabulary}) ||
      logits_record.value("dtype", "") != "torch.bfloat16" ||
      selected_rows.empty() || selected_rows.size() != target_token_ids.size()) {
    throw std::runtime_error("full-language oracle geometry differs");
  }

  const std::vector<unsigned char> injected =
      read_tensor(vl_root, injected_record);
  const std::vector<unsigned char> positions =
      read_tensor(vl_root, positions_record);
  const std::vector<unsigned char> expected_final_norm =
      read_tensor(vl_root, final_norm_record);
  const std::vector<unsigned char> expected_logits =
      read_tensor(vl_root, logits_record);
  const aima::NativeVlLogicalProjectionPrepareMetrics logical_metrics =
      logical_projections.prepare(prompt_tokens);
  if (!logical_metrics.prepared || logical_metrics.plan_count != 3) {
    throw std::runtime_error("full-language logical plans are incomplete");
  }

  aima::Bf16GemmPlan logits_plan(prompt_tokens - 1, kVocabulary,
                                 kLanguageHidden, 128ULL * 1024ULL * 1024ULL,
                                 true);
  DeviceAllocation final_norm_device(
      kBucketTokens * kLanguageHidden * kBf16Bytes);
  DeviceAllocation logits_device(
      (prompt_tokens - 1) * kVocabulary * kBf16Bytes);
  const auto started = std::chrono::steady_clock::now();
  const std::filesystem::path diagnostic_oracle_dir =
      diagnostic_oracle_root.empty()
          ? std::filesystem::path{}
          : diagnostic_oracle_root / case_id;
  const FullLanguageExecution warmup = execute_full_language(
      injected, positions, prompt_tokens, selected_rows, weights, bindings,
      workspace, invocations, gemm_plans, logical_projections, executor,
      provider, vl_unified_attention, logits_plan, positions_device,
      final_norm_device.get(), logits_device.get(), true,
      diagnostic_oracle_dir);
  std::vector<FullLanguageExecution> measured;
  measured.reserve(kMeasuredRuns);
  for (std::size_t run = 0; run < kMeasuredRuns; ++run) {
    measured.push_back(execute_full_language(
        injected, positions, prompt_tokens, selected_rows, weights, bindings,
        workspace, invocations, gemm_plans, logical_projections, executor,
        provider, vl_unified_attention, logits_plan, positions_device,
        final_norm_device.get(), logits_device.get(), false));
  }
  bool deterministic = true;
  std::vector<float> measured_ms;
  measured_ms.reserve(measured.size());
  for (const FullLanguageExecution& execution : measured) {
    deterministic = deterministic &&
                    execution.final_norm == warmup.final_norm &&
                    execution.selected_logits == warmup.selected_logits;
    measured_ms.push_back(execution.measured_ms);
  }
  std::sort(measured_ms.begin(), measured_ms.end());

  json diagnostic_comparisons = json::array();
  for (const aima::NativeOracleComparison& comparison :
       warmup.comparisons) {
    diagnostic_comparisons.push_back(comparison_json(comparison));
  }
  json router_expert_sets = json::array();
  bool all_router_expert_sets_exact = true;
  for (const Execution::RouterExpertSetComparison& comparison :
       warmup.router_expert_sets) {
    const bool exact = comparison.rows == comparison.exact_rows;
    all_router_expert_sets_exact = all_router_expert_sets_exact && exact;
    router_expert_sets.push_back({
        {"label", comparison.label},
        {"rows", comparison.rows},
        {"exact_rows", comparison.exact_rows},
        {"exact", exact},
    });
  }

  const Bf16Comparison final_norm =
      compare_bf16(warmup.final_norm, expected_final_norm);
  const std::size_t logits_row_bytes = kVocabulary * kBf16Bytes;
  json logits_rows = json::array();
  bool logits_passed = true;
  double maximum_kl_divergence = 0.0;
  for (std::size_t index = 0; index < selected_rows.size(); ++index) {
    const auto actual_begin =
        warmup.selected_logits.begin() + index * logits_row_bytes;
    const auto expected_begin =
        expected_logits.begin() + index * logits_row_bytes;
    const std::vector<unsigned char> actual(
        actual_begin, actual_begin + logits_row_bytes);
    const std::vector<unsigned char> expected(
        expected_begin, expected_begin + logits_row_bytes);
    const LogitsRowComparison comparison = compare_logits_row(
        actual, expected, index, selected_rows[index], target_token_ids[index]);
    logits_passed = logits_passed && comparison.passed();
    maximum_kl_divergence =
        std::max(maximum_kl_divergence, comparison.kl_divergence);
    logits_rows.push_back({
        {"selected_index", comparison.selected_index},
        {"prompt_row", comparison.prompt_row},
        {"teacher_forced_target_token_id", comparison.target_token_id},
        {"reference_top1_token_id", comparison.reference_top1_token_id},
        {"actual_top1_token_id", comparison.actual_top1_token_id},
        {"top1_match",
         comparison.reference_top1_token_id ==
             comparison.actual_top1_token_id},
        {"kl_divergence", comparison.kl_divergence},
        {"kl_divergence_threshold", 0.005},
        {"tensor", bf16_comparison_json(comparison.tensor)},
        {"passed", comparison.passed()},
    });
  }
  const bool production_shape =
      warmup.native_ck_fmha_launches == 0 &&
      warmup.native_vl_unified_attention_launches == 10 &&
      warmup.aot_launches != 0 && warmup.dense_gemm_launches != 0 &&
      warmup.native_pointwise_launches != 0;
  const bool complete = final_norm.passed() && logits_passed && deterministic &&
                        production_shape;
  const std::filesystem::path final_norm_output =
      output_dir / (case_id + "-final-norm.bin");
  const std::filesystem::path logits_output =
      output_dir / (case_id + "-selected-logits.bin");
  const std::filesystem::path layer_outputs =
      output_dir / (case_id + "-layer-outputs.bin");
  write_file(final_norm_output, warmup.final_norm);
  write_file(logits_output, warmup.selected_logits);
  write_file(layer_outputs, warmup.layer_outputs);
  const double wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  return {
      {"schema",
       "aima-amd395-qwen36/native-vl-language-full-case/v1"},
      {"complete", complete},
      {"case_id", case_id},
      {"prompt_tokens", prompt_tokens},
      {"bucket_tokens", kBucketTokens},
      {"padding_tokens", kBucketTokens - prompt_tokens},
      {"language_layers", 40},
      {"layer_diagnostics",
       {
           {"provided", !diagnostic_oracle_dir.empty()},
           {"comparisons", std::move(diagnostic_comparisons)},
           {"router_expert_sets", std::move(router_expert_sets)},
           {"all_router_expert_sets_exact",
            all_router_expert_sets_exact},
       }},
      {"final_norm", bf16_comparison_json(final_norm)},
      {"full_vocabulary_logits",
       {
           {"selected_row_count", selected_rows.size()},
           {"original_shape",
            json::array({prompt_tokens - 1, kVocabulary})},
           {"all_rows_passed", logits_passed},
           {"maximum_kl_divergence", maximum_kl_divergence},
           {"kl_divergence_threshold", 0.005},
           {"rows", std::move(logits_rows)},
       }},
      {"repeat_deterministic", deterministic},
      {"measured_runs", kMeasuredRuns},
      {"median_ms", measured_ms[measured_ms.size() / 2]},
      {"aot_launches", warmup.aot_launches},
      {"dense_gemm_launches", warmup.dense_gemm_launches},
      {"native_pointwise_launches", warmup.native_pointwise_launches},
      {"native_ck_fmha_launches", warmup.native_ck_fmha_launches},
      {"native_vl_unified_attention_launches",
       warmup.native_vl_unified_attention_launches},
      {"logical_projection_plan_count", logical_metrics.plan_count},
      {"logical_projection_workspace_bytes", logical_metrics.workspace_bytes},
      {"logical_projection_plan_build_wall_ms",
       logical_metrics.build_wall_ms},
      {"logical_projection_plan_reused", logical_metrics.reused},
      {"logits_gemm_workspace_bytes", logits_plan.workspace_bytes()},
      {"production_operation_shape", production_shape},
      {"final_norm_output", final_norm_output.filename().string()},
      {"selected_logits_output", logits_output.filename().string()},
      {"layer_outputs",
       {
           {"path", layer_outputs.filename().string()},
           {"shape", json::array({40, prompt_tokens, kLanguageHidden})},
           {"dtype", "bfloat16"},
           {"bytes", warmup.layer_outputs.size()},
           {"sha256",
            aima::sha256_bytes(warmup.layer_outputs.data(),
                               warmup.layer_outputs.size())},
       }},
      {"final_norm_output_sha256", final_norm.actual_sha256},
      {"selected_logits_output_sha256",
       aima::sha256_bytes(warmup.selected_logits.data(),
                          warmup.selected_logits.size())},
      {"case_wall_ms", wall_ms},
      {"injected_embeddings_sha256", injected_record.at("sha256")},
      {"mrope_positions_sha256", positions_record.at("sha256")},
      {"reference_final_norm_sha256", final_norm_record.at("sha256")},
      {"reference_selected_logits_sha256", logits_record.at("sha256")},
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 12 && argc != 13 && argc != 14) {
    std::cerr
        << "usage: native-vl-language-layer3-composed-probe MODEL_DIR "
           "VL_ORACLE_MANIFEST VL_ORACLE_ROOT PREFIX_MANIFEST PREFIX_ROOT "
           "LAYER3_MANIFEST LAYER3_ROOT FMHA_PROVIDER CASE_ID_OR_ALL "
           "LOAD_REPORT OUTPUT_DIR [full-language [DIAGNOSTIC_ORACLE_ROOT]]\n";
    return 2;
  }
  try {
    const bool full_language = argc >= 13;
    if (full_language && std::string_view(argv[12]) != "full-language") {
      throw std::runtime_error("unknown optional probe mode");
    }
    const std::filesystem::path vl_manifest_path =
        std::filesystem::absolute(argv[2]);
    const std::filesystem::path vl_root =
        std::filesystem::absolute(argv[3]);
    const std::filesystem::path prefix_manifest_path =
        std::filesystem::absolute(argv[4]);
    const std::filesystem::path prefix_root =
        std::filesystem::absolute(argv[5]);
    const std::filesystem::path layer3_manifest_path =
        std::filesystem::absolute(argv[6]);
    const std::filesystem::path layer3_root =
        std::filesystem::absolute(argv[7]);
    const std::filesystem::path provider_path =
        std::filesystem::absolute(argv[8]);
    const std::string selector = argv[9];
    const std::filesystem::path output_dir =
        std::filesystem::absolute(argv[11]);
    const std::filesystem::path full_language_diagnostic_oracle_root =
        argc == 14 ? std::filesystem::absolute(argv[13])
                   : std::filesystem::path{};
    const json vl_manifest = read_json(vl_manifest_path);
    const json prefix_manifest = read_json(prefix_manifest_path);
    const json layer3_manifest = read_json(layer3_manifest_path);
    if (vl_manifest.value("schema", "") !=
            "aima-amd395-qwen36/vl-oracle-manifest/v1" ||
        !vl_manifest.value("complete", false) ||
        prefix_manifest.value("schema", "") !=
            "aima-amd395-qwen36/vl-language-prefix-diagnostic-oracle/v3" ||
        !prefix_manifest.value("complete", false) ||
        !prefix_manifest.value("qualified_for_attribution_only", false) ||
        layer3_manifest.value("schema", "") !=
            "aima-amd395-qwen36/vl-language-layer3-mrope-diagnostic-oracle/v1" ||
        !layer3_manifest.value("complete", false) ||
        !layer3_manifest.value("qualified_for_attribution_only", false) ||
        !vl_manifest.at("cases").is_array() ||
        !prefix_manifest.at("cases").is_array() ||
        !layer3_manifest.at("cases").is_array()) {
      throw std::runtime_error("layer 0-3 oracle manifests are incomplete");
    }
    std::vector<std::string> selected;
    for (const json& value : vl_manifest.at("cases")) {
      const std::string case_id = value.value("case_id", "");
      if (selector == "all" || selector == case_id) selected.push_back(case_id);
    }
    if (selected.empty() || (selector == "all" && selected.size() != 5)) {
      throw std::runtime_error("layer 0-3 case selection is incomplete");
    }
    std::filesystem::create_directories(output_dir);

    const auto started = std::chrono::steady_clock::now();
    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[10]);
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
    aima::NativePrefillWorkspace workspace;
    const aima::NativePrefillWorkspaceMetrics workspace_metrics =
        workspace.build(options.device, kBucketTokens);
    aima::NativePrefillInvocations invocations;
    const aima::NativePrefillInvocationMetrics invocation_metrics =
        invocations.build(bindings, workspace, kBucketTokens);
    aima::NativeQ8192PrefillGemmPlans gemm_plans(kBucketTokens);
    aima::NativeDecodeExecutor executor;
    const aima::NativeDecodeExecutorMetrics executor_metrics = executor.load();
    aima::NativeQ8192CkProvider provider;
    const aima::NativeQ8192CkProviderMetrics provider_metrics =
        provider.load(provider_path, kBucketTokens);
    aima::NativeVlUnifiedAttentionPlan vl_unified_attention(
        executor, kBucketTokens, kBucketTokens, options.device);
    DeviceAllocation positions_device(
        3 * kBucketTokens * sizeof(std::int64_t));

    json cases = json::array();
    bool complete = true;
    for (const std::string& case_id : selected) {
      json result =
          full_language
              ? qualify_full_language_case(
                    find_case(vl_manifest, case_id), vl_root, output_dir,
                    full_language_diagnostic_oracle_root,
                    weights, bindings, workspace, invocations, gemm_plans,
                    logical_projections, executor, provider,
                    vl_unified_attention, positions_device.get())
              : qualify_case(
                    find_case(vl_manifest, case_id), vl_root,
                    find_case(prefix_manifest, case_id), prefix_root,
                    find_case(layer3_manifest, case_id), layer3_root,
                    output_dir, weights, bindings, workspace, invocations,
                    gemm_plans, logical_projections, executor, provider,
                    vl_unified_attention, positions_device.get());
      complete = complete && result.at("complete").get<bool>();
      cases.push_back(std::move(result));
    }
    const double wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();
    const json result = {
        {"schema",
         full_language
             ? "aima-amd395-qwen36/native-vl-language-full-qualification-run/v1"
             : "aima-amd395-qwen36/native-vl-language-layer3-composed-qualification-run/v1"},
        {"complete", complete},
        {"source_commit", AIMA_SOURCE_COMMIT},
        {"case_selector", selector},
        {"case_count", cases.size()},
        {"single_resident_weight_load", true},
        {"schedule_context_tokens", kBucketTokens},
        {"warmup_runs_per_case", 1},
        {"measured_runs_per_case", kMeasuredRuns},
        {"vl_oracle_manifest_sha256", aima::sha256_file(vl_manifest_path)},
        {"layer3_oracle_manifest_sha256",
         aima::sha256_file(layer3_manifest_path)},
        {"prefix_oracle_manifest_sha256",
         aima::sha256_file(prefix_manifest_path)},
        {"fmha_provider_sha256", aima::sha256_file(provider_path)},
        {"vl_unified_attention_image_bytes",
         vl_unified_attention.metrics().image_bytes},
        {"vl_unified_attention_metadata_bytes",
         vl_unified_attention.metrics().metadata_bytes},
        {"vl_unified_attention_launches",
         vl_unified_attention.metrics().launches},
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
        {"prefill_workspace_bytes", workspace_metrics.allocation_bytes},
        {"prefill_prepared_launches", invocation_metrics.launch_count},
        {"active_gemm_plan_count", gemm_plans.built_plan_count()},
        {"active_gemm_workspace_bytes", gemm_plans.workspace_bytes()},
        {"aot_loaded_modules", executor_metrics.loaded_modules},
        {"fmha_provider_loaded", provider_metrics.loaded},
        {"runtime_python", false},
        {"runtime_numpy", false},
        {"runtime_torch", false},
        {"runtime_vllm", false},
        {"runtime_triton", false},
        {"timing_scope",
         full_language
             ? "native q1024 language layers 0 through 39, final norm and "
               "teacher-forced full-vocabulary logits; excludes processor, "
               "vision tower, serving, TTFT and G4"
             : "native q1024 language layers 0 through 3 only; excludes "
               "processor, vision tower, serving, TTFT and G4"},
        {"total_wall_ms", wall_ms},
        {"cases", std::move(cases)},
    };
    std::cout << result.dump() << '\n';
    return complete ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native VL language layer 0-3 probe: " << error.what()
              << '\n';
    return 1;
  }
}
