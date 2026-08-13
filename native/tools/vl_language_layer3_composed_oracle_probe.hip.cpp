// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_bindings.h"
#include "aima/native_decode_executor.h"
#include "aima/native_derived_weights.h"
#include "aima/native_full_prefill.h"
#include "aima/native_linear_prefill.h"
#include "aima/native_lm_head.h"
#include "aima/native_moe_prefill.h"
#include "aima/native_prefill_gemm_plans.h"
#include "aima/native_prefill_invocation.h"
#include "aima/native_prefill_workspace.h"
#include "aima/native_weight_store.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
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
    aima::NativeDecodeExecutor& executor,
    aima::NativeQ8192CkProvider& provider, void* positions_device) {
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
  full_options.comparison_tokens = prompt_tokens;
  full_options.seed_layer_input = false;
  full_options.prepare_rotary_table = true;
  full_options.collect_oracle_comparisons = false;
  full_options.gemm_plans = &gemm_plans;
  full_options.bindings = &bindings;
  full_options.mrope_positions_i64 = positions_device;
  full_options.mrope_position_row_stride = kBucketTokens;
  if (!layer3_oracle_dir.empty()) {
    full_options.sequence_oracle_dir = layer3_oracle_dir;
    full_options.sequence_oracle_label_prefix = "layer-003-";
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
  if (!layer3_oracle_dir.empty()) {
    moe_options.chain_output_oracle_dir = layer3_oracle_dir;
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
  result.comparisons.insert(result.comparisons.end(),
                            full.boundary_comparisons.begin(),
                            full.boundary_comparisons.end());
  result.comparisons.insert(result.comparisons.end(),
                            moe.comparisons.begin(), moe.comparisons.end());
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
    aima::NativeDecodeExecutor& executor,
    aima::NativeQ8192CkProvider& provider, void* positions_device) {
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
      gemm_plans, executor, provider,
      positions_device);
  std::set<std::string> expected_labels = {
      "layer-003-attention_input_full_sequence",
      "layer-003-normalized_rotary_q_full_sequence",
      "layer-003-normalized_rotary_k_full_sequence",
      "layer-003-raw_v_full_sequence",
      "layer-003-projected_attention_full_sequence",
      "layer-003-post_attention_residual_full_sequence",
      "layer-003-post_attention_norm_full_sequence",
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
        workspace, invocations, gemm_plans, executor, provider,
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
      warmup.aot_launches == 34 &&
      warmup.dense_gemm_launches == 28 &&
      warmup.native_ck_fmha_launches == 1 &&
      (warmup.native_pointwise_launches == 35 ||
       warmup.native_pointwise_launches == 36);
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
      {"production_operation_shape", production_shape},
      {"case_wall_ms", wall_ms},
      {"injected_embeddings_sha256", injected_record.at("sha256")},
      {"mrope_positions_sha256", positions_record.at("sha256")},
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 12) {
    std::cerr
        << "usage: native-vl-language-layer3-composed-probe MODEL_DIR "
           "VL_ORACLE_MANIFEST VL_ORACLE_ROOT PREFIX_MANIFEST PREFIX_ROOT "
           "LAYER3_MANIFEST LAYER3_ROOT FMHA_PROVIDER CASE_ID_OR_ALL "
           "LOAD_REPORT OUTPUT_DIR\n";
    return 2;
  }
  try {
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
    const json vl_manifest = read_json(vl_manifest_path);
    const json prefix_manifest = read_json(prefix_manifest_path);
    const json layer3_manifest = read_json(layer3_manifest_path);
    if (vl_manifest.value("schema", "") !=
            "aima-amd395-qwen36/vl-oracle-manifest/v1" ||
        !vl_manifest.value("complete", false) ||
        prefix_manifest.value("schema", "") !=
            "aima-amd395-qwen36/vl-language-prefix-diagnostic-oracle/v1" ||
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
    DeviceAllocation positions_device(
        3 * kBucketTokens * sizeof(std::int64_t));

    json cases = json::array();
    bool complete = true;
    for (const std::string& case_id : selected) {
      json result = qualify_case(
          find_case(vl_manifest, case_id), vl_root,
          find_case(prefix_manifest, case_id), prefix_root,
          find_case(layer3_manifest, case_id), layer3_root, output_dir,
          weights, bindings, workspace, invocations, gemm_plans, executor,
          provider, positions_device.get());
      complete = complete && result.at("complete").get<bool>();
      cases.push_back(std::move(result));
    }
    const double wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();
    const json result = {
        {"schema",
         "aima-amd395-qwen36/native-vl-language-layer3-composed-qualification-run/v1"},
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
        {"language_weight_payload_bytes", load.payload_bytes},
        {"language_weight_load_wall_ms", load.load_wall_ms},
        {"derived_weight_payload_bytes", derived_metrics.payload_bytes},
        {"derived_weight_build_wall_ms", derived_metrics.build_wall_ms},
        {"lm_head_payload_bytes", lm_head_metrics.payload_bytes},
        {"lm_head_build_wall_ms", lm_head_metrics.build_wall_ms},
        {"decode_weight_bindings", binding_metrics.unique_bindings},
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
         "native q1024 language layers 0 through 3 only; excludes processor, "
         "vision tower, serving, TTFT and G4"},
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
