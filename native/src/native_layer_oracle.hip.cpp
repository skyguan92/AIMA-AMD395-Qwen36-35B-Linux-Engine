// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_layer_oracle.h"

#include "aima/sha256.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aima {
namespace {

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

std::vector<unsigned char> read_binary(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open native layer oracle tensor: " +
                             path.string());
  }
  const auto end = stream.tellg();
  if (end < 0) {
    throw std::runtime_error("cannot size native layer oracle tensor: " +
                             path.string());
  }
  std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
  stream.seekg(0, std::ios::beg);
  if (!bytes.empty() &&
      !stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("cannot read native layer oracle tensor: " +
                             path.string());
  }
  return bytes;
}

std::string json_string_field(const std::string& line,
                              const std::string& field) {
  const std::string prefix = "\"" + field + "\":\"";
  const std::size_t start = line.find(prefix);
  if (start == std::string::npos) return {};
  const std::size_t value_start = start + prefix.size();
  const std::size_t end = line.find('"', value_start);
  return end == std::string::npos
             ? std::string{}
             : line.substr(value_start, end - value_start);
}

std::unordered_map<std::string, std::filesystem::path> oracle_files(
    const std::filesystem::path& oracle_dir, std::size_t minimum_count = 60) {
  const std::filesystem::path manifest = oracle_dir / "oracle.jsonl";
  std::ifstream stream(manifest);
  if (!stream) {
    throw std::runtime_error("cannot open native layer oracle manifest: " +
                             manifest.string());
  }
  std::unordered_map<std::string, std::filesystem::path> files;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.find("\"event\":\"native_layer_oracle_tensor\"") ==
        std::string::npos) {
      continue;
    }
    const std::string label = json_string_field(line, "label");
    const std::string file = json_string_field(line, "file");
    if (!label.empty() && !file.empty()) files.emplace(label, oracle_dir / file);
  }
  if (files.size() < minimum_count) {
    throw std::runtime_error("native layer oracle manifest is incomplete");
  }
  return files;
}

const NativeDecodeWorkspaceView& require_view(
    const NativeDecodeWorkspace& workspace, const std::string& name,
    std::uint64_t bytes, DecodeTensorDtype dtype) {
  const NativeDecodeWorkspaceView* view = workspace.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes < bytes || view->dtype != dtype) {
    throw std::runtime_error("native oracle workspace mismatch: " + name);
  }
  return *view;
}

const std::filesystem::path& require_file(
    const std::unordered_map<std::string, std::filesystem::path>& files,
    const std::string& label) {
  const auto found = files.find(label);
  if (found == files.end()) {
    throw std::runtime_error("native layer oracle label is missing: " + label);
  }
  return found->second;
}

const char* schedule_binding(const PreparedDecodeInvocation& invocation,
                             const char* argument_name) {
  if (invocation.launch == nullptr) {
    throw std::runtime_error("native layer oracle invocation is missing");
  }
  for (std::size_t index = 0; index < invocation.launch->argument_count; ++index) {
    const DecodeArgument& argument = invocation.launch->arguments[index];
    if (argument.kind == DecodeArgumentKind::kTensor &&
        std::string(argument.name) == argument_name) {
      return argument.binding;
    }
  }
  throw std::runtime_error("native layer oracle schedule argument is missing: " +
                           std::string(argument_name));
}

float bf16_to_float(std::uint16_t bits) {
  std::uint32_t value = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0f;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

NativeOracleComparison compare_tensor(
    const std::string& label, const std::string& dtype, const void* actual_device,
    std::size_t bytes, const std::filesystem::path& expected_path,
    bool compare_expected_range = false,
    std::size_t expected_offset_bytes = 0) {
  std::vector<unsigned char> expected = read_binary(expected_path);
  if ((!compare_expected_range &&
       (expected_offset_bytes != 0 || expected.size() != bytes)) ||
      (compare_expected_range &&
       (expected_offset_bytes > expected.size() ||
        bytes > expected.size() - expected_offset_bytes))) {
    throw std::runtime_error("native layer oracle byte count mismatch: " + label);
  }
  if (expected_offset_bytes != 0) {
    expected.erase(expected.begin(),
                   expected.begin() + expected_offset_bytes);
  }
  if (expected.size() != bytes) expected.resize(bytes);
  std::vector<unsigned char> actual(bytes);
  check_hip(hipMemcpy(actual.data(), actual_device, bytes,
                      hipMemcpyDeviceToHost),
            "hipMemcpy native layer oracle comparison");
  NativeOracleComparison result;
  result.label = label;
  result.dtype = dtype;
  result.expected_sha256 = sha256_bytes(expected.data(), expected.size());
  result.actual_sha256 = sha256_bytes(actual.data(), actual.size());

  std::size_t element_bytes = 0;
  if (dtype == "bfloat16") {
    element_bytes = 2;
  } else if (dtype == "float32" || dtype == "int32") {
    element_bytes = 4;
  } else if (dtype == "int64") {
    element_bytes = 8;
  } else {
    throw std::runtime_error("unsupported native layer oracle dtype: " + dtype);
  }
  result.elements = bytes / element_bytes;
  double squared_error = 0.0;
  double squared_expected = 0.0;
  double squared_actual = 0.0;
  double dot = 0.0;
  for (std::size_t index = 0; index < result.elements; ++index) {
    const unsigned char* expected_element =
        expected.data() + index * element_bytes;
    const unsigned char* actual_element = actual.data() + index * element_bytes;
    const bool exact =
        std::memcmp(expected_element, actual_element, element_bytes) == 0;
    if (exact) {
      ++result.exact_elements;
    }
    if (dtype == "int32" || dtype == "int64") continue;
    double expected_value = 0.0;
    double actual_value = 0.0;
    if (dtype == "bfloat16") {
      std::uint16_t expected_bits = 0;
      std::uint16_t actual_bits = 0;
      std::memcpy(&expected_bits, expected_element, sizeof(expected_bits));
      std::memcpy(&actual_bits, actual_element, sizeof(actual_bits));
      expected_value = bf16_to_float(expected_bits);
      actual_value = bf16_to_float(actual_bits);
    } else {
      float expected_float = 0.0f;
      float actual_float = 0.0f;
      std::memcpy(&expected_float, expected_element, sizeof(expected_float));
      std::memcpy(&actual_float, actual_element, sizeof(actual_float));
      expected_value = expected_float;
      actual_value = actual_float;
    }
    if (!exact && !result.first_mismatch_provided) {
      result.first_mismatch_provided = true;
      result.first_mismatch_index = index;
      result.first_mismatch_expected = expected_value;
      result.first_mismatch_actual = actual_value;
    }
    if (std::isfinite(actual_value)) ++result.finite_elements;
    const double error = actual_value - expected_value;
    result.maximum_absolute_error =
        std::max(result.maximum_absolute_error, std::abs(error));
    squared_error += error * error;
    squared_expected += expected_value * expected_value;
    squared_actual += actual_value * actual_value;
    dot += expected_value * actual_value;
  }
  if (dtype == "int32" || dtype == "int64") {
    result.finite_elements = result.elements;
    result.cosine_similarity =
        result.exact_elements == result.elements ? 1.0 : 0.0;
    result.relative_l2_error =
        result.exact_elements == result.elements
            ? 0.0
            : std::numeric_limits<double>::infinity();
  } else {
    result.relative_l2_error =
        std::sqrt(squared_error / std::max(squared_expected, 1.0e-30));
    result.cosine_similarity =
        dot / std::sqrt(std::max(squared_expected * squared_actual, 1.0e-30));
  }
  return result;
}

}  // namespace

std::size_t seed_native_oracle_tensor(
    const std::filesystem::path& expected_path, void* actual_device,
    std::size_t expected_bytes) {
  if (actual_device == nullptr) {
    throw std::invalid_argument("native oracle seed device pointer is null");
  }
  const std::vector<unsigned char> bytes = read_binary(expected_path);
  if (bytes.size() != expected_bytes) {
    throw std::runtime_error("native oracle seed byte count mismatch: " +
                             expected_path.string());
  }
  check_hip(hipMemcpy(actual_device, bytes.data(), bytes.size(),
                      hipMemcpyHostToDevice),
            "hipMemcpy native oracle seed");
  return bytes.size();
}

NativeOracleComparison compare_native_oracle_tensor(
    const std::string& label, const std::string& dtype,
    const void* actual_device, std::size_t bytes,
    const std::filesystem::path& expected_path) {
  return compare_tensor(label, dtype, actual_device, bytes, expected_path);
}

NativeOracleComparison compare_native_oracle_tensor_prefix(
    const std::string& label, const std::string& dtype,
    const void* actual_device, std::size_t bytes,
    const std::filesystem::path& expected_path) {
  return compare_tensor(label, dtype, actual_device, bytes, expected_path,
                        true);
}

NativeOracleComparison compare_native_oracle_tensor_slice(
    const std::string& label, const std::string& dtype,
    const void* actual_device, std::size_t bytes,
    const std::filesystem::path& expected_path,
    std::size_t expected_offset_bytes) {
  return compare_tensor(label, dtype, actual_device, bytes, expected_path,
                        true, expected_offset_bytes);
}

NativeLogitsComparison compare_native_logits_fp32(
    const void* actual_device, std::size_t elements,
    const std::filesystem::path& expected_path) {
  if (actual_device == nullptr || elements == 0) {
    throw std::invalid_argument("native logits comparison input is invalid");
  }
  const std::size_t bytes = elements * sizeof(float);
  const std::vector<unsigned char> expected_bytes =
      read_binary(expected_path);
  if (expected_bytes.size() != bytes) {
    throw std::runtime_error(
        "native logits oracle byte count mismatch");
  }
  std::vector<float> expected(elements);
  std::vector<float> actual(elements);
  std::memcpy(expected.data(), expected_bytes.data(), bytes);
  check_hip(hipMemcpy(actual.data(), actual_device, bytes,
                      hipMemcpyDeviceToHost),
            "hipMemcpy native logits comparison");

  NativeLogitsComparison result;
  result.elements = elements;
  result.expected_sha256 =
      sha256_bytes(expected_bytes.data(), expected_bytes.size());
  result.actual_sha256 = sha256_bytes(
      reinterpret_cast<const unsigned char*>(actual.data()), bytes);
  double squared_error = 0.0;
  double squared_expected = 0.0;
  float expected_max = -std::numeric_limits<float>::infinity();
  float actual_max = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < elements; ++index) {
    if (std::memcmp(expected_bytes.data() + index * sizeof(float),
                    reinterpret_cast<const unsigned char*>(actual.data()) +
                        index * sizeof(float),
                    sizeof(float)) == 0) {
      ++result.exact_elements;
    }
    const double expected_value = expected[index];
    const double actual_value = actual[index];
    if (std::isfinite(actual_value)) ++result.finite_elements;
    const double error = actual_value - expected_value;
    result.maximum_absolute_error =
        std::max(result.maximum_absolute_error, std::abs(error));
    squared_error += error * error;
    squared_expected += expected_value * expected_value;
    if (expected[index] > expected_max) {
      expected_max = expected[index];
      result.reference_top1_token_id =
          static_cast<std::uint32_t>(index);
    }
    if (actual[index] > actual_max) {
      actual_max = actual[index];
      result.actual_top1_token_id = static_cast<std::uint32_t>(index);
    }
  }
  result.top1_match =
      result.reference_top1_token_id == result.actual_top1_token_id;
  result.relative_l2_error =
      std::sqrt(squared_error / std::max(squared_expected, 1.0e-30));
  if (result.finite_elements != elements || !std::isfinite(expected_max)) {
    result.kl_divergence = std::numeric_limits<double>::infinity();
    return result;
  }

  long double expected_partition = 0.0L;
  long double actual_partition = 0.0L;
  for (std::size_t index = 0; index < elements; ++index) {
    expected_partition += std::exp(
        static_cast<long double>(expected[index] - expected_max));
    actual_partition +=
        std::exp(static_cast<long double>(actual[index] - actual_max));
  }
  const long double expected_log_partition =
      static_cast<long double>(expected_max) + std::log(expected_partition);
  const long double actual_log_partition =
      static_cast<long double>(actual_max) + std::log(actual_partition);
  long double divergence = 0.0L;
  for (std::size_t index = 0; index < elements; ++index) {
    const long double reference_log_probability =
        static_cast<long double>(expected[index]) - expected_log_partition;
    const long double actual_log_probability =
        static_cast<long double>(actual[index]) - actual_log_partition;
    const long double probability = std::exp(reference_log_probability);
    divergence += probability *
                  (reference_log_probability - actual_log_probability);
  }
  result.kl_divergence =
      std::max(0.0, static_cast<double>(divergence));
  return result;
}

std::filesystem::path find_native_oracle_tensor_file(
    const std::filesystem::path& oracle_dir, const std::string& label) {
  const auto files = oracle_files(std::filesystem::absolute(oracle_dir), 0);
  return require_file(files, label);
}

std::filesystem::path find_native_oracle_tensor_file_if_present(
    const std::filesystem::path& oracle_dir, const std::string& label) {
  const auto files = oracle_files(std::filesystem::absolute(oracle_dir), 0);
  const auto found = files.find(label);
  return found == files.end() ? std::filesystem::path{} : found->second;
}

NativeFullAttentionCoreOracleResult probe_native_full_attention_core_oracle(
    const std::filesystem::path& oracle_dir, std::size_t layer_index,
    std::size_t cache_end, NativeFullAttentionState& state) {
  if (!state.built() || cache_end == 0 ||
      state.cache_capacity() < cache_end) {
    throw std::invalid_argument(
        "native full-attention oracle state geometry is invalid");
  }
  const auto files = oracle_files(std::filesystem::absolute(oracle_dir));
  char prefix_buffer[32] = {};
  std::snprintf(prefix_buffer, sizeof(prefix_buffer), "layer-%03zu-",
                layer_index);
  const std::string prefix(prefix_buffer);
  const auto label = [&prefix](const char* suffix) {
    return prefix + suffix;
  };
  NativeFullAttentionCoreOracleResult result;

  const std::size_t cache_bytes =
      cache_end * 2 * 256 * sizeof(std::uint16_t);
  struct Seed {
    std::string label;
    void* device;
    std::size_t bytes;
  };
  auto* q = static_cast<unsigned char*>(state.gated_attention());
  auto* k = static_cast<unsigned char*>(state.projected_attention());
  auto* v = k + 1024;
  const std::vector<Seed> seeds = {
      {label("return-full_attention-q"), q, 8192},
      {label("return-full_attention-k"), k, 1024},
      {label("return-full_attention-v"), v, 1024},
      {label("return-full_attention-k_cache"), state.k_cache(layer_index),
       cache_bytes},
      {label("return-full_attention-v_cache"), state.v_cache(layer_index),
       cache_bytes},
  };
  for (const Seed& seed : seeds) {
    const std::vector<unsigned char> bytes =
        read_binary(require_file(files, seed.label));
    if (bytes.size() != seed.bytes) {
      throw std::runtime_error(
          "native full-attention oracle seed byte count mismatch: " +
          seed.label);
    }
    check_hip(hipMemcpy(seed.device, bytes.data(), bytes.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy native full-attention oracle seed");
    ++result.seed_tensors;
    result.seed_bytes += bytes.size();
  }
  constexpr std::size_t kCacheTokenBytes = 2 * 256 * sizeof(std::uint16_t);
  auto* k_last = static_cast<unsigned char*>(state.k_cache(layer_index)) +
                 (cache_end - 1) * kCacheTokenBytes;
  auto* v_last = static_cast<unsigned char*>(state.v_cache(layer_index)) +
                 (cache_end - 1) * kCacheTokenBytes;
  check_hip(hipMemset(k_last, 0, kCacheTokenBytes),
            "hipMemset native full-attention K sentinel");
  check_hip(hipMemset(v_last, 0, kCacheTokenBytes),
            "hipMemset native full-attention V sentinel");

  result.core = launch_native_grouped_full_attention(
      layer_index, cache_end - 1, cache_end, q, k, v, state);
  check_hip(hipDeviceSynchronize(),
            "hipDeviceSynchronize native full-attention oracle");
  result.comparisons = {
      compare_tensor("k_cache", "bfloat16", state.k_cache(layer_index),
                     cache_bytes,
                     require_file(files,
                                  label("return-full_attention-k_cache"))),
      compare_tensor("v_cache", "bfloat16", state.v_cache(layer_index),
                     cache_bytes,
                     require_file(files,
                                  label("return-full_attention-v_cache"))),
      compare_tensor("scores", "bfloat16", state.scores(),
                     cache_end * 16 * sizeof(std::uint16_t),
                     require_file(files,
                                  label("return-full_attention-scores"))),
      compare_tensor("probabilities", "bfloat16", state.probabilities(),
                     cache_end * 16 * sizeof(std::uint16_t),
                     require_file(files,
                                  label("return-full_attention-probs"))),
      compare_tensor("attention", "bfloat16", state.attention_output(), 8192,
                     require_file(
                         files, label("return-full_attention-attn_grouped"))),
  };
  result.all_finite = true;
  for (const NativeOracleComparison& comparison : result.comparisons) {
    result.all_finite = result.all_finite &&
                        comparison.finite_elements == comparison.elements;
    if (comparison.label == "scores") {
      result.scores_relative_l2_error = comparison.relative_l2_error;
    } else if (comparison.label == "probabilities") {
      result.probabilities_relative_l2_error = comparison.relative_l2_error;
    } else if (comparison.label == "attention") {
      result.attention_relative_l2_error = comparison.relative_l2_error;
      result.attention_cosine_similarity = comparison.cosine_similarity;
    }
  }
  result.kv_cache_exact =
      result.comparisons[0].exact_elements == result.comparisons[0].elements &&
      result.comparisons[1].exact_elements == result.comparisons[1].elements;
  return result;
}

NativeFullLayerOracleResult probe_native_full_layer_oracle(
    const std::filesystem::path& oracle_dir, std::size_t layer_index,
    std::size_t cache_end, const NativeWeightStore& weights,
    const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, NativeFullAttentionState& attention_state,
    int cu_count) {
  if (!attention_state.built() || cache_end == 0 ||
      attention_state.cache_capacity() < cache_end) {
    throw std::invalid_argument("native full-layer oracle state is invalid");
  }
  const auto files = oracle_files(std::filesystem::absolute(oracle_dir));
  const auto& launches = invocations.launches();
  const std::size_t base = layer_index * 10;
  if (base + 10 >= launches.size()) {
    throw std::runtime_error("native full-layer oracle index is out of range");
  }
  char prefix_buffer[32] = {};
  std::snprintf(prefix_buffer, sizeof(prefix_buffer), "layer-%03zu-",
                layer_index);
  const std::string prefix(prefix_buffer);
  const auto label = [&prefix](const char* suffix) {
    return prefix + suffix;
  };
  NativeFullLayerOracleResult result;

  const std::size_t cache_bytes =
      cache_end * 2 * 256 * sizeof(std::uint16_t);
  struct Seed {
    std::string label;
    void* device;
    std::size_t bytes;
  };
  const auto& input = require_view(
      workspace, schedule_binding(launches[base], "x"), 4096,
      DecodeTensorDtype::kBfloat16);
  const auto& cos = require_view(
      workspace, schedule_binding(launches[base + 2], "cos"), 128,
      DecodeTensorDtype::kFloat32);
  const auto& sin = require_view(
      workspace, schedule_binding(launches[base + 2], "sin"), 128,
      DecodeTensorDtype::kFloat32);
  const std::vector<Seed> seeds = {
      {label("launch-000-x"), input.device_pointer, 4096},
      {label("launch-002-cos"), cos.device_pointer, 128},
      {label("launch-002-sin"), sin.device_pointer, 128},
      {label("return-full_attention-k_cache"),
       attention_state.k_cache(layer_index), cache_bytes},
      {label("return-full_attention-v_cache"),
       attention_state.v_cache(layer_index), cache_bytes},
  };
  for (const Seed& seed : seeds) {
    const std::vector<unsigned char> bytes =
        read_binary(require_file(files, seed.label));
    if (bytes.size() != seed.bytes) {
      throw std::runtime_error(
          "native full-layer oracle seed byte count mismatch: " + seed.label);
    }
    check_hip(hipMemcpy(seed.device, bytes.data(), bytes.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy native full-layer oracle seed");
    ++result.seed_tensors;
    result.seed_bytes += bytes.size();
  }
  constexpr std::size_t kCacheTokenBytes = 2 * 256 * sizeof(std::uint16_t);
  auto* k_last =
      static_cast<unsigned char*>(attention_state.k_cache(layer_index)) +
      (cache_end - 1) * kCacheTokenBytes;
  auto* v_last =
      static_cast<unsigned char*>(attention_state.v_cache(layer_index)) +
      (cache_end - 1) * kCacheTokenBytes;
  check_hip(hipMemset(k_last, 0, kCacheTokenBytes),
            "hipMemset native full-layer K sentinel");
  check_hip(hipMemset(v_last, 0, kCacheTokenBytes),
            "hipMemset native full-layer V sentinel");

  result.layer = run_native_full_layer(
      layer_index, cache_end - 1, cache_end, weights, workspace, invocations,
      executor, attention_state, cu_count);

  struct Check {
    std::string name;
    std::string expected_label;
    const void* device;
    std::size_t bytes;
    std::string dtype;
  };
  const auto& qkv = require_view(
      workspace, schedule_binding(launches[base + 1], "out"), 18432,
      DecodeTensorDtype::kBfloat16);
  const auto& q = require_view(
      workspace, schedule_binding(launches[base + 2], "out"), 8192,
      DecodeTensorDtype::kBfloat16);
  const auto& k = require_view(
      workspace, schedule_binding(launches[base + 3], "out"), 1024,
      DecodeTensorDtype::kBfloat16);
  const std::vector<Check> checks = {
      {"input_rmsnorm", label("launch-000-out"),
       require_view(workspace, schedule_binding(launches[base], "out"), 4096,
                    DecodeTensorDtype::kBfloat16)
           .device_pointer,
       4096, "bfloat16"},
      {"qkv_projection", label("launch-001-out"), qkv.device_pointer, 18432,
       "bfloat16"},
      {"q_norm_rope", label("launch-002-out"), q.device_pointer, 8192,
       "bfloat16"},
      {"k_norm_rope", label("launch-003-out"), k.device_pointer, 1024,
       "bfloat16"},
      {"k_cache", label("return-full_attention-k_cache"),
       attention_state.k_cache(layer_index), cache_bytes, "bfloat16"},
      {"v_cache", label("return-full_attention-v_cache"),
       attention_state.v_cache(layer_index), cache_bytes, "bfloat16"},
      {"scores", label("return-full_attention-scores"),
       attention_state.scores(), cache_end * 16 * sizeof(std::uint16_t),
       "bfloat16"},
      {"probabilities", label("return-full_attention-probs"),
       attention_state.probabilities(),
       cache_end * 16 * sizeof(std::uint16_t), "bfloat16"},
      {"attention", label("return-full_attention-attn_grouped"),
       attention_state.attention_output(), 8192, "bfloat16"},
      {"projected_attention", label("return-full_attention-output"),
       attention_state.projected_attention(), 4096, "bfloat16"},
      {"attention_residual", label("launch-004-x"),
       require_view(workspace, schedule_binding(launches[base + 4], "x"),
                    4096, DecodeTensorDtype::kBfloat16)
           .device_pointer,
       4096, "bfloat16"},
      {"post_attention_rmsnorm", label("launch-004-out"),
       require_view(workspace, schedule_binding(launches[base + 4], "out"),
                    4096, DecodeTensorDtype::kBfloat16)
           .device_pointer,
       4096, "bfloat16"},
      {"shared_input_projection", label("launch-005-out"),
       require_view(workspace, schedule_binding(launches[base + 5], "out"),
                    2050, DecodeTensorDtype::kBfloat16)
           .device_pointer,
       2050, "bfloat16"},
      {"shared_activation", label("return-shared_expert-activated"),
       require_view(workspace, "native.linear.shared_activation", 1024,
                    DecodeTensorDtype::kBfloat16)
           .device_pointer,
       1024, "bfloat16"},
      {"shared_down_projection", label("return-shared_expert-shared_out"),
       require_view(workspace, "native.linear.shared_down", 4096,
                    DecodeTensorDtype::kBfloat16)
           .device_pointer,
       4096, "bfloat16"},
      {"shared_gate_output", label("return-shared_expert-output"),
       require_view(workspace, "native.linear.shared_scaled", 4096,
                    DecodeTensorDtype::kBfloat16)
           .device_pointer,
       4096, "bfloat16"},
      {"router_scores", label("launch-007-out_values"),
       require_view(workspace,
                    schedule_binding(launches[base + 7], "out_values"), 16,
                    DecodeTensorDtype::kBfloat16)
           .device_pointer,
       16, "bfloat16"},
      {"router_ids", label("launch-007-out_ids"),
       require_view(workspace,
                    schedule_binding(launches[base + 7], "out_ids"), 32,
                    DecodeTensorDtype::kInt32)
           .device_pointer,
       32, "int32"},
      {"routed_activation", label("launch-008-act_out"),
       require_view(workspace,
                    schedule_binding(launches[base + 8], "act_out"), 8192,
                    DecodeTensorDtype::kBfloat16)
           .device_pointer,
       8192, "bfloat16"},
      {"routed_moe", label("launch-009-out"),
       require_view(workspace, schedule_binding(launches[base + 9], "out"),
                    4096, DecodeTensorDtype::kBfloat16)
           .device_pointer,
       4096, "bfloat16"},
      {"combined_moe", label("return-layer_body-moe_out"),
       require_view(workspace, "native.linear.combined_moe", 4096,
                    DecodeTensorDtype::kBfloat16)
           .device_pointer,
       4096, "bfloat16"},
      {"layer_output", label("return-layer_body-output"),
       require_view(workspace, schedule_binding(launches[base + 10], "x"),
                    4096, DecodeTensorDtype::kBfloat16)
           .device_pointer,
       4096, "bfloat16"},
  };
  result.comparisons.reserve(checks.size());
  result.all_finite = true;
  result.pre_attention_aot_exact = true;
  for (const Check& check : checks) {
    result.comparisons.push_back(compare_tensor(
        check.name, check.dtype, check.device, check.bytes,
        require_file(files, check.expected_label)));
    const NativeOracleComparison& comparison = result.comparisons.back();
    result.all_finite = result.all_finite &&
                        comparison.finite_elements == comparison.elements;
    if (check.name == "input_rmsnorm" || check.name == "qkv_projection" ||
        check.name == "q_norm_rope" || check.name == "k_norm_rope") {
      result.pre_attention_aot_exact =
          result.pre_attention_aot_exact &&
          comparison.exact_elements == comparison.elements;
    } else if (check.name == "router_ids") {
      result.router_ids_exact =
          comparison.exact_elements == comparison.elements;
    } else if (check.name == "attention") {
      result.attention_relative_l2_error = comparison.relative_l2_error;
    } else if (check.name == "projected_attention") {
      result.projected_attention_relative_l2_error =
          comparison.relative_l2_error;
    } else if (check.name == "layer_output") {
      result.final_relative_l2_error = comparison.relative_l2_error;
      result.final_cosine_similarity = comparison.cosine_similarity;
    }
  }
  const auto exact_named = [&result](const char* name) {
    const auto found = std::find_if(
        result.comparisons.begin(), result.comparisons.end(),
        [name](const NativeOracleComparison& value) {
          return value.label == name;
        });
    return found != result.comparisons.end() &&
           found->exact_elements == found->elements;
  };
  result.kv_cache_exact = exact_named("k_cache") && exact_named("v_cache");
  return result;
}

NativeDecodeOracleResult probe_native_decode_oracle(
    const std::filesystem::path& oracle_dir, std::size_t cache_end,
    const NativeWeightStore& weights, const NativeLmHeadStore& lm_head,
    const NativeDecodeWorkspace& workspace,
    NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, NativeFullAttentionState& attention_state,
    int cu_count, const NativeDecodeOracleOptions& options) {
  if (!attention_state.built() || cache_end == 0 ||
      attention_state.cache_capacity() < cache_end ||
      options.measured_runs == 0) {
    throw std::invalid_argument("native decode oracle state is invalid");
  }
  const auto files = oracle_files(std::filesystem::absolute(oracle_dir));
  const auto& launches = invocations.launches();
  if (launches.size() != 402) {
    throw std::runtime_error("native decode oracle schedule is incomplete");
  }
  NativeDecodeOracleResult result;
  std::unordered_map<std::string, std::vector<unsigned char>> seed_payloads;
  const auto seed = [&](const std::string& label, void* device,
                        std::size_t bytes) {
    auto found = seed_payloads.find(label);
    if (found == seed_payloads.end()) {
      found = seed_payloads
                  .emplace(label, read_binary(require_file(files, label)))
                  .first;
      ++result.seed_tensors;
      result.seed_bytes += found->second.size();
    }
    const std::vector<unsigned char>& payload = found->second;
    if (payload.size() != bytes) {
      throw std::runtime_error("native decode oracle seed byte drift: " + label);
    }
    check_hip(hipMemcpy(device, payload.data(), payload.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy native decode oracle seed");
  };
  const auto prefix = [](std::size_t layer_index) {
    char buffer[32] = {};
    std::snprintf(buffer, sizeof(buffer), "layer-%03zu-", layer_index);
    return std::string(buffer);
  };

  const std::size_t cache_bytes =
      cache_end * 2 * 256 * sizeof(std::uint16_t);
  constexpr std::size_t kCacheTokenBytes = 2 * 256 * sizeof(std::uint16_t);
  const auto seed_decode_state = [&]() {
    seed(prefix(0) + "launch-000-x",
         require_view(workspace, schedule_binding(launches[0], "x"), 4096,
                      DecodeTensorDtype::kBfloat16)
             .device_pointer,
         4096);
    seed(prefix(3) + "launch-002-cos",
         require_view(workspace, schedule_binding(launches[32], "cos"), 128,
                      DecodeTensorDtype::kFloat32)
             .device_pointer,
         128);
    seed(prefix(3) + "launch-002-sin",
         require_view(workspace, schedule_binding(launches[32], "sin"), 128,
                      DecodeTensorDtype::kFloat32)
             .device_pointer,
         128);
    for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
      const std::size_t base = layer_index * 10;
      const std::string layer_prefix = prefix(layer_index);
      if (std::string(launches[base + 1].launch->symbol) ==
          "triton_fused_input_proj_conv_kernel") {
        seed(layer_prefix + "launch-001-state_in",
             invocations.tensor_pointer(base + 1, "state_in"),
             49152);
        seed(layer_prefix + "launch-002-h0",
             invocations.tensor_pointer(base + 2, "h0"),
             2097152);
      } else {
        void* k_cache = attention_state.k_cache(layer_index);
        void* v_cache = attention_state.v_cache(layer_index);
        seed(layer_prefix + "return-full_attention-k_cache", k_cache,
             cache_bytes);
        seed(layer_prefix + "return-full_attention-v_cache", v_cache,
             cache_bytes);
        check_hip(hipMemset(static_cast<unsigned char*>(k_cache) +
                                (cache_end - 1) * kCacheTokenBytes,
                            0, kCacheTokenBytes),
                  "hipMemset native decode oracle K sentinel");
        check_hip(hipMemset(static_cast<unsigned char*>(v_cache) +
                                (cache_end - 1) * kCacheTokenBytes,
                            0, kCacheTokenBytes),
                  "hipMemset native decode oracle V sentinel");
      }
    }
  };

  result.warmup_decodes.reserve(options.warmup_runs);
  for (std::size_t run = 0; run < options.warmup_runs; ++run) {
    seed_decode_state();
    result.warmup_decodes.push_back(run_native_decode_token(
        cache_end - 1, cache_end, weights, lm_head, workspace, invocations,
        executor, attention_state, cu_count));
  }
  result.measured_decodes.reserve(options.measured_runs);
  for (std::size_t run = 0; run < options.measured_runs; ++run) {
    seed_decode_state();
    result.measured_decodes.push_back(run_native_decode_token(
        cache_end - 1, cache_end, weights, lm_head, workspace, invocations,
        executor, attention_state, cu_count));
  }
  result.decode = result.measured_decodes.back();
  result.comparisons.reserve(101);
  result.comparisons.push_back(compare_tensor(
      "final_hidden", "bfloat16",
      require_view(workspace, schedule_binding(launches[400], "x"), 4096,
                   DecodeTensorDtype::kBfloat16)
          .device_pointer,
      4096, require_file(files, prefix(39) + "return-layer_body-output")));
  result.final_hidden_relative_l2_error =
      result.comparisons.back().relative_l2_error;
  result.final_hidden_cosine_similarity =
      result.comparisons.back().cosine_similarity;

  for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
    const std::size_t base = layer_index * 10;
    const std::string layer_prefix = prefix(layer_index);
    result.comparisons.push_back(compare_tensor(
        "router_ids_layer_" + std::to_string(layer_index), "int32",
        require_view(workspace,
                     schedule_binding(launches[base + 7], "out_ids"), 32,
                     DecodeTensorDtype::kInt32)
            .device_pointer,
        32, require_file(files, layer_prefix + "launch-007-out_ids")));
    const auto& router = result.comparisons.back();
    result.router_layers_exact +=
        router.exact_elements == router.elements ? 1 : 0;
    if (std::string(launches[base + 1].launch->symbol) !=
        "triton_fused_input_proj_conv_kernel") {
      continue;
    }
    result.comparisons.push_back(compare_tensor(
        "recurrent_state_layer_" + std::to_string(layer_index), "float32",
        invocations.tensor_pointer(base + 2, "h0"),
        2097152, require_file(files, layer_prefix + "launch-002-ht")));
    const auto& recurrent = result.comparisons.back();
    result.recurrent_states_exact +=
        recurrent.exact_elements == recurrent.elements ? 1 : 0;
    result.comparisons.push_back(compare_tensor(
        "conv_state_layer_" + std::to_string(layer_index), "bfloat16",
        invocations.tensor_pointer(base + 1, "state_in"),
        49152,
        require_file(files, layer_prefix + "launch-001-state_out")));
  }
  result.all_finite = true;
  for (const NativeOracleComparison& comparison : result.comparisons) {
    result.all_finite = result.all_finite &&
                        comparison.finite_elements == comparison.elements;
  }
  return result;
}

NativeLinearLayerOracleResult probe_native_linear_layer_oracle(
    const std::filesystem::path& oracle_dir,
    std::size_t layer_index,
    const NativeWeightStore& weights,
    const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, int cu_count) {
  const auto files = oracle_files(std::filesystem::absolute(oracle_dir));
  const auto& launches = invocations.launches();
  const std::size_t base = layer_index * 10;
  if (base + 10 >= launches.size()) {
    throw std::runtime_error("native layer oracle layer index is out of range");
  }
  std::string prefix;
  {
    char buffer[32] = {};
    std::snprintf(buffer, sizeof(buffer), "layer-%03zu-", layer_index);
    if (files.find(std::string(buffer) + "launch-000-x") != files.end()) {
      prefix = buffer;
    } else if (layer_index != 0) {
      throw std::runtime_error("native layer oracle has no layer-qualified labels");
    }
  }
  const auto oracle_label = [&prefix](const char* suffix) {
    return prefix + suffix;
  };
  NativeLinearLayerOracleResult result;

  struct Seed {
    const char* label;
    const char* binding;
    std::size_t bytes;
    DecodeTensorDtype dtype;
  };
  const Seed seeds[] = {
      {"launch-000-x", schedule_binding(launches[base], "x"), 4096,
       DecodeTensorDtype::kBfloat16},
      {"launch-001-state_in", schedule_binding(launches[base + 1], "state_in"),
       49152, DecodeTensorDtype::kBfloat16},
      {"launch-002-h0", schedule_binding(launches[base + 2], "h0"), 2097152,
       DecodeTensorDtype::kFloat32},
  };
  for (const Seed& seed : seeds) {
    const auto& view = require_view(workspace, seed.binding, seed.bytes, seed.dtype);
    const std::vector<unsigned char> bytes =
        read_binary(require_file(files, oracle_label(seed.label)));
    if (bytes.size() != seed.bytes) {
      throw std::runtime_error("native layer oracle seed byte count mismatch");
    }
    check_hip(hipMemcpy(view.device_pointer, bytes.data(), bytes.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy native layer oracle seed");
    ++result.seed_tensors;
    result.seed_bytes += bytes.size();
  }

  result.layer = run_native_linear_layer(
      layer_index, weights, workspace, invocations, executor, cu_count);

  struct Check {
    std::string label;
    std::string expected_label;
    std::string binding;
    std::size_t bytes;
    DecodeTensorDtype native_dtype;
    std::string dtype;
  };
  const std::vector<Check> checks = {
      {"input_rmsnorm", oracle_label("launch-000-out"),
       schedule_binding(launches[base], "out"), 4096,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"input_projection", oracle_label("launch-001-out"),
       schedule_binding(launches[base + 1], "out"), 24704,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"conv_state", oracle_label("launch-001-state_out"),
       schedule_binding(launches[base + 1], "state_out"), 49152,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"recurrent_output", oracle_label("launch-002-o"),
       schedule_binding(launches[base + 2], "o"), 8192,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"recurrent_state", oracle_label("launch-002-ht"),
       schedule_binding(launches[base + 2], "ht"), 2097152,
       DecodeTensorDtype::kFloat32, "float32"},
      {"gated_norm", oracle_label("launch-003-out"),
       schedule_binding(launches[base + 3], "out"), 8192,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"attention_output", oracle_label("return-linear_attention-output"),
       "native.linear.attention_output",
       4096, DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"attention_residual", oracle_label("launch-004-x"),
       schedule_binding(launches[base + 4], "x"), 4096,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"post_attention_rmsnorm", oracle_label("launch-004-out"),
       schedule_binding(launches[base + 4], "out"), 4096,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"shared_input_projection", oracle_label("launch-005-out"),
       schedule_binding(launches[base + 5], "out"), 2050,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"shared_activation", oracle_label("return-shared_expert-activated"),
       "native.linear.shared_activation",
       1024, DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"shared_down_projection", oracle_label("return-shared_expert-shared_out"),
       "native.linear.shared_down", 4096, DecodeTensorDtype::kBfloat16,
       "bfloat16"},
      {"shared_gate_output", oracle_label("return-shared_expert-output"),
       "native.linear.shared_scaled",
       4096, DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"router_scores", oracle_label("launch-007-out_values"),
       schedule_binding(launches[base + 7], "out_values"), 16,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"router_ids", oracle_label("launch-007-out_ids"),
       schedule_binding(launches[base + 7], "out_ids"), 32,
       DecodeTensorDtype::kInt32, "int32"},
      {"routed_activation", oracle_label("launch-008-act_out"),
       schedule_binding(launches[base + 8], "act_out"), 8192,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"routed_moe", oracle_label("launch-009-out"),
       schedule_binding(launches[base + 9], "out"), 4096,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"combined_moe", oracle_label("return-layer_body-moe_out"),
       "native.linear.combined_moe", 4096,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
      {"layer_output", oracle_label("return-layer_body-output"),
       schedule_binding(launches[base + 10], "x"), 4096,
       DecodeTensorDtype::kBfloat16, "bfloat16"},
  };

  result.comparisons.reserve(checks.size());
  result.all_finite = true;
  for (const Check& check : checks) {
    const auto& view = require_view(
        workspace, check.binding, check.bytes, check.native_dtype);
    result.comparisons.push_back(compare_tensor(
        check.label, check.dtype, view.device_pointer, check.bytes,
        require_file(files, check.expected_label)));
    const NativeOracleComparison& comparison = result.comparisons.back();
    result.all_finite = result.all_finite &&
                        comparison.finite_elements == comparison.elements;
    if (comparison.label == "router_ids") {
      result.router_ids_exact =
          comparison.exact_elements == comparison.elements;
    }
    if (comparison.label == "layer_output") {
      result.final_relative_l2_error = comparison.relative_l2_error;
      result.final_cosine_similarity = comparison.cosine_similarity;
    }
  }

  static constexpr const char* aot_labels[] = {
      "input_rmsnorm",       "input_projection", "conv_state",
      "recurrent_output",    "recurrent_state",  "gated_norm",
      "post_attention_rmsnorm", "shared_input_projection",
      "router_scores",       "router_ids",       "routed_activation",
      "routed_moe",
  };
  result.aot_boundaries_exact = true;
  for (const char* label : aot_labels) {
    const auto found = std::find_if(
        result.comparisons.begin(), result.comparisons.end(),
        [label](const NativeOracleComparison& value) {
          return value.label == label;
        });
    result.aot_boundaries_exact =
        result.aot_boundaries_exact && found != result.comparisons.end() &&
        found->exact_elements == found->elements;
  }
  return result;
}

}  // namespace aima
