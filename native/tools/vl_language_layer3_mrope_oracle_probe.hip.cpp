// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_pointwise.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef AIMA_SOURCE_COMMIT
#define AIMA_SOURCE_COMMIT "unknown"
#endif

namespace {

using json = nlohmann::json;

constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);
constexpr std::size_t kPositionAxes = 3;
constexpr std::size_t kRotaryPairs = 32;
constexpr std::size_t kQGateWidth = 8192;
constexpr std::size_t kRawKWidth = 512;
constexpr std::size_t kRotaryQWidth = 4096;
constexpr std::size_t kRotaryKWidth = 512;
constexpr std::size_t kNormWeightWidth = 256;
constexpr std::size_t kMeasuredRuns = 5;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorName(status) + " (" +
                             hipGetErrorString(status) + ")");
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ == 0) throw std::invalid_argument("zero-sized device buffer");
    check_hip(hipMalloc(&pointer_, bytes_), "hipMalloc layer-3 M-RoPE");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      const hipError_t ignored = hipFree(pointer_);
      static_cast<void>(ignored);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void upload(const void* source, std::size_t bytes) {
    if (source == nullptr || bytes != bytes_) {
      throw std::invalid_argument("device upload size differs");
    }
    check_hip(hipMemcpy(pointer_, source, bytes_, hipMemcpyHostToDevice),
              "hipMemcpy layer-3 M-RoPE upload");
  }

  std::vector<unsigned char> download() const {
    std::vector<unsigned char> result(bytes_);
    check_hip(hipMemcpy(result.data(), pointer_, bytes_, hipMemcpyDeviceToHost),
              "hipMemcpy layer-3 M-RoPE download");
    return result;
  }

  void* get() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class Event {
 public:
  Event() { check_hip(hipEventCreate(&event_), "hipEventCreate M-RoPE"); }
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

std::filesystem::path checked_path(const std::filesystem::path& root,
                                   std::string_view relative_text) {
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
  const std::filesystem::path path = checked_path(
      root, record.at("path").get<std::string>());
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() < 0 ||
      static_cast<std::size_t>(stream.tellg()) != bytes) {
    throw std::runtime_error("oracle tensor size mismatch: " + path.string());
  }
  std::vector<unsigned char> result(bytes);
  stream.seekg(0);
  if (bytes != 0 &&
      !stream.read(reinterpret_cast<char*>(result.data()),
                   static_cast<std::streamsize>(bytes))) {
    throw std::runtime_error("oracle tensor read failed: " + path.string());
  }
  if (aima::sha256_bytes(result.data(), result.size()) !=
      record.at("sha256").get<std::string>()) {
    throw std::runtime_error("oracle tensor hash mismatch: " + path.string());
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
    throw std::runtime_error("M-RoPE output write failed: " + path.string());
  }
}

float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0f;
  std::memcpy(&result, &word, sizeof(result));
  return result;
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t word = 0;
  std::memcpy(&word, &value, sizeof(word));
  const std::uint32_t rounding = 0x7fffU + ((word >> 16U) & 1U);
  return static_cast<std::uint16_t>((word + rounding) >> 16U);
}

std::vector<float> bf16_to_fp32(
    const std::vector<unsigned char>& source) {
  if (source.size() % kBf16Bytes != 0) {
    throw std::invalid_argument("BF16 table byte count is invalid");
  }
  std::vector<float> result(source.size() / kBf16Bytes);
  for (std::size_t index = 0; index < result.size(); ++index) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, source.data() + index * kBf16Bytes, kBf16Bytes);
    result[index] = bf16_to_float(bits);
  }
  return result;
}

std::vector<unsigned char> fp32_to_bf16(
    const std::vector<unsigned char>& source) {
  if (source.size() % sizeof(float) != 0) {
    throw std::invalid_argument("FP32 table byte count is invalid");
  }
  const std::size_t elements = source.size() / sizeof(float);
  std::vector<unsigned char> result(elements * kBf16Bytes);
  for (std::size_t index = 0; index < elements; ++index) {
    float value = 0.0f;
    std::memcpy(&value, source.data() + index * sizeof(float), sizeof(value));
    const std::uint16_t bits = float_to_bf16(value);
    std::memcpy(result.data() + index * kBf16Bytes, &bits, kBf16Bytes);
  }
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
    throw std::invalid_argument("layer-3 M-RoPE comparison sizes differ");
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
                expected.data() + index * kBf16Bytes, kBf16Bytes);
    std::memcpy(&actual_bits, actual.data() + index * kBf16Bytes,
                kBf16Bytes);
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

json comparison_json(const Comparison& value) {
  json first_mismatch = nullptr;
  if (value.first_mismatch_index != std::numeric_limits<std::size_t>::max()) {
    first_mismatch = {
        {"index", value.first_mismatch_index},
        {"expected_bits", value.first_expected_bits},
        {"actual_bits", value.first_actual_bits},
    };
  }
  return {
      {"elements", value.elements},
      {"exact_elements", value.exact_elements},
      {"finite_elements", value.finite_elements},
      {"first_mismatch", std::move(first_mismatch)},
      {"maximum_absolute_error", value.maximum_absolute_error},
      {"relative_l2_error", value.relative_l2_error},
      {"cosine_similarity", value.cosine_similarity},
      {"expected_sha256", value.expected_sha256},
      {"actual_sha256", value.actual_sha256},
      {"passed", value.passed()},
  };
}

struct Execution {
  std::vector<unsigned char> cosine_fp32;
  std::vector<unsigned char> sine_fp32;
  std::vector<unsigned char> q;
  std::vector<unsigned char> k;
  float measured_ms = 0.0f;
};

Execution execute_generated(
    const DeviceBuffer& positions, const DeviceBuffer& q_gate,
    const DeviceBuffer& raw_k, const DeviceBuffer& q_weight,
    const DeviceBuffer& k_weight, DeviceBuffer& cosine, DeviceBuffer& sine,
    DeviceBuffer& q, DeviceBuffer& k, std::size_t tokens) {
  Event start;
  Event stop;
  check_hip(hipEventRecord(start), "hipEventRecord M-RoPE start");
  aima::launch_prefill_mrope_rotary_table(
      cosine.get(), sine.get(), positions.get(), tokens, tokens);
  aima::launch_full_attention_head_norm_mrope_prefill(
      q_gate.get(), raw_k.get(), nullptr, q_weight.get(), k_weight.get(),
      cosine.get(), sine.get(), q.get(), k.get(), nullptr, tokens,
      kQGateWidth, kRawKWidth, 0);
  check_hip(hipEventRecord(stop), "hipEventRecord M-RoPE stop");
  check_hip(hipEventSynchronize(stop), "hipEventSynchronize M-RoPE stop");
  Execution result;
  check_hip(hipEventElapsedTime(&result.measured_ms, start, stop),
            "hipEventElapsedTime M-RoPE");
  result.cosine_fp32 = cosine.download();
  result.sine_fp32 = sine.download();
  result.q = q.download();
  result.k = k.download();
  return result;
}

Execution execute_oracle_table(
    const DeviceBuffer& q_gate, const DeviceBuffer& raw_k,
    const DeviceBuffer& q_weight, const DeviceBuffer& k_weight,
    const DeviceBuffer& cosine, const DeviceBuffer& sine,
    DeviceBuffer& q, DeviceBuffer& k, std::size_t tokens) {
  aima::launch_full_attention_head_norm_mrope_prefill(
      q_gate.get(), raw_k.get(), nullptr, q_weight.get(), k_weight.get(),
      cosine.get(), sine.get(), q.get(), k.get(), nullptr, tokens,
      kQGateWidth, kRawKWidth, 0);
  check_hip(hipDeviceSynchronize(),
            "hipDeviceSynchronize oracle-table M-RoPE");
  Execution result;
  result.q = q.download();
  result.k = k.download();
  return result;
}

void require_component(const json& record, std::string_view dtype,
                       std::size_t rows, std::size_t columns) {
  if (record.value("dtype", "") != dtype ||
      !record.contains("shape") || !record.at("shape").is_array() ||
      record.at("shape").size() != 2 ||
      record.at("shape").at(0).get<std::size_t>() != rows ||
      record.at("shape").at(1).get<std::size_t>() != columns) {
    throw std::runtime_error("layer-3 M-RoPE component geometry differs");
  }
}

json qualify_case(const json& case_record,
                  const std::filesystem::path& oracle_root,
                  const std::filesystem::path& output_root) {
  const std::string case_id = case_record.at("case_id").get<std::string>();
  const std::size_t tokens = case_record.at("prompt_tokens").get<std::size_t>();
  if (tokens == 0 || tokens > 1024) {
    throw std::runtime_error("layer-3 M-RoPE prompt length is invalid");
  }
  const json& components = case_record.at("components");
  require_component(components.at("positions"), "torch.int64",
                    kPositionAxes, tokens);
  require_component(components.at("q_gate_projection"), "torch.bfloat16",
                    tokens, kQGateWidth);
  require_component(components.at("raw_k"), "torch.bfloat16", tokens,
                    kRawKWidth);
  require_component(components.at("effective_cos"), "torch.bfloat16", tokens,
                    kRotaryPairs);
  require_component(components.at("effective_sin"), "torch.bfloat16", tokens,
                    kRotaryPairs);
  require_component(components.at("rotary_q"), "torch.bfloat16", tokens,
                    kRotaryQWidth);
  require_component(components.at("rotary_k"), "torch.bfloat16", tokens,
                    kRotaryKWidth);
  require_component(components.at("normalized_q"), "torch.bfloat16", tokens,
                    kRotaryQWidth);
  require_component(components.at("normalized_k"), "torch.bfloat16", tokens,
                    kRotaryKWidth);
  if (components.at("q_norm_weight").value("dtype", "") !=
          "torch.bfloat16" ||
      components.at("q_norm_weight").at("shape") !=
          json::array({kNormWeightWidth}) ||
      components.at("k_norm_weight").value("dtype", "") !=
          "torch.bfloat16" ||
      components.at("k_norm_weight").at("shape") !=
          json::array({kNormWeightWidth})) {
    throw std::runtime_error("layer-3 head norm weight geometry differs");
  }

  const auto positions = read_tensor(oracle_root, components.at("positions"));
  const auto q_gate =
      read_tensor(oracle_root, components.at("q_gate_projection"));
  const auto raw_k = read_tensor(oracle_root, components.at("raw_k"));
  const auto q_weight =
      read_tensor(oracle_root, components.at("q_norm_weight"));
  const auto k_weight =
      read_tensor(oracle_root, components.at("k_norm_weight"));
  const auto expected_cos =
      read_tensor(oracle_root, components.at("effective_cos"));
  const auto expected_sin =
      read_tensor(oracle_root, components.at("effective_sin"));
  const auto expected_q = read_tensor(oracle_root, components.at("rotary_q"));
  const auto expected_k = read_tensor(oracle_root, components.at("rotary_k"));
  const auto expected_normalized_q =
      read_tensor(oracle_root, components.at("normalized_q"));
  const auto expected_normalized_k =
      read_tensor(oracle_root, components.at("normalized_k"));
  const auto q_norm_input =
      read_tensor(oracle_root, components.at("q_norm_input"));
  const auto raw_q = read_tensor(oracle_root, components.at("raw_q"));
  const auto k_norm_input =
      read_tensor(oracle_root, components.at("k_norm_input"));
  if (raw_q != q_norm_input || raw_k != k_norm_input) {
    throw std::runtime_error("captured head norm input boundary is inconsistent");
  }

  const std::size_t table_fp32_bytes =
      tokens * kRotaryPairs * sizeof(float);
  DeviceBuffer positions_device(positions.size());
  DeviceBuffer q_gate_device(q_gate.size());
  DeviceBuffer raw_k_device(raw_k.size());
  DeviceBuffer q_weight_device(q_weight.size());
  DeviceBuffer k_weight_device(k_weight.size());
  DeviceBuffer generated_cos_device(table_fp32_bytes);
  DeviceBuffer generated_sin_device(table_fp32_bytes);
  DeviceBuffer oracle_cos_device(table_fp32_bytes);
  DeviceBuffer oracle_sin_device(table_fp32_bytes);
  DeviceBuffer unity_cos_device(table_fp32_bytes);
  DeviceBuffer unity_sin_device(table_fp32_bytes);
  DeviceBuffer q_device(expected_q.size());
  DeviceBuffer k_device(expected_k.size());
  positions_device.upload(positions.data(), positions.size());
  q_gate_device.upload(q_gate.data(), q_gate.size());
  raw_k_device.upload(raw_k.data(), raw_k.size());
  q_weight_device.upload(q_weight.data(), q_weight.size());
  k_weight_device.upload(k_weight.data(), k_weight.size());
  const std::vector<float> oracle_cos_fp32 = bf16_to_fp32(expected_cos);
  const std::vector<float> oracle_sin_fp32 = bf16_to_fp32(expected_sin);
  oracle_cos_device.upload(oracle_cos_fp32.data(), table_fp32_bytes);
  oracle_sin_device.upload(oracle_sin_fp32.data(), table_fp32_bytes);
  const std::vector<float> unity_cos(tokens * kRotaryPairs, 1.0f);
  const std::vector<float> unity_sin(tokens * kRotaryPairs, 0.0f);
  unity_cos_device.upload(unity_cos.data(), table_fp32_bytes);
  unity_sin_device.upload(unity_sin.data(), table_fp32_bytes);

  const Execution head_norm = execute_oracle_table(
      q_gate_device, raw_k_device, q_weight_device, k_weight_device,
      unity_cos_device, unity_sin_device, q_device, k_device, tokens);
  const Execution oracle_table = execute_oracle_table(
      q_gate_device, raw_k_device, q_weight_device, k_weight_device,
      oracle_cos_device, oracle_sin_device, q_device, k_device, tokens);
  const Execution warmup = execute_generated(
      positions_device, q_gate_device, raw_k_device, q_weight_device,
      k_weight_device, generated_cos_device, generated_sin_device, q_device,
      k_device, tokens);
  std::vector<Execution> measured;
  measured.reserve(kMeasuredRuns);
  for (std::size_t run = 0; run < kMeasuredRuns; ++run) {
    measured.push_back(execute_generated(
        positions_device, q_gate_device, raw_k_device, q_weight_device,
        k_weight_device, generated_cos_device, generated_sin_device, q_device,
        k_device, tokens));
  }
  const Execution& actual = measured.front();
  bool deterministic =
      warmup.cosine_fp32 == actual.cosine_fp32 &&
      warmup.sine_fp32 == actual.sine_fp32 && warmup.q == actual.q &&
      warmup.k == actual.k;
  std::vector<float> measured_ms;
  measured_ms.reserve(measured.size());
  for (const Execution& value : measured) {
    deterministic = deterministic &&
                    value.cosine_fp32 == actual.cosine_fp32 &&
                    value.sine_fp32 == actual.sine_fp32 &&
                    value.q == actual.q && value.k == actual.k;
    measured_ms.push_back(value.measured_ms);
  }
  std::vector<float> sorted_ms = measured_ms;
  std::sort(sorted_ms.begin(), sorted_ms.end());

  const std::vector<unsigned char> actual_cos =
      fp32_to_bf16(actual.cosine_fp32);
  const std::vector<unsigned char> actual_sin =
      fp32_to_bf16(actual.sine_fp32);
  const Comparison cos_comparison = compare_bf16(actual_cos, expected_cos);
  const Comparison sin_comparison = compare_bf16(actual_sin, expected_sin);
  const Comparison oracle_q_comparison =
      compare_bf16(oracle_table.q, expected_q);
  const Comparison oracle_k_comparison =
      compare_bf16(oracle_table.k, expected_k);
  const Comparison head_norm_q_comparison =
      compare_bf16(head_norm.q, expected_normalized_q);
  const Comparison head_norm_k_comparison =
      compare_bf16(head_norm.k, expected_normalized_k);
  const Comparison generated_q_comparison = compare_bf16(actual.q, expected_q);
  const Comparison generated_k_comparison = compare_bf16(actual.k, expected_k);
  const bool complete = deterministic && cos_comparison.passed() &&
                        sin_comparison.passed() &&
                        head_norm_q_comparison.passed() &&
                        head_norm_k_comparison.passed() &&
                        oracle_q_comparison.passed() &&
                        oracle_k_comparison.passed() &&
                        generated_q_comparison.passed() &&
                        generated_k_comparison.passed();

  const std::filesystem::path case_output = output_root / case_id;
  write_file(case_output / "effective_cos.bf16.bin", actual_cos);
  write_file(case_output / "effective_sin.bf16.bin", actual_sin);
  write_file(case_output / "rotary_q.bf16.bin", actual.q);
  write_file(case_output / "rotary_k.bf16.bin", actual.k);

  return {
      {"schema",
       "aima-amd395-qwen36/native-vl-language-layer3-mrope-case/v1"},
      {"complete", complete},
      {"case_id", case_id},
      {"prompt_tokens", tokens},
      {"positions_sha256", components.at("positions").at("sha256")},
      {"generated_effective_cos", comparison_json(cos_comparison)},
      {"generated_effective_sin", comparison_json(sin_comparison)},
      {"head_norm_q", comparison_json(head_norm_q_comparison)},
      {"head_norm_k", comparison_json(head_norm_k_comparison)},
      {"oracle_table_rotary_q", comparison_json(oracle_q_comparison)},
      {"oracle_table_rotary_k", comparison_json(oracle_k_comparison)},
      {"generated_table_rotary_q", comparison_json(generated_q_comparison)},
      {"generated_table_rotary_k", comparison_json(generated_k_comparison)},
      {"deterministic", deterministic},
      {"warmup_runs", 1},
      {"measured_runs", kMeasuredRuns},
      {"measured_ms", measured_ms},
      {"median_ms", sorted_ms.at(sorted_ms.size() / 2)},
      {"output_sha256",
       {{"effective_cos", aima::sha256_bytes(actual_cos.data(), actual_cos.size())},
        {"effective_sin", aima::sha256_bytes(actual_sin.data(), actual_sin.size())},
        {"rotary_q", aima::sha256_bytes(actual.q.data(), actual.q.size())},
        {"rotary_k", aima::sha256_bytes(actual.k.data(), actual.k.size())}}},
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr
        << "usage: native-vl-language-layer3-mrope-probe "
           "CAPTURE_MANIFEST CAPTURE_ROOT OUTPUT_ROOT\n";
    return 2;
  }
  try {
    const std::filesystem::path manifest_path =
        std::filesystem::absolute(argv[1]);
    const std::filesystem::path oracle_root =
        std::filesystem::absolute(argv[2]);
    const std::filesystem::path output_root =
        std::filesystem::absolute(argv[3]);
    if (std::filesystem::exists(output_root) &&
        (!std::filesystem::is_directory(output_root) ||
         !std::filesystem::is_empty(output_root))) {
      throw std::runtime_error("M-RoPE output root must be empty");
    }
    std::filesystem::create_directories(output_root);
    const json manifest = read_json(manifest_path);
    if (manifest.value("schema", "") !=
            "aima-amd395-qwen36/vl-language-layer3-mrope-diagnostic-oracle/v1" ||
        !manifest.value("complete", false) ||
        !manifest.contains("cases") || !manifest.at("cases").is_array() ||
        manifest.at("cases").size() != 5 ||
        manifest.at("mrope").at("section") != json::array({11, 11, 10}) ||
        !manifest.at("mrope").value("interleaved", false)) {
      throw std::runtime_error("layer-3 M-RoPE capture manifest is incomplete");
    }
    check_hip(hipSetDevice(0), "hipSetDevice layer-3 M-RoPE");
    json cases = json::array();
    bool complete = true;
    std::size_t total_elements = 0;
    std::size_t total_exact_elements = 0;
    std::string q_weight_sha256;
    std::string k_weight_sha256;
    for (const json& case_record : manifest.at("cases")) {
      const json& components = case_record.at("components");
      const std::string current_q =
          components.at("q_norm_weight").at("sha256").get<std::string>();
      const std::string current_k =
          components.at("k_norm_weight").at("sha256").get<std::string>();
      if ((!q_weight_sha256.empty() && q_weight_sha256 != current_q) ||
          (!k_weight_sha256.empty() && k_weight_sha256 != current_k)) {
        throw std::runtime_error("layer-3 head norm weights changed by case");
      }
      q_weight_sha256 = current_q;
      k_weight_sha256 = current_k;
      json result = qualify_case(case_record, oracle_root, output_root);
      complete = complete && result.at("complete").get<bool>();
      for (const char* name : {"generated_effective_cos",
                               "generated_effective_sin", "head_norm_q",
                               "head_norm_k", "generated_table_rotary_q",
                               "generated_table_rotary_k"}) {
        total_elements += result.at(name).at("elements").get<std::size_t>();
        total_exact_elements +=
            result.at(name).at("exact_elements").get<std::size_t>();
      }
      cases.push_back(std::move(result));
    }
    const json result = {
        {"schema",
         "aima-amd395-qwen36/native-vl-language-layer3-mrope-qualification-run/v1"},
        {"complete", complete},
        {"source_commit", AIMA_SOURCE_COMMIT},
        {"capture_manifest_sha256", aima::sha256_file(manifest_path)},
        {"capture_source_commit", manifest.at("source").at("commit")},
        {"case_count", cases.size()},
        {"mrope_section", {11, 11, 10}},
        {"mrope_interleaved", true},
        {"rotary_dimension", 64},
        {"q_norm_weight_sha256", q_weight_sha256},
        {"k_norm_weight_sha256", k_weight_sha256},
        {"total_elements", total_elements},
        {"total_exact_elements", total_exact_elements},
        {"all_bit_exact", total_elements == total_exact_elements},
        {"acceptance",
         {{"relative_l2_maximum", 0.002},
          {"cosine_minimum", 0.999},
          {"all_finite", true}}},
        {"runtime_python", false},
        {"runtime_numpy", false},
        {"runtime_torch", false},
        {"runtime_vllm", false},
        {"runtime_triton", false},
        {"cases", std::move(cases)},
    };
    std::cout << result.dump() << '\n';
    return complete ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native VL language layer-3 M-RoPE probe: "
              << error.what() << '\n';
    return 1;
  }
}
