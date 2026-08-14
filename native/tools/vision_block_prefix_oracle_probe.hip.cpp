// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_encoder.h"
#include "aima/native_weight_store.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

std::vector<unsigned char> read_file(const std::filesystem::path& path,
                                     std::size_t expected_bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() < 0 ||
      static_cast<std::size_t>(stream.tellg()) != expected_bytes) {
    throw std::runtime_error("oracle file size mismatch: " + path.string());
  }
  std::vector<unsigned char> bytes(expected_bytes);
  stream.seekg(0);
  if (expected_bytes != 0 &&
      !stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(expected_bytes))) {
    throw std::runtime_error("oracle file read failed: " + path.string());
  }
  return bytes;
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

Comparison compare_bf16(const void* actual_device, std::size_t bytes,
                        const std::filesystem::path& expected_path,
                        const char* copy_description) {
  const std::vector<unsigned char> expected = read_file(expected_path, bytes);
  std::vector<unsigned char> actual(bytes);
  check_hip(hipMemcpy(actual.data(), actual_device, bytes,
                      hipMemcpyDeviceToHost),
            copy_description);
  Comparison result;
  result.elements = bytes / sizeof(std::uint16_t);
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

void print_comparison(const char* name, const Comparison& comparison) {
  std::cout << '"' << name << "\":{";
  std::cout << "\"passed\":" << (comparison.passed() ? "true" : "false")
            << ',' << "\"elements\":" << comparison.elements << ','
            << "\"exact_elements\":" << comparison.exact_elements << ','
            << "\"finite_elements\":" << comparison.finite_elements << ','
            << "\"first_mismatch_index\":"
            << (comparison.first_mismatch_index ==
                        std::numeric_limits<std::size_t>::max()
                    ? -1LL
                    : static_cast<long long>(
                          comparison.first_mismatch_index))
            << ',' << "\"first_expected_bits\":"
            << comparison.first_expected_bits << ','
            << "\"first_actual_bits\":" << comparison.first_actual_bits
            << ',' << "\"maximum_absolute_error\":"
            << comparison.maximum_absolute_error << ','
            << "\"relative_l2_error\":" << comparison.relative_l2_error
            << ',' << "\"cosine_similarity\":"
            << comparison.cosine_similarity << ','
            << "\"expected_sha256\":\"" << comparison.expected_sha256
            << "\",\"actual_sha256\":\"" << comparison.actual_sha256
            << "\"}";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: native-vision-block-prefix-probe MODEL_DIR "
                 "BLOCK_INPUT NORM1_ORACLE QKV_ORACLE PATCHES LOAD_REPORT\n";
    return 2;
  }
  void* input = nullptr;
  void* norm1_output = nullptr;
  void* qkv_output = nullptr;
  try {
    constexpr std::size_t kVisionHidden = 1152;
    constexpr std::size_t kVisionQkvHidden = 3456;
    const std::size_t patches = std::stoull(argv[5]);
    if (patches == 0 ||
        patches > aima::kNativeVlVisionBatchPatchLimit) {
      throw std::invalid_argument("patch count is outside the probe domain");
    }
    const std::size_t input_bytes =
        patches * kVisionHidden * sizeof(std::uint16_t);
    const std::size_t qkv_bytes =
        patches * kVisionQkvHidden * sizeof(std::uint16_t);

    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[6]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load_visual(options);
    check_hip(hipMalloc(&input, input_bytes), "hipMalloc block input");
    check_hip(hipMalloc(&norm1_output, input_bytes),
              "hipMalloc norm1 output");
    check_hip(hipMalloc(&qkv_output, qkv_bytes), "hipMalloc QKV output");
    const std::vector<unsigned char> input_host =
        read_file(std::filesystem::absolute(argv[2]), input_bytes);
    check_hip(hipMemcpy(input, input_host.data(), input_bytes,
                        hipMemcpyHostToDevice),
              "hipMemcpy block input");

    aima::NativeVisionBlockPrefixPlan plan(weights, 0, patches);
    plan.launch(input, norm1_output, qkv_output);
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize vision block prefix");
    const Comparison norm1 = compare_bf16(
        norm1_output, input_bytes, std::filesystem::absolute(argv[3]),
        "hipMemcpy norm1 comparison");
    const Comparison qkv = compare_bf16(
        qkv_output, qkv_bytes, std::filesystem::absolute(argv[4]),
        "hipMemcpy QKV comparison");
    const bool passed = norm1.passed() && qkv.passed();
    std::cout << std::setprecision(17)
              << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-vision-block-prefix-oracle/v1\","
              << "\"complete\":" << (passed ? "true" : "false") << ','
              << "\"block_index\":0,\"patches\":" << patches << ','
              << "\"weight_payload_bytes\":" << load.payload_bytes << ','
              << "\"workspace_bytes\":" << plan.workspace_bytes() << ','
              << "\"comparisons\":{";
    print_comparison("norm1", norm1);
    std::cout << ',';
    print_comparison("qkv_linear", qkv);
    std::cout << "}}\n";

    check_hip(hipFree(qkv_output), "hipFree QKV output");
    qkv_output = nullptr;
    check_hip(hipFree(norm1_output), "hipFree norm1 output");
    norm1_output = nullptr;
    check_hip(hipFree(input), "hipFree block input");
    input = nullptr;
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    if (qkv_output != nullptr) (void)hipFree(qkv_output);
    if (norm1_output != nullptr) (void)hipFree(norm1_output);
    if (input != nullptr) (void)hipFree(input);
    std::cerr << "native vision block prefix probe: " << error.what() << '\n';
    return 1;
  }
}
