// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_segmented_attention.h"
#include "aima/native_vl_processor.h"
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

std::vector<std::uint32_t> parse_cu_seqlens(const std::string& text,
                                            std::size_t patches) {
  std::vector<std::uint32_t> values;
  std::size_t offset = 0;
  while (offset < text.size()) {
    const std::size_t comma = text.find(',', offset);
    const std::string part = text.substr(offset, comma - offset);
    if (part.empty()) {
      throw std::invalid_argument("cu_seqlens contains an empty value");
    }
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(part, &consumed);
    if (consumed != part.size() ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("cu_seqlens value is invalid");
    }
    values.push_back(static_cast<std::uint32_t>(parsed));
    if (comma == std::string::npos) break;
    offset = comma + 1;
  }
  if (values.size() < 2 || values.front() != 0 ||
      values.back() != patches) {
    throw std::invalid_argument("cu_seqlens does not span the patch tensor");
  }
  return values;
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
                        const std::filesystem::path& expected_path) {
  const std::vector<unsigned char> expected = read_file(expected_path, bytes);
  std::vector<unsigned char> actual(bytes);
  check_hip(hipMemcpy(actual.data(), actual_device, bytes,
                      hipMemcpyDeviceToHost),
            "hipMemcpy attention comparison");
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

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: native-vision-segmented-attention-probe QUERY KEY "
                 "VALUE ATTENTION_ORACLE CU_SEQLENS PATCHES CASE_ID\n";
    return 2;
  }
  void* query = nullptr;
  void* key = nullptr;
  void* value = nullptr;
  void* output = nullptr;
  try {
    constexpr std::size_t kVisionHidden = 1152;
    const std::size_t patches = std::stoull(argv[6]);
    if (patches == 0 || patches > 4 * aima::kNativeVlAggregateTokenLimit) {
      throw std::invalid_argument("patch count is outside the probe domain");
    }
    const std::vector<std::uint32_t> cu_seqlens =
        parse_cu_seqlens(argv[5], patches);
    if (std::string(argv[7]).empty()) {
      throw std::invalid_argument("case id is required");
    }
    const std::size_t tensor_bytes =
        patches * kVisionHidden * sizeof(std::uint16_t);
    const std::vector<unsigned char> query_host =
        read_file(std::filesystem::absolute(argv[1]), tensor_bytes);
    const std::vector<unsigned char> key_host =
        read_file(std::filesystem::absolute(argv[2]), tensor_bytes);
    const std::vector<unsigned char> value_host =
        read_file(std::filesystem::absolute(argv[3]), tensor_bytes);
    check_hip(hipMalloc(&query, tensor_bytes), "hipMalloc attention query");
    check_hip(hipMalloc(&key, tensor_bytes), "hipMalloc attention key");
    check_hip(hipMalloc(&value, tensor_bytes), "hipMalloc attention value");
    check_hip(hipMalloc(&output, tensor_bytes), "hipMalloc attention output");
    check_hip(hipMemcpy(query, query_host.data(), tensor_bytes,
                        hipMemcpyHostToDevice),
              "hipMemcpy attention query");
    check_hip(hipMemcpy(key, key_host.data(), tensor_bytes,
                        hipMemcpyHostToDevice),
              "hipMemcpy attention key");
    check_hip(hipMemcpy(value, value_host.data(), tensor_bytes,
                        hipMemcpyHostToDevice),
              "hipMemcpy attention value");

    aima::NativeVisionSegmentedAttentionPlan plan(patches, cu_seqlens);
    plan.launch(query, key, value, output);
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize segmented attention");
    const Comparison comparison = compare_bf16(
        output, tensor_bytes, std::filesystem::absolute(argv[4]));

    const bool isolation_applicable = cu_seqlens.size() > 2;
    std::size_t isolation_elements = 0;
    std::size_t isolation_exact_elements = 0;
    if (isolation_applicable) {
      const std::size_t first_segment_bytes =
          static_cast<std::size_t>(cu_seqlens[1]) * kVisionHidden *
          sizeof(std::uint16_t);
      std::vector<unsigned char> baseline(first_segment_bytes);
      std::vector<unsigned char> isolated(first_segment_bytes);
      check_hip(hipMemcpy(baseline.data(), output, first_segment_bytes,
                          hipMemcpyDeviceToHost),
                "hipMemcpy attention isolation baseline");
      const std::size_t tail_bytes = tensor_bytes - first_segment_bytes;
      check_hip(hipMemset(static_cast<unsigned char*>(query) +
                              first_segment_bytes,
                          0, tail_bytes),
                "hipMemset attention isolation query");
      check_hip(hipMemset(static_cast<unsigned char*>(key) +
                              first_segment_bytes,
                          0, tail_bytes),
                "hipMemset attention isolation key");
      check_hip(hipMemset(static_cast<unsigned char*>(value) +
                              first_segment_bytes,
                          0, tail_bytes),
                "hipMemset attention isolation value");
      plan.launch(query, key, value, output);
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize attention isolation");
      check_hip(hipMemcpy(isolated.data(), output, first_segment_bytes,
                          hipMemcpyDeviceToHost),
                "hipMemcpy attention isolation output");
      isolation_elements = first_segment_bytes / sizeof(std::uint16_t);
      for (std::size_t index = 0; index < isolation_elements; ++index) {
        std::uint16_t before = 0;
        std::uint16_t after = 0;
        std::memcpy(&before,
                    baseline.data() + index * sizeof(std::uint16_t),
                    sizeof(before));
        std::memcpy(&after,
                    isolated.data() + index * sizeof(std::uint16_t),
                    sizeof(after));
        isolation_exact_elements += before == after ? 1 : 0;
      }
    }
    const bool isolation_passed =
        !isolation_applicable || isolation_exact_elements == isolation_elements;
    const bool passed = comparison.passed() && isolation_passed;
    std::cout << std::setprecision(17)
              << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-vision-segmented-attention-oracle/v1\","
              << "\"complete\":" << (passed ? "true" : "false") << ','
              << "\"patches\":" << patches << ','
              << "\"case_id\":\"" << argv[7] << "\","
              << "\"cu_seqlens\":[";
    for (std::size_t index = 0; index < cu_seqlens.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << cu_seqlens[index];
    }
    std::cout << "],\"segment_count\":" << plan.segment_count() << ','
              << "\"workspace_bytes\":" << plan.workspace_bytes() << ','
              << "\"elements\":" << comparison.elements << ','
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
              << "\",\"segment_isolation_applicable\":"
              << (isolation_applicable ? "true" : "false") << ','
              << "\"segment_isolation_elements\":" << isolation_elements
              << ',' << "\"segment_isolation_exact_elements\":"
              << isolation_exact_elements << ','
              << "\"segment_isolation_passed\":"
              << (isolation_passed ? "true" : "false") << "}\n";

    check_hip(hipFree(output), "hipFree attention output");
    output = nullptr;
    check_hip(hipFree(value), "hipFree attention value");
    value = nullptr;
    check_hip(hipFree(key), "hipFree attention key");
    key = nullptr;
    check_hip(hipFree(query), "hipFree attention query");
    query = nullptr;
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    if (output != nullptr) (void)hipFree(output);
    if (value != nullptr) (void)hipFree(value);
    if (key != nullptr) (void)hipFree(key);
    if (query != nullptr) (void)hipFree(query);
    std::cerr << "native vision segmented attention probe: " << error.what()
              << '\n';
    return 1;
  }
}
