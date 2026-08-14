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
};

Comparison compare_bf16(const void* actual_device, std::size_t bytes,
                        const std::filesystem::path& expected_path) {
  const std::vector<unsigned char> expected = read_file(expected_path, bytes);
  std::vector<unsigned char> actual(bytes);
  check_hip(hipMemcpy(actual.data(), actual_device, bytes,
                      hipMemcpyDeviceToHost),
            "hipMemcpy position comparison");
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
  if (argc != 7) {
    std::cerr << "usage: native-vision-position-probe MODEL_DIR ORACLE "
                 "TEMPORAL HEIGHT WIDTH LOAD_REPORT\n";
    return 2;
  }
  void* output = nullptr;
  try {
    const aima::NativeVlGrid grid{std::stoull(argv[3]), std::stoull(argv[4]),
                                  std::stoull(argv[5])};
    if (grid.temporal == 0 || grid.height == 0 || grid.width == 0 ||
        grid.height % aima::kNativeVlMergeSize != 0 ||
        grid.width % aima::kNativeVlMergeSize != 0) {
      throw std::invalid_argument("grid is outside the probe domain");
    }
    if (grid.height > std::numeric_limits<std::size_t>::max() / grid.width) {
      throw std::invalid_argument("spatial grid overflows");
    }
    const std::size_t spatial = grid.height * grid.width;
    if (grid.temporal > std::numeric_limits<std::size_t>::max() / spatial) {
      throw std::invalid_argument("patch grid overflows");
    }
    constexpr std::size_t kVisionHidden = 1152;
    const std::size_t patches = grid.temporal * spatial;
    if (patches > aima::kNativeVlVisionBatchPatchLimit / 2) {
      throw std::invalid_argument(
          "duplicated probe grid exceeds the serving budget");
    }
    const std::size_t output_bytes =
        patches * kVisionHidden * sizeof(std::uint16_t);

    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[6]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load_visual(options);
    check_hip(hipMalloc(&output, output_bytes), "hipMalloc position output");

    aima::NativeVisionPositionPlan plan(weights, {grid});
    plan.launch(output);
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize position");
    const Comparison comparison = compare_bf16(
        output, output_bytes, std::filesystem::absolute(argv[2]));
    aima::NativeVisionPositionPlan concatenated_plan(weights, {grid, grid});
    check_hip(hipFree(output), "hipFree single position output");
    output = nullptr;
    check_hip(hipMalloc(&output, 2 * output_bytes),
              "hipMalloc concatenated position output");
    concatenated_plan.launch(output);
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize concatenated position");
    const Comparison concatenated_first = compare_bf16(
        output, output_bytes, std::filesystem::absolute(argv[2]));
    const Comparison concatenated_second = compare_bf16(
        static_cast<unsigned char*>(output) + output_bytes, output_bytes,
        std::filesystem::absolute(argv[2]));
    check_hip(hipMemset(output, 0, 2 * output_bytes),
              "hipMemset zero add input");
    concatenated_plan.launch_add(output, output);
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize position add");
    const Comparison zero_add_first = compare_bf16(
        output, output_bytes, std::filesystem::absolute(argv[2]));
    const Comparison zero_add_second = compare_bf16(
        static_cast<unsigned char*>(output) + output_bytes, output_bytes,
        std::filesystem::absolute(argv[2]));
    const std::size_t concatenated_exact =
        concatenated_first.exact_elements + concatenated_second.exact_elements;
    const std::size_t zero_add_exact =
        zero_add_first.exact_elements + zero_add_second.exact_elements;
    const bool passed = comparison.finite_elements == comparison.elements &&
                        comparison.relative_l2_error <= 0.002 &&
                        comparison.cosine_similarity >= 0.999 &&
                        concatenated_exact == 2 * comparison.elements &&
                        zero_add_exact == 2 * comparison.elements;
    std::cout << std::setprecision(17)
              << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-vision-position-oracle/v2\","
              << "\"complete\":" << (passed ? "true" : "false") << ','
              << "\"grid_thw\":[" << grid.temporal << ',' << grid.height
              << ',' << grid.width << "],"
              << "\"weight_payload_bytes\":" << load.payload_bytes << ','
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
              << ','
              << "\"concatenated_exact_elements\":" << concatenated_exact
              << ',' << "\"zero_add_exact_elements\":" << zero_add_exact
              << ','
              << "\"maximum_absolute_error\":"
              << comparison.maximum_absolute_error << ','
              << "\"relative_l2_error\":" << comparison.relative_l2_error
              << ',' << "\"cosine_similarity\":"
              << comparison.cosine_similarity << ','
              << "\"expected_sha256\":\"" << comparison.expected_sha256
              << "\",\"actual_sha256\":\"" << comparison.actual_sha256
              << "\"}\n";
    check_hip(hipFree(output), "hipFree position output");
    output = nullptr;
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    if (output != nullptr) (void)hipFree(output);
    std::cerr << "native vision position probe: " << error.what() << '\n';
    return 1;
  }
}
