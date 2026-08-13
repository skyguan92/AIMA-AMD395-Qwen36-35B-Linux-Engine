// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_block_suffix.h"
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

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) {
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc suffix probe tensor");
  }
  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      const hipError_t ignored = hipFree(pointer_);
      static_cast<void>(ignored);
    }
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  void* get() const { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

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

void upload_file(void* device, const std::filesystem::path& path,
                 std::size_t bytes, const char* operation) {
  const std::vector<unsigned char> host = read_file(path, bytes);
  check_hip(hipMemcpy(device, host.data(), bytes, hipMemcpyHostToDevice),
            operation);
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
                        const char* operation) {
  const std::vector<unsigned char> expected = read_file(expected_path, bytes);
  std::vector<unsigned char> actual(bytes);
  check_hip(hipMemcpy(actual.data(), actual_device, bytes,
                      hipMemcpyDeviceToHost),
            operation);
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

struct NamedComparison {
  const char* name;
  Comparison comparison;
};

void print_comparison(const NamedComparison& named) {
  const Comparison& value = named.comparison;
  std::cout << '"' << named.name << "\":{";
  std::cout << "\"passed\":" << (value.passed() ? "true" : "false") << ','
            << "\"elements\":" << value.elements << ','
            << "\"exact_elements\":" << value.exact_elements << ','
            << "\"finite_elements\":" << value.finite_elements << ','
            << "\"first_mismatch_index\":"
            << (value.first_mismatch_index ==
                        std::numeric_limits<std::size_t>::max()
                    ? -1LL
                    : static_cast<long long>(value.first_mismatch_index))
            << ',' << "\"first_expected_bits\":"
            << value.first_expected_bits << ','
            << "\"first_actual_bits\":" << value.first_actual_bits << ','
            << "\"maximum_absolute_error\":"
            << value.maximum_absolute_error << ','
            << "\"relative_l2_error\":" << value.relative_l2_error << ','
            << "\"cosine_similarity\":" << value.cosine_similarity << ','
            << "\"expected_sha256\":\"" << value.expected_sha256
            << "\",\"actual_sha256\":\"" << value.actual_sha256
            << "\"}";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 13) {
    std::cerr
        << "usage: native-vision-block-suffix-probe MODEL_DIR BLOCK_INPUT "
           "ATTENTION ATTENTION_PROJECTION ATTENTION_RESIDUAL NORM2 MLP_FC1 "
           "MLP_ACTIVATION MLP_FC2 BLOCK_OUTPUT PATCHES LOAD_REPORT\n";
    return 2;
  }
  try {
    constexpr std::size_t kVisionHidden = 1152;
    constexpr std::size_t kVisionIntermediate = 4304;
    const std::size_t patches = std::stoull(argv[11]);
    if (patches == 0 || patches > 65536) {
      throw std::invalid_argument("patch count is outside the probe domain");
    }
    const std::size_t hidden_bytes =
        patches * kVisionHidden * sizeof(std::uint16_t);
    const std::size_t intermediate_bytes =
        patches * kVisionIntermediate * sizeof(std::uint16_t);

    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[12]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load_visual(options);
    DeviceAllocation block_input(hidden_bytes);
    DeviceAllocation attention(hidden_bytes);
    DeviceAllocation attention_projection(hidden_bytes);
    DeviceAllocation attention_residual(hidden_bytes);
    DeviceAllocation norm2(hidden_bytes);
    DeviceAllocation mlp_fc1(intermediate_bytes);
    DeviceAllocation mlp_activation(intermediate_bytes);
    DeviceAllocation mlp_fc2(hidden_bytes);
    DeviceAllocation block_output(hidden_bytes);
    upload_file(block_input.get(), std::filesystem::absolute(argv[2]),
                hidden_bytes, "hipMemcpy block input");
    upload_file(attention.get(), std::filesystem::absolute(argv[3]),
                hidden_bytes, "hipMemcpy attention input");

    aima::NativeVisionBlockSuffixPlan plan(weights, 0, patches);
    std::vector<NamedComparison> comparisons;
    comparisons.reserve(8);
    plan.launch_attention_projection(attention.get(),
                                     attention_projection.get());
    comparisons.push_back(NamedComparison{
        "attention_projection",
        compare_bf16(attention_projection.get(), hidden_bytes,
                     std::filesystem::absolute(argv[4]),
                     "hipMemcpy attention projection comparison")});

    upload_file(attention_projection.get(), std::filesystem::absolute(argv[4]),
                hidden_bytes, "hipMemcpy oracle attention projection");
    plan.launch_residual(block_input.get(), attention_projection.get(),
                         attention_residual.get());
    comparisons.push_back(NamedComparison{
        "attention_residual",
        compare_bf16(attention_residual.get(), hidden_bytes,
                     std::filesystem::absolute(argv[5]),
                     "hipMemcpy attention residual comparison")});

    upload_file(attention_residual.get(), std::filesystem::absolute(argv[5]),
                hidden_bytes, "hipMemcpy oracle attention residual");
    plan.launch_norm2(attention_residual.get(), norm2.get());
    comparisons.push_back(NamedComparison{
        "norm2", compare_bf16(norm2.get(), hidden_bytes,
                              std::filesystem::absolute(argv[6]),
                              "hipMemcpy norm2 comparison")});

    upload_file(norm2.get(), std::filesystem::absolute(argv[6]), hidden_bytes,
                "hipMemcpy oracle norm2");
    plan.launch_mlp_fc1(norm2.get(), mlp_fc1.get());
    comparisons.push_back(NamedComparison{
        "mlp_fc1", compare_bf16(mlp_fc1.get(), intermediate_bytes,
                                std::filesystem::absolute(argv[7]),
                                "hipMemcpy MLP FC1 comparison")});

    upload_file(mlp_fc1.get(), std::filesystem::absolute(argv[7]),
                intermediate_bytes, "hipMemcpy oracle MLP FC1");
    plan.launch_gelu(mlp_fc1.get(), mlp_activation.get());
    comparisons.push_back(NamedComparison{
        "mlp_activation",
        compare_bf16(mlp_activation.get(), intermediate_bytes,
                     std::filesystem::absolute(argv[8]),
                     "hipMemcpy MLP activation comparison")});

    upload_file(mlp_activation.get(), std::filesystem::absolute(argv[8]),
                intermediate_bytes, "hipMemcpy oracle MLP activation");
    plan.launch_mlp_fc2(mlp_activation.get(), mlp_fc2.get());
    comparisons.push_back(NamedComparison{
        "mlp_fc2", compare_bf16(mlp_fc2.get(), hidden_bytes,
                                std::filesystem::absolute(argv[9]),
                                "hipMemcpy MLP FC2 comparison")});

    upload_file(attention_residual.get(), std::filesystem::absolute(argv[5]),
                hidden_bytes, "hipMemcpy oracle final residual input");
    upload_file(mlp_fc2.get(), std::filesystem::absolute(argv[9]), hidden_bytes,
                "hipMemcpy oracle MLP FC2 residual input");
    plan.launch_residual(attention_residual.get(), mlp_fc2.get(),
                         block_output.get());
    comparisons.push_back(NamedComparison{
        "block_output_isolated",
        compare_bf16(block_output.get(), hidden_bytes,
                     std::filesystem::absolute(argv[10]),
                     "hipMemcpy isolated block output comparison")});

    plan.launch(block_input.get(), attention.get(), attention_projection.get(),
                attention_residual.get(), norm2.get(), mlp_fc1.get(),
                mlp_activation.get(), mlp_fc2.get(), block_output.get());
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize chained block suffix");
    comparisons.push_back(NamedComparison{
        "block_output_chained",
        compare_bf16(block_output.get(), hidden_bytes,
                     std::filesystem::absolute(argv[10]),
                     "hipMemcpy chained block output comparison")});

    const bool passed = std::all_of(
        comparisons.begin(), comparisons.end(),
        [](const NamedComparison& item) { return item.comparison.passed(); });
    std::cout << std::setprecision(17)
              << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-vision-block-suffix-oracle/v1\","
              << "\"complete\":" << (passed ? "true" : "false") << ','
              << "\"block_index\":0,\"patches\":" << patches << ','
              << "\"weight_payload_bytes\":" << load.payload_bytes << ','
              << "\"workspace_bytes\":" << plan.workspace_bytes() << ','
              << "\"comparisons\":{";
    for (std::size_t index = 0; index < comparisons.size(); ++index) {
      if (index != 0) std::cout << ',';
      print_comparison(comparisons[index]);
    }
    std::cout << "}}\n";
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native vision block suffix probe: " << error.what() << '\n';
    return 1;
  }
}
