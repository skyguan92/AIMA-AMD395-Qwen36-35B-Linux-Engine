// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_block.h"
#include "aima/native_vl_processor.h"
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
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc block probe tensor");
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

class Event {
 public:
  Event() { check_hip(hipEventCreate(&event_), "hipEventCreate"); }
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

Comparison compare_bf16(const std::vector<unsigned char>& actual,
                        const std::filesystem::path& expected_path) {
  const std::vector<unsigned char> expected =
      read_file(expected_path, actual.size());
  Comparison result;
  result.elements = actual.size() / sizeof(std::uint16_t);
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
  if (argc != 9) {
    std::cerr << "usage: native-vision-block-probe MODEL_DIR BLOCK_INPUT COS "
                 "SIN BLOCK_OUTPUT CU_SEQLENS PATCHES LOAD_REPORT\n";
    return 2;
  }
  try {
    constexpr std::size_t kVisionHidden = 1152;
    constexpr std::size_t kRotaryHalfDimension = 36;
    const std::size_t patches = std::stoull(argv[7]);
    if (patches == 0 ||
        patches > aima::kNativeVlVisionBatchPatchLimit) {
      throw std::invalid_argument("patch count is outside the probe domain");
    }
    const std::vector<std::uint32_t> cu_seqlens =
        parse_cu_seqlens(argv[6], patches);
    const std::size_t hidden_bytes =
        patches * kVisionHidden * sizeof(std::uint16_t);
    const std::size_t cos_sin_bytes =
        patches * kRotaryHalfDimension * sizeof(std::uint16_t);

    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[8]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load_visual(options);
    aima::NativeVisionBlockPlan plan(weights, 0, patches, cu_seqlens);
    DeviceAllocation input(hidden_bytes);
    DeviceAllocation cosine(cos_sin_bytes);
    DeviceAllocation sine(cos_sin_bytes);
    DeviceAllocation output(hidden_bytes);
    DeviceAllocation temporary(plan.temporary_bytes());
    upload_file(input.get(), std::filesystem::absolute(argv[2]), hidden_bytes,
                "hipMemcpy block input");
    upload_file(cosine.get(), std::filesystem::absolute(argv[3]),
                cos_sin_bytes, "hipMemcpy rotary cosine");
    upload_file(sine.get(), std::filesystem::absolute(argv[4]), cos_sin_bytes,
                "hipMemcpy rotary sine");

    plan.launch(input.get(), cosine.get(), sine.get(), output.get(),
                temporary.get(), plan.temporary_bytes());
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize block warmup");
    std::vector<unsigned char> first_output(hidden_bytes);
    check_hip(hipMemcpy(first_output.data(), output.get(), hidden_bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy first block output");
    const std::string first_sha256 =
        aima::sha256_bytes(first_output.data(), first_output.size());

    Event start;
    Event stop;
    std::vector<double> measured_ms;
    measured_ms.reserve(7);
    for (std::size_t repetition = 0; repetition < 7; ++repetition) {
      check_hip(hipEventRecord(start), "hipEventRecord block start");
      plan.launch(input.get(), cosine.get(), sine.get(), output.get(),
                  temporary.get(), plan.temporary_bytes());
      check_hip(hipEventRecord(stop), "hipEventRecord block stop");
      check_hip(hipEventSynchronize(stop), "hipEventSynchronize block stop");
      float milliseconds = 0.0f;
      check_hip(hipEventElapsedTime(&milliseconds, start, stop),
                "hipEventElapsedTime block");
      measured_ms.push_back(milliseconds);
    }
    std::vector<unsigned char> actual(hidden_bytes);
    check_hip(hipMemcpy(actual.data(), output.get(), hidden_bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy block output");
    const Comparison comparison = compare_bf16(
        actual, std::filesystem::absolute(argv[5]));
    std::vector<double> sorted_ms = measured_ms;
    std::sort(sorted_ms.begin(), sorted_ms.end());
    const double median_ms = sorted_ms[sorted_ms.size() / 2];
    const bool deterministic = first_sha256 == comparison.actual_sha256;
    const bool passed = comparison.passed() && deterministic;

    std::cout << std::setprecision(17)
              << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-vision-block-oracle/v1\","
              << "\"complete\":" << (passed ? "true" : "false") << ','
              << "\"block_index\":0,\"patches\":" << patches << ','
              << "\"cu_seqlens\":[";
    for (std::size_t index = 0; index < cu_seqlens.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << cu_seqlens[index];
    }
    std::cout << "],\"weight_payload_bytes\":" << load.payload_bytes << ','
              << "\"temporary_bytes\":" << plan.temporary_bytes() << ','
              << "\"library_workspace_bytes\":"
              << plan.library_workspace_bytes() << ','
              << "\"measured_ms\":[";
    for (std::size_t index = 0; index < measured_ms.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << measured_ms[index];
    }
    std::cout << "],\"median_ms\":" << median_ms << ','
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
              << "\",\"repeat_actual_sha256\":\"" << first_sha256
              << "\",\"repeat_deterministic\":"
              << (deterministic ? "true" : "false") << "}\n";
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native vision block probe: " << error.what() << '\n';
    return 1;
  }
}
