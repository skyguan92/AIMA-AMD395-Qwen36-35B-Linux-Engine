// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_merger.h"
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
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc vision merger tensor");
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
  if (!stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("oracle file read failed: " + path.string());
  }
  return bytes;
}

void write_file(const std::filesystem::path& path,
                const std::vector<unsigned char>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream ||
      !stream.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("merger output write failed: " + path.string());
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
    if (expected_bits == actual_bits) ++result.exact_elements;
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
    std::cerr << "usage: native-vision-merger-probe MODEL_DIR BLOCK26_INPUT "
                 "EXPECTED_OUTPUT PATCHES LOAD_REPORT ACTUAL_OUTPUT\n";
    return 2;
  }
  try {
    constexpr std::size_t kVisionHidden = 1152;
    constexpr std::size_t kLanguageHidden = 2048;
    constexpr std::size_t kMergeArea = 4;
    const std::size_t patches = std::stoull(argv[4]);
    if (patches == 0 || patches % kMergeArea != 0 ||
        patches > aima::kNativeVlVisionBatchPatchLimit) {
      throw std::invalid_argument("merger probe patch count is invalid");
    }
    const std::size_t input_bytes =
        patches * kVisionHidden * sizeof(std::uint16_t);
    const std::size_t output_bytes =
        (patches / kMergeArea) * kLanguageHidden * sizeof(std::uint16_t);
    const std::vector<unsigned char> input =
        read_file(std::filesystem::absolute(argv[2]), input_bytes);
    const std::vector<unsigned char> expected =
        read_file(std::filesystem::absolute(argv[3]), output_bytes);

    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[5]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load_visual(options);
    aima::NativeVisionMergerPlan plan(weights, patches);
    DeviceAllocation input_device(input_bytes);
    DeviceAllocation output_device(output_bytes);
    DeviceAllocation temporary(plan.temporary_bytes());
    check_hip(hipMemcpy(input_device.get(), input.data(), input.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy vision merger input");

    plan.launch(input_device.get(), output_device.get(), temporary.get(),
                plan.temporary_bytes());
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize merger warmup");
    std::vector<unsigned char> first(output_bytes);
    check_hip(hipMemcpy(first.data(), output_device.get(), output_bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy first merger output");
    const std::string first_sha256 =
        aima::sha256_bytes(first.data(), first.size());

    Event start;
    Event stop;
    std::vector<double> measured_ms;
    measured_ms.reserve(5);
    for (std::size_t repetition = 0; repetition < 5; ++repetition) {
      check_hip(hipEventRecord(start), "hipEventRecord merger start");
      plan.launch(input_device.get(), output_device.get(), temporary.get(),
                  plan.temporary_bytes());
      check_hip(hipEventRecord(stop), "hipEventRecord merger stop");
      check_hip(hipEventSynchronize(stop), "hipEventSynchronize merger stop");
      float milliseconds = 0.0f;
      check_hip(hipEventElapsedTime(&milliseconds, start, stop),
                "hipEventElapsedTime merger");
      measured_ms.push_back(milliseconds);
    }
    std::vector<unsigned char> actual(output_bytes);
    check_hip(hipMemcpy(actual.data(), output_device.get(), output_bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy merger output");
    write_file(std::filesystem::absolute(argv[6]), actual);
    const Comparison comparison = compare_bf16(actual, expected);
    std::vector<double> sorted_ms = measured_ms;
    std::sort(sorted_ms.begin(), sorted_ms.end());
    const double median_ms = sorted_ms[sorted_ms.size() / 2];
    const bool deterministic = first_sha256 == comparison.actual_sha256;
    const bool passed = comparison.passed() && deterministic;

    std::cout << std::setprecision(17)
              << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-vision-merger-oracle/v1\","
              << "\"complete\":" << (passed ? "true" : "false")
              << ",\"patches\":" << patches
              << ",\"merged_tokens\":" << plan.merged_token_count()
              << ",\"weight_payload_bytes\":" << load.payload_bytes
              << ",\"temporary_bytes\":" << plan.temporary_bytes()
              << ",\"library_workspace_bytes\":"
              << plan.library_workspace_bytes() << ",\"measured_ms\":[";
    for (std::size_t index = 0; index < measured_ms.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << measured_ms[index];
    }
    std::cout << "],\"median_ms\":" << median_ms
              << ",\"elements\":" << comparison.elements
              << ",\"exact_elements\":" << comparison.exact_elements
              << ",\"finite_elements\":" << comparison.finite_elements
              << ",\"maximum_absolute_error\":"
              << comparison.maximum_absolute_error
              << ",\"relative_l2_error\":" << comparison.relative_l2_error
              << ",\"cosine_similarity\":" << comparison.cosine_similarity
              << ",\"expected_sha256\":\"" << comparison.expected_sha256
              << "\",\"actual_sha256\":\"" << comparison.actual_sha256
              << "\",\"repeat_actual_sha256\":\"" << first_sha256
              << "\",\"repeat_deterministic\":"
              << (deterministic ? "true" : "false") << "}\n";
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native vision merger probe: " << error.what() << '\n';
    return 1;
  }
}
