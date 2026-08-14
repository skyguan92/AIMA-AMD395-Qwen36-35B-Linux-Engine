// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_exact_layer_norm.h"
#include "aima/native_vl_processor.h"
#include "aima/native_weight_store.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
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
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc exact LayerNorm tensor");
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
  if (!stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("oracle file read failed: " + path.string());
  }
  return bytes;
}

std::string block_tensor_name(std::size_t block_index,
                              std::string_view suffix) {
  return "model.visual.blocks." + std::to_string(block_index) + "." +
         std::string(suffix);
}

const aima::NativeTensorView& require_tensor(
    const aima::NativeWeightStore& weights, std::string_view name,
    std::uint64_t expected_bytes) {
  const aima::NativeTensorView* tensor = weights.find(name);
  if (tensor == nullptr || tensor->device_pointer == nullptr ||
      tensor->payload_bytes != expected_bytes) {
    throw std::runtime_error("exact LayerNorm weight mismatch");
  }
  return *tensor;
}

struct Result {
  std::size_t exact_elements = 0;
  std::string actual_sha256;
  bool exact = false;
};

Result compare(const void* output_device,
               const std::vector<unsigned char>& expected) {
  std::vector<unsigned char> actual(expected.size());
  check_hip(hipMemcpy(actual.data(), output_device, actual.size(),
                      hipMemcpyDeviceToHost),
            "hipMemcpy exact LayerNorm output");
  Result result;
  result.actual_sha256 = aima::sha256_bytes(actual.data(), actual.size());
  for (std::size_t offset = 0; offset < actual.size();
       offset += sizeof(std::uint16_t)) {
    if (actual[offset] == expected[offset] &&
        actual[offset + 1] == expected[offset + 1]) {
      ++result.exact_elements;
    }
  }
  result.exact = actual == expected;
  return result;
}

void print_result(const char* name, const Result& result) {
  std::cout << '"' << name << "\":{\"exact\":"
            << (result.exact ? "true" : "false")
            << ",\"exact_elements\":" << result.exact_elements
            << ",\"actual_sha256\":\"" << result.actual_sha256 << "\"}";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: native-vision-exact-layer-norm-probe MODEL_DIR "
                 "BLOCK_INDEX INPUT EXPECTED_OUTPUT ROWS LOAD_REPORT\n";
    return 2;
  }
  try {
    constexpr std::size_t kVisionBlockCount = 27;
    constexpr std::size_t kVisionHidden = 1152;
    const std::size_t block_index = std::stoull(argv[2]);
    const std::size_t rows = std::stoull(argv[5]);
    if (block_index >= kVisionBlockCount || rows == 0 ||
        rows > aima::kNativeVlVisionBatchPatchLimit) {
      throw std::invalid_argument("exact LayerNorm probe shape is invalid");
    }
    const std::size_t tensor_bytes =
        rows * kVisionHidden * sizeof(std::uint16_t);
    const std::size_t affine_bytes =
        kVisionHidden * sizeof(std::uint16_t);
    const std::vector<unsigned char> input =
        read_file(std::filesystem::absolute(argv[3]), tensor_bytes);
    const std::vector<unsigned char> expected =
        read_file(std::filesystem::absolute(argv[4]), tensor_bytes);

    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[6]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load_visual(options);
    const aima::NativeTensorView& weight = require_tensor(
        weights, block_tensor_name(block_index, "norm1.weight"), affine_bytes);
    const aima::NativeTensorView& bias = require_tensor(
        weights, block_tensor_name(block_index, "norm1.bias"), affine_bytes);
    DeviceAllocation input_device(tensor_bytes);
    DeviceAllocation division_output(tensor_bytes);
    DeviceAllocation fast_output(tensor_bytes);
    check_hip(hipMemcpy(input_device.get(), input.data(), input.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy exact LayerNorm input");

    aima::NativeVisionExactLayerNormPlan division(
        rows, aima::NativeVisionLayerNormReciprocal::kDivision);
    aima::NativeVisionExactLayerNormPlan fast(
        rows, aima::NativeVisionLayerNormReciprocal::kFastAmdReciprocal);
    division.launch(input_device.get(), weight.device_pointer,
                    bias.device_pointer, division_output.get());
    fast.launch(input_device.get(), weight.device_pointer, bias.device_pointer,
                fast_output.get());
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize exact LayerNorm candidates");
    const Result division_result = compare(division_output.get(), expected);
    const Result fast_result = compare(fast_output.get(), expected);
    const bool complete = fast_result.exact;
    const std::string expected_sha256 =
        aima::sha256_bytes(expected.data(), expected.size());
    std::cout << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-vision-exact-layer-norm-oracle/v1\","
              << "\"complete\":" << (complete ? "true" : "false")
              << ",\"block_index\":" << block_index
              << ",\"rows\":" << rows
              << ",\"elements\":" << tensor_bytes / sizeof(std::uint16_t)
              << ",\"weight_payload_bytes\":" << load.payload_bytes
              << ",\"selected_mode\":\"fast_amd_reciprocal\""
              << ",\"expected_sha256\":\"" << expected_sha256
              << "\",\"candidates\":{";
    print_result("division", division_result);
    std::cout << ',';
    print_result("fast_amd_reciprocal", fast_result);
    std::cout << "}}\n";
    return complete ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native vision exact LayerNorm probe: " << error.what()
              << '\n';
    return 1;
  }
}
