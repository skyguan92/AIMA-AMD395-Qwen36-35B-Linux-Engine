// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_aot_attention.h"
#include "aima/native_vl_processor.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
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
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc AOT attention tensor");
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

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: native-vision-aot-attention-probe AOT_IMAGE QUERY "
                 "KEY VALUE EXPECTED_OUTPUT CU_SEQLENS PATCHES\n";
    return 2;
  }
  try {
    constexpr std::size_t kElementsPerPatch = 16 * 72;
    const std::size_t patches = std::stoull(argv[7]);
    if (patches == 0 ||
        patches > aima::kNativeVlVisionBatchPatchLimit) {
      throw std::invalid_argument("patch count is outside the probe domain");
    }
    const std::vector<std::uint32_t> cu_seqlens =
        parse_cu_seqlens(argv[6], patches);
    const std::size_t tensor_bytes =
        patches * kElementsPerPatch * sizeof(std::uint16_t);
    aima::NativeVisionAotAttentionPlan plan(
        std::filesystem::absolute(argv[1]), patches, cu_seqlens);
    DeviceAllocation query(tensor_bytes);
    DeviceAllocation key(tensor_bytes);
    DeviceAllocation value(tensor_bytes);
    DeviceAllocation output(tensor_bytes);
    upload_file(query.get(), std::filesystem::absolute(argv[2]), tensor_bytes,
                "hipMemcpy AOT attention query");
    upload_file(key.get(), std::filesystem::absolute(argv[3]), tensor_bytes,
                "hipMemcpy AOT attention key");
    upload_file(value.get(), std::filesystem::absolute(argv[4]), tensor_bytes,
                "hipMemcpy AOT attention value");

    plan.launch(query.get(), key.get(), value.get(), output.get());
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize AOT attention first");
    std::vector<unsigned char> first(tensor_bytes);
    check_hip(hipMemcpy(first.data(), output.get(), tensor_bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy first AOT attention output");
    plan.launch(query.get(), key.get(), value.get(), output.get());
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize AOT attention repeat");
    std::vector<unsigned char> actual(tensor_bytes);
    check_hip(hipMemcpy(actual.data(), output.get(), tensor_bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy AOT attention output");
    const std::vector<unsigned char> expected =
        read_file(std::filesystem::absolute(argv[5]), tensor_bytes);
    const std::string expected_sha256 =
        aima::sha256_bytes(expected.data(), expected.size());
    const std::string actual_sha256 =
        aima::sha256_bytes(actual.data(), actual.size());
    const std::string first_sha256 =
        aima::sha256_bytes(first.data(), first.size());
    std::size_t exact_elements = 0;
    for (std::size_t offset = 0; offset < tensor_bytes;
         offset += sizeof(std::uint16_t)) {
      if (expected[offset] == actual[offset] &&
          expected[offset + 1] == actual[offset + 1]) {
        ++exact_elements;
      }
    }
    const bool deterministic = first_sha256 == actual_sha256;
    const bool exact = actual == expected;
    const bool passed = exact && deterministic;
    std::cout << std::setprecision(17)
              << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-vision-aot-attention-oracle/v1\","
              << "\"complete\":" << (passed ? "true" : "false") << ','
              << "\"patches\":" << patches
              << ",\"segment_count\":" << plan.segment_count()
              << ",\"workspace_bytes\":" << plan.workspace_bytes()
              << ",\"elements\":"
              << tensor_bytes / sizeof(std::uint16_t)
              << ",\"exact_elements\":" << exact_elements
              << ",\"expected_sha256\":\"" << expected_sha256
              << "\",\"actual_sha256\":\"" << actual_sha256
              << "\",\"repeat_actual_sha256\":\"" << first_sha256
              << "\",\"repeat_deterministic\":"
              << (deterministic ? "true" : "false")
              << ",\"aot_image_sha256\":\"" << plan.image_sha256()
              << "\",\"exact\":" << (exact ? "true" : "false") << "}\n";
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native vision AOT attention probe: " << error.what() << '\n';
    return 1;
  }
}
