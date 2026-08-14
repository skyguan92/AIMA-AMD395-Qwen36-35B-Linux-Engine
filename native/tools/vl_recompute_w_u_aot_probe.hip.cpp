// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/aot_kernel.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kPromptTokens = 131;
constexpr std::size_t kBucketTokens = 1024;
constexpr std::size_t kKeyHeads = 16;
constexpr std::size_t kValueHeads = 32;
constexpr std::size_t kKey = 128;
constexpr std::size_t kValue = 128;
constexpr std::size_t kChunk = 64;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

template <typename T>
std::vector<T> read_exact(const std::filesystem::path& path,
                          std::size_t elements) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() !=
                     static_cast<std::streamoff>(elements * sizeof(T))) {
    throw std::runtime_error("raw tensor size mismatch: " + path.string());
  }
  std::vector<T> values(elements);
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(elements * sizeof(T)))) {
    throw std::runtime_error("raw tensor read failed: " + path.string());
  }
  return values;
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      static_cast<void>(hipFree(pointer_));
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

template <typename T>
void copy_padded_prefix(const std::filesystem::path& path,
                        std::size_t prefix_elements,
                        DeviceBuffer& destination) {
  const std::vector<T> prefix = read_exact<T>(path, prefix_elements);
  std::vector<T> padded(destination.bytes() / sizeof(T), T{});
  if (prefix.size() > padded.size()) {
    throw std::runtime_error("raw tensor exceeds padded destination");
  }
  std::copy(prefix.begin(), prefix.end(), padded.begin());
  check_hip(hipMemcpy(destination.get(), padded.data(), destination.bytes(),
                      hipMemcpyHostToDevice),
            "hipMemcpy padded tensor");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: vl-recompute-w-u-aot-probe IMAGE K V BETA A G "
                 "EXPECTED_U\n";
    return 2;
  }
  try {
    const std::size_t key_elements =
        kBucketTokens * kKeyHeads * kKey;
    const std::size_t value_elements =
        kBucketTokens * kValueHeads * kValue;
    const std::size_t gate_elements = kBucketTokens * kValueHeads;
    const std::size_t matrix_elements =
        kBucketTokens * kValueHeads * kChunk;
    DeviceBuffer key(key_elements * sizeof(std::uint16_t));
    DeviceBuffer value(value_elements * sizeof(std::uint16_t));
    DeviceBuffer beta(gate_elements * sizeof(float));
    DeviceBuffer w(value_elements * sizeof(std::uint16_t));
    DeviceBuffer u(value_elements * sizeof(std::uint16_t));
    DeviceBuffer matrix(matrix_elements * sizeof(std::uint16_t));
    DeviceBuffer g(gate_elements * sizeof(float));
    copy_padded_prefix<std::uint16_t>(
        argv[2], kPromptTokens * kKeyHeads * kKey, key);
    copy_padded_prefix<std::uint16_t>(
        argv[3], kPromptTokens * kValueHeads * kValue, value);
    copy_padded_prefix<float>(
        argv[4], kPromptTokens * kValueHeads, beta);
    copy_padded_prefix<std::uint16_t>(
        argv[5], kPromptTokens * kValueHeads * kChunk, matrix);
    copy_padded_prefix<float>(argv[6], kPromptTokens * kValueHeads, g);
    check_hip(hipMemset(w.get(), 0, w.bytes()), "hipMemset W");
    check_hip(hipMemset(u.get(), 0, u.bytes()), "hipMemset U");

    void* key_pointer = key.get();
    void* value_pointer = value.get();
    void* beta_pointer = beta.get();
    void* w_pointer = w.get();
    void* u_pointer = u.get();
    void* matrix_pointer = matrix.get();
    void* g_pointer = g.get();
    std::int32_t tokens = static_cast<std::int32_t>(kBucketTokens);
    std::vector<void*> parameters = {
        &key_pointer, &value_pointer, &beta_pointer, &w_pointer,
        &u_pointer,   &matrix_pointer, &g_pointer,  &tokens,
    };
    aima::AotKernel kernel = aima::AotKernel::from_file(
        argv[1], "recompute_w_u_fwd_kernel");
    kernel.launch(aima::AotLaunchConfig{16, 32, 1, 4, 32, 8192},
                  parameters);
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

    const std::vector<std::uint16_t> expected = read_exact<std::uint16_t>(
        argv[7], kPromptTokens * kValueHeads * kValue);
    std::vector<std::uint16_t> actual(expected.size());
    check_hip(hipMemcpy(actual.data(), u.get(),
                        actual.size() * sizeof(actual[0]),
                        hipMemcpyDeviceToHost),
              "hipMemcpy U output");
    std::size_t exact = 0;
    std::size_t first = actual.size();
    for (std::size_t index = 0; index < actual.size(); ++index) {
      if (actual[index] == expected[index]) {
        ++exact;
      } else if (first == actual.size()) {
        first = index;
      }
    }
    std::cout << "{\"elements\":" << actual.size()
              << ",\"exact_elements\":" << exact
              << ",\"first_mismatch_index\":";
    if (first == actual.size()) {
      std::cout << "null";
    } else {
      std::cout << first;
    }
    std::cout << ",\"passed\":"
              << (exact == actual.size() ? "true" : "false") << "}\n";
    return exact == actual.size() ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "VL recompute W/U AOT probe: " << error.what() << '\n';
    return 1;
  }
}
