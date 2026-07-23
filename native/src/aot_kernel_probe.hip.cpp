// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/aot_kernel.h"

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
  }
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) {
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc");
  }
  ~DeviceAllocation() {
    if (pointer_) {
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

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2 && argc != 4) {
    throw std::runtime_error(
        "usage: aot-kernel-probe MATVEC.hsaco | "
        "aot-kernel-probe --load-only IMAGE.hsaco KERNEL");
  }

  constexpr std::size_t k = 2048;
  constexpr std::size_t n = 1025;
  constexpr std::uint16_t bf16_one = 0x3f80;
  constexpr std::uint16_t bf16_expected = 0x4500;  // 2048.0

  hipDeviceProp_t properties{};
  check_hip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
  if (std::string(properties.gcnArchName).find("gfx1151") != 0) {
    throw std::runtime_error("probe requires gfx1151, got " +
                             std::string(properties.gcnArchName));
  }

  if (argc == 4) {
    if (std::string(argv[1]) != "--load-only") {
      throw std::runtime_error("unknown probe mode: " + std::string(argv[1]));
    }
    auto kernel = aima::AotKernel::from_file(argv[2], argv[3]);
    std::cout << "{\"schema\":\"aima-amd395-qwen36/native-aot-load/v1\","
              << "\"target\":\"" << properties.gcnArchName << "\","
              << "\"kernel\":\"" << argv[3] << "\",\"loaded\":true}\n";
    return 0;
  }

  std::vector<std::uint16_t> host_x(k, bf16_one);
  std::vector<std::uint16_t> host_weight(k * n, bf16_one);
  std::vector<std::uint16_t> host_output(n, 0);
  DeviceAllocation device_x(host_x.size() * sizeof(host_x[0]));
  DeviceAllocation device_weight(host_weight.size() * sizeof(host_weight[0]));
  DeviceAllocation device_output(host_output.size() * sizeof(host_output[0]));
  check_hip(hipMemcpy(device_x.get(), host_x.data(),
                      host_x.size() * sizeof(host_x[0]), hipMemcpyHostToDevice),
            "hipMemcpy x");
  check_hip(hipMemcpy(device_weight.get(), host_weight.data(),
                      host_weight.size() * sizeof(host_weight[0]),
                      hipMemcpyHostToDevice),
            "hipMemcpy weight");

  auto kernel = aima::AotKernel::from_file(argv[1], "triton_matvec_kernel");
  aima::AotLaunchConfig config{
      65, 1, 1, 2, 32, 512,
  };
  void* x = device_x.get();
  void* weight = device_weight.get();
  void* output = device_output.get();
  std::vector<void*> params{&x, &weight, &output};

  const auto started = std::chrono::steady_clock::now();
  kernel.launch(config, params);
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
  const auto elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started);
  check_hip(hipMemcpy(host_output.data(), device_output.get(),
                      host_output.size() * sizeof(host_output[0]),
                      hipMemcpyDeviceToHost),
            "hipMemcpy output");

  std::size_t exact = 0;
  for (const auto value : host_output) {
    exact += value == bf16_expected ? 1 : 0;
  }
  if (exact != n) {
    throw std::runtime_error("AOT matvec mismatch: exact=" +
                             std::to_string(exact) + "/" + std::to_string(n));
  }
  std::cout << "{\"schema\":\"aima-amd395-qwen36/native-aot-probe/v1\","
            << "\"target\":\"" << properties.gcnArchName << "\","
            << "\"kernel\":\"triton_matvec_kernel\","
            << "\"grid\":[65,1,1],\"block\":[64,1,1],"
            << "\"shared_memory_bytes\":512,\"exact_bf16\":" << exact
            << ",\"elements\":" << n << ",\"elapsed_ms\":"
            << elapsed.count() << "}\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "aot-kernel-probe: " << error.what() << '\n';
  return 1;
}
