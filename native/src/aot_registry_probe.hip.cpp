// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/aot_kernel.h"
#include "aima/aot_registry.h"

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima {
namespace {

constexpr const char* kExactMatvecHash =
    "6f535daa92ead7e115e435151ef80d8b72e3d791eaabc28f641359534b9d12c2";

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

std::vector<unsigned char> image_copy(const EmbeddedAotImage& image) {
  return std::vector<unsigned char>(image.image, image.image + image.image_bytes);
}

}  // namespace

AotClosureProbeResult probe_embedded_aot_closure() {
  AotClosureProbeResult result;
  hipDeviceProp_t properties{};
  check_hip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
  result.gpu_arch = properties.gcnArchName;
  if (result.gpu_arch.find("gfx1151") != 0) {
    throw std::runtime_error("embedded AOT closure requires gfx1151, got " +
                             result.gpu_arch);
  }

  const EmbeddedAotImage* images = embedded_aot_images(&result.image_count);
  if (images == nullptr || result.image_count == 0) {
    throw std::runtime_error("embedded AOT registry is empty");
  }
  for (std::size_t index = 0; index < result.image_count; ++index) {
    const EmbeddedAotImage& record = images[index];
    AotKernel kernel(image_copy(record), record.symbol);
    ++result.loaded_count;
    result.image_bytes += record.image_bytes;
  }

  const EmbeddedAotImage* matvec = find_embedded_aot_image(kExactMatvecHash);
  if (matvec == nullptr || std::string(matvec->symbol) != "triton_matvec_kernel" ||
      matvec->num_warps != 2 || matvec->warp_size != 32 ||
      matvec->shared_memory_bytes != 512) {
    throw std::runtime_error("embedded exact-matvec contract is missing or changed");
  }

  constexpr std::size_t k = 2048;
  constexpr std::size_t n = 1025;
  constexpr std::uint16_t bf16_one = 0x3f80;
  constexpr std::uint16_t bf16_expected = 0x4500;
  std::vector<std::uint16_t> host_x(k, bf16_one);
  std::vector<std::uint16_t> host_weight(k * n, bf16_one);
  std::vector<std::uint16_t> host_output(n, 0);
  DeviceAllocation device_x(host_x.size() * sizeof(host_x[0]));
  DeviceAllocation device_weight(host_weight.size() * sizeof(host_weight[0]));
  DeviceAllocation device_output(host_output.size() * sizeof(host_output[0]));
  check_hip(hipMemcpy(device_x.get(), host_x.data(), host_x.size() * sizeof(host_x[0]),
                      hipMemcpyHostToDevice),
            "hipMemcpy x");
  check_hip(hipMemcpy(device_weight.get(), host_weight.data(),
                      host_weight.size() * sizeof(host_weight[0]), hipMemcpyHostToDevice),
            "hipMemcpy weight");

  AotKernel kernel(image_copy(*matvec), matvec->symbol);
  AotLaunchConfig config{65, 1, 1, matvec->num_warps, matvec->warp_size,
                         matvec->shared_memory_bytes};
  void* x = device_x.get();
  void* weight = device_weight.get();
  void* output = device_output.get();
  std::vector<void*> params{&x, &weight, &output};
  const auto started = std::chrono::steady_clock::now();
  kernel.launch(config, params);
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
  result.exact_probe_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
  check_hip(hipMemcpy(host_output.data(), device_output.get(),
                      host_output.size() * sizeof(host_output[0]), hipMemcpyDeviceToHost),
            "hipMemcpy output");
  result.expected_bf16_elements = n;
  for (const std::uint16_t value : host_output) {
    result.exact_bf16_elements += value == bf16_expected ? 1 : 0;
  }
  if (result.exact_bf16_elements != result.expected_bf16_elements) {
    throw std::runtime_error("embedded AOT exact-matvec comparison failed");
  }
  return result;
}

}  // namespace aima
