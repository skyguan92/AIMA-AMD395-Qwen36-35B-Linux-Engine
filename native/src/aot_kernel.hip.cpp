// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/aot_kernel.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace aima {
namespace {

void check_hip(hipError_t status, const char* operation) {
  if (status == hipSuccess) {
    return;
  }
  throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

std::vector<unsigned char> read_binary(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open AOT image: " + path.string());
  }
  const auto end = stream.tellg();
  if (end <= 0) {
    throw std::runtime_error("AOT image is empty: " + path.string());
  }
  std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("cannot read AOT image: " + path.string());
  }
  return bytes;
}

}  // namespace

AotKernel::AotKernel(std::vector<unsigned char> image, std::string kernel_name)
    : image_(std::move(image)), kernel_name_(std::move(kernel_name)) {
  if (image_.empty() || kernel_name_.empty()) {
    throw std::invalid_argument("AOT image and kernel name must be non-empty");
  }
  check_hip(hipModuleLoadData(&module_, image_.data()), "hipModuleLoadData");
  try {
    check_hip(hipModuleGetFunction(&function_, module_, kernel_name_.c_str()),
              "hipModuleGetFunction");
  } catch (...) {
    unload();
    throw;
  }
}

AotKernel::~AotKernel() { unload(); }

AotKernel::AotKernel(AotKernel&& other) noexcept
    : image_(std::move(other.image_)),
      kernel_name_(std::move(other.kernel_name_)),
      module_(std::exchange(other.module_, nullptr)),
      function_(std::exchange(other.function_, nullptr)) {}

AotKernel& AotKernel::operator=(AotKernel&& other) noexcept {
  if (this != &other) {
    unload();
    image_ = std::move(other.image_);
    kernel_name_ = std::move(other.kernel_name_);
    module_ = std::exchange(other.module_, nullptr);
    function_ = std::exchange(other.function_, nullptr);
  }
  return *this;
}

AotKernel AotKernel::from_file(const std::filesystem::path& path,
                               std::string kernel_name) {
  return AotKernel(read_binary(path), std::move(kernel_name));
}

void AotKernel::launch(const AotLaunchConfig& config,
                       const std::vector<void*>& regular_kernel_params,
                       hipStream_t stream) const {
  if (!module_ || !function_) {
    throw std::runtime_error("AOT kernel is not loaded");
  }
  if (config.grid_x == 0 || config.grid_y == 0 || config.grid_z == 0 ||
      config.num_warps == 0 || config.warp_size == 0) {
    throw std::invalid_argument("AOT launch dimensions must be non-zero");
  }

  // Triton AMD code objects append two architecture-level pointer arguments
  // after the user ABI: global scratch and profile scratch.  The qualified
  // kernels use neither, but the zero placeholders are ABI-mandatory.
  hipDeviceptr_t global_scratch = 0;
  hipDeviceptr_t profile_scratch = 0;
  std::vector<void*> params = regular_kernel_params;
  params.push_back(&global_scratch);
  params.push_back(&profile_scratch);

  check_hip(
      hipModuleLaunchKernel(function_, config.grid_x, config.grid_y,
                            config.grid_z, config.num_warps * config.warp_size,
                            1, 1, config.shared_memory_bytes, stream,
                            params.data(), nullptr),
      "hipModuleLaunchKernel");
}

void AotKernel::unload() noexcept {
  function_ = nullptr;
  if (module_) {
    const hipError_t ignored = hipModuleUnload(module_);
    static_cast<void>(ignored);
    module_ = nullptr;
  }
}

}  // namespace aima
