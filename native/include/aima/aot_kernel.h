#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace aima {

struct AotLaunchConfig {
  std::uint32_t grid_x = 1;
  std::uint32_t grid_y = 1;
  std::uint32_t grid_z = 1;
  std::uint32_t num_warps = 1;
  std::uint32_t warp_size = 32;
  std::uint32_t shared_memory_bytes = 0;
};

class AotKernel {
 public:
  AotKernel(std::vector<unsigned char> image, std::string kernel_name);
  ~AotKernel();

  AotKernel(const AotKernel&) = delete;
  AotKernel& operator=(const AotKernel&) = delete;
  AotKernel(AotKernel&& other) noexcept;
  AotKernel& operator=(AotKernel&& other) noexcept;

  static AotKernel from_file(const std::filesystem::path& path,
                             std::string kernel_name);

  void launch(const AotLaunchConfig& config,
              const std::vector<void*>& regular_kernel_params,
              hipStream_t stream = nullptr) const;

 private:
  void unload() noexcept;

  std::vector<unsigned char> image_;
  std::string kernel_name_;
  hipModule_t module_ = nullptr;
  hipFunction_t function_ = nullptr;
};

}  // namespace aima
