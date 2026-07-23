// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_executor.h"

#include "aima/aot_registry.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace aima {
namespace {

std::vector<unsigned char> image_copy(const EmbeddedAotImage& image) {
  return std::vector<unsigned char>(image.image, image.image + image.image_bytes);
}

}  // namespace

NativeDecodeExecutorMetrics NativeDecodeExecutor::load() {
  if (loaded() || !hash_to_index_.empty()) {
    throw std::runtime_error("native decode executor is already loaded");
  }
  const auto started = std::chrono::steady_clock::now();
  std::size_t image_count = 0;
  const EmbeddedAotImage* images = embedded_aot_images(&image_count);
  if (images == nullptr || image_count == 0) {
    throw std::runtime_error("embedded native decode AOT registry is empty");
  }
  kernels_.reserve(image_count);
  hash_to_index_.reserve(image_count);
  for (std::size_t index = 0; index < image_count; ++index) {
    const EmbeddedAotImage& image = images[index];
    if (image.kernel_hash == nullptr || image.symbol == nullptr ||
        image.image == nullptr || image.image_bytes == 0) {
      throw std::runtime_error("invalid embedded native decode AOT image");
    }
    const auto inserted = hash_to_index_.emplace(image.kernel_hash, index);
    if (!inserted.second) {
      throw std::runtime_error("duplicate embedded native decode kernel hash");
    }
    kernels_.push_back(
        std::make_unique<AotKernel>(image_copy(image), image.symbol));
  }
  metrics_.embedded_images = image_count;
  metrics_.loaded_modules = kernels_.size();
  metrics_.module_load_wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  return metrics_;
}

void NativeDecodeExecutor::launch(
    const PreparedDecodeInvocation& invocation, void* stream) {
  if (!loaded() || invocation.launch == nullptr ||
      invocation.kernel_params.size() != invocation.launch->argument_count) {
    throw std::runtime_error("native decode executor received an incomplete invocation");
  }
  const auto found = hash_to_index_.find(invocation.launch->kernel_hash);
  if (found == hash_to_index_.end() || found->second >= kernels_.size()) {
    throw std::runtime_error("native decode schedule references an unloaded AOT image");
  }
  const DecodeLaunchConfig& source = invocation.launch->config;
  const AotLaunchConfig config{
      source.grid_x,
      source.grid_y,
      source.grid_z,
      source.num_warps,
      source.warp_size,
      source.shared_memory_bytes,
  };
  kernels_[found->second]->launch(
      config, invocation.kernel_params, static_cast<hipStream_t>(stream));
  ++metrics_.launched_kernels;
  metrics_.launched_abi_arguments += invocation.kernel_params.size();
}

}  // namespace aima
