// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_weight_store.h"

#include "aima/sha256.h"
#include "model_layout.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

extern "C" int torch_owned_safetensors_tensor_scatter_ingest(
    const char* const* shard_paths, const std::uint64_t* shard_bytes,
    std::size_t shard_count, const std::uint32_t* tensor_shard_indices,
    const std::uint64_t* source_offsets, const std::uint64_t* payload_bytes,
    const std::uint64_t* destination_ptrs, std::size_t tensor_count,
    std::size_t chunk_bytes, std::size_t requested_worker_count,
    std::uint64_t expected_xor, std::uint64_t expected_sum,
    const char* output_path);

namespace aima {
namespace {

void hip_check(hipError_t status, const char* expression) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             hipGetErrorName(status) + " (" +
                             hipGetErrorString(status) + ")");
  }
}

#define AIMA_HIP_CHECK(expression) hip_check((expression), #expression)

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

NativeWeightStore::~NativeWeightStore() { reset(); }

NativeWeightLoadMetrics NativeWeightStore::load(
    const NativeWeightLoadOptions& options) {
  if (loaded_ || !allocations_.empty()) {
    throw std::runtime_error("native weight store is already loaded");
  }
  if (options.worker_count == 0 || options.worker_count > 16 ||
      options.chunk_bytes == 0 || options.chunk_bytes % 4096 != 0) {
    throw std::runtime_error("invalid native weight-loader worker or chunk configuration");
  }
  const auto started = std::chrono::steady_clock::now();
  const std::filesystem::path model_dir =
      std::filesystem::weakly_canonical(options.model_dir);
  const std::filesystem::path config_path = model_dir / "config.json";
  const std::filesystem::path index_path =
      model_dir / "model.safetensors.index.json";
  if (!std::filesystem::is_regular_file(config_path) ||
      !std::filesystem::is_regular_file(index_path)) {
    throw std::runtime_error("model directory is missing config.json or checkpoint index");
  }

  NativeWeightLoadMetrics metrics;
  metrics.model_config_sha256 = sha256_file(config_path);
  metrics.checkpoint_index_sha256 = sha256_file(index_path);
  if (metrics.model_config_sha256 != generated::kModelConfigSha256) {
    throw std::runtime_error("model config SHA-256 does not match the native product contract");
  }
  if (metrics.checkpoint_index_sha256 != generated::kCheckpointIndexSha256) {
    throw std::runtime_error("checkpoint index SHA-256 does not match the native product contract");
  }

  device_ = options.device;
  AIMA_HIP_CHECK(hipSetDevice(device_));
  hipDeviceProp_t properties{};
  AIMA_HIP_CHECK(hipGetDeviceProperties(&properties, device_));
  metrics.device_name = properties.name;
  metrics.gpu_arch = properties.gcnArchName;
  if (metrics.gpu_arch.rfind("gfx1151", 0) != 0) {
    throw std::runtime_error("native release requires gfx1151, got " + metrics.gpu_arch);
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  AIMA_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
  metrics.free_bytes_before = free_bytes;
  metrics.total_device_bytes = total_bytes;
  constexpr std::uint64_t kAllocationHeadroom = 64ULL * 1024ULL * 1024ULL;
  if (free_bytes < generated::kPayloadBytes + kAllocationHeadroom) {
    throw std::runtime_error(
        "insufficient GPU/GTT memory for the native resident weight store");
  }

  std::vector<std::string> shard_storage;
  std::vector<const char*> shard_paths;
  std::vector<std::uint64_t> shard_bytes;
  shard_storage.reserve(generated::kShardNames.size());
  shard_paths.reserve(generated::kShardNames.size());
  shard_bytes.reserve(generated::kShardNames.size());
  for (const char* name : generated::kShardNames) {
    const std::filesystem::path path = model_dir / name;
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("checkpoint shard is missing: " + path.string());
    }
    shard_storage.push_back(path.string());
    shard_bytes.push_back(std::filesystem::file_size(path));
  }
  for (const std::string& path : shard_storage) shard_paths.push_back(path.c_str());

  std::vector<std::uint32_t> tensor_shards;
  std::vector<std::uint64_t> source_offsets;
  std::vector<std::uint64_t> payload_bytes;
  std::vector<std::uint64_t> destination_pointers;
  tensor_shards.reserve(generated::kTensorSpecs.size());
  source_offsets.reserve(generated::kTensorSpecs.size());
  payload_bytes.reserve(generated::kTensorSpecs.size());
  destination_pointers.reserve(generated::kTensorSpecs.size());
  allocations_.assign(generated::kTensorSpecs.size(), nullptr);

  try {
    const auto allocation_started = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < generated::kTensorSpecs.size(); ++index) {
      const auto& tensor = generated::kTensorSpecs[index];
      AIMA_HIP_CHECK(hipMalloc(&allocations_[index], tensor.payload_bytes));
      tensor_shards.push_back(tensor.shard_index);
      source_offsets.push_back(tensor.source_offset_bytes);
      payload_bytes.push_back(tensor.payload_bytes);
      destination_pointers.push_back(static_cast<std::uint64_t>(
          reinterpret_cast<std::uintptr_t>(allocations_[index])));
    }
    AIMA_HIP_CHECK(hipDeviceSynchronize());
    metrics.allocation_ms = elapsed_ms(allocation_started);

    if (!options.native_report.parent_path().empty()) {
      std::filesystem::create_directories(options.native_report.parent_path());
    }
    const auto ingest_started = std::chrono::steady_clock::now();
    const int result = torch_owned_safetensors_tensor_scatter_ingest(
        shard_paths.data(), shard_bytes.data(), shard_paths.size(),
        tensor_shards.data(), source_offsets.data(), payload_bytes.data(),
        destination_pointers.data(), destination_pointers.size(),
        options.chunk_bytes, options.worker_count,
        generated::kExpectedPayloadXor, generated::kExpectedPayloadSum,
        options.native_report.c_str());
    metrics.ingest_ms = elapsed_ms(ingest_started);
    if (result != 0) {
      throw std::runtime_error(
          "native Safetensors ingest failed; inspect " +
          options.native_report.string());
    }

    views_.reserve(generated::kTensorSpecs.size());
    name_to_index_.reserve(generated::kTensorSpecs.size());
    for (std::size_t index = 0; index < generated::kTensorSpecs.size(); ++index) {
      const auto& tensor = generated::kTensorSpecs[index];
      views_.push_back(NativeTensorView{
          tensor.name,
          allocations_[index],
          tensor.payload_bytes,
          tensor.rank,
          tensor.shape,
      });
      name_to_index_.emplace(views_.back().name, index);
    }
    if (name_to_index_.size() != generated::kTensorSpecs.size()) {
      throw std::runtime_error("native tensor registry contains duplicate names");
    }
    loaded_ = true;
    AIMA_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    metrics.free_bytes_after = free_bytes;
    metrics.payload_bytes = generated::kPayloadBytes;
    metrics.tensor_count = generated::kTensorSpecs.size();
    metrics.shard_count = generated::kShardNames.size();
    metrics.load_wall_ms = elapsed_ms(started);
    return metrics;
  } catch (...) {
    reset();
    throw;
  }
}

const NativeTensorView* NativeWeightStore::find(std::string_view name) const {
  const auto found = name_to_index_.find(name);
  return found == name_to_index_.end() ? nullptr : &views_[found->second];
}

void NativeWeightStore::reset() noexcept {
  (void)hipSetDevice(device_);
  for (auto iterator = allocations_.rbegin(); iterator != allocations_.rend();
       ++iterator) {
    if (*iterator != nullptr) (void)hipFree(*iterator);
  }
  allocations_.clear();
  views_.clear();
  name_to_index_.clear();
  loaded_ = false;
}

}  // namespace aima
