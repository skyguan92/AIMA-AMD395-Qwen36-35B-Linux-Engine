// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aima {

struct NativeTensorView {
  std::string_view name;
  void* device_pointer = nullptr;
  std::uint64_t payload_bytes = 0;
  std::uint8_t rank = 0;
  std::array<std::uint32_t, 5> shape{};
};

struct NativeWeightLoadOptions {
  std::filesystem::path model_dir;
  std::filesystem::path native_report;
  int device = 0;
  std::size_t worker_count = 1;
  std::size_t chunk_bytes = 536870912ULL;
};

struct NativeWeightLoadMetrics {
  std::string weight_set;
  std::string layout_manifest_sha256;
  std::string language_layout_manifest_sha256;
  std::string visual_layout_manifest_sha256;
  std::string device_name;
  std::string gpu_arch;
  std::string checkpoint_index_sha256;
  std::string model_config_sha256;
  std::uint64_t free_bytes_before = 0;
  std::uint64_t total_device_bytes = 0;
  std::uint64_t free_bytes_after = 0;
  std::uint64_t payload_bytes = 0;
  std::size_t tensor_count = 0;
  std::size_t shard_count = 0;
  std::uint64_t language_payload_bytes = 0;
  std::size_t language_tensor_count = 0;
  std::size_t language_shard_count = 0;
  std::uint64_t visual_payload_bytes = 0;
  std::size_t visual_tensor_count = 0;
  std::size_t visual_shard_count = 0;
  double allocation_ms = 0.0;
  double ingest_ms = 0.0;
  double load_wall_ms = 0.0;
};

class NativeWeightStore {
 public:
  NativeWeightStore() = default;
  ~NativeWeightStore();
  NativeWeightStore(const NativeWeightStore&) = delete;
  NativeWeightStore& operator=(const NativeWeightStore&) = delete;

  NativeWeightLoadMetrics load(const NativeWeightLoadOptions& options);
  NativeWeightLoadMetrics load_visual(const NativeWeightLoadOptions& options);
  NativeWeightLoadMetrics load_resident(const NativeWeightLoadOptions& options);
  const NativeTensorView* find(std::string_view name) const;
  const std::vector<NativeTensorView>& tensors() const { return views_; }
  bool loaded() const { return loaded_; }
  void reset() noexcept;

 private:
  struct LayoutEntry {
    std::string_view name;
    std::uint32_t shard_index = 0;
    std::uint64_t source_offset_bytes = 0;
    std::uint64_t payload_bytes = 0;
    std::uint8_t rank = 0;
    std::array<std::uint32_t, 5> shape{};
  };

  NativeWeightLoadMetrics load_layout(
      const NativeWeightLoadOptions& options, std::string_view weight_set,
      std::string_view layout_manifest_sha256,
      std::string_view model_config_sha256,
      std::string_view checkpoint_index_sha256,
      const std::vector<std::string_view>& shard_names,
      const std::vector<LayoutEntry>& entries,
      std::uint64_t expected_payload_bytes,
      std::uint64_t expected_payload_xor,
      std::uint64_t expected_payload_sum);

  int device_ = 0;
  bool loaded_ = false;
  std::vector<void*> allocations_;
  std::vector<NativeTensorView> views_;
  std::unordered_map<std::string_view, std::size_t> name_to_index_;
};

}  // namespace aima
