// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aima {

class NativeWeightStore;

// Executes all 27 Qwen3.6 vision transformer blocks in checkpoint order. All
// block plans remain resident, while one hidden-state buffer and one block
// arena are reused across the sequential launches.
class NativeVisionBlockStackPlan {
 public:
  NativeVisionBlockStackPlan(
      const NativeWeightStore& weights, std::size_t patch_count,
      const std::vector<std::uint32_t>& cu_seqlens);
  ~NativeVisionBlockStackPlan();
  NativeVisionBlockStackPlan(const NativeVisionBlockStackPlan&) = delete;
  NativeVisionBlockStackPlan& operator=(
      const NativeVisionBlockStackPlan&) = delete;
  NativeVisionBlockStackPlan(NativeVisionBlockStackPlan&&) noexcept;
  NativeVisionBlockStackPlan& operator=(
      NativeVisionBlockStackPlan&&) noexcept;

  // input/output are distinct BF16 [patch_count,1152] tensors. cos/sin are
  // BF16 [patch_count,36]. temporary_device must provide temporary_bytes(); it
  // holds one intermediate hidden state followed by the shared per-block arena.
  void launch(const void* input_device, const void* cos_device,
              const void* sin_device, void* output_device,
              void* temporary_device, std::size_t temporary_bytes,
              void* stream = nullptr) const;
  // Diagnostic/qualification entry point that executes blocks [0,last]. The
  // final selected block always lands in output_device regardless of parity.
  void launch_through(std::size_t last_block_index, const void* input_device,
                      const void* cos_device, const void* sin_device,
                      void* output_device, void* temporary_device,
                      std::size_t temporary_bytes,
                      void* stream = nullptr) const;
  std::size_t patch_count() const;
  std::size_t block_count() const;
  std::size_t temporary_bytes() const;
  std::size_t library_workspace_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
