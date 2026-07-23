#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_weight_store.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace aima {

struct NativeLmHeadMetrics {
  std::uint64_t payload_bytes = 0;
  std::uint64_t free_bytes_before = 0;
  std::uint64_t free_bytes_after = 0;
  double allocation_ms = 0.0;
  double quantize_ms = 0.0;
  double hash_ms = 0.0;
  double build_wall_ms = 0.0;
  std::string q_weight_sha256;
  std::string scales_sha256;
  std::string residual_l2_sha256;
  bool q_weight_reference_exact = false;
  bool scales_reference_exact = false;
  bool residual_l2_reference_exact = false;
  std::array<std::int8_t, 6> q_weight_samples{};
  std::array<float, 5> scale_samples{};
  std::array<float, 5> residual_l2_samples{};
};

class NativeLmHeadStore {
 public:
  NativeLmHeadStore() = default;
  ~NativeLmHeadStore();
  NativeLmHeadStore(const NativeLmHeadStore&) = delete;
  NativeLmHeadStore& operator=(const NativeLmHeadStore&) = delete;

  NativeLmHeadMetrics build(const NativeWeightStore& weights, int device = 0);
  void* q_weight() const { return q_weight_; }
  void* scales() const { return scales_; }
  void* residual_l2() const { return residual_l2_; }
  bool built() const { return allocation_ != nullptr; }
  void write_scales_for_validation(const std::filesystem::path& path) const;
  void write_residual_l2_for_validation(
      const std::filesystem::path& path) const;
  void reset() noexcept;

 private:
  int device_ = 0;
  void* allocation_ = nullptr;
  void* q_weight_ = nullptr;
  void* scales_ = nullptr;
  void* residual_l2_ = nullptr;
};

}  // namespace aima
