#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aima {

class Bf16GemmPlan {
 public:
  Bf16GemmPlan(std::size_t m, std::size_t n, std::size_t k,
               std::size_t workspace_limit_bytes = 128ULL * 1024 * 1024,
               bool right_operand_is_transposed = false,
               bool bias_epilogue = false);
  ~Bf16GemmPlan();

  Bf16GemmPlan(const Bf16GemmPlan&) = delete;
  Bf16GemmPlan& operator=(const Bf16GemmPlan&) = delete;
  Bf16GemmPlan(Bf16GemmPlan&&) noexcept;
  Bf16GemmPlan& operator=(Bf16GemmPlan&&) noexcept;

  // Row-major D[M,N] = A[M,K] * B[K,N]. When
  // right_operand_is_transposed is true, the supplied right-hand storage is
  // W[N,K] and the mathematical B is W^T. All matrices are BF16 and
  // accumulation is FP32, matching the qualified projection surface.
  void launch(const void* a, const void* b, void* d,
              void* stream = nullptr) const;
  void launch_with_bias(const void* a, const void* b, const void* bias,
                        void* d, void* stream = nullptr) const;

  std::size_t m() const;
  std::size_t n() const;
  std::size_t k() const;
  std::size_t workspace_bytes() const;
  int heuristic_count() const;
  int library_version() const;
  bool bias_epilogue() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct Bf16GemmProbeResult {
  std::string gpu_arch;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  std::size_t workspace_bytes = 0;
  int heuristic_count = 0;
  int library_version = 0;
  std::size_t exact_bf16_elements = 0;
  std::size_t expected_bf16_elements = 0;
  std::vector<double> measured_ms;
  double median_ms = 0.0;
  double tflops = 0.0;
};

Bf16GemmProbeResult probe_bf16_gemm();

}  // namespace aima
