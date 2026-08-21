// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace aima {

// Decode-specialized BF16 projection:
//   output[1,M] = activation[1,K] * weight[M,K]^T
//
// This is the batch-1 gfx1151 surface used by the qualified shared-expert-down
// and linear-attention-output owners. K must be a multiple of 8, M must be an
// even number larger than 8, and the inputs/outputs are device-resident BF16.
void launch_bf16_wvsplitk(const void* weight_mk, const void* activation_1k,
                          const void* bias_m, void* output_1m,
                          std::size_t m, std::size_t k, int cu_count,
                          void* stream = nullptr);

// Multiple singleton projections that consume the same activation. Each
// logical row preserves launch_bf16_wvsplitk's FP32 accumulation and BF16
// output boundary; grouping only amortizes activation staging and short-matrix
// wave tails. The projection count is limited to four and bias is unsupported.
struct Bf16WvSplitKProjection {
  const void* weight_mk = nullptr;
  void* output_1m = nullptr;
  std::size_t m = 0;
};

void launch_bf16_wvsplitk_grouped(
    const Bf16WvSplitKProjection* projections,
    std::size_t projection_count, const void* activation_1k,
    std::size_t k, int cu_count, void* stream = nullptr);

struct Bf16WvSplitKCaseResult {
  std::size_t m = 0;
  std::size_t k = 0;
  int cu_count = 0;
  int active_waves_per_group = 0;
  int launches_per_sample = 0;
  std::vector<double> measured_ms;
  double median_ms = 0.0;
  double effective_weight_bandwidth_gbs = 0.0;
  double maximum_absolute_error = 0.0;
  double relative_l2_error = 0.0;
  std::size_t finite_elements = 0;
  std::size_t expected_elements = 0;
  std::string output_bf16_sha256;
};

struct Bf16WvSplitKProbeResult {
  std::string gpu_arch;
  std::vector<Bf16WvSplitKCaseResult> cases;
};

Bf16WvSplitKProbeResult probe_bf16_wvsplitk();

}  // namespace aima
