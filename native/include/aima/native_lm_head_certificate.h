// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors
#pragma once

#include <cstddef>
#include <cstdint>

namespace aima {

constexpr std::size_t kNativeLmHeadCandidateCapacity = 1024;
constexpr std::size_t kNativeLmHeadCandidateWeightBytes =
    kNativeLmHeadCandidateCapacity * 2048 * sizeof(std::uint16_t);
constexpr std::size_t kNativeLmHeadCandidateLogitBytes =
    kNativeLmHeadCandidateCapacity * sizeof(std::uint16_t);
constexpr std::size_t kNativeLmHeadCertificateScratchBytes = 8192;

// Device/host wire stored at the beginning of the certificate scratch.
// Every field is populated before the decode stream is synchronized.
struct NativeLmHeadCertificateWire {
  float hidden_l2 = 0.0f;
  float maximum_lower_bound = 0.0f;
  float exact_top1_logit = 0.0f;
  std::uint32_t exact_top1_token_id = 0;
  std::uint32_t candidate_count = 0;
  std::uint32_t overflow = 0;
};

struct NativeLmHeadCertificateLaunchMetrics {
  std::size_t native_kernel_launches = 0;
  const NativeLmHeadCertificateWire* device_wire = nullptr;
};

// Certifies the global top-1 without a full BF16 vocabulary GEMV:
// 1. form conservative Cauchy-Schwarz bounds around the resident int8 logits,
// 2. retain every row whose upper bound can beat the global lower bound,
// 3. gather at most 1024 raw BF16 rows and evaluate them with the qualified
//    native BF16 N=1 provider, and
// 4. select the exact BF16 top-1 with deterministic token-id tie breaking.
NativeLmHeadCertificateLaunchMetrics launch_native_lm_head_certificate(
    const void* raw_weight_bf16, const void* residual_l2_fp32,
    const void* hidden_bf16, void* approximate_logits_fp32,
    void* candidate_weights_bf16, std::size_t candidate_weight_bytes,
    void* candidate_logits_bf16, std::size_t candidate_logit_bytes,
    void* scratch, std::size_t scratch_bytes, int cu_count,
    const std::uint8_t* allowed_token_mask = nullptr,
    void* stream = nullptr);

}  // namespace aima
