// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors
#pragma once

#include <cstddef>
#include <cstdint>

namespace aima {

struct NativeSamplingParameters {
  double temperature = 1.0;
  double top_p = 1.0;
  std::uint64_t seed = 0;
};

// Stable request-local stochastic sampler. The PRNG and nucleus ordering are
// specified here instead of delegated to the C++ standard library so an
// explicit seed produces the same token sequence across supported toolchains.
class NativeLogitSampler {
 public:
  explicit NativeLogitSampler(NativeSamplingParameters parameters);

  std::uint32_t sample_fp32(
      const float* logits, std::size_t count,
      const std::uint8_t* allowed_token_mask = nullptr,
      std::size_t allowed_token_mask_count = 0);
  std::uint32_t sample_bf16(
      const std::uint16_t* logits, std::size_t count,
      const std::uint8_t* allowed_token_mask = nullptr,
      std::size_t allowed_token_mask_count = 0);

  const NativeSamplingParameters& parameters() const { return parameters_; }

 private:
  double next_unit_interval();

  NativeSamplingParameters parameters_;
  std::uint64_t state_ = 0;
};

// Produces an entropy-backed seed for requests that do not specify one. The
// effective value is returned in request metrics so the generation can still
// be reproduced exactly.
std::uint64_t native_random_sampling_seed() noexcept;

}  // namespace aima
