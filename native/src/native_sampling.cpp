// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima {
namespace {

struct SamplingCandidate {
  std::uint32_t token_id = 0;
  float logit = 0.0f;
  double weight = 0.0;
};

float bf16_bits_to_float(std::uint16_t bits) {
  const std::uint32_t fp32_bits = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0f;
  std::memcpy(&result, &fp32_bits, sizeof(result));
  return result;
}

template <typename LogitAt>
std::uint32_t sample_logits(
    std::size_t count, const std::uint8_t* allowed_token_mask,
    std::size_t allowed_token_mask_count, double temperature, double top_p,
    double draw, LogitAt logit_at) {
  if (count == 0 ||
      count > static_cast<std::size_t>(
                  std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument("native sampling logits are empty or too large");
  }
  if (allowed_token_mask != nullptr && allowed_token_mask_count != count) {
    throw std::invalid_argument("native sampling token mask size is invalid");
  }

  std::vector<SamplingCandidate> candidates;
  candidates.reserve(count);
  float maximum = -std::numeric_limits<float>::infinity();
  bool have_positive_infinity = false;
  for (std::size_t index = 0; index < count; ++index) {
    if (allowed_token_mask != nullptr && allowed_token_mask[index] == 0) {
      continue;
    }
    const float value = logit_at(index);
    if (std::isnan(value) || value == -std::numeric_limits<float>::infinity()) {
      continue;
    }
    have_positive_infinity =
        have_positive_infinity ||
        value == std::numeric_limits<float>::infinity();
    maximum = std::max(maximum, value);
    candidates.push_back(
        {static_cast<std::uint32_t>(index), value, 0.0});
  }
  if (candidates.empty()) {
    throw std::runtime_error("native sampling has no finite allowed logits");
  }

  if (have_positive_infinity) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [](const SamplingCandidate& candidate) {
                         return candidate.logit !=
                                std::numeric_limits<float>::infinity();
                       }),
        candidates.end());
    for (SamplingCandidate& candidate : candidates) candidate.weight = 1.0;
  } else {
    for (SamplingCandidate& candidate : candidates) {
      candidate.weight = std::exp(
          (static_cast<double>(candidate.logit) -
           static_cast<double>(maximum)) /
          temperature);
    }
  }

  if (top_p < 1.0) {
    std::sort(candidates.begin(), candidates.end(),
              [](const SamplingCandidate& left,
                 const SamplingCandidate& right) {
                return left.logit > right.logit ||
                       (left.logit == right.logit &&
                        left.token_id < right.token_id);
              });
  }
  double total = 0.0;
  for (const SamplingCandidate& candidate : candidates) {
    total += candidate.weight;
  }
  if (!(total > 0.0) || !std::isfinite(total)) {
    throw std::runtime_error("native sampling probability mass is invalid");
  }

  std::size_t retained = candidates.size();
  double retained_total = total;
  if (top_p < 1.0) {
    const double threshold = top_p * total;
    retained_total = 0.0;
    retained = 0;
    do {
      retained_total += candidates[retained].weight;
      ++retained;
    } while (retained < candidates.size() && retained_total < threshold);
  }

  const double target = draw * retained_total;
  double cumulative = 0.0;
  for (std::size_t index = 0; index < retained; ++index) {
    cumulative += candidates[index].weight;
    if (target < cumulative) return candidates[index].token_id;
  }
  // The unit draw is strictly below one, but retain a deterministic guard for
  // platforms whose final addition rounds down by one ULP.
  return candidates[retained - 1].token_id;
}

}  // namespace

NativeLogitSampler::NativeLogitSampler(NativeSamplingParameters parameters)
    : parameters_(parameters), state_(parameters.seed) {
  if (!std::isfinite(parameters_.temperature) ||
      parameters_.temperature <= 0.0 || parameters_.temperature > 2.0) {
    throw std::invalid_argument(
        "native sampling temperature must be in (0, 2]");
  }
  if (!std::isfinite(parameters_.top_p) || parameters_.top_p <= 0.0 ||
      parameters_.top_p > 1.0) {
    throw std::invalid_argument("native sampling top_p must be in (0, 1]");
  }
}

double NativeLogitSampler::next_unit_interval() {
  // SplitMix64 with the upper 53 bits projected exactly onto [0, 1).
  std::uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  value ^= value >> 31U;
  return static_cast<double>(value >> 11U) * 0x1.0p-53;
}

std::uint32_t NativeLogitSampler::sample_fp32(
    const float* logits, std::size_t count,
    const std::uint8_t* allowed_token_mask,
    std::size_t allowed_token_mask_count) {
  if (logits == nullptr) {
    throw std::invalid_argument("native FP32 sampling logits are null");
  }
  return sample_logits(
      count, allowed_token_mask, allowed_token_mask_count,
      parameters_.temperature, parameters_.top_p, next_unit_interval(),
      [logits](std::size_t index) { return logits[index]; });
}

std::uint32_t NativeLogitSampler::sample_bf16(
    const std::uint16_t* logits, std::size_t count,
    const std::uint8_t* allowed_token_mask,
    std::size_t allowed_token_mask_count) {
  if (logits == nullptr) {
    throw std::invalid_argument("native BF16 sampling logits are null");
  }
  return sample_logits(
      count, allowed_token_mask, allowed_token_mask_count,
      parameters_.temperature, parameters_.top_p, next_unit_interval(),
      [logits](std::size_t index) {
        return bf16_bits_to_float(logits[index]);
      });
}

std::uint64_t native_random_sampling_seed() noexcept {
  std::uint64_t seed = static_cast<std::uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  try {
    std::random_device entropy;
    seed ^= static_cast<std::uint64_t>(entropy()) << 32U;
    seed ^= static_cast<std::uint64_t>(entropy());
  } catch (...) {
    // The high-resolution clock plus address-space/process jitter remains a
    // valid nondeterministic default when a platform has no random_device.
  }
  seed ^= reinterpret_cast<std::uintptr_t>(&seed);
  return seed == 0 ? 0x6a09e667f3bcc909ULL : seed;
}

}  // namespace aima
