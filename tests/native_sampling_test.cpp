// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_sampling.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_throws(Function function, const char* message) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

}  // namespace

int main() {
  const std::vector<float> logits{2.0f, 1.0f, 0.0f, -1.0f};
  aima::NativeSamplingParameters parameters{1.0, 1.0, 123456789ULL};
  aima::NativeLogitSampler first(parameters);
  aima::NativeLogitSampler second(parameters);
  std::vector<std::uint32_t> first_sequence;
  std::vector<std::uint32_t> second_sequence;
  for (int index = 0; index < 64; ++index) {
    first_sequence.push_back(first.sample_fp32(logits.data(), logits.size()));
    second_sequence.push_back(second.sample_fp32(logits.data(), logits.size()));
  }
  require(first_sequence == second_sequence,
          "the seeded sampling sequence is not stable");

  aima::NativeLogitSampler different_seed({1.0, 1.0, 987654321ULL});
  std::vector<std::uint32_t> different_sequence;
  for (int index = 0; index < 64; ++index) {
    different_sequence.push_back(
        different_seed.sample_fp32(logits.data(), logits.size()));
  }
  require(first_sequence != different_sequence,
          "different seeds did not change the sampling sequence");

  aima::NativeLogitSampler nucleus({1.0, 0.5, 7ULL});
  for (int index = 0; index < 32; ++index) {
    require(nucleus.sample_fp32(logits.data(), logits.size()) == 0,
            "top_p retained a token outside the nucleus");
  }

  const std::vector<std::uint8_t> mask{0, 1, 0, 0};
  aima::NativeLogitSampler constrained({2.0, 1.0, 11ULL});
  for (int index = 0; index < 32; ++index) {
    require(constrained.sample_fp32(logits.data(), logits.size(), mask.data(),
                                    mask.size()) == 1,
            "the allowed-token mask was not enforced");
  }

  // 0x4000 and 0x3f80 are BF16 encodings of 2.0 and 1.0.
  const std::vector<std::uint16_t> bf16_logits{0x4000U, 0x3f80U, 0x0000U,
                                               0xbf80U};
  aima::NativeLogitSampler bf16_sampler({1.0, 0.5, 7ULL});
  require(bf16_sampler.sample_bf16(bf16_logits.data(), bf16_logits.size()) ==
              0,
          "BF16 logits did not preserve nucleus ordering");

  require_throws(
      [] { aima::NativeLogitSampler invalid({0.0, 1.0, 0}); },
      "zero stochastic temperature was accepted");
  require_throws(
      [] { aima::NativeLogitSampler invalid({1.0, 0.0, 0}); },
      "zero top_p was accepted");
  require_throws(
      [&] {
        aima::NativeLogitSampler invalid_mask({1.0, 1.0, 0});
        invalid_mask.sample_fp32(logits.data(), logits.size(), mask.data(),
                                 mask.size() - 1);
      },
      "an incorrectly sized token mask was accepted");

  std::cout << "native sampling: PASS\n";
  return 0;
}
