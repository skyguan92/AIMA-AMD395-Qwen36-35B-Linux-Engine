// SPDX-License-Identifier: Apache-2.0

#include "aima/native_prompt_plan.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_prompt_plan_test: " << message << '\n';
    std::exit(1);
  }
}

void require_plan(std::size_t input_tokens, std::size_t matched_tokens,
                  aima::NativePromptExecutionMode expected_mode,
                  std::size_t expected_aot_tokens,
                  std::size_t expected_bucket_tokens,
                  const std::vector<std::size_t>& expected_buckets) {
  const auto plan = aima::plan_native_prompt_execution(
      input_tokens, matched_tokens, {1024, 2048, 4096, 8192});
  if (plan.mode != expected_mode) {
    std::cerr << "native_prompt_plan_test: input=" << input_tokens
              << " matched=" << matched_tokens << " expected="
              << aima::native_prompt_execution_mode_name(expected_mode)
              << " actual="
              << aima::native_prompt_execution_mode_name(plan.mode) << '\n';
    std::exit(1);
  }
  require(plan.cold_aot_tokens == expected_aot_tokens,
          "AOT input length changed");
  require(plan.aot_bucket_tokens == expected_bucket_tokens,
          "AOT bucket total changed");
  require(plan.prompt_decode_start == input_tokens,
          "planned prompt still requires decode");
  require(!plan.prompt_decode_required(input_tokens),
          "planned prompt reports decode work");
  require(plan.aot_segments.size() == expected_buckets.size(),
          "AOT segment count changed");
  std::size_t expected_offset = matched_tokens;
  std::size_t remaining = input_tokens - matched_tokens;
  for (std::size_t index = 0; index < expected_buckets.size(); ++index) {
    const auto& segment = plan.aot_segments[index];
    const std::size_t expected_tokens =
        std::min(remaining, expected_buckets[index]);
    require(segment.input_offset == expected_offset,
            "AOT segment input offset changed");
    require(segment.input_tokens == expected_tokens,
            "AOT segment input length changed");
    require(segment.bucket_tokens == expected_buckets[index],
            "AOT segment bucket changed");
    expected_offset += expected_tokens;
    remaining -= expected_tokens;
  }
  require(remaining == 0, "AOT segments did not cover the prompt");
  require(
      std::string(aima::native_prompt_execution_mode_name(plan.mode)).size() !=
          0,
      "execution mode has no metric name");
}

}  // namespace

int main() {
  using Mode = aima::NativePromptExecutionMode;
  require_plan(13, 0, Mode::kColdAotPadded, 13, 1024, {1024});
  require_plan(502, 0, Mode::kColdAotPadded, 502, 1024, {1024});
  require_plan(1024, 0, Mode::kColdAot, 1024, 1024, {1024});
  require_plan(1050, 0, Mode::kColdAotPadded, 1050, 2048, {2048});
  require_plan(1536, 0, Mode::kColdAotPadded, 1536, 2048, {2048});
  require_plan(1993, 0, Mode::kColdAotPadded, 1993, 2048, {2048});
  require_plan(3005, 0, Mode::kColdAotComposed, 3005, 3072, {2048, 1024});
  require_plan(3996, 0, Mode::kColdAotPadded, 3996, 4096, {4096});
  require_plan(4096, 0, Mode::kColdAot, 4096, 4096, {4096});
  require_plan(4152, 0, Mode::kColdAotComposed, 4152, 5120, {4096, 1024});
  require_plan(8192, 0, Mode::kColdAot, 8192, 8192, {8192});
  require_plan(8212, 0, Mode::kColdAotComposed, 8212, 9216, {8192, 1024});
  require_plan(13, 13, Mode::kPrefixCacheExact, 0, 0, {});
  require_plan(20, 13, Mode::kPrefixCachePlusAot, 7, 1024, {1024});
  bool rejected = false;
  try {
    (void)aima::plan_native_prompt_execution(4096, 0, {2048, 1024, 8192});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "unordered resident buckets were admitted");
  require(aima::native_request_fits_capacity(2047, 1, 2048),
          "maximum admitted window was rejected");
  require(!aima::native_request_fits_capacity(2048, 1, 2048),
          "prompt plus output overflow was admitted");
  require(!aima::native_request_fits_capacity(0, 1, 2048),
          "empty prompt was admitted");
  std::cout << "native_prompt_plan_test: PASS\n";
  return 0;
}
