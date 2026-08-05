// SPDX-License-Identifier: Apache-2.0

#include "aima/native_prompt_plan.h"

#include <cstdlib>
#include <iostream>
#include <string>

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
                  std::size_t expected_decode_start) {
  const auto plan = aima::plan_native_prompt_execution(
      input_tokens, matched_tokens, {1024, 2048, 4096, 8192});
  require(plan.mode == expected_mode, "execution mode changed");
  require(plan.cold_aot_tokens == expected_aot_tokens,
          "AOT prefix length changed");
  require(plan.prompt_decode_start == expected_decode_start,
          "prompt decode start changed");
  require(std::string(aima::native_prompt_execution_mode_name(plan.mode))
              .size() != 0,
          "execution mode has no metric name");
}

}  // namespace

int main() {
  using Mode = aima::NativePromptExecutionMode;
  require_plan(13, 0, Mode::kColdDecodeFallback, 0, 0);
  require_plan(1024, 0, Mode::kColdAot, 1024, 1024);
  require_plan(1536, 0, Mode::kColdAotPlusDecode, 1024, 1024);
  require_plan(4096, 0, Mode::kColdAot, 4096, 4096);
  require_plan(5000, 0, Mode::kColdAotPlusDecode, 4096, 4096);
  require_plan(8192, 0, Mode::kColdAot, 8192, 8192);
  require_plan(8212, 0, Mode::kColdAotPlusDecode, 8192, 8192);
  require_plan(13, 13, Mode::kPrefixCacheExact, 0, 13);
  require_plan(20, 13, Mode::kPrefixCachePlusDecode, 0, 13);
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
