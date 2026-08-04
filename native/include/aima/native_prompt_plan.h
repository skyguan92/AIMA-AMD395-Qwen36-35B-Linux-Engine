#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <cstddef>
#include <stdexcept>

namespace aima {

enum class NativePromptExecutionMode {
  kColdDecodeFallback,
  kColdAot,
  kColdAotPlusDecode,
  kPrefixCacheExact,
  kPrefixCachePlusDecode,
};

struct NativePromptExecutionPlan {
  NativePromptExecutionMode mode =
      NativePromptExecutionMode::kColdDecodeFallback;
  std::size_t matched_prefix_tokens = 0;
  std::size_t cold_aot_tokens = 0;
  std::size_t prompt_decode_start = 0;
  bool prefix_hit = false;
  bool exact_prefix_hit = false;
  bool prefix_extension_hit = false;

  bool prompt_decode_required(std::size_t input_tokens) const {
    return prompt_decode_start < input_tokens;
  }
};

inline bool native_request_fits_capacity(std::size_t input_tokens,
                                         std::size_t output_tokens,
                                         std::size_t capacity) {
  return input_tokens != 0 && output_tokens != 0 &&
         input_tokens <= capacity &&
         output_tokens <= capacity - input_tokens;
}

inline NativePromptExecutionPlan plan_native_prompt_execution(
    std::size_t input_tokens, std::size_t matched_prefix_tokens,
    std::size_t static_prefill_tokens) {
  if (input_tokens == 0 || static_prefill_tokens == 0 ||
      matched_prefix_tokens > input_tokens) {
    throw std::invalid_argument("native prompt execution geometry is invalid");
  }

  NativePromptExecutionPlan plan;
  plan.matched_prefix_tokens = matched_prefix_tokens;
  plan.prefix_hit = matched_prefix_tokens != 0;
  plan.exact_prefix_hit =
      plan.prefix_hit && matched_prefix_tokens == input_tokens;
  plan.prefix_extension_hit =
      plan.prefix_hit && matched_prefix_tokens < input_tokens;
  plan.cold_aot_tokens =
      !plan.prefix_hit && input_tokens >= static_prefill_tokens
          ? static_prefill_tokens
          : 0;
  plan.prompt_decode_start =
      plan.prefix_extension_hit
          ? matched_prefix_tokens
          : (!plan.prefix_hit ? plan.cold_aot_tokens : input_tokens);

  if (plan.exact_prefix_hit) {
    plan.mode = NativePromptExecutionMode::kPrefixCacheExact;
  } else if (plan.prefix_extension_hit) {
    plan.mode = NativePromptExecutionMode::kPrefixCachePlusDecode;
  } else if (plan.cold_aot_tokens == 0) {
    plan.mode = NativePromptExecutionMode::kColdDecodeFallback;
  } else if (plan.prompt_decode_required(input_tokens)) {
    plan.mode = NativePromptExecutionMode::kColdAotPlusDecode;
  } else {
    plan.mode = NativePromptExecutionMode::kColdAot;
  }
  return plan;
}

inline const char* native_prompt_execution_mode_name(
    NativePromptExecutionMode mode) {
  switch (mode) {
    case NativePromptExecutionMode::kColdDecodeFallback:
      return "cold-decode-fallback";
    case NativePromptExecutionMode::kColdAot:
      return "cold-aot";
    case NativePromptExecutionMode::kColdAotPlusDecode:
      return "cold-aot-plus-decode";
    case NativePromptExecutionMode::kPrefixCacheExact:
      return "prefix-cache-exact";
    case NativePromptExecutionMode::kPrefixCachePlusDecode:
      return "prefix-cache-plus-decode";
  }
  throw std::invalid_argument("native prompt execution mode is invalid");
}

}  // namespace aima
