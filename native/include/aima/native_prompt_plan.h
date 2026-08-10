#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aima {

enum class NativePromptExecutionMode {
  kColdDecodeFallback,
  kColdAot,
  kColdAotPadded,
  kColdAotComposed,
  kColdAotPlusDecode,
  kPrefixCacheExact,
  kPrefixCachePlusAot,
  kPrefixCachePlusDecode,
};

struct NativePromptAotSegment {
  std::size_t input_offset = 0;
  std::size_t input_tokens = 0;
  std::size_t bucket_tokens = 0;

  bool padded() const { return input_tokens < bucket_tokens; }
};

struct NativePromptExecutionPlan {
  NativePromptExecutionMode mode =
      NativePromptExecutionMode::kColdDecodeFallback;
  std::size_t matched_prefix_tokens = 0;
  std::size_t cold_aot_tokens = 0;
  std::size_t aot_bucket_tokens = 0;
  std::size_t prompt_decode_start = 0;
  std::vector<NativePromptAotSegment> aot_segments;
  bool prefix_hit = false;
  bool exact_prefix_hit = false;
  bool prefix_extension_hit = false;

  bool prompt_decode_required(std::size_t input_tokens) const {
    return prompt_decode_start < input_tokens;
  }

  bool padded_aot() const {
    return std::any_of(
        aot_segments.begin(), aot_segments.end(),
        [](const NativePromptAotSegment& segment) { return segment.padded(); });
  }
};

inline bool native_request_fits_capacity(std::size_t input_tokens,
                                         std::size_t output_tokens,
                                         std::size_t capacity) {
  return input_tokens != 0 && output_tokens != 0 && input_tokens <= capacity &&
         output_tokens <= capacity - input_tokens;
}

inline NativePromptExecutionPlan plan_native_prompt_execution(
    std::size_t input_tokens, std::size_t matched_prefix_tokens,
    const std::vector<std::size_t>& resident_prefill_tokens) {
  if (resident_prefill_tokens.empty()) {
    throw std::invalid_argument(
        "native prompt execution has no resident prefill bucket");
  }
  if (input_tokens == 0 || matched_prefix_tokens > input_tokens) {
    throw std::invalid_argument("native prompt execution geometry is invalid");
  }
  std::size_t previous_bucket = 0;
  for (const std::size_t tokens : resident_prefill_tokens) {
    if (tokens == 0 || tokens > 262144 || tokens <= previous_bucket) {
      throw std::invalid_argument(
          "native resident prefill buckets must be strictly increasing");
    }
    previous_bucket = tokens;
  }

  NativePromptExecutionPlan plan;
  plan.matched_prefix_tokens = matched_prefix_tokens;
  plan.prefix_hit = matched_prefix_tokens != 0;
  plan.exact_prefix_hit =
      plan.prefix_hit && matched_prefix_tokens == input_tokens;
  plan.prefix_extension_hit =
      plan.prefix_hit && matched_prefix_tokens < input_tokens;
  if (plan.exact_prefix_hit) {
    plan.prompt_decode_start = input_tokens;
    plan.mode = NativePromptExecutionMode::kPrefixCacheExact;
    return plan;
  }

  const std::size_t unmatched_tokens = input_tokens - matched_prefix_tokens;
  const std::size_t maximum_bucket = resident_prefill_tokens.back();
  if (unmatched_tokens >
      std::numeric_limits<std::size_t>::max() - maximum_bucket) {
    throw std::invalid_argument("native prompt execution geometry overflows");
  }
  const std::size_t search_limit = unmatched_tokens + maximum_bucket - 1;
  const std::size_t unreachable = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> segment_counts(search_limit + 1, unreachable);
  std::vector<std::size_t> previous_totals(search_limit + 1, unreachable);
  std::vector<std::size_t> selected_buckets(search_limit + 1, 0);
  segment_counts[0] = 0;
  for (std::size_t total = 1; total <= search_limit; ++total) {
    for (auto bucket = resident_prefill_tokens.rbegin();
         bucket != resident_prefill_tokens.rend(); ++bucket) {
      if (*bucket > total || segment_counts[total - *bucket] == unreachable) {
        continue;
      }
      const std::size_t candidate = segment_counts[total - *bucket] + 1;
      if (candidate < segment_counts[total]) {
        segment_counts[total] = candidate;
        previous_totals[total] = total - *bucket;
        selected_buckets[total] = *bucket;
      }
    }
  }
  std::size_t scheduled_tokens = unmatched_tokens;
  while (scheduled_tokens <= search_limit &&
         segment_counts[scheduled_tokens] == unreachable) {
    ++scheduled_tokens;
  }
  if (scheduled_tokens > search_limit) {
    throw std::invalid_argument(
        "native prompt execution has no composable prefill plan");
  }
  std::vector<std::size_t> buckets;
  for (std::size_t cursor = scheduled_tokens; cursor != 0;
       cursor = previous_totals[cursor]) {
    const std::size_t bucket = selected_buckets[cursor];
    if (bucket == 0 || previous_totals[cursor] == unreachable) {
      throw std::runtime_error("native prompt prefill plan is incomplete");
    }
    buckets.push_back(bucket);
  }
  std::sort(buckets.begin(), buckets.end(), std::greater<std::size_t>());

  std::size_t input_offset = matched_prefix_tokens;
  std::size_t remaining = unmatched_tokens;
  for (const std::size_t bucket : buckets) {
    const std::size_t tokens = std::min(remaining, bucket);
    plan.aot_segments.push_back({input_offset, tokens, bucket});
    input_offset += tokens;
    remaining -= tokens;
  }
  if (remaining != 0 || input_offset != input_tokens) {
    throw std::runtime_error("native prompt prefill plan did not cover input");
  }
  plan.cold_aot_tokens = unmatched_tokens;
  plan.aot_bucket_tokens = scheduled_tokens;
  plan.prompt_decode_start = input_tokens;
  if (plan.prefix_extension_hit) {
    plan.mode = NativePromptExecutionMode::kPrefixCachePlusAot;
  } else if (plan.aot_segments.size() > 1) {
    plan.mode = NativePromptExecutionMode::kColdAotComposed;
  } else if (plan.padded_aot()) {
    plan.mode = NativePromptExecutionMode::kColdAotPadded;
  } else {
    plan.mode = NativePromptExecutionMode::kColdAot;
  }
  return plan;
}

inline NativePromptExecutionPlan plan_native_prompt_execution(
    std::size_t input_tokens, std::size_t matched_prefix_tokens,
    std::size_t static_prefill_tokens) {
  return plan_native_prompt_execution(
      input_tokens, matched_prefix_tokens,
      std::vector<std::size_t>{static_prefill_tokens});
}

inline const char* native_prompt_execution_mode_name(
    NativePromptExecutionMode mode) {
  switch (mode) {
    case NativePromptExecutionMode::kColdDecodeFallback:
      return "cold-decode-fallback";
    case NativePromptExecutionMode::kColdAot:
      return "cold-aot";
    case NativePromptExecutionMode::kColdAotPadded:
      return "cold-aot-padded";
    case NativePromptExecutionMode::kColdAotComposed:
      return "cold-aot-composed";
    case NativePromptExecutionMode::kColdAotPlusDecode:
      return "cold-aot-plus-decode";
    case NativePromptExecutionMode::kPrefixCacheExact:
      return "prefix-cache-exact";
    case NativePromptExecutionMode::kPrefixCachePlusAot:
      return "prefix-cache-plus-aot";
    case NativePromptExecutionMode::kPrefixCachePlusDecode:
      return "prefix-cache-plus-decode";
  }
  throw std::invalid_argument("native prompt execution mode is invalid");
}

}  // namespace aima
