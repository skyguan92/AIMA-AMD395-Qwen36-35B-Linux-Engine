#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/decode_schedule.h"

#include <cstddef>

namespace aima {

enum class NativePrefillScheduleKind {
  kDefault,
  kFrozenText,
};

// Prefill and decode share the same low-level AOT ABI descriptor.  Separate
// registries keep their launch orders and qualification hashes independent.
const DecodeLaunch* native_prefill_schedule(std::size_t context_tokens,
                                            std::size_t* count);
const char* native_prefill_schedule_sha256(std::size_t context_tokens);

// The no-context overload remains the q8192 compatibility surface used by
// the original closure probe. Product execution always selects explicitly.
const DecodeLaunch* native_prefill_schedule(std::size_t* count);
const char* native_prefill_schedule_sha256();

// q1024 VL work uses a recaptured closure, while text-only requests retain
// the exact v1.5.1 schedule. Both registries are embedded and share one union
// workspace; request dispatch selects the invocation table explicitly.
const DecodeLaunch* native_frozen_text_prefill_schedule(
    std::size_t context_tokens, std::size_t* count);
const char* native_frozen_text_prefill_schedule_sha256(
    std::size_t context_tokens);
const DecodeLaunch* native_frozen_text_prefill_schedule(std::size_t* count);
const char* native_frozen_text_prefill_schedule_sha256();

DecodeScheduleProbeResult probe_native_prefill_schedule();

}  // namespace aima
