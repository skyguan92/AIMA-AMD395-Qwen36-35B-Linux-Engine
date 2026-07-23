#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <cstddef>
#include <cstdint>
#include <string>

namespace aima {

struct DecodeLaunchConfig {
  std::uint32_t grid_x = 1;
  std::uint32_t grid_y = 1;
  std::uint32_t grid_z = 1;
  std::uint32_t num_warps = 1;
  std::uint32_t warp_size = 32;
  std::uint32_t shared_memory_bytes = 0;
};

enum class DecodeArgumentKind : std::uint8_t {
  kTensor,
  kFloat32,
  kInt32,
  kInt64,
};

enum class DecodeTensorDtype : std::uint8_t {
  kNone,
  kBfloat16,
  kFloat32,
  kInt32,
  kInt8,
  kBool,
};

enum class DecodeBindingKind : std::uint8_t {
  kNone,
  kModelOrDerivedWeight,
  kResidentStateOrWorkspace,
  kTransientWorkspace,
};

struct DecodeArgument {
  const char* name;
  const char* abi_type;
  DecodeArgumentKind kind;
  DecodeTensorDtype tensor_dtype;
  DecodeBindingKind binding_kind;
  const char* binding;
  std::uint64_t storage_bytes;
  std::uint64_t byte_offset;
  float float32_value;
  std::int32_t int32_value;
  std::int64_t int64_value;
};

struct DecodeLaunch {
  std::uint16_t sequence;
  std::int16_t layer_index;
  const char* kernel_hash;
  const char* symbol;
  DecodeLaunchConfig config;
  const DecodeArgument* arguments;
  std::uint16_t argument_count;
};

const DecodeLaunch* native_decode_schedule(std::size_t* count);
const char* native_decode_schedule_sha256();

struct DecodeScheduleProbeResult {
  std::size_t launch_count = 0;
  std::size_t layer_launch_count = 0;
  std::size_t final_logit_launch_count = 0;
  std::size_t tensor_argument_count = 0;
  std::size_t scalar_argument_count = 0;
  std::size_t model_binding_arguments = 0;
  std::size_t resident_binding_arguments = 0;
  std::size_t transient_binding_arguments = 0;
  std::size_t unique_kernel_count = 0;
  std::size_t embedded_kernel_matches = 0;
  std::string schedule_sha256;
};

DecodeScheduleProbeResult probe_native_decode_schedule();

}  // namespace aima
