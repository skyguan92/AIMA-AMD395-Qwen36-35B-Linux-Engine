// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/decode_schedule.h"
#include "aima/prefill_schedule.h"

#include "aima/aot_registry.h"

#include <set>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

DecodeScheduleProbeResult probe_schedule(
    const DecodeLaunch* launches, std::size_t count,
    std::size_t expected_launches, std::size_t expected_layer_launches,
    std::size_t expected_final_launches, std::size_t expected_tensors,
    std::size_t expected_scalars, const char* schedule_sha256,
    const char* phase) {
  DecodeScheduleProbeResult result;
  if (launches == nullptr || count != expected_launches) {
    throw std::runtime_error(std::string("native ") + phase +
                             " schedule launch count mismatch");
  }
  std::set<std::string> hashes;
  for (std::size_t index = 0; index < count; ++index) {
    const DecodeLaunch& launch = launches[index];
    if (launch.sequence != index || launch.kernel_hash == nullptr ||
        launch.symbol == nullptr || launch.arguments == nullptr ||
        launch.argument_count == 0) {
      throw std::runtime_error(std::string("native ") + phase +
                               " schedule structural mismatch");
    }
    const EmbeddedAotImage* image = find_embedded_aot_image(launch.kernel_hash);
    if (image == nullptr || std::string(image->symbol) != launch.symbol ||
        image->num_warps != launch.config.num_warps ||
        image->warp_size != launch.config.warp_size ||
        image->shared_memory_bytes != launch.config.shared_memory_bytes) {
      throw std::runtime_error(std::string("native ") + phase +
                               " schedule AOT registry mismatch");
    }
    ++result.embedded_kernel_matches;
    hashes.emplace(launch.kernel_hash);
    if (launch.layer_index < 0) {
      ++result.final_logit_launch_count;
    } else {
      if (launch.layer_index >= 40) {
        throw std::runtime_error(std::string("native ") + phase +
                                 " schedule layer index out of range");
      }
      ++result.layer_launch_count;
    }
    for (std::size_t argument_index = 0;
         argument_index < launch.argument_count; ++argument_index) {
      const DecodeArgument& argument = launch.arguments[argument_index];
      if (argument.name == nullptr || argument.abi_type == nullptr) {
        throw std::runtime_error(std::string("native ") + phase +
                                 " schedule argument metadata is missing");
      }
      if (argument.kind != DecodeArgumentKind::kTensor) {
        ++result.scalar_argument_count;
        continue;
      }
      ++result.tensor_argument_count;
      if (argument.binding == nullptr || argument.binding[0] == '\0' ||
          argument.storage_bytes == 0 ||
          argument.tensor_dtype == DecodeTensorDtype::kNone) {
        throw std::runtime_error(std::string("native ") + phase +
                                 " tensor binding is incomplete");
      }
      switch (argument.binding_kind) {
        case DecodeBindingKind::kModelOrDerivedWeight:
          ++result.model_binding_arguments;
          break;
        case DecodeBindingKind::kResidentStateOrWorkspace:
          ++result.resident_binding_arguments;
          break;
        case DecodeBindingKind::kTransientWorkspace:
          ++result.transient_binding_arguments;
          break;
        case DecodeBindingKind::kNone:
          throw std::runtime_error(std::string("native ") + phase +
                                   " tensor binding kind is absent");
      }
    }
  }
  result.launch_count = count;
  result.unique_kernel_count = hashes.size();
  result.schedule_sha256 = schedule_sha256;
  if (result.layer_launch_count != expected_layer_launches ||
      result.final_logit_launch_count != expected_final_launches ||
      result.tensor_argument_count != expected_tensors ||
      result.scalar_argument_count != expected_scalars ||
      result.embedded_kernel_matches != count) {
    throw std::runtime_error(std::string("native ") + phase +
                             " schedule closure count mismatch");
  }
  return result;
}

}  // namespace

DecodeScheduleProbeResult probe_native_decode_schedule() {
  std::size_t count = 0;
  const DecodeLaunch* launches = native_decode_schedule(&count);
  return probe_schedule(launches, count, 402, 400, 2, 1777, 150,
                        native_decode_schedule_sha256(), "decode");
}

DecodeScheduleProbeResult probe_native_prefill_schedule() {
  std::size_t count = 0;
  const DecodeLaunch* launches = native_prefill_schedule(&count);
  return probe_schedule(launches, count, 431, 430, 1, 2474, 1650,
                        native_prefill_schedule_sha256(), "prefill");
}

}  // namespace aima
