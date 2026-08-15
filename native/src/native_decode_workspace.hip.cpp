// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_workspace.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aima {
namespace {

constexpr std::uint64_t kAlignment = 256;
constexpr std::uint64_t kRecurrentStateBytes =
    32ULL * 128ULL * 128ULL * sizeof(float);
constexpr std::size_t kLinearLayerCount = 30;
constexpr std::size_t kModelLayerCount = 40;
constexpr std::uint64_t kPackedStateIndexBytes =
    kModelLayerCount * sizeof(std::int32_t);
constexpr std::uint64_t kNativeRuntimeScratchBytes =
    4221952ULL + kRecurrentStateBytes + kPackedStateIndexBytes;
constexpr std::string_view kInitialRecurrentPrefix =
    "linear_attention_initial_ssm_states_vllm.";

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorName(status) + " (" +
                             hipGetErrorString(status) + ")");
  }
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

std::uint64_t align_up(std::uint64_t value) {
  return (value + kAlignment - 1) / kAlignment * kAlignment;
}

struct Plan {
  std::string name;
  std::uint64_t bytes = 0;
  std::uint64_t offset = 0;
  DecodeTensorDtype dtype = DecodeTensorDtype::kNone;
  DecodeBindingKind binding_kind = DecodeBindingKind::kNone;
};

bool is_initial_recurrent_state(const Plan& plan) {
  return plan.name.rfind(kInitialRecurrentPrefix, 0) == 0;
}

std::size_t recurrent_state_layer(const Plan& plan) {
  return std::stoul(plan.name.substr(kInitialRecurrentPrefix.size()));
}

const std::array<std::int32_t, kModelLayerCount>& packed_state_indices() {
  static const std::array<std::int32_t, kModelLayerCount> indices = [] {
    std::array<std::int32_t, kModelLayerCount> result{};
    std::int32_t state_index = 1;
    for (std::size_t layer_index = 0; layer_index < result.size();
         ++layer_index) {
      if (layer_index % 4 != 3) result[layer_index] = state_index++;
    }
    return result;
  }();
  return indices;
}

}  // namespace

NativeDecodeWorkspace::~NativeDecodeWorkspace() { reset(); }

NativeDecodeWorkspaceMetrics NativeDecodeWorkspace::build(int device) {
  if (built() || !views_.empty() || !name_to_index_.empty()) {
    throw std::runtime_error("native decode workspace is already built");
  }
  std::vector<Plan> plans;
  std::unordered_map<std::string, std::size_t> plan_indices;
  plans.reserve(576);
  plan_indices.reserve(566);
  NativeDecodeWorkspaceMetrics metrics;
  std::size_t launch_count = 0;
  const DecodeLaunch* launches = native_decode_schedule(&launch_count);
  for (std::size_t launch_index = 0; launch_index < launch_count; ++launch_index) {
    const DecodeLaunch& launch = launches[launch_index];
    for (std::size_t argument_index = 0;
         argument_index < launch.argument_count; ++argument_index) {
      const DecodeArgument& argument = launch.arguments[argument_index];
      if (argument.kind != DecodeArgumentKind::kTensor ||
          argument.binding_kind == DecodeBindingKind::kModelOrDerivedWeight) {
        continue;
      }
      ++metrics.schedule_tensor_arguments;
      const std::string name(argument.binding);
      const auto found = plan_indices.find(name);
      if (found != plan_indices.end()) {
        const Plan& plan = plans[found->second];
        if (plan.bytes != argument.storage_bytes ||
            plan.dtype != argument.tensor_dtype ||
            plan.binding_kind != argument.binding_kind) {
          throw std::runtime_error(
              "native decode workspace alias geometry drift: " + name);
        }
        continue;
      }
      plan_indices.emplace(name, plans.size());
      plans.push_back(
          {name, argument.storage_bytes, 0, argument.tensor_dtype,
           argument.binding_kind});
    }
  }

  std::vector<std::size_t> recurrent_state_plans;
  recurrent_state_plans.reserve(kLinearLayerCount);
  std::uint64_t allocation_bytes = 0;
  for (Plan& plan : plans) {
    metrics.logical_payload_bytes += plan.bytes;
    if (plan.binding_kind == DecodeBindingKind::kResidentStateOrWorkspace) {
      ++metrics.resident_bindings;
    } else if (plan.binding_kind == DecodeBindingKind::kTransientWorkspace) {
      ++metrics.transient_bindings;
    } else {
      throw std::runtime_error("invalid non-weight decode workspace binding kind");
    }
    if (is_initial_recurrent_state(plan)) {
      if (plan.bytes != kRecurrentStateBytes ||
          plan.dtype != DecodeTensorDtype::kFloat32) {
        throw std::runtime_error(
            "native packed recurrent state geometry drift");
      }
      recurrent_state_plans.push_back(&plan - plans.data());
    } else {
      plan.offset = allocation_bytes;
      allocation_bytes += align_up(plan.bytes);
    }
  }
  metrics.unique_bindings = plans.size();
  metrics.allocation_bytes = allocation_bytes;
  if (metrics.schedule_tensor_arguments != 1354 ||
      metrics.unique_bindings != 566 || metrics.resident_bindings != 556 ||
      metrics.transient_bindings != 10 ||
      metrics.logical_payload_bytes != 131922896ULL) {
    throw std::runtime_error("native decode workspace closure count mismatch");
  }

  std::sort(recurrent_state_plans.begin(), recurrent_state_plans.end(),
            [&](std::size_t left, std::size_t right) {
              return recurrent_state_layer(plans[left]) <
                     recurrent_state_layer(plans[right]);
            });
  if (recurrent_state_plans.size() != kLinearLayerCount) {
    throw std::runtime_error(
        "native packed recurrent state layer closure mismatch");
  }
  const std::uint64_t packed_state_base_offset = allocation_bytes;
  allocation_bytes += align_up(kRecurrentStateBytes);
  for (std::size_t ordinal = 0; ordinal < recurrent_state_plans.size();
       ++ordinal) {
    Plan& plan = plans[recurrent_state_plans[ordinal]];
    const std::size_t layer_index = recurrent_state_layer(plan);
    const std::size_t expected_state_index =
        layer_index - layer_index / 4 + 1;
    if (layer_index % 4 == 3 || expected_state_index != ordinal + 1) {
      throw std::runtime_error(
          "native packed recurrent state ordering drift");
    }
    plan.offset = allocation_bytes;
    allocation_bytes += align_up(plan.bytes);
  }

  plans.push_back(
      {"native.linear.packed_ssm_state_base", kRecurrentStateBytes,
       packed_state_base_offset, DecodeTensorDtype::kFloat32,
       DecodeBindingKind::kResidentStateOrWorkspace});
  ++metrics.runtime_scratch_bindings;
  metrics.runtime_scratch_payload_bytes += kRecurrentStateBytes;

  // These O(1) scratch buffers cover the non-AOT layer boundaries and the
  // certified LM-head shortlist.  They are deliberately outside the captured
  // schedule binding counts so the schedule closure remains independently
  // auditable while the executor never aliases live schedule tensors.
  const std::vector<Plan> runtime_scratch = {
      {"native.linear.attention_output", 4096, 0,
       DecodeTensorDtype::kBfloat16, DecodeBindingKind::kTransientWorkspace},
      {"native.linear.shared_activation", 1024, 0,
       DecodeTensorDtype::kBfloat16, DecodeBindingKind::kTransientWorkspace},
      {"native.linear.shared_down", 4096, 0,
       DecodeTensorDtype::kBfloat16, DecodeBindingKind::kTransientWorkspace},
      {"native.linear.shared_scaled", 4096, 0,
       DecodeTensorDtype::kBfloat16, DecodeBindingKind::kTransientWorkspace},
      {"native.linear.combined_moe", 4096, 0,
       DecodeTensorDtype::kBfloat16, DecodeBindingKind::kTransientWorkspace},
      {"native.lm_head.candidate_weights", 4194304, 0,
       DecodeTensorDtype::kBfloat16, DecodeBindingKind::kTransientWorkspace},
      {"native.lm_head.candidate_logits", 2048, 0,
       DecodeTensorDtype::kBfloat16, DecodeBindingKind::kTransientWorkspace},
      {"native.lm_head.certificate_scratch", 8192, 0,
       DecodeTensorDtype::kNone, DecodeBindingKind::kTransientWorkspace},
      {"native.linear.packed_ssm_state_indices", kPackedStateIndexBytes, 0,
       DecodeTensorDtype::kInt32,
       DecodeBindingKind::kResidentStateOrWorkspace},
  };
  for (const Plan& scratch : runtime_scratch) {
    Plan plan = scratch;
    plan.offset = allocation_bytes;
    allocation_bytes += align_up(plan.bytes);
    plans.push_back(std::move(plan));
    ++metrics.runtime_scratch_bindings;
    metrics.runtime_scratch_payload_bytes += scratch.bytes;
  }
  metrics.total_bindings =
      metrics.unique_bindings + metrics.runtime_scratch_bindings;
  metrics.total_logical_payload_bytes =
      metrics.logical_payload_bytes + metrics.runtime_scratch_payload_bytes;
  if (metrics.runtime_scratch_bindings != 10 ||
      metrics.runtime_scratch_payload_bytes != kNativeRuntimeScratchBytes ||
      metrics.total_bindings != 576 ||
      metrics.total_logical_payload_bytes != 138242160ULL) {
    throw std::runtime_error("native decode runtime scratch closure mismatch");
  }
  metrics.allocation_bytes = allocation_bytes;

  const auto started = std::chrono::steady_clock::now();
  device_ = device;
  check_hip(hipSetDevice(device_), "hipSetDevice native decode workspace");
  try {
    check_hip(hipMalloc(&allocation_, allocation_bytes),
              "hipMalloc native decode workspace");
    allocation_bytes_ = allocation_bytes;
    check_hip(hipMemset(allocation_, 0, allocation_bytes),
              "hipMemset native decode workspace");
    auto* base = static_cast<unsigned char*>(allocation_);
    views_.reserve(plans.size());
    name_to_index_.reserve(plans.size());
    for (const Plan& plan : plans) {
      const std::size_t index = views_.size();
      views_.push_back({plan.name, base + plan.offset, plan.bytes, plan.dtype,
                        plan.binding_kind});
      name_to_index_.emplace(plan.name, index);
    }
    const NativeDecodeWorkspaceView* packed_indices =
        find("native.linear.packed_ssm_state_indices");
    if (packed_indices == nullptr || packed_indices->device_pointer == nullptr ||
        packed_indices->payload_bytes != kPackedStateIndexBytes ||
        packed_indices->dtype != DecodeTensorDtype::kInt32) {
      throw std::runtime_error(
          "native packed recurrent state indices are incomplete");
    }
    check_hip(hipMemcpy(packed_indices->device_pointer,
                        packed_state_indices().data(),
                        kPackedStateIndexBytes, hipMemcpyHostToDevice),
              "hipMemcpy native packed recurrent state indices");
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native decode workspace");
    metrics.allocation_and_zero_ms = elapsed_ms(started);
    return metrics;
  } catch (...) {
    reset();
    throw;
  }
}

const NativeDecodeWorkspaceView* NativeDecodeWorkspace::find(
    std::string_view name) const {
  const auto found = name_to_index_.find(std::string(name));
  return found == name_to_index_.end() ? nullptr : &views_[found->second];
}

std::uint64_t NativeDecodeWorkspace::clear(void* stream_value) {
  if (!built()) {
    throw std::runtime_error("native decode workspace is not built");
  }
  check_hip(hipMemsetAsync(allocation_, 0, allocation_bytes_,
                           static_cast<hipStream_t>(stream_value)),
            "hipMemsetAsync native decode workspace");
  const NativeDecodeWorkspaceView* packed_indices =
      find("native.linear.packed_ssm_state_indices");
  if (packed_indices == nullptr || packed_indices->device_pointer == nullptr ||
      packed_indices->payload_bytes != kPackedStateIndexBytes) {
    throw std::runtime_error(
        "native packed recurrent state indices are incomplete");
  }
  check_hip(hipMemcpyAsync(packed_indices->device_pointer,
                           packed_state_indices().data(),
                           kPackedStateIndexBytes, hipMemcpyHostToDevice,
                           static_cast<hipStream_t>(stream_value)),
            "hipMemcpyAsync native packed recurrent state indices");
  return allocation_bytes_;
}

void NativeDecodeWorkspace::reset() noexcept {
  (void)hipSetDevice(device_);
  if (allocation_) (void)hipFree(allocation_);
  allocation_ = nullptr;
  allocation_bytes_ = 0;
  views_.clear();
  name_to_index_.clear();
}

}  // namespace aima
