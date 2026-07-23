// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_prefill_workspace.h"

#include "aima/prefill_schedule.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aima {
namespace {

constexpr std::uint64_t kAlignment = 256;

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
  DecodeTensorDtype first_dtype = DecodeTensorDtype::kNone;
  bool mixed_dtype = false;
  DecodeBindingKind binding_kind = DecodeBindingKind::kNone;
};

}  // namespace

NativePrefillWorkspace::~NativePrefillWorkspace() { reset(); }

NativePrefillWorkspaceMetrics NativePrefillWorkspace::build(
    int device, std::size_t context_tokens) {
  if (built() || !views_.empty() || !name_to_index_.empty()) {
    throw std::runtime_error("native prefill workspace is already built");
  }
  if (context_tokens == 0 || context_tokens > 262144) {
    throw std::invalid_argument("unsupported native prefill context");
  }

  std::vector<Plan> plans;
  std::unordered_map<std::string, std::size_t> plan_indices;
  plans.reserve(74);
  plan_indices.reserve(74);
  NativePrefillWorkspaceMetrics metrics;
  std::size_t launch_count = 0;
  const DecodeLaunch* launches =
      native_prefill_schedule(context_tokens, &launch_count);
  if (launches == nullptr) {
    throw std::runtime_error("native prefill schedule is unavailable");
  }
  const bool split_projection_tail =
      context_tokens != 8192 && launch_count == 401 && launch_count > 1 &&
      launches[1].symbol != nullptr &&
      std::string(launches[1].symbol) == "_causal_conv1d_fwd_kernel";
  for (std::size_t launch_index = 0; launch_index < launch_count;
       ++launch_index) {
    const DecodeLaunch& launch = launches[launch_index];
    for (std::size_t argument_index = 0;
         argument_index < launch.argument_count; ++argument_index) {
      const DecodeArgument& argument = launch.arguments[argument_index];
      if (argument.kind != DecodeArgumentKind::kTensor ||
          argument.binding_kind == DecodeBindingKind::kModelOrDerivedWeight) {
        continue;
      }
      ++metrics.schedule_tensor_arguments;
      if (argument.binding == nullptr || argument.storage_bytes == 0 ||
          argument.byte_offset >= argument.storage_bytes) {
        throw std::runtime_error(
            "invalid native prefill workspace schedule geometry");
      }
      const std::string name(argument.binding);
      const auto found = plan_indices.find(name);
      if (found != plan_indices.end()) {
        Plan& plan = plans[found->second];
        if (plan.binding_kind != argument.binding_kind) {
          throw std::runtime_error(
              "native prefill workspace lifetime drift: " + name);
        }
        plan.bytes = std::max(plan.bytes, argument.storage_bytes);
        plan.mixed_dtype =
            plan.mixed_dtype || plan.first_dtype != argument.tensor_dtype;
        continue;
      }
      plan_indices.emplace(name, plans.size());
      plans.push_back({name, argument.storage_bytes, 0, argument.tensor_dtype,
                       false, argument.binding_kind});
    }
  }

  std::uint64_t allocation_bytes = 0;
  for (Plan& plan : plans) {
    plan.offset = allocation_bytes;
    allocation_bytes += align_up(plan.bytes);
    metrics.logical_payload_bytes += plan.bytes;
    if (plan.binding_kind == DecodeBindingKind::kResidentStateOrWorkspace) {
      ++metrics.resident_bindings;
    } else if (plan.binding_kind == DecodeBindingKind::kTransientWorkspace) {
      ++metrics.transient_bindings;
    } else {
      throw std::runtime_error(
          "invalid non-weight native prefill workspace binding kind");
    }
    if (plan.mixed_dtype) ++metrics.mixed_dtype_bindings;
  }
  metrics.unique_bindings = plans.size();
  metrics.allocation_bytes = allocation_bytes;
  const bool q8192_closure =
      context_tokens == 8192 && launch_count == 431 &&
      metrics.schedule_tensor_arguments == 2192 &&
      metrics.unique_bindings == 73 && metrics.resident_bindings == 35 &&
      metrics.transient_bindings == 38 && metrics.mixed_dtype_bindings == 2 &&
      metrics.logical_payload_bytes == 1407481841ULL &&
      metrics.allocation_bytes == 1407482880ULL;
  const bool direct_closure =
      context_tokens != 8192 && !split_projection_tail &&
      launch_count == 401 &&
      metrics.schedule_tensor_arguments == 1952 &&
      metrics.unique_bindings != 0 && metrics.resident_bindings != 0 &&
      metrics.transient_bindings != 0 &&
      metrics.resident_bindings + metrics.transient_bindings ==
          metrics.unique_bindings &&
      metrics.mixed_dtype_bindings <= metrics.unique_bindings &&
      metrics.logical_payload_bytes != 0 &&
      metrics.allocation_bytes >= metrics.logical_payload_bytes &&
      metrics.allocation_bytes - metrics.logical_payload_bytes <
          metrics.unique_bindings * kAlignment;
  const bool split_projection_tail_closure =
      split_projection_tail && metrics.schedule_tensor_arguments == 2072 &&
      metrics.resident_bindings == 34 &&
      ((context_tokens == 8191 && metrics.unique_bindings == 79 &&
        metrics.transient_bindings == 45 &&
        metrics.mixed_dtype_bindings == 3) ||
       (context_tokens != 8191 && metrics.unique_bindings == 71 &&
        metrics.transient_bindings == 37 &&
        metrics.mixed_dtype_bindings == 1)) &&
      metrics.logical_payload_bytes != 0 &&
      metrics.allocation_bytes >= metrics.logical_payload_bytes &&
      metrics.allocation_bytes - metrics.logical_payload_bytes <
          metrics.unique_bindings * kAlignment;
  if (!q8192_closure && !direct_closure &&
      !split_projection_tail_closure) {
    throw std::runtime_error("native prefill workspace closure count mismatch");
  }

  // Non-power-of-two near-capacity tails retain the source split-projection
  // convolution path but use native gated norm and FMHA.  Their Torch-only
  // intermediates are not present in the captured Triton lifetime plan, so
  // add one parameterized O(1) scratch set to the native owner.
  auto add_runtime_scratch = [&](const char* name, std::uint64_t bytes,
                                 DecodeTensorDtype dtype) {
    plans.push_back({name, bytes, allocation_bytes, dtype, false,
                     DecodeBindingKind::kTransientWorkspace});
    allocation_bytes += align_up(bytes);
    ++metrics.runtime_scratch_bindings;
    metrics.runtime_scratch_payload_bytes += bytes;
  };
  if (split_projection_tail) {
    add_runtime_scratch(
        "native.tail_linear_gate",
        context_tokens * 4096ULL * sizeof(std::uint16_t),
        DecodeTensorDtype::kBfloat16);
    add_runtime_scratch(
        "native.tail_full_q_gate",
        context_tokens * 8192ULL * sizeof(std::uint16_t),
        DecodeTensorDtype::kBfloat16);
    add_runtime_scratch(
        "native.tail_full_q",
        context_tokens * 4096ULL * sizeof(std::uint16_t),
        DecodeTensorDtype::kBfloat16);
    add_runtime_scratch(
        "native.tail_full_raw_k",
        context_tokens * 512ULL * sizeof(std::uint16_t),
        DecodeTensorDtype::kBfloat16);
    add_runtime_scratch(
        "native.tail_full_attention_f32",
        context_tokens * 4096ULL * sizeof(float),
        DecodeTensorDtype::kFloat32);
    add_runtime_scratch(
        "native.tail_rotary_cos",
        context_tokens * 32ULL * sizeof(float),
        DecodeTensorDtype::kFloat32);
    add_runtime_scratch(
        "native.tail_rotary_sin",
        context_tokens * 32ULL * sizeof(float),
        DecodeTensorDtype::kFloat32);
  }
  // One resident upload buffer turns host token ids into the layer-0 input.
  add_runtime_scratch(
      "native.prompt_token_ids",
      context_tokens * sizeof(std::uint32_t), DecodeTensorDtype::kInt32);
  metrics.total_bindings =
      metrics.unique_bindings + metrics.runtime_scratch_bindings;
  metrics.total_logical_payload_bytes =
      metrics.logical_payload_bytes + metrics.runtime_scratch_payload_bytes;
  metrics.allocation_bytes = allocation_bytes;
  const bool q8192_total =
      context_tokens == 8192 && metrics.total_bindings == 74 &&
      metrics.total_logical_payload_bytes == 1407514609ULL &&
      metrics.allocation_bytes == 1407515648ULL;
  const bool direct_total =
      context_tokens != 8192 &&
      metrics.total_bindings ==
          metrics.unique_bindings + (split_projection_tail ? 8 : 1) &&
      metrics.total_logical_payload_bytes ==
          metrics.logical_payload_bytes + metrics.runtime_scratch_payload_bytes &&
      metrics.allocation_bytes >= metrics.total_logical_payload_bytes &&
      metrics.allocation_bytes - metrics.total_logical_payload_bytes <
          metrics.total_bindings * kAlignment;
  if (!q8192_total && !direct_total) {
    throw std::runtime_error(
        "native prefill runtime scratch closure count mismatch");
  }

  const auto started = std::chrono::steady_clock::now();
  device_ = device;
  check_hip(hipSetDevice(device_), "hipSetDevice native prefill workspace");
  try {
    check_hip(hipMalloc(&allocation_, allocation_bytes),
              "hipMalloc native prefill workspace");
    allocation_bytes_ = allocation_bytes;
    context_tokens_ = context_tokens;
    check_hip(hipMemset(allocation_, 0, allocation_bytes),
              "hipMemset native prefill workspace");
    auto* base = static_cast<unsigned char*>(allocation_);
    views_.reserve(plans.size());
    name_to_index_.reserve(plans.size());
    for (const Plan& plan : plans) {
      const std::size_t index = views_.size();
      views_.push_back(
          {plan.name, base + plan.offset, plan.bytes, plan.binding_kind});
      name_to_index_.emplace(plan.name, index);
    }
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native prefill workspace");
    metrics.allocation_and_zero_ms = elapsed_ms(started);
    return metrics;
  } catch (...) {
    reset();
    throw;
  }
}

const NativePrefillWorkspaceView* NativePrefillWorkspace::find(
    std::string_view name) const {
  const auto found = name_to_index_.find(std::string(name));
  return found == name_to_index_.end() ? nullptr : &views_[found->second];
}

void NativePrefillWorkspace::reset() noexcept {
  (void)hipSetDevice(device_);
  if (allocation_) (void)hipFree(allocation_);
  allocation_ = nullptr;
  allocation_bytes_ = 0;
  context_tokens_ = 0;
  views_.clear();
  name_to_index_.clear();
}

}  // namespace aima
