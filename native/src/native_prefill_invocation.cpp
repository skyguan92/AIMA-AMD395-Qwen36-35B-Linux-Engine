// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_prefill_invocation.h"

#include "aima/prefill_schedule.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace aima {

NativePrefillInvocationMetrics NativePrefillInvocations::build(
    const NativeDecodeBindings& bindings,
    const NativePrefillWorkspace& workspace,
    std::size_t context_tokens) {
  if (!launches_.empty() || bindings.views().empty() || !workspace.built()) {
    throw std::runtime_error(
        "native prefill invocations require fresh complete owners");
  }
  if (workspace.context_tokens() != context_tokens) {
    throw std::invalid_argument(
        "native prefill invocation context does not match workspace");
  }
  NativePrefillInvocationMetrics metrics;
  std::size_t launch_count = 0;
  const DecodeLaunch* schedule =
      native_prefill_schedule(context_tokens, &launch_count);
  if (schedule == nullptr) {
    throw std::runtime_error("native prefill schedule is unavailable");
  }
  launches_.reserve(launch_count);
  for (std::size_t launch_index = 0; launch_index < launch_count;
       ++launch_index) {
    const DecodeLaunch& launch = schedule[launch_index];
    PreparedDecodeInvocation prepared;
    prepared.launch = &launch;
    prepared.slots.reserve(launch.argument_count);
    for (std::size_t argument_index = 0;
         argument_index < launch.argument_count; ++argument_index) {
      const DecodeArgument& argument = launch.arguments[argument_index];
      NativeDecodeAbiSlot slot{};
      if (argument.kind == DecodeArgumentKind::kTensor) {
        ++metrics.tensor_argument_count;
        void* base_pointer = nullptr;
        std::uint64_t payload_bytes = 0;
        if (argument.binding_kind == DecodeBindingKind::kModelOrDerivedWeight) {
          const NativeDecodeBindingView* view = bindings.find(argument.binding);
          if (view == nullptr || view->dtype != argument.tensor_dtype) {
            throw std::runtime_error(
                "native prefill model invocation binding mismatch: " +
                std::string(argument.binding));
          }
          base_pointer = view->device_pointer;
          payload_bytes = view->payload_bytes;
          ++metrics.model_tensor_arguments;
        } else {
          const NativePrefillWorkspaceView* view =
              workspace.find(argument.binding);
          if (view == nullptr ||
              view->binding_kind != argument.binding_kind) {
            throw std::runtime_error(
                "native prefill workspace invocation binding mismatch: " +
                std::string(argument.binding));
          }
          base_pointer = view->device_pointer;
          payload_bytes = view->payload_bytes;
          ++metrics.workspace_tensor_arguments;
        }
        if (base_pointer == nullptr || payload_bytes < argument.storage_bytes ||
            argument.byte_offset >= argument.storage_bytes) {
          throw std::runtime_error(
              "native prefill invocation pointer offset is out of range: " +
              std::string(argument.binding));
        }
        slot.device_pointer =
            static_cast<unsigned char*>(base_pointer) + argument.byte_offset;
        ++metrics.pointer_offset_checks;
      } else if (argument.kind == DecodeArgumentKind::kFloat32) {
        slot.float32_value = argument.float32_value;
        ++metrics.scalar_argument_count;
      } else if (argument.kind == DecodeArgumentKind::kInt32) {
        slot.int32_value = argument.int32_value;
        ++metrics.scalar_argument_count;
      } else if (argument.kind == DecodeArgumentKind::kInt64) {
        slot.int64_value = argument.int64_value;
        ++metrics.scalar_argument_count;
      } else {
        throw std::runtime_error("unsupported native prefill invocation ABI kind");
      }
      prepared.slots.push_back(slot);
    }
    prepared.kernel_params.reserve(prepared.slots.size());
    for (NativeDecodeAbiSlot& slot : prepared.slots) {
      prepared.kernel_params.push_back(&slot);
    }
    metrics.abi_argument_count += prepared.kernel_params.size();
    launches_.push_back(std::move(prepared));
  }
  metrics.launch_count = launches_.size();
  const bool q8192_closure =
      context_tokens == 8192 && metrics.launch_count == 431 &&
      metrics.abi_argument_count == 4124 &&
      metrics.tensor_argument_count == 2474 &&
      metrics.model_tensor_arguments == 282 &&
      metrics.workspace_tensor_arguments == 2192 &&
      metrics.scalar_argument_count == 1650 &&
      metrics.pointer_offset_checks == 2474;
  const bool direct_closure =
      context_tokens != 8192 && metrics.launch_count == 401 &&
      metrics.abi_argument_count == 3824 &&
      metrics.tensor_argument_count == 2204 &&
      metrics.model_tensor_arguments == 252 &&
      metrics.workspace_tensor_arguments == 1952 &&
      metrics.scalar_argument_count == 1620 &&
      metrics.pointer_offset_checks == 2204;
  const bool split_projection_tail_closure =
      context_tokens != 8192 && metrics.launch_count == 401 &&
      metrics.abi_argument_count == 3974 &&
      metrics.tensor_argument_count == 2324 &&
      metrics.model_tensor_arguments == 252 &&
      metrics.workspace_tensor_arguments == 2072 &&
      metrics.scalar_argument_count == 1650 &&
      metrics.pointer_offset_checks == 2324;
  if (!q8192_closure && !direct_closure &&
      !split_projection_tail_closure) {
    throw std::runtime_error("native prefill invocation ABI closure count mismatch");
  }
  return metrics;
}

std::uint64_t NativePrefillInvocations::tensor_storage_bytes(
    std::size_t launch_index, std::string_view argument_name) const {
  if (launch_index >= launches_.size()) {
    throw std::out_of_range("native prefill invocation index is out of range");
  }
  const PreparedDecodeInvocation& invocation = launches_[launch_index];
  for (std::size_t index = 0; index < invocation.launch->argument_count;
       ++index) {
    const DecodeArgument& argument = invocation.launch->arguments[index];
    if (argument.kind == DecodeArgumentKind::kTensor &&
        argument_name == argument.name) {
      return argument.storage_bytes;
    }
  }
  throw std::runtime_error("native prefill tensor argument is missing: " +
                           std::string(argument_name));
}

void* NativePrefillInvocations::tensor_pointer(
    std::size_t launch_index, std::string_view argument_name) const {
  if (launch_index >= launches_.size()) {
    throw std::out_of_range("native prefill invocation index is out of range");
  }
  const PreparedDecodeInvocation& invocation = launches_[launch_index];
  for (std::size_t index = 0; index < invocation.launch->argument_count;
       ++index) {
    const DecodeArgument& argument = invocation.launch->arguments[index];
    if (argument.kind == DecodeArgumentKind::kTensor &&
        argument_name == argument.name) {
      return invocation.slots[index].device_pointer;
    }
  }
  throw std::runtime_error("native prefill tensor argument is missing: " +
                           std::string(argument_name));
}

void NativePrefillInvocations::rebind_tensor(
    std::size_t launch_index, std::string_view argument_name,
    void* device_pointer) {
  if (device_pointer == nullptr || launch_index >= launches_.size()) {
    throw std::invalid_argument(
        "native prefill tensor rebind has invalid geometry");
  }
  PreparedDecodeInvocation& invocation = launches_[launch_index];
  for (std::size_t index = 0; index < invocation.launch->argument_count;
       ++index) {
    const DecodeArgument& argument = invocation.launch->arguments[index];
    if (argument.kind == DecodeArgumentKind::kTensor &&
        argument_name == argument.name) {
      invocation.slots[index].device_pointer = device_pointer;
      return;
    }
  }
  throw std::runtime_error("native prefill tensor rebind argument is missing: " +
                           std::string(argument_name));
}

}  // namespace aima
