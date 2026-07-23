// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_invocation.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace aima {

NativeDecodeInvocationMetrics NativeDecodeInvocations::build(
    const NativeDecodeBindings& bindings,
    const NativeDecodeWorkspace& workspace) {
  if (!launches_.empty() || !workspace.built()) {
    throw std::runtime_error("native decode invocations require fresh complete owners");
  }
  linear_state_buffers_swapped_ = false;
  NativeDecodeInvocationMetrics metrics;
  std::size_t launch_count = 0;
  const DecodeLaunch* schedule = native_decode_schedule(&launch_count);
  launches_.reserve(launch_count);
  for (std::size_t launch_index = 0; launch_index < launch_count; ++launch_index) {
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
                "native decode model invocation binding mismatch: " +
                std::string(argument.binding));
          }
          base_pointer = view->device_pointer;
          payload_bytes = view->payload_bytes;
          ++metrics.model_tensor_arguments;
        } else {
          const NativeDecodeWorkspaceView* view = workspace.find(argument.binding);
          if (view == nullptr || view->dtype != argument.tensor_dtype ||
              view->binding_kind != argument.binding_kind) {
            throw std::runtime_error(
                "native decode workspace invocation binding mismatch: " +
                std::string(argument.binding));
          }
          base_pointer = view->device_pointer;
          payload_bytes = view->payload_bytes;
          ++metrics.workspace_tensor_arguments;
        }
        if (base_pointer == nullptr || payload_bytes != argument.storage_bytes ||
            argument.byte_offset >= payload_bytes) {
          throw std::runtime_error(
              "native decode invocation pointer offset is out of range: " +
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
        throw std::runtime_error("unsupported native decode invocation ABI kind");
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
  if (metrics.launch_count != 402 || metrics.abi_argument_count != 1927 ||
      metrics.tensor_argument_count != 1777 ||
      metrics.model_tensor_arguments != 423 ||
      metrics.workspace_tensor_arguments != 1354 ||
      metrics.scalar_argument_count != 150 ||
      metrics.pointer_offset_checks != 1777) {
    throw std::runtime_error("native decode invocation ABI closure count mismatch");
  }
  return metrics;
}

void* NativeDecodeInvocations::tensor_pointer(
    std::size_t launch_index, std::string_view argument_name) const {
  if (launch_index >= launches_.size()) {
    throw std::out_of_range("native decode invocation index is out of range");
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
  throw std::runtime_error("native decode tensor argument is missing: " +
                           std::string(argument_name));
}

std::size_t NativeDecodeInvocations::swap_linear_decode_state_buffers() {
  if (launches_.size() != 402) {
    throw std::runtime_error("native decode state swap requires 402 launches");
  }
  const auto swap_named = [](PreparedDecodeInvocation& invocation,
                             std::string_view left,
                             std::string_view right) {
    std::size_t left_index = invocation.slots.size();
    std::size_t right_index = invocation.slots.size();
    for (std::size_t index = 0; index < invocation.launch->argument_count;
         ++index) {
      const DecodeArgument& argument = invocation.launch->arguments[index];
      if (argument.kind != DecodeArgumentKind::kTensor) continue;
      if (left == argument.name) left_index = index;
      if (right == argument.name) right_index = index;
    }
    if (left_index == invocation.slots.size() ||
        right_index == invocation.slots.size()) {
      throw std::runtime_error("native decode state swap argument is missing");
    }
    std::swap(invocation.slots[left_index].device_pointer,
              invocation.slots[right_index].device_pointer);
  };

  std::size_t swaps = 0;
  for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
    const std::size_t base = layer_index * 10;
    if (std::string(launches_[base + 1].launch->symbol) !=
        "triton_fused_input_proj_conv_kernel") {
      continue;
    }
    swap_named(launches_[base + 1], "state_in", "state_out");
    swap_named(launches_[base + 2], "h0", "ht");
    swaps += 2;
  }
  if (swaps != 60) {
    throw std::runtime_error("native decode state swap closure is incomplete");
  }
  linear_state_buffers_swapped_ = !linear_state_buffers_swapped_;
  return swaps;
}

std::size_t NativeDecodeInvocations::reset_linear_decode_state_buffers() {
  return linear_state_buffers_swapped_
             ? swap_linear_decode_state_buffers()
             : 0;
}

}  // namespace aima
