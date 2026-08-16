#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/decode_schedule.h"
#include "aima/native_decode_bindings.h"
#include "aima/native_decode_workspace.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace aima {

union NativeDecodeAbiSlot {
  void* device_pointer;
  float float32_value;
  std::int32_t int32_value;
  std::int64_t int64_value;
};

struct PreparedDecodeInvocation {
  const DecodeLaunch* launch = nullptr;
  std::vector<NativeDecodeAbiSlot> slots;
  std::vector<void*> kernel_params;
};

struct NativeDecodeInvocationMetrics {
  std::size_t launch_count = 0;
  std::size_t abi_argument_count = 0;
  std::size_t tensor_argument_count = 0;
  std::size_t model_tensor_arguments = 0;
  std::size_t workspace_tensor_arguments = 0;
  std::size_t scalar_argument_count = 0;
  std::size_t pointer_offset_checks = 0;
};

class NativeDecodeInvocations {
 public:
  NativeDecodeInvocationMetrics build(
      const NativeDecodeBindings& bindings,
      const NativeDecodeWorkspace& workspace);
  const std::vector<PreparedDecodeInvocation>& launches() const {
    return launches_;
  }
  void* tensor_pointer(std::size_t launch_index,
                       std::string_view argument_name) const;
  std::size_t swap_linear_decode_conv_state_buffers();
  std::size_t swap_linear_decode_recurrent_state_buffers();
  // Prefill writes the canonical convolution owner while the current packed
  // recurrent kernel updates its canonical state in place. Restore only the
  // convolution pointer orientation before an unrelated resident request.
  std::size_t reset_linear_decode_conv_state_buffers();
  std::size_t reset_linear_decode_recurrent_state_buffers();
  bool linear_decode_conv_state_buffers_swapped() const {
    return linear_conv_state_buffers_swapped_;
  }
  bool linear_decode_recurrent_state_buffers_swapped() const {
    return linear_recurrent_state_buffers_swapped_;
  }

 private:
  std::vector<PreparedDecodeInvocation> launches_;
  bool linear_conv_state_buffers_swapped_ = false;
  bool linear_recurrent_state_buffers_swapped_ = false;
};

}  // namespace aima
