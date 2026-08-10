#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_bindings.h"
#include "aima/native_decode_invocation.h"
#include "aima/native_prefill_workspace.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace aima {

struct NativePrefillInvocationMetrics {
  std::size_t launch_count = 0;
  std::size_t abi_argument_count = 0;
  std::size_t tensor_argument_count = 0;
  std::size_t model_tensor_arguments = 0;
  std::size_t workspace_tensor_arguments = 0;
  std::size_t scalar_argument_count = 0;
  std::size_t pointer_offset_checks = 0;
};

class NativePrefillInvocations {
 public:
  NativePrefillInvocationMetrics build(
      const NativeDecodeBindings& bindings,
      const NativePrefillWorkspace& workspace,
      std::size_t context_tokens = 8192);
  const std::vector<PreparedDecodeInvocation>& launches() const {
    return launches_;
  }
  void* tensor_pointer(std::size_t launch_index,
                       std::string_view argument_name) const;
  std::uint64_t tensor_storage_bytes(
      std::size_t launch_index, std::string_view argument_name) const;
  void rebind_tensor(std::size_t launch_index, std::string_view argument_name,
                     void* device_pointer);
  void set_int32_argument(std::size_t launch_index,
                          std::string_view argument_name,
                          std::int32_t value);

 private:
  std::vector<PreparedDecodeInvocation> launches_;
};

}  // namespace aima
