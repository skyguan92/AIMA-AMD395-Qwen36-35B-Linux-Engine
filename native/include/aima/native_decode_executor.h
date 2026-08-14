#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/aot_kernel.h"
#include "aima/native_decode_invocation.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace aima {

struct NativeDecodeExecutorMetrics {
  std::size_t embedded_images = 0;
  std::size_t loaded_modules = 0;
  std::size_t launched_kernels = 0;
  std::size_t launched_abi_arguments = 0;
  double module_load_wall_ms = 0.0;
};

// Persistent owner for the embedded Triton/HIP code objects used by the
// captured decode schedule.  Modules are loaded once and addressed by the
// schedule's content hash; request execution performs no file or JSON I/O.
class NativeDecodeExecutor {
 public:
  NativeDecodeExecutor() = default;
  ~NativeDecodeExecutor() = default;
  NativeDecodeExecutor(const NativeDecodeExecutor&) = delete;
  NativeDecodeExecutor& operator=(const NativeDecodeExecutor&) = delete;

  NativeDecodeExecutorMetrics load();
  void launch(const PreparedDecodeInvocation& invocation,
              void* stream = nullptr);
  // Launches an embedded code object that is resident in this executor but is
  // not part of a captured decode/prefill schedule.
  void launch_embedded(const std::string& kernel_hash,
                       const AotLaunchConfig& config,
                       const std::vector<void*>& kernel_params,
                       void* stream = nullptr);
  bool loaded() const { return !kernels_.empty(); }
  const NativeDecodeExecutorMetrics& metrics() const { return metrics_; }

 private:
  std::vector<std::unique_ptr<AotKernel>> kernels_;
  std::unordered_map<std::string, std::size_t> hash_to_index_;
  NativeDecodeExecutorMetrics metrics_;
};

}  // namespace aima
