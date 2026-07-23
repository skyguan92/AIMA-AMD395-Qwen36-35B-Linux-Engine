#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

namespace aima {

// Opaque HIP resources used to run the shared-expert branch concurrently with
// the routed-expert branch. The resident engine owns these resources.
struct NativeMoeOverlapResources {
  void* auxiliary_stream = nullptr;
  void* branch_ready_event = nullptr;
  void* shared_done_event = nullptr;

  bool valid() const {
    return auxiliary_stream != nullptr && branch_ready_event != nullptr &&
           shared_done_event != nullptr;
  }
};

}  // namespace aima
