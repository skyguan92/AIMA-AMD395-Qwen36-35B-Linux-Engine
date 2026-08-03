#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

namespace aima {

// Runs non-destructive host, GPU, memory-layout, bundle and optional model
// checks without loading model weights or allocating the inference workspace.
int run_native_doctor(int argc, char** argv, const char* version,
                      const char* source_commit);

}  // namespace aima
