#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

namespace aima {

// Runs the dependency-free, single-request-at-a-time resident HTTP service.
// Argument parsing is owned here so the same surface is used by the real
// binary inside the portable bundle.
int run_native_http_server(int argc, char** argv);

}  // namespace aima
