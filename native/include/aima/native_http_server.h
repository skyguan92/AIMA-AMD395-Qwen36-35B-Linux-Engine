#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

namespace aima {

// Runs the dependency-free, single-request-at-a-time resident HTTP service.
// Argument parsing is owned here so the same surface is used by the real
// binary inside the portable bundle.
int run_native_http_server(int argc, char** argv);

// Qualification-only resident VL generation diagnosis. The cases file binds
// real chat/media requests to frozen full-vocabulary reference rows; no oracle
// path is accepted by or reachable from the product HTTP service.
int run_native_vl_generation_logits_probe(int argc, char** argv);

}  // namespace aima
