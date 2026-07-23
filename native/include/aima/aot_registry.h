#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <cstddef>
#include <cstdint>
#include <string>

namespace aima {

struct EmbeddedAotImage {
  const char* kernel_hash;
  const char* symbol;
  const unsigned char* image;
  std::size_t image_bytes;
  std::uint32_t num_warps;
  std::uint32_t warp_size;
  std::uint32_t shared_memory_bytes;
};

const EmbeddedAotImage* embedded_aot_images(std::size_t* count);
const EmbeddedAotImage* find_embedded_aot_image(const std::string& kernel_hash);

struct AotClosureProbeResult {
  std::string gpu_arch;
  std::size_t image_count = 0;
  std::size_t loaded_count = 0;
  std::size_t image_bytes = 0;
  std::size_t exact_bf16_elements = 0;
  std::size_t expected_bf16_elements = 0;
  double exact_probe_ms = 0.0;
};

AotClosureProbeResult probe_embedded_aot_closure();

}  // namespace aima
