// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_chat_protocol.h"
#include "aima/native_mrope.h"
#include "aima/native_vl_embedding.h"
#include "aima/native_vl_processor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace aima {

class NativeTokenizer;
struct NativeVlPreparedRequest;

inline constexpr std::uint64_t kNativeVlDefaultMediaCacheBytes =
    4ULL * 1024ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kNativeVlDefaultMediaCacheEntries = 64;

// Byte- and entry-bounded LRU of decoded/processed media. Lookup happens only
// after the source bytes have been loaded and hashed, so a URL or pathname
// whose content changes cannot reuse stale pixels.
class NativeVlMediaCache {
 public:
  explicit NativeVlMediaCache(
      std::uint64_t capacity_bytes = kNativeVlDefaultMediaCacheBytes,
      std::size_t capacity_entries = kNativeVlDefaultMediaCacheEntries);
  ~NativeVlMediaCache();
  NativeVlMediaCache(const NativeVlMediaCache&) = delete;
  NativeVlMediaCache& operator=(const NativeVlMediaCache&) = delete;

  std::uint64_t capacity_bytes() const;
  std::uint64_t resident_bytes() const;
  std::size_t capacity_entries() const;
  std::size_t entries() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend NativeVlPreparedRequest prepare_native_vl_request(
      NativeTokenizer&, const NativePreparedChat&, const NativeMediaPolicy&,
      NativeVlMediaCache*);
};

struct NativeVlPreparationMetrics {
  std::size_t media_count = 0;
  std::size_t image_count = 0;
  std::size_t video_count = 0;
  std::uint64_t source_bytes = 0;
  std::size_t vision_patches = 0;
  std::size_t visual_tokens = 0;
  std::size_t media_cache_hits = 0;
  std::size_t media_cache_misses = 0;
  std::size_t media_cache_entries = 0;
  std::uint64_t media_cache_resident_bytes = 0;
  double media_load_wall_ms = 0.0;
  double media_decode_wall_ms = 0.0;
  double media_load_decode_wall_ms = 0.0;
  double processor_wall_ms = 0.0;
};

// Fully prepared CPU-side input for one native VL request. Pixel values are
// concatenated in ordered-media/grid order and remain BF16 [patches,1536].
// No source URL or filename crosses this boundary.
struct NativeVlPreparedRequest {
  std::vector<std::uint32_t> prompt_token_ids;
  std::vector<NativeVlGrid> grids;
  std::vector<std::uint16_t> pixel_values_bf16;
  std::vector<NativeVlEmbeddingSpan> embedding_spans;
  std::optional<NativeMropePlan> mrope_plan;
  std::string multimodal_cache_namespace;
  NativeVlPreparationMetrics metrics;
};

// Loads, decodes and processes every ordered media part, expands the frozen
// Qwen3-VL prompt, locates the resulting placeholder spans, and seals the
// request's M-RoPE and multimodal-prefix identities.
NativeVlPreparedRequest prepare_native_vl_request(
    NativeTokenizer& tokenizer, const NativePreparedChat& chat,
    const NativeMediaPolicy& policy, NativeVlMediaCache* media_cache = nullptr);

}  // namespace aima
