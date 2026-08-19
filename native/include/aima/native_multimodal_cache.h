// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_media.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aima {

inline constexpr std::uint32_t kNativeImagePadTokenId = 248056;
inline constexpr std::uint32_t kNativeVideoPadTokenId = 248057;

// A processed media span in the final language-model prompt. Source URLs and
// filenames are deliberately absent: cache identity follows decoded content.
struct NativeMultimodalCacheItem {
  NativeMediaKind kind = NativeMediaKind::kImage;
  std::string content_sha256;
  std::uint32_t placeholder_token_id = kNativeImagePadTokenId;
  std::size_t token_offset = 0;
  std::size_t token_length = 0;
};

struct NativeMultimodalCacheIdentityInput {
  // SHA-256 of the canonical effective processor configuration, including
  // resize/normalize/patch/sample parameters and relevant special tokens.
  std::string processor_config_sha256;
  // Request order is semantic and is preserved in the namespace digest.
  std::vector<NativeMultimodalCacheItem> media;
};

// Returns the versioned SHA-256 namespace used alongside the prompt-token
// vector. Both values form the prefix-cache identity.
std::string build_native_multimodal_cache_namespace(
    const NativeMultimodalCacheIdentityInput& input);

// Returns a distinct prompt-independent namespace for resident vision output
// reuse. Callers describe token_offset/token_length in ordered visual-output
// coordinates rather than final language-prompt coordinates.
std::string build_native_vision_embedding_cache_namespace(
    const NativeMultimodalCacheIdentityInput& input);

// Empty namespaces retain the v1.5.1 text-only behavior. A non-empty
// namespace must be one canonical lowercase SHA-256 digest.
bool valid_native_multimodal_cache_namespace(std::string_view value);

// Pure matching primitive shared by the GPU cache and CPU regression tests.
// Text tokens are compared as an exact prefix while the media namespace must
// match exactly. A changed media item/configuration therefore conservatively
// misses even when placeholder tokens are identical.
std::size_t native_prefix_cache_matched_tokens(
    const std::vector<std::uint32_t>& cached_tokens,
    std::string_view cached_multimodal_namespace,
    const std::vector<std::uint32_t>& request_tokens,
    std::string_view request_multimodal_namespace);

}  // namespace aima
