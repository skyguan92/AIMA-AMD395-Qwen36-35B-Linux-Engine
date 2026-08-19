// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_multimodal_cache.h"

#include "aima/sha256.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

bool canonical_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char byte) {
           return std::isdigit(byte) != 0 ||
                  (byte >= 'a' && byte <= 'f');
         });
}

std::string build_cache_namespace(
    const NativeMultimodalCacheIdentityInput& input,
    std::string_view schema) {
  if (!canonical_sha256(input.processor_config_sha256)) {
    throw std::invalid_argument(
        "multimodal cache processor identity must be lowercase SHA-256");
  }
  if (input.media.empty()) {
    throw std::invalid_argument(
        "multimodal cache identity requires at least one media item");
  }
  std::ostringstream canonical;
  canonical << schema << '\n'
            << "processor=" << input.processor_config_sha256 << '\n'
            << "media_count=" << input.media.size() << '\n';
  for (std::size_t index = 0; index < input.media.size(); ++index) {
    const NativeMultimodalCacheItem& item = input.media[index];
    if (!canonical_sha256(item.content_sha256)) {
      throw std::invalid_argument(
          "multimodal cache content identity must be lowercase SHA-256");
    }
    const std::uint32_t expected_token =
        item.kind == NativeMediaKind::kImage ? kNativeImagePadTokenId
                                             : kNativeVideoPadTokenId;
    if (item.placeholder_token_id != expected_token) {
      throw std::invalid_argument(
          "multimodal cache placeholder token does not match media kind");
    }
    if (item.token_length == 0 ||
        item.token_offset >
            std::numeric_limits<std::size_t>::max() - item.token_length) {
      throw std::invalid_argument(
          "multimodal cache token span is empty or overflows");
    }
    canonical << index << ':' << native_media_kind_name(item.kind) << ':'
              << item.content_sha256 << ':' << item.placeholder_token_id
              << ':' << item.token_offset << ':' << item.token_length << '\n';
  }
  const std::string payload = canonical.str();
  return sha256_bytes(payload.data(), payload.size());
}

}  // namespace

std::string build_native_multimodal_cache_namespace(
    const NativeMultimodalCacheIdentityInput& input) {
  return build_cache_namespace(
      input, "aima-amd395-qwen36/native-multimodal-cache/v1");
}

std::string build_native_vision_embedding_cache_namespace(
    const NativeMultimodalCacheIdentityInput& input) {
  return build_cache_namespace(
      input, "aima-amd395-qwen36/native-vision-embedding-cache/v1");
}

bool valid_native_multimodal_cache_namespace(std::string_view value) {
  return value.empty() || canonical_sha256(value);
}

std::size_t native_prefix_cache_matched_tokens(
    const std::vector<std::uint32_t>& cached_tokens,
    std::string_view cached_multimodal_namespace,
    const std::vector<std::uint32_t>& request_tokens,
    std::string_view request_multimodal_namespace) {
  if (!valid_native_multimodal_cache_namespace(cached_multimodal_namespace) ||
      !valid_native_multimodal_cache_namespace(request_multimodal_namespace)) {
    throw std::invalid_argument(
        "prefix cache multimodal namespace is not canonical SHA-256");
  }
  if (cached_tokens.empty() ||
      cached_multimodal_namespace != request_multimodal_namespace ||
      request_tokens.size() < cached_tokens.size() ||
      !std::equal(cached_tokens.begin(), cached_tokens.end(),
                  request_tokens.begin())) {
    return 0;
  }
  return cached_tokens.size();
}

}  // namespace aima
