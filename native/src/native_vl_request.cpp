// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vl_request.h"

#include "aima/native_image_decoder.h"
#include "aima/native_multimodal_cache.h"
#include "aima/native_tokenizer.h"
#include "aima/native_video_decoder.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kPixelColumns = 1536;

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char* description) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::invalid_argument(std::string(description) + " overflows");
  }
  return left + right;
}

std::size_t find_token(const std::vector<std::uint32_t>& tokens,
                       std::uint32_t token, std::size_t begin) {
  for (std::size_t index = begin; index < tokens.size(); ++index) {
    if (tokens[index] == token) return index;
  }
  return tokens.size();
}

std::size_t find_media_marker(
    const std::vector<std::uint32_t>& tokens, std::uint32_t pad,
    std::size_t begin) {
  for (std::size_t index = begin; index + 2 < tokens.size(); ++index) {
    if (tokens[index] == kNativeVisionStartTokenId &&
        tokens[index + 1] == pad &&
        tokens[index + 2] == kNativeVisionEndTokenId) {
      return index;
    }
  }
  return tokens.size();
}

std::vector<std::uint32_t> expand_media_prompt_tokens(
    NativeTokenizer& tokenizer, const std::string& prompt,
    const std::vector<NativeVlPromptMedia>& media) {
  std::vector<std::uint32_t> tokens = tokenizer.encode(prompt);
  std::size_t search = 0;
  for (const NativeVlPromptMedia& item : media) {
    const std::uint32_t pad =
        item.kind == NativeMediaKind::kImage ? kNativeImagePadTokenId
                                             : kNativeVideoPadTokenId;
    const std::size_t marker = find_media_marker(tokens, pad, search);
    if (marker == tokens.size()) {
      throw std::invalid_argument(
          "native VL media marker is missing from the rendered prompt");
    }

    const std::string wrapped =
        item.kind == NativeMediaKind::kImage
            ? "<|vision_start|><|image_pad|><|vision_end|>"
            : "<|vision_start|><|video_pad|><|vision_end|>";
    const std::string expanded =
        native_qwen36_expand_media_prompt(wrapped, {item});
    const std::vector<std::uint32_t> replacement =
        tokenizer.encode(expanded);
    if (replacement.empty()) {
      throw std::runtime_error(
          "native VL media marker expansion produced no tokens");
    }
    tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(marker),
                 tokens.begin() + static_cast<std::ptrdiff_t>(marker + 3));
    tokens.insert(tokens.begin() + static_cast<std::ptrdiff_t>(marker),
                  replacement.begin(), replacement.end());
    search = marker + replacement.size();
  }
  return tokens;
}

struct LocatedMediaSpan {
  std::size_t offset = 0;
  std::size_t length = 0;
};

LocatedMediaSpan locate_image_span(
    const std::vector<std::uint32_t>& tokens, std::size_t begin,
    std::size_t visual_tokens) {
  const std::size_t offset =
      find_token(tokens, kNativeImagePadTokenId, begin);
  if (offset == tokens.size() || visual_tokens == 0 ||
      visual_tokens > tokens.size() - offset) {
    throw std::invalid_argument(
        "native VL image placeholder span is missing or truncated");
  }
  for (std::size_t index = 0; index < visual_tokens; ++index) {
    if (tokens[offset + index] != kNativeImagePadTokenId) {
      throw std::invalid_argument(
          "native VL image placeholder span is not contiguous");
    }
  }
  return {offset, visual_tokens};
}

LocatedMediaSpan locate_video_span(
    const std::vector<std::uint32_t>& tokens, std::size_t begin,
    const NativeVlGrid& grid) {
  if (grid.temporal == 0 || grid.height % kNativeVlMergeSize != 0 ||
      grid.width % kNativeVlMergeSize != 0) {
    throw std::invalid_argument("native VL video grid is invalid");
  }
  const std::size_t tokens_per_frame =
      grid.height * grid.width /
      (kNativeVlMergeSize * kNativeVlMergeSize);
  std::size_t cursor = begin;
  std::size_t span_begin = tokens.size();
  for (std::size_t frame = 0; frame < grid.temporal; ++frame) {
    const std::size_t vision_start =
        find_token(tokens, kNativeVisionStartTokenId, cursor);
    if (vision_start == tokens.size()) {
      throw std::invalid_argument(
          "native VL video frame is missing vision_start");
    }
    const std::size_t pad =
        find_token(tokens, kNativeVideoPadTokenId, vision_start + 1);
    if (pad == tokens.size() || tokens_per_frame == 0 ||
        tokens_per_frame > tokens.size() - pad) {
      throw std::invalid_argument(
          "native VL video placeholder span is missing or truncated");
    }
    for (std::size_t index = 0; index < tokens_per_frame; ++index) {
      if (tokens[pad + index] != kNativeVideoPadTokenId) {
        throw std::invalid_argument(
            "native VL video placeholder span is not contiguous");
      }
    }
    const std::size_t vision_end = pad + tokens_per_frame;
    if (vision_end >= tokens.size() ||
        tokens[vision_end] != kNativeVisionEndTokenId) {
      throw std::invalid_argument(
          "native VL video frame is missing vision_end");
    }
    if (span_begin == tokens.size()) span_begin = vision_start;
    cursor = vision_end + 1;
  }
  return {span_begin, cursor - span_begin};
}

struct ProcessedMedia {
  NativeMediaKind kind = NativeMediaKind::kImage;
  std::string content_sha256;
  NativeVlPromptMedia prompt;
  NativeVlPixelTensor pixels;
};

ProcessedMedia process_media(const NativeMediaPayload& payload,
                             const NativeMediaPolicy& policy,
                             NativeVlPreparationMetrics* metrics) {
  ProcessedMedia result;
  result.kind = payload.kind;
  result.content_sha256 = payload.content_sha256;
  result.prompt.kind = payload.kind;
  const auto decode_started = std::chrono::steady_clock::now();
  if (payload.kind == NativeMediaKind::kImage) {
    NativeRgbFrame frame = decode_native_image(payload, policy);
    metrics->media_decode_wall_ms += elapsed_ms(decode_started);
    const auto processor_started = std::chrono::steady_clock::now();
    const NativeVlResizeGeometry geometry =
        native_qwen36_image_geometry(frame.height, frame.width);
    result.prompt.grid = geometry.grid;
    result.pixels = native_qwen36_process_rgb({frame}, geometry);
    metrics->processor_wall_ms += elapsed_ms(processor_started);
  } else {
    NativeDecodedVideo video = decode_native_video(payload, policy);
    metrics->media_decode_wall_ms += elapsed_ms(decode_started);
    const auto processor_started = std::chrono::steady_clock::now();
    const NativeVlResizeGeometry geometry = native_qwen36_video_geometry(
        video.frames.size(), video.height, video.width);
    result.prompt.grid = geometry.grid;
    result.prompt.frame_indices = video.frame_indices;
    result.prompt.source_fps = video.source_fps;
    result.pixels = native_qwen36_process_rgb(video.frames, geometry);
    metrics->processor_wall_ms += elapsed_ms(processor_started);
  }
  if (result.pixels.grid.temporal != result.prompt.grid.temporal ||
      result.pixels.grid.height != result.prompt.grid.height ||
      result.pixels.grid.width != result.prompt.grid.width ||
      result.pixels.rows != result.prompt.grid.patch_count() ||
      result.pixels.columns != kPixelColumns) {
    throw std::runtime_error(
        "native VL processor output disagrees with its prompt grid");
  }
  return result;
}

}  // namespace

struct NativeVlMediaCache::Impl {
  struct Entry {
    std::string key;
    std::shared_ptr<const ProcessedMedia> media;
    std::uint64_t bytes = 0;
    std::uint64_t use = 0;
  };

  explicit Impl(std::uint64_t byte_capacity,
                std::size_t entry_capacity)
      : capacity_bytes(byte_capacity),
        capacity_entries(entry_capacity) {
    if (capacity_bytes == 0 || capacity_entries == 0) {
      capacity_bytes = 0;
      capacity_entries = 0;
    }
  }

  std::string key(const NativeMediaPayload& payload,
                  std::string_view processor_identity) const {
    return std::string(processor_identity) + ':' +
           (payload.kind == NativeMediaKind::kImage ? "image:" : "video:") +
           payload.content_sha256;
  }

  std::shared_ptr<const ProcessedMedia> find(const std::string& key_value) {
    std::lock_guard<std::mutex> lock(mutex);
    for (Entry& entry : values) {
      if (entry.key != key_value) continue;
      entry.use = ++clock;
      return entry.media;
    }
    return {};
  }

  void insert(const std::string& key_value,
              std::shared_ptr<const ProcessedMedia> media) {
    std::lock_guard<std::mutex> lock(mutex);
    if (capacity_bytes == 0 || capacity_entries == 0 || !media) return;
    const std::uint64_t pixel_bytes =
        media->pixels.values.size() * sizeof(std::uint16_t);
    const std::uint64_t frame_index_bytes =
        media->prompt.frame_indices.size() * sizeof(std::size_t);
    const std::uint64_t bytes = sizeof(Entry) + sizeof(ProcessedMedia) +
                                key_value.size() +
                                media->content_sha256.size() + pixel_bytes +
                                frame_index_bytes;
    if (bytes > capacity_bytes) return;
    while (!values.empty() &&
           (values.size() >= capacity_entries ||
            resident_bytes > capacity_bytes - bytes)) {
      const auto oldest = std::min_element(
          values.begin(), values.end(),
          [](const Entry& left, const Entry& right) {
            return left.use < right.use;
          });
      resident_bytes -= oldest->bytes;
      values.erase(oldest);
    }
    values.push_back(Entry{key_value, std::move(media), bytes, ++clock});
    resident_bytes += bytes;
  }

  std::uint64_t capacity_bytes = 0;
  std::size_t capacity_entries = 0;
  mutable std::mutex mutex;
  std::vector<Entry> values;
  std::uint64_t resident_bytes = 0;
  std::uint64_t clock = 0;
};

NativeVlMediaCache::NativeVlMediaCache(std::uint64_t capacity_bytes,
                                       std::size_t capacity_entries)
    : impl_(std::make_unique<Impl>(capacity_bytes, capacity_entries)) {}

NativeVlMediaCache::~NativeVlMediaCache() = default;

std::uint64_t NativeVlMediaCache::capacity_bytes() const {
  return impl_->capacity_bytes;
}

std::uint64_t NativeVlMediaCache::resident_bytes() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->resident_bytes;
}

std::size_t NativeVlMediaCache::capacity_entries() const {
  return impl_->capacity_entries;
}

std::size_t NativeVlMediaCache::entries() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->values.size();
}

NativeVlPreparedRequest prepare_native_vl_request(
    NativeTokenizer& tokenizer, const NativePreparedChat& chat,
    const NativeMediaPolicy& policy, NativeVlMediaCache* media_cache) {
  if (chat.media.empty()) {
    throw std::invalid_argument(
        "native VL request preparation requires media");
  }

  NativeVlPreparedRequest result;
  result.metrics.media_count = chat.media.size();
  const std::string processor_identity =
      native_qwen36_processor_config_sha256(policy.video_io, policy.image_io);
  std::vector<std::shared_ptr<const ProcessedMedia>> processed;
  processed.reserve(chat.media.size());
  std::vector<NativeVlPromptMedia> prompt_media;
  prompt_media.reserve(chat.media.size());
  for (const NativeMediaPart& media : chat.media) {
    const auto load_started = std::chrono::steady_clock::now();
    NativeMediaPayload payload = load_native_media_payload(media, policy);
    result.metrics.media_load_wall_ms += elapsed_ms(load_started);
    result.metrics.source_bytes += payload.bytes.size();
    const std::string cache_key =
        media_cache != nullptr
            ? media_cache->impl_->key(payload, processor_identity)
            : "";
    std::shared_ptr<const ProcessedMedia> item =
        media_cache != nullptr ? media_cache->impl_->find(cache_key) : nullptr;
    if (item) {
      ++result.metrics.media_cache_hits;
    } else {
      ++result.metrics.media_cache_misses;
      item = std::make_shared<const ProcessedMedia>(
          process_media(payload, policy, &result.metrics));
      if (media_cache != nullptr) {
        media_cache->impl_->insert(cache_key, item);
      }
    }
    if (item->kind == NativeMediaKind::kImage) {
      ++result.metrics.image_count;
    } else {
      ++result.metrics.video_count;
    }
    result.metrics.vision_patches = checked_add(
        result.metrics.vision_patches, item->pixels.rows,
        "native VL patch count");
    result.metrics.visual_tokens = checked_add(
        result.metrics.visual_tokens,
        item->prompt.grid.language_token_count(),
        "native VL visual-token count");
    result.grids.push_back(item->prompt.grid);
    result.pixel_values_bf16.insert(result.pixel_values_bf16.end(),
                                    item->pixels.values.begin(),
                                    item->pixels.values.end());
    prompt_media.push_back(item->prompt);
    processed.push_back(std::move(item));
  }
  result.metrics.media_load_decode_wall_ms =
      result.metrics.media_load_wall_ms + result.metrics.media_decode_wall_ms;
  if (media_cache != nullptr) {
    result.metrics.media_cache_entries = media_cache->entries();
    result.metrics.media_cache_resident_bytes =
        media_cache->resident_bytes();
  }
  if (result.metrics.visual_tokens > kNativeVlAggregateTokenLimit ||
      result.metrics.vision_patches > kNativeVlAggregatePatchLimit) {
    throw std::invalid_argument(
        "native VL request exceeds the aggregate visual budget");
  }

  // The fixed vLLM VL server leaves Qwen thinking enabled for named tool
  // choice and constrains the generated function arguments at the decoder.
  // `required` retains the existing native XML-tool contract until its
  // multi-tool JSON-array grammar is implemented and qualified separately.
  bool disable_thinking =
      chat.tool_choice == NativeToolChoiceMode::kRequired;
  if (chat.thinking_mode == NativeThinkingMode::kEnabled) {
    disable_thinking = false;
  } else if (chat.thinking_mode == NativeThinkingMode::kDisabled) {
    disable_thinking = true;
  }
  const std::string prompt = tokenizer.render_chat_prompt(
      chat.vl_prompt_messages, chat.vl_prompt_tools, disable_thinking);
  // vLLM tokenizes each chat/content segment before replacing multimodal
  // markers. Preserve that boundary here: encoding one fully concatenated
  // string can merge a preceding text token with a video's leading timestamp
  // (for example, `.<`), changing the frozen prompt by one token.
  result.prompt_token_ids =
      expand_media_prompt_tokens(tokenizer, prompt, prompt_media);

  std::vector<NativeMropeMedia> mrope_media;
  mrope_media.reserve(processed.size());
  NativeMultimodalCacheIdentityInput cache_identity;
  cache_identity.processor_config_sha256 = processor_identity;
  cache_identity.media.reserve(processed.size());
  NativeMultimodalCacheIdentityInput vision_cache_identity;
  vision_cache_identity.processor_config_sha256 = processor_identity;
  vision_cache_identity.media.reserve(processed.size());
  std::size_t search = 0;
  std::size_t visual_offset = 0;
  for (const std::shared_ptr<const ProcessedMedia>& item_pointer : processed) {
    const ProcessedMedia& item = *item_pointer;
    const std::size_t visual_count = item.prompt.grid.language_token_count();
    const LocatedMediaSpan span =
        item.kind == NativeMediaKind::kImage
            ? locate_image_span(result.prompt_token_ids, search, visual_count)
            : locate_video_span(result.prompt_token_ids, search,
                                item.prompt.grid);
    result.embedding_spans.push_back(NativeVlEmbeddingSpan{
        item.kind, span.offset, span.length, visual_offset, visual_count});
    mrope_media.push_back(NativeMropeMedia{
        item.kind, span.offset, span.length, item.prompt.grid});
    const std::uint32_t placeholder =
        item.kind == NativeMediaKind::kImage ? kNativeImagePadTokenId
                                             : kNativeVideoPadTokenId;
    cache_identity.media.push_back(NativeMultimodalCacheItem{
        item.kind, item.content_sha256, placeholder, span.offset,
        span.length});
    vision_cache_identity.media.push_back(NativeMultimodalCacheItem{
        item.kind, item.content_sha256, placeholder, visual_offset,
        visual_count});
    visual_offset = checked_add(visual_offset, visual_count,
                                "native VL visual embedding offset");
    search = checked_add(span.offset, span.length,
                         "native VL prompt media span");
  }

  (void)build_native_vl_embedding_plan(
      result.prompt_token_ids, result.embedding_spans, visual_offset);
  result.mrope_plan =
      build_native_mrope_plan(result.prompt_token_ids, mrope_media);
  result.multimodal_cache_namespace =
      build_native_multimodal_cache_namespace(cache_identity);
  if (media_cache != nullptr && media_cache->capacity_bytes() != 0 &&
      media_cache->capacity_entries() != 0) {
    result.vision_embedding_cache_namespace =
        build_native_vision_embedding_cache_namespace(
            vision_cache_identity);
  }
  return result;
}

}  // namespace aima
