// SPDX-License-Identifier: Apache-2.0

#include "aima/native_multimodal_cache.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_multimodal_cache_test: " << message << '\n';
    std::exit(1);
  }
}

template <typename Function>
void require_invalid(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  require(false, message);
}

aima::NativeMultimodalCacheIdentityInput fixture() {
  aima::NativeMultimodalCacheIdentityInput input;
  input.processor_config_sha256 = std::string(64, '1');
  input.media = {
      {aima::NativeMediaKind::kImage, std::string(64, 'a'),
       aima::kNativeImagePadTokenId, 9, 64},
      {aima::NativeMediaKind::kVideo, std::string(64, 'b'),
       aima::kNativeVideoPadTokenId, 80, 24},
  };
  return input;
}

}  // namespace

int main() {
  const auto input = fixture();
  const std::string identity =
      aima::build_native_multimodal_cache_namespace(input);
  require(identity ==
                  "7e147ebd61a3e2256b1756c2664cce6f52d0103220bad67a158915e67e66e2ba" &&
              aima::valid_native_multimodal_cache_namespace(identity),
          "canonical namespace contract changed");
  require(identity == aima::build_native_multimodal_cache_namespace(input),
          "identical media identity is unstable");
  const std::string vision_identity =
      aima::build_native_vision_embedding_cache_namespace(input);
  require(
      vision_identity ==
              "607b470d44616e991ba30f34c484ad350e3fcb8668d03b3c7193726a96ce8c5c" &&
          vision_identity != identity &&
          aima::valid_native_multimodal_cache_namespace(vision_identity),
      "vision embedding namespace contract changed");
  require(vision_identity ==
              aima::build_native_vision_embedding_cache_namespace(input),
          "identical vision media identity is unstable");

  auto content_b = input;
  content_b.media[0].content_sha256 = std::string(64, 'c');
  require(aima::build_native_multimodal_cache_namespace(content_b) != identity,
          "different media bytes reused a namespace");
  require(aima::build_native_vision_embedding_cache_namespace(content_b) !=
              vision_identity,
          "different media bytes reused a vision embedding namespace");

  auto reordered = input;
  std::swap(reordered.media[0], reordered.media[1]);
  require(aima::build_native_multimodal_cache_namespace(reordered) != identity,
          "media ordering was omitted from the namespace");
  require(aima::build_native_vision_embedding_cache_namespace(reordered) !=
              vision_identity,
          "media ordering was omitted from the vision embedding namespace");

  auto processor_b = input;
  processor_b.processor_config_sha256 = std::string(64, '2');
  require(aima::build_native_multimodal_cache_namespace(processor_b) != identity,
          "processor configuration was omitted from the namespace");
  require(aima::build_native_vision_embedding_cache_namespace(processor_b) !=
              vision_identity,
          "processor configuration was omitted from the vision namespace");

  auto span_b = input;
  ++span_b.media[0].token_length;
  require(aima::build_native_multimodal_cache_namespace(span_b) != identity,
          "media token span was omitted from the namespace");
  require(aima::build_native_vision_embedding_cache_namespace(span_b) !=
              vision_identity,
          "visual embedding span was omitted from the vision namespace");

  // Transport and filename are intentionally not part of the descriptor:
  // local-file and data-URI payloads with the same decoded SHA are equivalent.
  const auto equivalent_transport = fixture();
  require(aima::build_native_multimodal_cache_namespace(equivalent_transport) ==
              identity,
          "equivalent decoded media failed to share identity");

  const std::vector<std::uint32_t> cached = {1, 2, 3};
  const std::vector<std::uint32_t> extended = {1, 2, 3, 4};
  require(aima::native_prefix_cache_matched_tokens(
              cached, identity, extended, identity) == cached.size(),
          "same-media text extension missed its prefix");
  require(aima::native_prefix_cache_matched_tokens(
              cached, identity, extended,
              aima::build_native_multimodal_cache_namespace(content_b)) == 0,
          "A/B media collision reused a prefix");
  require(aima::build_native_multimodal_cache_namespace(input) == identity,
          "A/B/A media identity did not recover A");
  require(aima::native_prefix_cache_matched_tokens(
              cached, "", extended, "") == cached.size(),
          "text-only prefix behavior regressed");
  require(aima::native_prefix_cache_matched_tokens(
              cached, "", std::vector<std::uint32_t>{1, 9, 3}, "") == 0,
          "changed text tokens reused a prefix");

  // A common token is reusable only if a complete hybrid-state checkpoint
  // exists at that boundary. Never rewind a full-request recurrent snapshot.
  require(aima::native_prefix_cache_matched_tokens(
              cached, "", {1, 9, 3}, "", {1, 2}) == 1,
          "safe common-prefix checkpoint was not selected");
  require(aima::native_prefix_cache_matched_tokens(
              cached, "", {1, 2, 9}, "", {1}) == 1,
          "matched count exceeded the captured state boundary");
  require(aima::native_prefix_cache_matched_tokens(
              cached, "", {1, 2}, "", {1, 2}) == 2,
          "a shorter request failed to select its exact checkpoint");
  require(aima::native_prefix_cache_matched_tokens(
              cached, "", extended, "", {1, 2}) == 3,
          "whole-request append hit regressed with checkpoints");
  require(aima::native_prefix_cache_matched_tokens(
              cached, "", cached, "", {1, 2}) == 3,
          "exact-repeat hit regressed with checkpoints");
  require(aima::native_prefix_cache_matched_tokens(
              cached, "", {9, 2, 3}, "", {1, 2}) == 0,
          "divergent first token was matched");
  require(aima::native_prefix_cache_matched_tokens(
              cached, "", {}, "", {1, 2}) == 0,
          "an empty request matched a checkpoint");
  require(aima::native_prefix_cache_matched_tokens(
              cached, "", cached, identity, {1, 2}) == 0,
          "text and media namespaces shared a checkpoint");
  require(aima::native_prefix_cache_matched_tokens(
              cached, identity, {1, 2, 9}, std::string(64, 'f'), {1, 2}) == 0,
          "changed media identity shared a checkpoint");
  for (const std::vector<std::size_t>& boundaries :
       std::vector<std::vector<std::size_t>>{{0}, {3}, {4}, {2, 1}, {1, 1}}) {
    require_invalid([&]() {
      (void)aima::native_prefix_cache_matched_tokens(cached, "", extended, "", boundaries);
    }, "invalid checkpoint set was admitted");
  }

  const std::vector<std::uint32_t> chat = {
      248045, 8948, 198, 111, 248046, 198,
      248045, 872, 198, 109266, 248046, 198, 248045, 74455, 198};
  const auto checkpoints = aima::native_chat_prefix_checkpoint_tokens(chat, "");
  require(checkpoints == std::vector<std::size_t>({4, 10}),
          "message checkpoints included the divergent terminator");
  auto generation_prompt = chat;
  generation_prompt.insert(generation_prompt.end(), {248068, 271, 248069, 271});
  const auto generation_checkpoints = aima::native_chat_prefix_checkpoint_tokens(generation_prompt, "");
  require(generation_checkpoints == std::vector<std::size_t>({4, 10, 15}),
          "assistant generation header was not checkpointed");
  auto completed_answer = chat;
  completed_answer.insert(completed_answer.end(), {109266, 248046});
  require(aima::native_prefix_cache_matched_tokens(generation_prompt, "", completed_answer, "",
                                                  generation_checkpoints) == 15,
          "answer history failed to reuse the complete assistant header");
  auto longer_chat = chat;
  longer_chat.insert(longer_chat.begin() + 10, {3709, 144810});
  require(aima::native_prefix_cache_matched_tokens(
              chat, "", longer_chat, "", checkpoints) == 10,
          "divergent final chat message missed its content checkpoint");
  longer_chat[9] = 999;
  require(aima::native_prefix_cache_matched_tokens(
              chat, "", longer_chat, "", checkpoints) == 4,
          "a different user message failed to reuse the shared system state");
  auto multi_turn = chat;
  multi_turn.insert(multi_turn.end(), {123, 248046, 198, 248045, 872, 198, 999, 248046});
  require(aima::native_chat_prefix_checkpoint_tokens(multi_turn, "") ==
              std::vector<std::size_t>({4, multi_turn.size() - 1}),
          "multi-turn checkpoint count was not bounded to first and last");
  require(aima::native_chat_prefix_checkpoint_tokens(chat, identity).empty(),
          "media requests were admitted to text-only partial checkpoints");
  require(aima::native_chat_prefix_checkpoint_tokens({1, 2, 3}, "").empty(),
          "raw tokens invented a message checkpoint");
  std::vector<std::uint32_t> short_owner(30, 1);
  short_owner.back() = 2;
  require(aima::native_prefix_cache_matched_tokens(short_owner, "",
              std::vector<std::uint32_t>(39, 1), "", {9, 20}) == 0,
          "unaligned short checkpoint crossed an FLA chunk boundary");
  std::vector<std::uint32_t> long_owner(80, 1);
  long_owner[47] = 248046;
  long_owner[70] = 248046;
  require(aima::native_chat_prefix_checkpoint_tokens(long_owner, "") ==
              std::vector<std::size_t>({32, 64}),
          "long message checkpoints were not aligned to FLA chunk state");

  // Longest restored boundary wins across owners, regardless of full prompt
  // length; eviction uses request-owner LRU and also removes its checkpoints.
  const std::vector<std::vector<std::uint32_t>> owners = {{1, 2, 7}, {1, 2, 3, 4, 5}};
  const std::vector<std::vector<std::size_t>> owner_checkpoints = {{1}, {1, 3}};
  std::size_t best_tokens = 0;
  std::size_t best_owner = owners.size();
  for (std::size_t index = 0; index < owners.size(); ++index) {
    const auto matched = aima::native_prefix_cache_matched_tokens(
        owners[index], "", {1, 2, 3, 9}, "", owner_checkpoints[index]);
    if (matched > best_tokens) {
      best_tokens = matched;
      best_owner = index;
    }
  }
  require(best_tokens == 3 && best_owner == 1, "longest safe owner selection failed");
  require(aima::native_prefix_cache_capture_index({true, false, true}, {9, 4, 1}) == 1,
          "LRU evicted a live owner before using an empty slot");
  require(aima::native_prefix_cache_capture_index({true, true, true}, {9, 4, 1}) == 2,
          "LRU did not evict the oldest request owner");
  require(aima::native_prefix_cache_capture_index({true}, {9}) == 0,
          "one-entry long-window cache cannot be replaced");
  require_invalid([&]() { (void)aima::native_prefix_cache_capture_index({}, {}); },
                  "zero-capacity cache admitted a capture");
  require_invalid([&]() { (void)aima::native_prefix_cache_capture_index({true}, {}); },
                  "mismatched LRU metadata was admitted");

  auto invalid = input;
  invalid.media[0].content_sha256 = std::string(64, 'A');
  require_invalid(
      [&]() { (void)aima::build_native_multimodal_cache_namespace(invalid); },
      "non-canonical media digest was admitted");
  invalid = input;
  invalid.media[0].placeholder_token_id = aima::kNativeVideoPadTokenId;
  require_invalid(
      [&]() { (void)aima::build_native_multimodal_cache_namespace(invalid); },
      "kind/token mismatch was admitted");
  require_invalid(
      [&]() {
        (void)aima::native_prefix_cache_matched_tokens(
            cached, "not-a-digest", extended, "not-a-digest");
      },
      "malformed cache namespace was admitted");

  std::cout << "native_multimodal_cache_test: PASS\n";
  return 0;
}
