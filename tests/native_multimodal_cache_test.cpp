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

  auto content_b = input;
  content_b.media[0].content_sha256 = std::string(64, 'c');
  require(aima::build_native_multimodal_cache_namespace(content_b) != identity,
          "different media bytes reused a namespace");

  auto reordered = input;
  std::swap(reordered.media[0], reordered.media[1]);
  require(aima::build_native_multimodal_cache_namespace(reordered) != identity,
          "media ordering was omitted from the namespace");

  auto processor_b = input;
  processor_b.processor_config_sha256 = std::string(64, '2');
  require(aima::build_native_multimodal_cache_namespace(processor_b) != identity,
          "processor configuration was omitted from the namespace");

  auto span_b = input;
  ++span_b.media[0].token_length;
  require(aima::build_native_multimodal_cache_namespace(span_b) != identity,
          "media token span was omitted from the namespace");

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
