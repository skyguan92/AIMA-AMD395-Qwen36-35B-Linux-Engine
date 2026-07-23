// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_tokenizer.h"

#include "aima/sha256.h"
#include "model_layout.h"

#include <nlohmann/json.hpp>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/stringpiece.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/utf16.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace aima {
namespace {

using Json = nlohmann::json;

Json load_json(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("could not open tokenizer JSON: " + path.string());
  Json value;
  stream >> value;
  if (!value.is_object()) {
    throw std::runtime_error("tokenizer JSON root must be an object: " + path.string());
  }
  return value;
}

std::string utf8_for_codepoint(UChar32 codepoint) {
  icu::UnicodeString value;
  value.append(codepoint);
  std::string output;
  value.toUTF8String(output);
  return output;
}

std::array<std::string, 256> byte_encoder() {
  std::array<bool, 256> direct{};
  for (int value = '!'; value <= '~'; ++value) direct[value] = true;
  for (int value = 0xA1; value <= 0xAC; ++value) direct[value] = true;
  for (int value = 0xAE; value <= 0xFF; ++value) direct[value] = true;
  std::array<std::string, 256> result;
  int extra = 0;
  for (int value = 0; value < 256; ++value) {
    const UChar32 mapped = direct[value] ? value : 256 + extra++;
    result[value] = utf8_for_codepoint(mapped);
  }
  return result;
}

std::unordered_map<UChar32, unsigned char> byte_decoder(
    const std::array<std::string, 256>& encoder) {
  std::unordered_map<UChar32, unsigned char> result;
  result.reserve(256);
  for (std::size_t index = 0; index < encoder.size(); ++index) {
    const icu::UnicodeString symbol = icu::UnicodeString::fromUTF8(encoder[index]);
    result.emplace(symbol.char32At(0), static_cast<unsigned char>(index));
  }
  return result;
}

std::string trim_unicode(std::string_view text) {
  const icu::UnicodeString value = icu::UnicodeString::fromUTF8(
      icu::StringPiece(text.data(), static_cast<int32_t>(text.size())));
  int32_t begin = 0;
  int32_t end = value.length();
  while (begin < end) {
    const UChar32 point = value.char32At(begin);
    if (!u_isUWhiteSpace(point)) break;
    begin += U16_LENGTH(point);
  }
  while (end > begin) {
    const int32_t previous = value.moveIndex32(end, -1);
    const UChar32 point = value.char32At(previous);
    if (!u_isUWhiteSpace(point)) break;
    end = previous;
  }
  std::string output;
  value.tempSubStringBetween(begin, end).toUTF8String(output);
  return output;
}

std::string pair_key(std::string_view left, std::string_view right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1);
  key.append(left);
  key.push_back('\0');
  key.append(right);
  return key;
}

}  // namespace

struct NativeTokenizer::Impl {
  struct AddedToken {
    std::uint32_t id = 0;
    std::string content;
  };

  bool loaded = false;
  const icu::Normalizer2* normalizer = nullptr;
  std::unique_ptr<icu::RegexPattern> regex;
  std::unordered_map<std::string, std::uint32_t> vocab;
  std::vector<std::string> id_to_token;
  std::unordered_set<std::uint32_t> added_ids;
  std::vector<AddedToken> added_tokens;
  std::unordered_map<std::string, std::size_t> merge_ranks;
  std::unordered_map<std::string, std::vector<std::uint32_t>> bpe_cache;
  std::array<std::string, 256> bytes_to_unicode = byte_encoder();
  std::unordered_map<UChar32, unsigned char> unicode_to_bytes =
      byte_decoder(bytes_to_unicode);
  std::uint32_t eos = 248046;
  std::uint32_t pad = 248044;

  void require_loaded() const {
    if (!loaded) throw std::runtime_error("native tokenizer is not loaded");
  }

  void register_added(std::uint32_t id, const std::string& content) {
    if (content.empty()) throw std::runtime_error("empty added tokenizer token");
    if (id >= id_to_token.size()) id_to_token.resize(static_cast<std::size_t>(id) + 1);
    if (!id_to_token[id].empty() && id_to_token[id] != content) {
      throw std::runtime_error("conflicting tokenizer token id");
    }
    id_to_token[id] = content;
    if (added_ids.insert(id).second) added_tokens.push_back(AddedToken{id, content});
  }

  std::vector<std::uint32_t> bpe(std::string_view piece) {
    const std::string cache_key(piece);
    const auto cached = bpe_cache.find(cache_key);
    if (cached != bpe_cache.end()) return cached->second;

    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (const unsigned char byte : piece) symbols.push_back(bytes_to_unicode[byte]);
    while (symbols.size() > 1) {
      std::size_t best_rank = std::numeric_limits<std::size_t>::max();
      std::string best_left;
      std::string best_right;
      for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
        const auto found = merge_ranks.find(pair_key(symbols[index], symbols[index + 1]));
        if (found != merge_ranks.end() && found->second < best_rank) {
          best_rank = found->second;
          best_left = symbols[index];
          best_right = symbols[index + 1];
        }
      }
      if (best_rank == std::numeric_limits<std::size_t>::max()) break;
      std::vector<std::string> merged;
      merged.reserve(symbols.size());
      for (std::size_t index = 0; index < symbols.size();) {
        if (index + 1 < symbols.size() && symbols[index] == best_left &&
            symbols[index + 1] == best_right) {
          merged.push_back(symbols[index] + symbols[index + 1]);
          index += 2;
        } else {
          merged.push_back(std::move(symbols[index]));
          ++index;
        }
      }
      symbols = std::move(merged);
    }

    std::vector<std::uint32_t> ids;
    ids.reserve(symbols.size());
    for (const std::string& symbol : symbols) {
      const auto found = vocab.find(symbol);
      if (found == vocab.end()) {
        throw std::runtime_error("native BPE produced a symbol outside the vocabulary");
      }
      ids.push_back(found->second);
    }
    bpe_cache.emplace(cache_key, ids);
    return ids;
  }

  void encode_normal_span(std::string_view text, std::vector<std::uint32_t>* output) {
    if (text.empty()) return;
    UErrorCode status = U_ZERO_ERROR;
    const icu::UnicodeString input = icu::UnicodeString::fromUTF8(
        icu::StringPiece(text.data(), static_cast<int32_t>(text.size())));
    icu::UnicodeString normalized;
    normalizer->normalize(input, normalized, status);
    if (U_FAILURE(status)) throw std::runtime_error("ICU NFC normalization failed");

    std::unique_ptr<icu::RegexMatcher> matcher(regex->matcher(normalized, status));
    if (U_FAILURE(status) || matcher == nullptr) {
      throw std::runtime_error("could not create native tokenizer regex matcher");
    }
    int32_t consumed = 0;
    while (matcher->find(status)) {
      if (U_FAILURE(status)) throw std::runtime_error("native tokenizer regex failed");
      const int32_t begin = matcher->start(status);
      const int32_t end = matcher->end(status);
      auto append_range = [&](int32_t from, int32_t to) {
        if (to <= from) return;
        std::string piece;
        normalized.tempSubStringBetween(from, to).toUTF8String(piece);
        const std::vector<std::uint32_t> ids = bpe(piece);
        output->insert(output->end(), ids.begin(), ids.end());
      };
      append_range(consumed, begin);
      append_range(begin, end);
      consumed = end;
    }
    if (U_FAILURE(status)) throw std::runtime_error("native tokenizer regex failed");
    if (consumed < normalized.length()) {
      std::string tail;
      normalized.tempSubString(consumed).toUTF8String(tail);
      const std::vector<std::uint32_t> ids = bpe(tail);
      output->insert(output->end(), ids.begin(), ids.end());
    }
  }

  std::vector<std::uint32_t> encode(std::string_view text) {
    require_loaded();
    std::vector<std::uint32_t> output;
    std::size_t cursor = 0;
    while (cursor < text.size()) {
      std::size_t next_position = std::string_view::npos;
      const AddedToken* next_token = nullptr;
      for (const AddedToken& token : added_tokens) {
        const std::size_t position = text.find(token.content, cursor);
        if (position == std::string_view::npos) continue;
        if (next_token == nullptr || position < next_position ||
            (position == next_position && token.content.size() > next_token->content.size())) {
          next_position = position;
          next_token = &token;
        }
      }
      if (next_token == nullptr) {
        encode_normal_span(text.substr(cursor), &output);
        break;
      }
      encode_normal_span(text.substr(cursor, next_position - cursor), &output);
      output.push_back(next_token->id);
      cursor = next_position + next_token->content.size();
    }
    return output;
  }
};

NativeTokenizer::NativeTokenizer() : impl_(std::make_unique<Impl>()) {}
NativeTokenizer::~NativeTokenizer() = default;
NativeTokenizer::NativeTokenizer(NativeTokenizer&&) noexcept = default;
NativeTokenizer& NativeTokenizer::operator=(NativeTokenizer&&) noexcept = default;

void NativeTokenizer::load(const std::filesystem::path& model_dir) {
  if (impl_->loaded) throw std::runtime_error("native tokenizer is already loaded");
  const std::filesystem::path tokenizer_path = model_dir / "tokenizer.json";
  const std::filesystem::path config_path = model_dir / "tokenizer_config.json";
  if (sha256_file(tokenizer_path) != generated::kTokenizerSha256 ||
      sha256_file(config_path) != generated::kTokenizerConfigSha256) {
    throw std::runtime_error("tokenizer identity does not match the native product contract");
  }
  const Json tokenizer = load_json(tokenizer_path);
  const Json config = load_json(config_path);
  if (tokenizer.at("normalizer").at("type") != "NFC" ||
      tokenizer.at("model").at("type") != "BPE" ||
      tokenizer.at("decoder").at("type") != "ByteLevel") {
    throw std::runtime_error("unsupported tokenizer pipeline in qualified model");
  }

  const auto& vocab = tokenizer.at("model").at("vocab");
  impl_->vocab.reserve(vocab.size());
  impl_->id_to_token.resize(vocab.size());
  for (auto item = vocab.begin(); item != vocab.end(); ++item) {
    const std::uint32_t id = item.value().get<std::uint32_t>();
    if (id >= impl_->id_to_token.size()) {
      throw std::runtime_error("BPE vocabulary ids are not dense");
    }
    impl_->vocab.emplace(item.key(), id);
    impl_->id_to_token[id] = item.key();
  }
  if (impl_->vocab.size() != 248044) {
    throw std::runtime_error("qualified native tokenizer requires 248044 BPE entries");
  }

  const auto& merges = tokenizer.at("model").at("merges");
  impl_->merge_ranks.reserve(merges.size());
  for (std::size_t rank = 0; rank < merges.size(); ++rank) {
    const std::string merge = merges[rank].get<std::string>();
    const std::size_t separator = merge.find(' ');
    if (separator == std::string::npos) throw std::runtime_error("invalid BPE merge");
    impl_->merge_ranks.emplace(
        pair_key(merge.substr(0, separator), merge.substr(separator + 1)), rank);
  }

  for (const auto& token : tokenizer.at("added_tokens")) {
    impl_->register_added(token.at("id").get<std::uint32_t>(),
                          token.at("content").get<std::string>());
  }
  for (auto item = config.at("added_tokens_decoder").begin();
       item != config.at("added_tokens_decoder").end(); ++item) {
    impl_->register_added(static_cast<std::uint32_t>(std::stoul(item.key())),
                          item.value().at("content").get<std::string>());
  }
  if (impl_->id_to_token.size() != 248077 || impl_->added_tokens.size() != 33) {
    throw std::runtime_error("qualified native tokenizer requires 33 added tokens");
  }

  const std::string regex_text = tokenizer.at("pre_tokenizer")
                                     .at("pretokenizers")[0]
                                     .at("pattern")
                                     .at("Regex")
                                     .get<std::string>();
  UErrorCode status = U_ZERO_ERROR;
  impl_->regex.reset(icu::RegexPattern::compile(
      icu::UnicodeString::fromUTF8(regex_text), 0, status));
  if (U_FAILURE(status) || impl_->regex == nullptr) {
    throw std::runtime_error("could not compile qualified tokenizer Unicode regex");
  }
  impl_->normalizer = icu::Normalizer2::getNFCInstance(status);
  if (U_FAILURE(status) || impl_->normalizer == nullptr) {
    throw std::runtime_error("could not initialize ICU NFC normalizer");
  }
  impl_->bpe_cache.reserve(32768);
  impl_->loaded = true;
}

std::vector<std::uint32_t> NativeTokenizer::encode(std::string_view text) {
  return impl_->encode(text);
}

std::string NativeTokenizer::decode(
    const std::vector<std::uint32_t>& token_ids) const {
  impl_->require_loaded();
  std::string output;
  std::string pending_bytes;
  auto flush = [&]() {
    if (pending_bytes.empty()) return;
    const icu::UnicodeString decoded = icu::UnicodeString::fromUTF8(
        icu::StringPiece(pending_bytes.data(), static_cast<int32_t>(pending_bytes.size())));
    decoded.toUTF8String(output);
    pending_bytes.clear();
  };
  for (const std::uint32_t id : token_ids) {
    if (id >= impl_->id_to_token.size() || impl_->id_to_token[id].empty()) {
      throw std::runtime_error("token id is outside the native tokenizer vocabulary");
    }
    if (impl_->added_ids.count(id) != 0) {
      flush();
      output += impl_->id_to_token[id];
      continue;
    }
    const icu::UnicodeString symbols =
        icu::UnicodeString::fromUTF8(impl_->id_to_token[id]);
    for (int32_t offset = 0; offset < symbols.length();) {
      const UChar32 point = symbols.char32At(offset);
      offset += U16_LENGTH(point);
      const auto found = impl_->unicode_to_bytes.find(point);
      if (found == impl_->unicode_to_bytes.end()) {
        throw std::runtime_error("BPE vocabulary contains an invalid byte-level symbol");
      }
      pending_bytes.push_back(static_cast<char>(found->second));
    }
  }
  flush();
  return output;
}

std::string NativeTokenizer::render_chat_prompt(
    std::string_view system_prompt, std::string_view user_prompt,
    bool disable_thinking) const {
  impl_->require_loaded();
  const std::string system = trim_unicode(system_prompt);
  const std::string user = trim_unicode(user_prompt);
  if (user.empty()) throw std::runtime_error("native chat prompt requires a user message");
  std::string prompt;
  if (!system.empty()) {
    prompt += "<|im_start|>system\n" + system + "<|im_end|>\n";
  }
  prompt += "<|im_start|>user\n" + user + "<|im_end|>\n";
  prompt += "<|im_start|>assistant\n";
  prompt += disable_thinking ? "<think>\n\n</think>\n\n" : "<think>\n";
  return prompt;
}

std::vector<std::uint32_t> NativeTokenizer::encode_chat(
    std::string_view system_prompt, std::string_view user_prompt,
    bool disable_thinking) {
  return encode(render_chat_prompt(system_prompt, user_prompt, disable_thinking));
}

std::size_t NativeTokenizer::size() const {
  impl_->require_loaded();
  return impl_->id_to_token.size();
}

std::uint32_t NativeTokenizer::eos_token_id() const {
  impl_->require_loaded();
  return impl_->eos;
}

std::uint32_t NativeTokenizer::pad_token_id() const {
  impl_->require_loaded();
  return impl_->pad;
}

}  // namespace aima
