// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace aima {

struct NativeChatTool {
  // Exact JSON representation consumed by the qualified Qwen chat template.
  std::string serialized_json;
};

struct NativeChatToolCallArgument {
  std::string name;
  // String arguments are stored verbatim. Other JSON values use compact JSON.
  std::string rendered_value;
};

struct NativeChatToolCall {
  std::string name;
  std::vector<NativeChatToolCallArgument> arguments;
};

struct NativeChatMessage {
  std::string role;
  std::string content;
  bool reasoning_content_provided = false;
  std::string reasoning_content;
  std::vector<NativeChatToolCall> tool_calls;
};

class NativeTokenizer {
 public:
  NativeTokenizer();
  ~NativeTokenizer();
  NativeTokenizer(NativeTokenizer&&) noexcept;
  NativeTokenizer& operator=(NativeTokenizer&&) noexcept;
  NativeTokenizer(const NativeTokenizer&) = delete;
  NativeTokenizer& operator=(const NativeTokenizer&) = delete;

  void load(const std::filesystem::path& model_dir);
  std::vector<std::uint32_t> encode(std::string_view text);
  std::string decode(const std::vector<std::uint32_t>& token_ids) const;
  // Returns the raw byte-level BPE payload for one token. Added tokens are
  // returned as their UTF-8 spelling. Callers doing incremental output must
  // preserve incomplete UTF-8 sequences between tokens.
  std::string decode_token_bytes(std::uint32_t token_id) const;
  std::string render_chat_prompt(
      const std::vector<NativeChatMessage>& messages,
      const std::vector<NativeChatTool>& tools, bool disable_thinking,
      bool preserve_thinking = false) const;
  std::vector<std::uint32_t> encode_chat(
      const std::vector<NativeChatMessage>& messages,
      const std::vector<NativeChatTool>& tools, bool disable_thinking,
      bool preserve_thinking = false);
  std::string render_chat_prompt(std::string_view system_prompt,
                                 std::string_view user_prompt,
                                 bool disable_thinking) const;
  std::vector<std::uint32_t> encode_chat(std::string_view system_prompt,
                                         std::string_view user_prompt,
                                         bool disable_thinking);
  std::size_t size() const;
  std::uint32_t eos_token_id() const;
  std::uint32_t pad_token_id() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
