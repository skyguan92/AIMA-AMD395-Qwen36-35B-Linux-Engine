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
