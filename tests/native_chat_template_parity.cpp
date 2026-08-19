// SPDX-License-Identifier: Apache-2.0

#include "aima/native_chat_protocol.h"
#include "aima/native_tokenizer.h"
#include "aima/sha256.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = aima::NativeOrderedJson;

void check_case(aima::NativeTokenizer* tokenizer, const char* name,
                const Json& request, const char* expected_sha256,
                std::size_t expected_tokens) {
  const aima::NativePreparedChat prepared =
      aima::prepare_native_chat(request);
  const std::string prompt = tokenizer->render_chat_prompt(
      prepared.messages, prepared.prompt_tools, true);
  const std::string actual_sha256 =
      aima::sha256_bytes(prompt.data(), prompt.size());
  const std::size_t actual_tokens = tokenizer->encode(prompt).size();
  if (actual_sha256 != expected_sha256 ||
      actual_tokens != expected_tokens) {
    std::cerr << name << " parity failed: sha256=" << actual_sha256
              << " tokens=" << actual_tokens << '\n';
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: native_chat_template_parity MODEL_DIR\n";
    return 2;
  }
  aima::NativeTokenizer tokenizer;
  tokenizer.load(std::filesystem::absolute(argv[1]));
  const Json tool = {
      {"type", "function"},
      {"function",
       {{"name", "weather"},
        {"description", "天气"},
        {"parameters",
         {{"type", "object"},
          {"properties",
           {{"city", {{"type", "string"}}},
            {"days", {{"type", "integer"}}}}}}}}}};

  check_case(
      &tokenizer, "simple",
      {{"messages",
        Json::array(
            {{{"role", "system"}, {"content", " Be concise. "}},
             {{"role", "user"}, {"content", " Hello 世界 "}}})}},
      "7bedf5ff847e89b2836218398d22fa6db5197ebbcff83cf98dbbcef420c2ba9d",
      23);
  check_case(
      &tokenizer, "tool",
      {{"messages",
        Json::array(
            {{{"role", "system"}, {"content", " Be concise. "}},
             {{"role", "user"}, {"content", "Weather in Paris?"}}})},
       {"tools", Json::array({tool})},
       {"tool_choice", "auto"}},
      "f8300b39c0e87c79e052cdd4cd152a143a94db95268d0217d440e731b05a5914",
      274);
  check_case(
      &tokenizer, "history",
      {{"messages",
        Json::array(
            {{{"role", "user"}, {"content", "Use it"}},
             {{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array(
                   {{{"id", "call_1"},
                     {"type", "function"},
                     {"function",
                      {{"name", "weather"},
                       {"arguments",
                        R"({"city":"Paris","days":2})"}}}}})}},
             {{"role", "tool"},
              {"tool_call_id", "call_1"},
              {"content", R"({"temperature":20})"}},
             {{"role", "user"}, {"content", "Summarize"}}})},
       {"tools", Json::array({tool})}},
      "e763e00178be9cbb2df3de13e9b393d815e976ecdd4e8e9ddea01260b165566c",
      333);

  constexpr std::size_t kNearWindowVisualTokens = 245760;
  constexpr std::uint32_t kImagePadTokenId = 248056;
  std::string near_window_media_tokens;
  near_window_media_tokens.reserve(
      kNearWindowVisualTokens * std::string("<|image_pad|>").size());
  for (std::size_t index = 0; index < kNearWindowVisualTokens; ++index) {
    near_window_media_tokens += "<|image_pad|>";
  }
  const std::vector<std::uint32_t> near_window_ids =
      tokenizer.encode(near_window_media_tokens);
  if (near_window_ids.size() != kNearWindowVisualTokens ||
      !std::all_of(near_window_ids.begin(), near_window_ids.end(),
                   [](std::uint32_t id) { return id == kImagePadTokenId; })) {
    std::cerr << "near-window added-token scan drifted\n";
    return 1;
  }

  const Json named_tool_request = {
      {"messages",
       Json::array({{{"role", "user"}, {"content", "Inspect it"}}})},
      {"tools",
       Json::array(
           {{{"type", "function"},
             {"function",
              {{"name", "inspect_visual"},
               {"description", "Record a label."},
               {"parameters",
                {{"type", "object"},
                 {"properties", {{"label", {{"type", "string"}}}}},
                 {"required", Json::array({"label"})},
                 {"additionalProperties", false}}}}}}})},
      {"tool_choice",
       {{"type", "function"},
        {"function", {{"name", "inspect_visual"}}}}}};
  const aima::NativePreparedChat named =
      aima::prepare_native_chat(named_tool_request);
  aima::NativeNamedToolJsonConstraint constraint(
      tokenizer, named.function_tools[0]);
  const std::vector<std::uint32_t> expected_arguments = tokenizer.encode(
      R"({"label": "Colorful diagonal stripes"})");
  std::vector<std::uint32_t> generated;
  std::vector<std::uint8_t> mask;
  for (const std::uint32_t token_id : expected_arguments) {
    constraint.allowed_token_mask(generated, &mask);
    if (mask.size() != aima::kNativeModelVocabularySize ||
        mask[token_id] == 0 || mask[tokenizer.eos_token_id()] != 0) {
      std::cerr << "named-tool JSON token grammar rejected a valid prefix\n";
      return 1;
    }
    generated.push_back(token_id);
  }
  constraint.allowed_token_mask(generated, &mask);
  if (mask[tokenizer.eos_token_id()] == 0 ||
      std::count(mask.begin(), mask.end(), std::uint8_t{1}) != 1) {
    std::cerr << "named-tool JSON terminal mask was not EOS-only\n";
    return 1;
  }

  std::cout << "native_chat_template_parity: PASS\n";
  return 0;
}
