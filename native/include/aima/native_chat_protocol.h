// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_media.h"
#include "aima/native_tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace aima {

using NativeOrderedJson = nlohmann::ordered_json;

struct NativeFunctionTool {
  std::string name;
  NativeOrderedJson definition;
  NativeOrderedJson parameters;
};

enum class NativeToolChoiceMode {
  kAuto,
  kNone,
  kRequired,
  kSpecific,
};

struct NativePreparedChat {
  // Baseline text serving retains its established synthetic tool-choice
  // directives and content-array concatenation in messages/prompt_tools. VL
  // rendering uses vLLM's auto-resolved string content layout and
  // Pydantic-normalized tools: supplied media markers are grouped and prepended
  // per message, text parts are newline-joined, every tool stays visible, and
  // no tool_choice prose is injected.
  std::vector<NativeChatMessage> vl_prompt_messages;
  std::vector<NativeChatTool> vl_prompt_tools;
  std::vector<NativeChatMessage> messages;
  std::vector<NativeChatTool> prompt_tools;
  std::vector<NativeFunctionTool> function_tools;
  std::vector<NativeMediaPart> media;
  // A non-empty request-level media_io_kwargs object replaces the frozen
  // launch-level media IO mapping in vLLM. Empty/absent objects retain the
  // launch default and therefore leave this override unset.
  std::optional<NativeVideoIoPolicy> video_io_override;
  NativeToolChoiceMode tool_choice = NativeToolChoiceMode::kAuto;
  std::string required_function_name;
  bool parallel_tool_calls = true;
};

struct NativeParsedToolCall {
  std::string id;
  std::string name;
  NativeOrderedJson arguments;
  std::string serialized_arguments;
};

struct NativeAssistantOutput {
  std::string content;
  std::vector<NativeParsedToolCall> tool_calls;
};

// Matches the whitespace and Unicode behavior of the Qwen tokenizer template's
// Jinja `tojson` filter while preserving the request's object field order.
std::string render_qwen_json(const NativeOrderedJson& value);

// Converts an OpenAI Chat Completions message/tool request into the exact
// Qwen-native template structures. Unsupported multimodal/deprecated shapes
// fail with invalid_argument rather than being silently changed.
NativePreparedChat prepare_native_chat(const NativeOrderedJson& request);

// Parses Qwen3 XML function calls and converts parameter values according to
// the supplied JSON schemas. Plain assistant text is preserved when no
// complete function call is present.
NativeAssistantOutput parse_qwen_tool_output(
    std::string_view model_output,
    const std::vector<NativeFunctionTool>& tools,
    std::string_view call_id_prefix);

// Fixed-vLLM named tool_choice constrains generation to the selected
// function's parameters JSON Schema, then wraps the resulting JSON as that
// function's arguments. The current frozen capability schema is a closed
// object with exactly one required string property; unsupported schemas fail
// closed instead of silently becoming unconstrained generation.
class NativeNamedToolJsonConstraint {
 public:
  NativeNamedToolJsonConstraint(const NativeTokenizer& tokenizer,
                                const NativeFunctionTool& tool);

  void allowed_token_mask(
      const std::vector<std::uint32_t>& generated_token_ids,
      std::vector<std::uint8_t>* mask) const;
  NativeAssistantOutput parse_output(
      std::string_view model_output,
      std::string_view call_id_prefix) const;

 private:
  const NativeTokenizer* tokenizer_ = nullptr;
  std::string function_name_;
  std::string property_name_;
  std::string rendered_property_;
  std::vector<std::string> token_bytes_;
};

// Pure prefix oracle used by CPU tests and the token-mask implementation.
// `complete` is true only when the prefix is already a complete JSON object
// (possibly followed by JSON whitespace).
bool native_single_string_json_prefix_viable(
    std::string_view property_name, std::string_view prefix,
    bool* complete);

NativeAssistantOutput parse_native_named_tool_json_output(
    std::string_view model_output, const NativeFunctionTool& tool,
    std::string_view call_id_prefix);

// Converts raw byte-level tokenizer fragments to valid UTF-8 without emitting
// an incomplete code point. Invalid terminal bytes become U+FFFD.
class NativeIncrementalUtf8Decoder {
 public:
  std::string push(std::string_view bytes);
  std::string finish();

 private:
  std::string pending_;
};

// Streams ordinary assistant content while retaining only the small suffix
// that could become `<tool_call>`. Once the marker is seen, the XML region is
// held for structured parsing and never leaks into `delta.content`.
class NativeToolStreamGate {
 public:
  std::string push(std::string_view utf8);
  std::string finish(bool parsed_tool_calls);
  const std::string& complete_text() const { return complete_text_; }

 private:
  std::string complete_text_;
  std::size_t emitted_bytes_ = 0;
  bool tool_marker_seen_ = false;
};

}  // namespace aima
