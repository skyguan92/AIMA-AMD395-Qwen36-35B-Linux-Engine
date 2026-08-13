// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstddef>
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

enum class NativeMediaKind {
  kImage,
  kVideo,
};

// An ordered OpenAI content part. The source is retained only for the native
// media loader; prompts receive the model's canonical special-token marker.
struct NativeMediaPart {
  NativeMediaKind kind = NativeMediaKind::kImage;
  std::string source;
  std::size_t message_index = 0;
  std::size_t content_part_index = 0;
  std::size_t media_index = 0;
};

struct NativePreparedChat {
  std::vector<NativeChatMessage> messages;
  std::vector<NativeChatTool> prompt_tools;
  std::vector<NativeFunctionTool> function_tools;
  std::vector<NativeMediaPart> media;
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
