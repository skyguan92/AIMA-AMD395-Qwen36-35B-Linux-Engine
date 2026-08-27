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

enum class NativeThinkingMode {
  // Preserve the pre-thinking-API behavior: text prompts use the answer-only
  // template while VL keeps its frozen vLLM-compatible tool-choice behavior.
  kDefault,
  kEnabled,
  kDisabled,
};

struct NativeHistoricalToolCall {
  std::string name;
  std::string serialized_arguments;
  std::string normalized_signature;
  std::size_t call_count = 0;
  std::size_t result_count = 0;
  std::size_t no_progress_result_count = 0;
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
  // Request-level media_io_kwargs are merged per modality with the frozen
  // launch mapping. Empty modality objects are no-ops; fps and num_frames use
  // vLLM's cross-field clearing rule.
  std::optional<NativeImageIoPolicy> image_io_override;
  std::optional<NativeVideoIoPolicy> video_io_override;
  NativeToolChoiceMode tool_choice = NativeToolChoiceMode::kAuto;
  std::string required_function_name;
  bool parallel_tool_calls = true;
  NativeThinkingMode thinking_mode = NativeThinkingMode::kDefault;
  // This is a validated declaration within the total max_tokens bound.  The
  // current native decoder does not impose a second, independent stop point.
  std::optional<std::size_t> thinking_budget_tokens;
  std::vector<NativeHistoricalToolCall> historical_tool_calls;
};

struct NativeParsedToolCall {
  std::string id;
  std::string name;
  NativeOrderedJson arguments;
  std::string serialized_arguments;
};

struct NativeToolProgress {
  std::size_t parsed_tool_calls = 0;
  std::size_t duplicate_calls_suppressed = 0;
  std::size_t parallel_calls_suppressed = 0;
  std::size_t history_signature_occurrences = 0;
  std::size_t history_no_progress_results = 0;
  std::size_t exhausted_history_calls_suppressed = 0;
  bool no_progress = false;
};

struct NativeAssistantOutput {
  bool reasoning_content_provided = false;
  std::string reasoning_content;
  std::string content;
  std::vector<NativeParsedToolCall> tool_calls;
  NativeToolProgress tool_progress;
};

struct NativeThinkingOutput {
  bool reasoning_content_provided = false;
  std::string reasoning_content;
  std::string content;
};

struct NativeThinkingStreamDelta {
  std::string reasoning_content;
  std::string content;
};

// Matches the whitespace and Unicode behavior of the Qwen tokenizer template's
// Jinja `tojson` filter while preserving the request's object field order.
std::string render_qwen_json(const NativeOrderedJson& value);

// Converts an OpenAI Chat Completions message/tool request into the exact
// Qwen-native template structures. Unsupported multimodal/deprecated shapes
// fail with invalid_argument rather than being silently changed.
NativePreparedChat prepare_native_chat(const NativeOrderedJson& request);

// budget_tokens declares a reasoning budget inside max_tokens; max_tokens
// remains the single hard generation limit for reasoning plus final content.
void validate_native_thinking_budget(const NativePreparedChat& chat,
                                     std::size_t max_tokens);

// Parses Qwen3 XML function calls and converts parameter values according to
// the supplied JSON schemas. Plain assistant text is preserved when no
// complete function call is present.
NativeAssistantOutput parse_qwen_tool_output(
    std::string_view model_output,
    const std::vector<NativeFunctionTool>& tools,
    std::string_view call_id_prefix);

// Splits Qwen's optional thinking region, parses any following tool markup,
// and applies the same duplicate/history/parallel policy used by both HTTP
// response modes.
NativeAssistantOutput parse_native_assistant_output(
    std::string_view model_output, const NativePreparedChat& chat,
    std::string_view call_id_prefix);

// Applies request-history, named-choice and parallel-call admission to output
// produced by either XML parsing or named-tool constrained decoding.
void apply_native_tool_call_policy(const NativePreparedChat& chat,
                                   NativeAssistantOutput* output);

// Conservative classification used by the bounded same-signature retry
// policy. Empty results and explicit failures are no progress; useful payloads
// are never rejected merely because they contain the word "error".
bool native_tool_result_is_no_progress(std::string_view result);

NativeThinkingOutput split_qwen_thinking_output(
    std::string_view model_output, bool enabled);

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
// `complete` is true only when the prefix ends at the complete JSON object.
// Trailing whitespace is rejected so the terminal decoder mask contains only
// EOS, matching the frozen named-tool generation boundary.
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

// Incrementally separates Qwen reasoning from final content. The optional
// opening marker and the closing marker are withheld even when split across
// token/UTF-8 boundaries, and post-marker CR/LF separators are suppressed.
class NativeThinkingStreamGate {
 public:
  explicit NativeThinkingStreamGate(bool enabled) : enabled_(enabled) {}

  NativeThinkingStreamDelta push(std::string_view utf8);
  NativeThinkingStreamDelta finish();

 private:
  NativeThinkingStreamDelta process(bool terminal);

  bool enabled_ = false;
  bool opening_decided_ = false;
  bool stripping_reasoning_newlines_ = false;
  bool reasoning_finished_ = false;
  bool stripping_content_newlines_ = false;
  std::string pending_;
};

}  // namespace aima
