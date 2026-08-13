// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_chat_protocol.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace aima {
namespace {

constexpr std::string_view kToolCallStart = "<tool_call>";
constexpr std::string_view kToolCallEnd = "</tool_call>";
constexpr std::string_view kFunctionStart = "<function=";
constexpr std::string_view kFunctionEnd = "</function>";
constexpr std::string_view kParameterStart = "<parameter=";
constexpr std::string_view kParameterEnd = "</parameter>";
constexpr std::string_view kImagePlaceholder =
    "<|vision_start|><|image_pad|><|vision_end|>";
constexpr std::string_view kVideoPlaceholder =
    "<|vision_start|><|video_pad|><|vision_end|>";
constexpr std::size_t kMaximumImagesPerPrompt = 16;
constexpr std::size_t kMaximumVideosPerPrompt = 21;

std::string trim_ascii_copy(std::string_view input) {
  std::size_t begin = 0;
  std::size_t end = input.size();
  while (begin < end &&
         std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
    --end;
  }
  return std::string(input.substr(begin, end - begin));
}

std::string json_string(std::string_view value) {
  return NativeOrderedJson(std::string(value)).dump(
      -1, ' ', false, NativeOrderedJson::error_handler_t::strict);
}

bool valid_function_name(std::string_view name) {
  if (name.empty() || name.size() > 64) return false;
  return std::all_of(name.begin(), name.end(), [](unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '_' || ch == '-';
  });
}

std::string media_source(const NativeOrderedJson& part,
                         std::string_view field) {
  const std::string key(field);
  if (!part.contains(key)) {
    throw std::invalid_argument(std::string(field) +
                                " content part requires a source");
  }
  const NativeOrderedJson& value = part[key];
  if (value.is_string() && !value.get_ref<const std::string&>().empty()) {
    return value.get<std::string>();
  }
  if (value.is_object() && value.contains("url") &&
      value["url"].is_string() &&
      !value["url"].get_ref<const std::string&>().empty()) {
    return value["url"].get<std::string>();
  }
  throw std::invalid_argument(std::string(field) +
                              " content part requires a non-empty URL");
}

std::string content_text(const NativeOrderedJson& message,
                         std::string_view role,
                         std::size_t message_index,
                         NativePreparedChat* prepared) {
  if (!message.contains("content") || message["content"].is_null()) return {};
  const NativeOrderedJson& content = message["content"];
  if (content.is_string()) return content.get<std::string>();
  if (!content.is_array()) {
    throw std::invalid_argument(
        "message content must be a string, null, or a text-part array");
  }
  std::string result;
  for (std::size_t part_index = 0; part_index < content.size(); ++part_index) {
    const NativeOrderedJson& part = content[part_index];
    if (!part.is_object() || !part.contains("type") ||
        !part["type"].is_string()) {
      throw std::invalid_argument(
          "message content parts require a string type");
    }
    const std::string type = part["type"].get<std::string>();
    if (type == "text") {
      if (!part.contains("text") || !part["text"].is_string()) {
        throw std::invalid_argument("text content part requires string text");
      }
      result += part["text"].get<std::string>();
      continue;
    }
    NativeMediaKind kind;
    std::string_view field;
    std::string_view placeholder;
    if (type == "image_url") {
      kind = NativeMediaKind::kImage;
      field = "image_url";
      placeholder = kImagePlaceholder;
    } else if (type == "video_url") {
      kind = NativeMediaKind::kVideo;
      field = "video_url";
      placeholder = kVideoPlaceholder;
    } else {
      throw std::invalid_argument("unsupported message content part type");
    }
    if (role != "user") {
      throw std::invalid_argument(
          "image and video content parts are supported in user messages only");
    }
    NativeMediaPart media;
    media.kind = kind;
    media.source = media_source(part, field);
    media.message_index = message_index;
    media.content_part_index = part_index;
    media.media_index = prepared->media.size();
    prepared->media.push_back(std::move(media));
    result += placeholder;
  }
  return result;
}

const NativeFunctionTool* find_tool(
    const std::vector<NativeFunctionTool>& tools, std::string_view name) {
  const auto found = std::find_if(
      tools.begin(), tools.end(), [&](const NativeFunctionTool& tool) {
        return tool.name == name;
      });
  return found == tools.end() ? nullptr : &*found;
}

NativeOrderedJson convert_parameter(
    std::string value, std::string_view parameter_name,
    const NativeFunctionTool* tool) {
  if (!value.empty() && value.front() == '\n') value.erase(value.begin());
  if (!value.empty() && value.back() == '\n') value.pop_back();
  if (value == "null" || value == "NULL" || value == "Null") return nullptr;

  const NativeOrderedJson* schema = nullptr;
  if (tool != nullptr && tool->parameters.is_object() &&
      tool->parameters.contains("properties") &&
      tool->parameters["properties"].is_object()) {
    const auto found =
        tool->parameters["properties"].find(std::string(parameter_name));
    if (found != tool->parameters["properties"].end() && found->is_object()) {
      schema = &*found;
    }
  }
  std::string type = "string";
  if (schema != nullptr && schema->contains("type") &&
      (*schema)["type"].is_string()) {
    type = (*schema)["type"].get<std::string>();
    std::transform(type.begin(), type.end(), type.begin(),
                   [](unsigned char ch) {
                     return static_cast<char>(std::tolower(ch));
                   });
  } else if (schema != nullptr && schema->contains("anyOf")) {
    type = "object";
  }

  if (type == "string" || type == "str" || type == "text" ||
      type == "varchar" || type == "char" || type == "enum") {
    return value;
  }
  if (type.rfind("int", 0) == 0 || type.rfind("uint", 0) == 0 ||
      type.rfind("long", 0) == 0 || type.rfind("short", 0) == 0 ||
      type.rfind("unsigned", 0) == 0) {
    std::int64_t parsed = 0;
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec == std::errc{} &&
        result.ptr == value.data() + value.size()) {
      return parsed;
    }
    return value;
  }
  if (type.rfind("num", 0) == 0 || type.rfind("float", 0) == 0 ||
      type == "number" || type == "double") {
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (errno == 0 && end == value.c_str() + value.size() &&
        std::isfinite(parsed)) {
      const double integral = std::trunc(parsed);
      if (integral == parsed &&
          integral >= static_cast<double>(
                          std::numeric_limits<std::int64_t>::min()) &&
          integral <= static_cast<double>(
                          std::numeric_limits<std::int64_t>::max())) {
        return static_cast<std::int64_t>(integral);
      }
      return parsed;
    }
    return value;
  }
  if (type == "boolean" || type == "bool" || type == "binary") {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                     return static_cast<char>(std::tolower(ch));
                   });
    return value == "true";
  }
  if (type == "object" || type == "array" || type == "arr" ||
      type.rfind("dict", 0) == 0 || type.rfind("list", 0) == 0) {
    try {
      return NativeOrderedJson::parse(value);
    } catch (const NativeOrderedJson::exception&) {
      return value;
    }
  }
  try {
    return NativeOrderedJson::parse(value);
  } catch (const NativeOrderedJson::exception&) {
    return value;
  }
}

bool parse_tool_block(std::string_view block,
                      const std::vector<NativeFunctionTool>& tools,
                      std::string_view call_id, NativeParsedToolCall* output) {
  const std::size_t function = block.find(kFunctionStart);
  if (function == std::string_view::npos) return false;
  const std::size_t name_begin = function + kFunctionStart.size();
  const std::size_t name_end = block.find('>', name_begin);
  if (name_end == std::string_view::npos) return false;
  const std::string name = trim_ascii_copy(
      block.substr(name_begin, name_end - name_begin));
  if (!valid_function_name(name)) return false;
  const std::size_t function_end = block.find(kFunctionEnd, name_end + 1);
  if (function_end == std::string_view::npos) return false;

  NativeOrderedJson arguments = NativeOrderedJson::object();
  const NativeFunctionTool* tool = find_tool(tools, name);
  if (!tools.empty() && tool == nullptr) return false;
  std::size_t cursor = name_end + 1;
  while (cursor < function_end) {
    const std::size_t parameter = block.find(kParameterStart, cursor);
    if (parameter == std::string_view::npos || parameter >= function_end) break;
    const std::size_t parameter_name_begin =
        parameter + kParameterStart.size();
    const std::size_t parameter_name_end =
        block.find('>', parameter_name_begin);
    if (parameter_name_end == std::string_view::npos ||
        parameter_name_end >= function_end) {
      return false;
    }
    const std::string parameter_name = trim_ascii_copy(
        block.substr(parameter_name_begin,
                     parameter_name_end - parameter_name_begin));
    if (parameter_name.empty() ||
        parameter_name.find_first_of(">\r\n") != std::string::npos) {
      return false;
    }
    const std::size_t parameter_end =
        block.find(kParameterEnd, parameter_name_end + 1);
    if (parameter_end == std::string_view::npos ||
        parameter_end > function_end) {
      return false;
    }
    std::string value(block.substr(parameter_name_end + 1,
                                   parameter_end - parameter_name_end - 1));
    arguments[parameter_name] =
        convert_parameter(std::move(value), parameter_name, tool);
    cursor = parameter_end + kParameterEnd.size();
  }

  output->id = std::string(call_id);
  output->name = name;
  output->arguments = std::move(arguments);
  output->serialized_arguments = render_qwen_json(output->arguments);
  return true;
}

std::size_t longest_marker_prefix_suffix(std::string_view value) {
  const std::size_t maximum =
      std::min(value.size(), kToolCallStart.size() - 1);
  for (std::size_t length = maximum; length != 0; --length) {
    if (value.substr(value.size() - length) ==
        kToolCallStart.substr(0, length)) {
      return length;
    }
  }
  return 0;
}

}  // namespace

std::string render_qwen_json(const NativeOrderedJson& value) {
  if (value.is_object()) {
    std::string output = "{";
    bool first = true;
    for (auto item = value.begin(); item != value.end(); ++item) {
      if (!first) output += ", ";
      first = false;
      output += json_string(item.key());
      output += ": ";
      output += render_qwen_json(item.value());
    }
    output += '}';
    return output;
  }
  if (value.is_array()) {
    std::string output = "[";
    for (std::size_t index = 0; index < value.size(); ++index) {
      if (index != 0) output += ", ";
      output += render_qwen_json(value[index]);
    }
    output += ']';
    return output;
  }
  return value.dump(-1, ' ', false,
                    NativeOrderedJson::error_handler_t::strict);
}

NativePreparedChat prepare_native_chat(const NativeOrderedJson& request) {
  if (!request.is_object()) {
    throw std::invalid_argument("request body must be a JSON object");
  }
  if (!request.contains("messages") || !request["messages"].is_array() ||
      request["messages"].empty()) {
    throw std::invalid_argument("messages must be a non-empty array");
  }

  NativePreparedChat prepared;
  std::unordered_set<std::string> tool_names;
  if (request.contains("tools")) {
    if (!request["tools"].is_array()) {
      throw std::invalid_argument("tools must be an array");
    }
    for (const NativeOrderedJson& tool : request["tools"]) {
      if (!tool.is_object() || tool.value("type", "") != "function" ||
          !tool.contains("function") || !tool["function"].is_object() ||
          !tool["function"].contains("name") ||
          !tool["function"]["name"].is_string()) {
        throw std::invalid_argument(
            "each tool must be an OpenAI function tool with a string name");
      }
      const std::string name =
          tool["function"]["name"].get<std::string>();
      if (!valid_function_name(name) || !tool_names.insert(name).second) {
        throw std::invalid_argument(
            "tool names must be unique 1-64 character identifiers using "
            "letters, digits, underscores, or hyphens");
      }
      NativeFunctionTool parsed;
      parsed.name = name;
      parsed.definition = tool;
      parsed.parameters =
          tool["function"].contains("parameters")
              ? tool["function"]["parameters"]
              : NativeOrderedJson::object();
      if (!parsed.parameters.is_object()) {
        throw std::invalid_argument(
            "function tool parameters must be a JSON Schema object");
      }
      prepared.function_tools.push_back(std::move(parsed));
    }
  }

  if (request.contains("tool_choice")) {
    const NativeOrderedJson& choice = request["tool_choice"];
    if (choice.is_string()) {
      const std::string value = choice.get<std::string>();
      if (value == "auto") {
        prepared.tool_choice = NativeToolChoiceMode::kAuto;
      } else if (value == "none") {
        prepared.tool_choice = NativeToolChoiceMode::kNone;
      } else if (value == "required") {
        prepared.tool_choice = NativeToolChoiceMode::kRequired;
      } else {
        throw std::invalid_argument(
            "tool_choice must be auto, none, required, or a function object");
      }
    } else if (choice.is_object() &&
               choice.value("type", "") == "function" &&
               choice.contains("function") &&
               choice["function"].is_object() &&
               choice["function"].contains("name") &&
               choice["function"]["name"].is_string()) {
      prepared.tool_choice = NativeToolChoiceMode::kSpecific;
      prepared.required_function_name =
          choice["function"]["name"].get<std::string>();
      if (tool_names.count(prepared.required_function_name) == 0) {
        throw std::invalid_argument(
            "tool_choice names a function absent from tools");
      }
    } else {
      throw std::invalid_argument(
          "tool_choice must be auto, none, required, or a function object");
    }
  }
  if ((prepared.tool_choice == NativeToolChoiceMode::kRequired ||
       prepared.tool_choice == NativeToolChoiceMode::kSpecific) &&
      prepared.function_tools.empty()) {
    throw std::invalid_argument("tool_choice requires at least one tool");
  }
  if (request.contains("parallel_tool_calls")) {
    if (!request["parallel_tool_calls"].is_boolean()) {
      throw std::invalid_argument("parallel_tool_calls must be boolean");
    }
    prepared.parallel_tool_calls =
        request["parallel_tool_calls"].get<bool>();
  }

  std::string leading_system;
  bool left_leading_system = false;
  bool saw_user = false;
  std::unordered_set<std::string> known_call_ids;
  for (std::size_t message_index = 0;
       message_index < request["messages"].size(); ++message_index) {
    const NativeOrderedJson& source = request["messages"][message_index];
    if (!source.is_object() || !source.contains("role") ||
        !source["role"].is_string()) {
      throw std::invalid_argument("each message requires a string role");
    }
    const std::string role = source["role"].get<std::string>();
    if (role == "system" || role == "developer") {
      if (left_leading_system) {
        throw std::invalid_argument(
            "system and developer messages must precede conversation messages");
      }
      const std::string content = content_text(
          source, role, message_index, &prepared);
      if (!leading_system.empty() && !content.empty()) leading_system += '\n';
      leading_system += content;
      continue;
    }
    left_leading_system = true;
    if (prepared.messages.empty() && !leading_system.empty()) {
      prepared.messages.push_back(
          {"system", leading_system, false, std::string(), {}});
    }

    NativeChatMessage message;
    message.role = role;
    message.content = content_text(source, role, message_index, &prepared);
    if (role == "user") {
      saw_user = true;
    } else if (role == "assistant") {
      if (source.contains("reasoning_content") &&
          !source["reasoning_content"].is_null()) {
        if (!source["reasoning_content"].is_string()) {
          throw std::invalid_argument(
              "assistant reasoning_content must be a string or null");
        }
        message.reasoning_content_provided = true;
        message.reasoning_content =
            source["reasoning_content"].get<std::string>();
      }
      if (source.contains("tool_calls") &&
          !source["tool_calls"].is_null()) {
        if (!source["tool_calls"].is_array()) {
          throw std::invalid_argument(
              "assistant tool_calls must be an array");
        }
        for (const NativeOrderedJson& source_call : source["tool_calls"]) {
          if (!source_call.is_object() ||
              !source_call.contains("id") ||
              !source_call["id"].is_string() ||
              source_call["id"].get<std::string>().empty() ||
              source_call.value("type", "function") != "function" ||
              !source_call.contains("function") ||
              !source_call["function"].is_object() ||
              !source_call["function"].contains("name") ||
              !source_call["function"]["name"].is_string() ||
              !source_call["function"].contains("arguments") ||
              !source_call["function"]["arguments"].is_string()) {
            throw std::invalid_argument(
                "assistant tool_calls require function name and JSON arguments");
          }
          if (!known_call_ids
                   .insert(source_call["id"].get<std::string>())
                   .second) {
            throw std::invalid_argument(
                "assistant tool call ids must be unique");
          }
          NativeChatToolCall call;
          call.name =
              source_call["function"]["name"].get<std::string>();
          if (!valid_function_name(call.name)) {
            throw std::invalid_argument(
                "assistant function name is invalid");
          }
          NativeOrderedJson arguments;
          try {
            arguments = NativeOrderedJson::parse(
                source_call["function"]["arguments"].get<std::string>());
          } catch (const NativeOrderedJson::exception&) {
            throw std::invalid_argument(
                "assistant function arguments must contain valid JSON");
          }
          if (!arguments.is_object()) {
            throw std::invalid_argument(
                "assistant function arguments must encode a JSON object");
          }
          for (auto item = arguments.begin(); item != arguments.end(); ++item) {
            if (item.key().empty() ||
                item.key().find_first_of(">\r\n") != std::string::npos) {
              throw std::invalid_argument(
                  "assistant function argument name is invalid");
            }
            call.arguments.push_back(
                {item.key(),
                 item.value().is_string()
                     ? item.value().get<std::string>()
                     : render_qwen_json(item.value())});
          }
          message.tool_calls.push_back(std::move(call));
        }
      }
    } else if (role == "tool") {
      if (!source.contains("tool_call_id") ||
          !source["tool_call_id"].is_string() ||
          source["tool_call_id"].get<std::string>().empty()) {
        throw std::invalid_argument(
            "tool messages require a string tool_call_id");
      }
      const std::string id = source["tool_call_id"].get<std::string>();
      if (known_call_ids.count(id) == 0) {
        throw std::invalid_argument(
            "tool message tool_call_id has no preceding assistant tool call");
      }
    } else {
      throw std::invalid_argument(
          "message role must be system, developer, user, assistant, or tool");
    }
    prepared.messages.push_back(std::move(message));
  }
  if (prepared.messages.empty() && !leading_system.empty()) {
    prepared.messages.push_back(
        {"system", leading_system, false, std::string(), {}});
  }
  if (!saw_user) {
    throw std::invalid_argument("at least one user message is required");
  }
  const std::size_t image_count = static_cast<std::size_t>(std::count_if(
      prepared.media.begin(), prepared.media.end(),
      [](const NativeMediaPart& media) {
        return media.kind == NativeMediaKind::kImage;
      }));
  const std::size_t video_count = prepared.media.size() - image_count;
  if (image_count > kMaximumImagesPerPrompt) {
    throw std::invalid_argument("image count exceeds the fixed limit of 16");
  }
  if (video_count > kMaximumVideosPerPrompt) {
    throw std::invalid_argument("video count exceeds the fixed limit of 21");
  }

  std::string directive;
  if (prepared.tool_choice == NativeToolChoiceMode::kRequired) {
    directive =
        "You must call one or more of the available functions. Do not answer "
        "normally.";
  } else if (prepared.tool_choice == NativeToolChoiceMode::kSpecific) {
    directive = "You must call the function `" +
                prepared.required_function_name +
                "`. Do not call another function and do not answer normally.";
  }
  if (!prepared.parallel_tool_calls && !prepared.function_tools.empty() &&
      prepared.tool_choice != NativeToolChoiceMode::kNone) {
    if (!directive.empty()) directive += '\n';
    directive += "Call at most one function.";
  }
  if (!directive.empty()) {
    if (prepared.messages.front().role != "system") {
      prepared.messages.insert(prepared.messages.begin(),
                               NativeChatMessage{
                                   "system", directive, false,
                                   std::string(), {}});
    } else {
      if (!prepared.messages.front().content.empty()) {
        prepared.messages.front().content += "\n\n";
      }
      prepared.messages.front().content += directive;
    }
  }

  if (prepared.tool_choice != NativeToolChoiceMode::kNone) {
    for (const NativeFunctionTool& tool : prepared.function_tools) {
      if (prepared.tool_choice == NativeToolChoiceMode::kSpecific &&
          tool.name != prepared.required_function_name) {
        continue;
      }
      prepared.prompt_tools.push_back({render_qwen_json(tool.definition)});
    }
  }
  return prepared;
}

NativeAssistantOutput parse_qwen_tool_output(
    std::string_view model_output,
    const std::vector<NativeFunctionTool>& tools,
    std::string_view call_id_prefix) {
  NativeAssistantOutput output;
  std::size_t cursor = 0;
  std::size_t first_call = std::string_view::npos;
  while (cursor < model_output.size()) {
    const std::size_t start = model_output.find(kToolCallStart, cursor);
    if (start == std::string_view::npos) break;
    const std::size_t end = model_output.find(kToolCallEnd,
                                              start + kToolCallStart.size());
    if (end == std::string_view::npos) break;
    const std::size_t block_end = end + kToolCallEnd.size();
    NativeParsedToolCall call;
    const std::string id = std::string(call_id_prefix) +
                           std::to_string(output.tool_calls.size());
    if (parse_tool_block(model_output.substr(start, block_end - start), tools,
                         id, &call)) {
      if (first_call == std::string_view::npos) first_call = start;
      output.tool_calls.push_back(std::move(call));
    }
    cursor = block_end;
  }
  if (output.tool_calls.empty()) {
    output.content = std::string(model_output);
  } else {
    output.content = std::string(model_output.substr(0, first_call));
  }
  return output;
}

std::string NativeIncrementalUtf8Decoder::push(std::string_view bytes) {
  pending_.append(bytes);
  std::string output;
  std::size_t index = 0;
  while (index < pending_.size()) {
    const unsigned char lead =
        static_cast<unsigned char>(pending_[index]);
    std::size_t width = 0;
    if (lead <= 0x7f) {
      width = 1;
    } else if (lead >= 0xc2 && lead <= 0xdf) {
      width = 2;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      width = 3;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      width = 4;
    } else {
      output += "\xef\xbf\xbd";
      ++index;
      continue;
    }
    if (index + width > pending_.size()) break;
    bool valid = true;
    for (std::size_t offset = 1; offset < width; ++offset) {
      const unsigned char continuation =
          static_cast<unsigned char>(pending_[index + offset]);
      if ((continuation & 0xc0) != 0x80) {
        valid = false;
        break;
      }
    }
    if (valid && width == 3) {
      const unsigned char second =
          static_cast<unsigned char>(pending_[index + 1]);
      valid = !((lead == 0xe0 && second < 0xa0) ||
                (lead == 0xed && second >= 0xa0));
    } else if (valid && width == 4) {
      const unsigned char second =
          static_cast<unsigned char>(pending_[index + 1]);
      valid = !((lead == 0xf0 && second < 0x90) ||
                (lead == 0xf4 && second >= 0x90));
    }
    if (!valid) {
      output += "\xef\xbf\xbd";
      ++index;
      continue;
    }
    output.append(pending_, index, width);
    index += width;
  }
  pending_.erase(0, index);
  return output;
}

std::string NativeIncrementalUtf8Decoder::finish() {
  if (pending_.empty()) return {};
  std::string output;
  output.reserve(pending_.size() * 3);
  for (std::size_t index = 0; index < pending_.size(); ++index) {
    output += "\xef\xbf\xbd";
  }
  pending_.clear();
  return output;
}

std::string NativeToolStreamGate::push(std::string_view utf8) {
  complete_text_.append(utf8);
  if (tool_marker_seen_) return {};
  const std::size_t marker = complete_text_.find(kToolCallStart, emitted_bytes_);
  if (marker != std::string::npos) {
    tool_marker_seen_ = true;
    std::string output =
        complete_text_.substr(emitted_bytes_, marker - emitted_bytes_);
    emitted_bytes_ = marker;
    return output;
  }
  const std::string_view unissued(complete_text_.data() + emitted_bytes_,
                                  complete_text_.size() - emitted_bytes_);
  const std::size_t keep = longest_marker_prefix_suffix(unissued);
  const std::size_t send = unissued.size() - keep;
  std::string output(unissued.substr(0, send));
  emitted_bytes_ += send;
  return output;
}

std::string NativeToolStreamGate::finish(bool parsed_tool_calls) {
  if (parsed_tool_calls) return {};
  std::string output = complete_text_.substr(emitted_bytes_);
  emitted_bytes_ = complete_text_.size();
  return output;
}

}  // namespace aima
