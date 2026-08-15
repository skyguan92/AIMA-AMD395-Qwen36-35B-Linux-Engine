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

std::int64_t media_io_integer(const NativeOrderedJson& value,
                              std::string_view field) {
  if (value.is_number_unsigned()) {
    const std::uint64_t parsed = value.get<std::uint64_t>();
    if (parsed <=
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return static_cast<std::int64_t>(parsed);
    }
  } else if (value.is_number_integer()) {
    return value.get<std::int64_t>();
  }
  throw std::invalid_argument(std::string(field) + " must be an integer");
}

void parse_media_io_kwargs(const NativeOrderedJson& request,
                           NativePreparedChat* prepared) {
  if (prepared == nullptr) {
    throw std::invalid_argument("media_io_kwargs output is null");
  }
  if (!request.contains("media_io_kwargs")) return;
  const NativeOrderedJson& mapping = request["media_io_kwargs"];
  if (!mapping.is_object()) {
    throw std::invalid_argument("media_io_kwargs must be an object");
  }
  // vLLM falls back to the engine-level mapping for an empty runtime mapping.
  if (mapping.empty()) return;

  for (auto item = mapping.begin(); item != mapping.end(); ++item) {
    if (item.key() != "image" && item.key() != "video") {
      throw std::invalid_argument(
          "media_io_kwargs supports only image and video mappings");
    }
    if (!item.value().is_object()) {
      throw std::invalid_argument(
          "media_io_kwargs modality values must be objects");
    }
  }

  const auto image = mapping.find("image");
  if (image != mapping.end() && !image->empty()) {
    NativeImageIoPolicy effective;
    for (auto item = image->begin(); item != image->end(); ++item) {
      if (item.key() != "rgba_background_color") {
        throw std::invalid_argument(
            "unsupported media_io_kwargs.image field: " + item.key());
      }
      if (!item.value().is_array() || item.value().size() != 3) {
        throw std::invalid_argument(
            "media_io_kwargs.image.rgba_background_color must contain three integers");
      }
      for (std::size_t channel = 0; channel < 3; ++channel) {
        const NativeOrderedJson& value = item.value()[channel];
        const std::int64_t parsed = media_io_integer(
            value, "media_io_kwargs.image.rgba_background_color");
        if (parsed < 0 || parsed > 255) {
          throw std::invalid_argument(
              "media_io_kwargs.image.rgba_background_color channels must be in [0,255]");
        }
        effective.rgba_background_color[channel] =
            static_cast<std::uint8_t>(parsed);
      }
    }
    prepared->image_io_override = effective;
  }

  const auto video = mapping.find("video");
  if (video == mapping.end() || video->empty()) return;
  NativeVideoIoPolicy effective;
  bool has_num_frames = false;
  bool has_fps = false;
  for (auto item = video->begin(); item != video->end(); ++item) {
    if (item.key() == "num_frames") {
      effective.num_frames =
          media_io_integer(item.value(), "media_io_kwargs.video.num_frames");
      has_num_frames = true;
    } else if (item.key() == "fps") {
      if (!item.value().is_number()) {
        throw std::invalid_argument(
            "media_io_kwargs.video.fps must be a number");
      }
      effective.fps = item.value().get<double>();
      if (!std::isfinite(effective.fps)) {
        throw std::invalid_argument(
            "media_io_kwargs.video.fps must be finite");
      }
      has_fps = true;
    } else if (item.key() == "video_backend") {
      if (!item.value().is_string() ||
          item.value().get<std::string>() != "opencv") {
        throw std::invalid_argument(
            "media_io_kwargs.video.video_backend must be opencv");
      }
    } else {
      throw std::invalid_argument(
          "unsupported media_io_kwargs.video field: " + item.key());
    }
  }
  // VideoMediaIO shallow-merges request fields with the launch mapping, then
  // clears the other sampling field when exactly one is supplied. num_frames
  // is a constructor default rather than a launch-mapping field, so an fps
  // override still retains the 32-frame cap.
  if (has_num_frames && !has_fps) effective.fps = -1.0;
  prepared->video_io_override = effective;
}

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

struct ParsedMessageContent {
  std::string baseline;
  std::string vl;
};

std::size_t count_nonoverlapping(std::string_view value,
                                 std::string_view needle) {
  std::size_t count = 0;
  std::size_t cursor = 0;
  while (cursor <= value.size()) {
    const std::size_t found = value.find(needle, cursor);
    if (found == std::string_view::npos) break;
    ++count;
    cursor = found + needle.size();
  }
  return count;
}

std::string join_with_newlines(const std::vector<std::string>& parts) {
  std::string result;
  for (std::size_t index = 0; index < parts.size(); ++index) {
    if (index != 0) result += '\n';
    result += parts[index];
  }
  return result;
}

std::string_view media_placeholder(NativeMediaKind kind) {
  return kind == NativeMediaKind::kImage ? kImagePlaceholder
                                         : kVideoPlaceholder;
}

ParsedMessageContent parse_message_content(
    const NativeOrderedJson& message, std::string_view role,
    std::size_t message_index, NativePreparedChat* prepared) {
  if (!message.contains("content") || message["content"].is_null()) return {};
  const NativeOrderedJson& content = message["content"];
  if (content.is_string()) {
    const std::string value = content.get<std::string>();
    return {value, value};
  }
  if (!content.is_array()) {
    throw std::invalid_argument(
        "message content must be a string, null, or a text-part array");
  }

  ParsedMessageContent result;
  std::vector<std::string> text_parts;
  std::vector<NativeMediaPart> message_media;
  std::vector<NativeMediaKind> modality_order;
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
      const std::string text = part["text"].get<std::string>();
      result.baseline += text;
      text_parts.push_back(text);
      continue;
    }
    NativeMediaKind kind;
    std::string_view field;
    if (type == "image_url") {
      kind = NativeMediaKind::kImage;
      field = "image_url";
    } else if (type == "video_url") {
      kind = NativeMediaKind::kVideo;
      field = "video_url";
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
    message_media.push_back(std::move(media));
    if (std::find(modality_order.begin(), modality_order.end(), kind) ==
        modality_order.end()) {
      modality_order.push_back(kind);
    }
    result.baseline += media_placeholder(kind);
  }

  const std::string text_prompt = join_with_newlines(text_parts);
  std::vector<std::string> vl_parts;
  for (NativeMediaKind kind : modality_order) {
    const std::string_view placeholder = media_placeholder(kind);
    const std::size_t media_count = static_cast<std::size_t>(std::count_if(
        message_media.begin(), message_media.end(),
        [kind](const NativeMediaPart& media) { return media.kind == kind; }));
    const std::size_t existing = count_nonoverlapping(text_prompt, placeholder);
    if (existing > media_count) {
      throw std::invalid_argument(
          "message text contains more media placeholders than media items");
    }
    for (std::size_t index = existing; index < media_count; ++index) {
      vl_parts.emplace_back(placeholder);
    }
  }
  if (!text_prompt.empty()) vl_parts.push_back(text_prompt);
  result.vl = join_with_newlines(vl_parts);

  // vLLM stores media per modality, then binds each modality's items to that
  // modality's placeholder occurrences. Flatten into prompt occurrence order
  // so the native sequential expander preserves the same association even
  // when a caller supplied a valid placeholder explicitly in a text part.
  std::vector<std::size_t> image_indices;
  std::vector<std::size_t> video_indices;
  for (std::size_t index = 0; index < message_media.size(); ++index) {
    (message_media[index].kind == NativeMediaKind::kImage ? image_indices
                                                          : video_indices)
        .push_back(index);
  }
  std::size_t next_image = 0;
  std::size_t next_video = 0;
  std::size_t cursor = 0;
  while (next_image + next_video < message_media.size()) {
    const std::size_t image_position =
        next_image < image_indices.size()
            ? result.vl.find(kImagePlaceholder, cursor)
            : std::string::npos;
    const std::size_t video_position =
        next_video < video_indices.size()
            ? result.vl.find(kVideoPlaceholder, cursor)
            : std::string::npos;
    const bool use_image = image_position < video_position;
    const std::size_t position = use_image ? image_position : video_position;
    if (position == std::string::npos) {
      throw std::logic_error(
          "VL media placeholder association could not be constructed");
    }
    const std::size_t media_index =
        use_image ? image_indices[next_image++] : video_indices[next_video++];
    NativeMediaPart& media = message_media[media_index];
    media.media_index = prepared->media.size();
    prepared->media.push_back(std::move(media));
    cursor = position + media_placeholder(
                            use_image ? NativeMediaKind::kImage
                                      : NativeMediaKind::kVideo)
                            .size();
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

bool json_whitespace(unsigned char value) {
  return value == ' ' || value == '\t' || value == '\n' || value == '\r';
}

bool hex_digit(unsigned char value) {
  return (value >= '0' && value <= '9') ||
         (value >= 'a' && value <= 'f') ||
         (value >= 'A' && value <= 'F');
}

std::string single_required_string_property(
    const NativeFunctionTool& tool) {
  const NativeOrderedJson& schema = tool.parameters;
  if (!schema.is_object() || schema.size() != 4 ||
      schema.value("type", "") != "object" ||
      !schema.contains("properties") ||
      !schema["properties"].is_object() ||
      schema["properties"].size() != 1 ||
      !schema.contains("required") || !schema["required"].is_array() ||
      schema["required"].size() != 1 ||
      !schema["required"][0].is_string() ||
      !schema.contains("additionalProperties") ||
      !schema["additionalProperties"].is_boolean() ||
      schema["additionalProperties"].get<bool>()) {
    throw std::invalid_argument(
        "named VL tool JSON Schema must be a closed object with one "
        "required string property");
  }
  const std::string property = schema["required"][0].get<std::string>();
  const auto found = schema["properties"].find(property);
  if (property.empty() || found == schema["properties"].end() ||
      !found->is_object() || found->size() != 1 ||
      found->value("type", "") != "string") {
    throw std::invalid_argument(
        "named VL tool JSON Schema property must be exactly type string");
  }
  return property;
}

class SingleStringJsonPrefixState {
 public:
  explicit SingleStringJsonPrefixState(std::string rendered_property)
      : rendered_property_(std::move(rendered_property)) {}

  bool feed(std::string_view bytes) {
    for (const unsigned char value : bytes) {
      if (!feed_byte(value)) return false;
    }
    return true;
  }

  bool viable() const { return phase_ != Phase::kInvalid; }
  bool complete() const { return phase_ == Phase::kComplete; }

 private:
  enum class Phase {
    kLeading,
    kProperty,
    kAfterProperty,
    kAfterColon,
    kString,
    kAfterString,
    kComplete,
    kInvalid,
  };

  bool reject() {
    phase_ = Phase::kInvalid;
    return false;
  }

  bool feed_string_byte(unsigned char value) {
    if (unicode_digits_ != 0) {
      if (!hex_digit(value)) return reject();
      --unicode_digits_;
      return true;
    }
    if (escaped_) {
      escaped_ = false;
      if (value == 'u') {
        unicode_digits_ = 4;
        return true;
      }
      return value == '"' || value == '\\' || value == '/' ||
             value == 'b' || value == 'f' || value == 'n' ||
             value == 'r' || value == 't' || reject();
    }
    if (utf8_remaining_ != 0) {
      if (value < utf8_minimum_ || value > utf8_maximum_) return reject();
      --utf8_remaining_;
      utf8_minimum_ = 0x80;
      utf8_maximum_ = 0xbf;
      return true;
    }
    if (value < 0x20) return reject();
    if (value == '"') {
      phase_ = Phase::kAfterString;
      return true;
    }
    if (value == '\\') {
      escaped_ = true;
      return true;
    }
    if (value < 0x80) return true;
    if (value >= 0xc2 && value <= 0xdf) {
      utf8_remaining_ = 1;
    } else if (value == 0xe0) {
      utf8_remaining_ = 2;
      utf8_minimum_ = 0xa0;
    } else if (value >= 0xe1 && value <= 0xec) {
      utf8_remaining_ = 2;
    } else if (value == 0xed) {
      utf8_remaining_ = 2;
      utf8_maximum_ = 0x9f;
    } else if (value >= 0xee && value <= 0xef) {
      utf8_remaining_ = 2;
    } else if (value == 0xf0) {
      utf8_remaining_ = 3;
      utf8_minimum_ = 0x90;
    } else if (value >= 0xf1 && value <= 0xf3) {
      utf8_remaining_ = 3;
    } else if (value == 0xf4) {
      utf8_remaining_ = 3;
      utf8_maximum_ = 0x8f;
    } else {
      return reject();
    }
    return true;
  }

  bool feed_byte(unsigned char value) {
    if (phase_ == Phase::kInvalid) return false;
    if (phase_ == Phase::kLeading) {
      if (json_whitespace(value)) return true;
      if (value != '{') return reject();
      phase_ = Phase::kProperty;
      return true;
    }
    if (phase_ == Phase::kProperty) {
      if (property_offset_ == 0 && json_whitespace(value)) return true;
      if (property_offset_ >= rendered_property_.size() ||
          value != static_cast<unsigned char>(
                       rendered_property_[property_offset_])) {
        return reject();
      }
      ++property_offset_;
      if (property_offset_ == rendered_property_.size()) {
        phase_ = Phase::kAfterProperty;
      }
      return true;
    }
    if (phase_ == Phase::kAfterProperty) {
      if (json_whitespace(value)) return true;
      if (value != ':') return reject();
      phase_ = Phase::kAfterColon;
      return true;
    }
    if (phase_ == Phase::kAfterColon) {
      if (json_whitespace(value)) return true;
      if (value != '"') return reject();
      phase_ = Phase::kString;
      return true;
    }
    if (phase_ == Phase::kString) return feed_string_byte(value);
    if (phase_ == Phase::kAfterString) {
      if (json_whitespace(value)) return true;
      if (value != '}') return reject();
      phase_ = Phase::kComplete;
      return true;
    }
    if (phase_ == Phase::kComplete) {
      return json_whitespace(value) || reject();
    }
    return reject();
  }

  Phase phase_ = Phase::kLeading;
  std::string rendered_property_;
  std::size_t property_offset_ = 0;
  bool escaped_ = false;
  unsigned unicode_digits_ = 0;
  unsigned utf8_remaining_ = 0;
  unsigned char utf8_minimum_ = 0x80;
  unsigned char utf8_maximum_ = 0xbf;
};

NativeOrderedJson normalize_vllm_prompt_tool(
    const NativeFunctionTool& tool) {
  // vLLM validates tools through its Pydantic request model before applying
  // the Qwen chat template.  model_dump() fixes these model-field positions
  // while retaining the insertion order inside the caller's JSON Schema.
  // Reproduce that boundary for VL prompts so canonical wire JSON and direct
  // Python clients render identically to the frozen reference server.
  const NativeOrderedJson& source = tool.definition["function"];
  NativeOrderedJson function = NativeOrderedJson::object();
  function["name"] = tool.name;
  function["description"] =
      source.contains("description") ? source["description"]
                                     : NativeOrderedJson(nullptr);
  function["parameters"] =
      source.contains("parameters") ? source["parameters"]
                                    : NativeOrderedJson(nullptr);

  NativeOrderedJson normalized = NativeOrderedJson::object();
  normalized["type"] = "function";
  normalized["function"] = std::move(function);
  return normalized;
}

}  // namespace

bool native_single_string_json_prefix_viable(
    std::string_view property_name, std::string_view prefix,
    bool* complete) {
  if (property_name.empty() || complete == nullptr) {
    throw std::invalid_argument("JSON prefix oracle input is invalid");
  }
  const std::string rendered_property =
      NativeOrderedJson(std::string(property_name)).dump();
  SingleStringJsonPrefixState state(rendered_property);
  const bool viable = state.feed(prefix) && state.viable();
  *complete = viable && state.complete();
  return viable;
}

NativeAssistantOutput parse_native_named_tool_json_output(
    std::string_view model_output, const NativeFunctionTool& tool,
    std::string_view call_id_prefix) {
  const std::string property = single_required_string_property(tool);
  NativeOrderedJson arguments;
  try {
    arguments = NativeOrderedJson::parse(model_output);
  } catch (const NativeOrderedJson::exception&) {
    throw std::runtime_error(
        "named VL tool constrained output is not valid JSON");
  }
  if (!arguments.is_object() || arguments.size() != 1 ||
      !arguments.contains(property) || !arguments[property].is_string()) {
    throw std::runtime_error(
        "named VL tool constrained output violated its JSON Schema");
  }
  NativeParsedToolCall call;
  call.id = std::string(call_id_prefix) + "0";
  call.name = tool.name;
  call.arguments = std::move(arguments);
  call.serialized_arguments = std::string(model_output);
  return {std::string(), {std::move(call)}};
}

NativeNamedToolJsonConstraint::NativeNamedToolJsonConstraint(
    const NativeTokenizer& tokenizer, const NativeFunctionTool& tool)
    : tokenizer_(&tokenizer),
      function_name_(tool.name),
      property_name_(single_required_string_property(tool)),
      rendered_property_(NativeOrderedJson(property_name_).dump()) {
  if (tokenizer.pad_token_id() != 248044 ||
      tokenizer.eos_token_id() >= kNativeModelVocabularySize) {
    throw std::runtime_error(
        "named VL tool tokenizer boundary is not qualified");
  }
  token_bytes_.reserve(tokenizer.pad_token_id());
  for (std::uint32_t token_id = 0;
       token_id < tokenizer.pad_token_id(); ++token_id) {
    token_bytes_.push_back(tokenizer.decode_token_bytes(token_id));
  }
}

void NativeNamedToolJsonConstraint::allowed_token_mask(
    const std::vector<std::uint32_t>& generated_token_ids,
    std::vector<std::uint8_t>* mask) const {
  if (tokenizer_ == nullptr || mask == nullptr) {
    throw std::invalid_argument("named VL tool token mask is invalid");
  }
  SingleStringJsonPrefixState prefix(rendered_property_);
  for (const std::uint32_t token_id : generated_token_ids) {
    if (token_id >= token_bytes_.size() ||
        !prefix.feed(token_bytes_[token_id])) {
      throw std::runtime_error(
          "named VL tool generated prefix escaped its JSON grammar");
    }
  }
  mask->assign(kNativeModelVocabularySize, 0);
  std::size_t admitted = 0;
  for (std::size_t token_id = 0; token_id < token_bytes_.size(); ++token_id) {
    SingleStringJsonPrefixState candidate = prefix;
    if (!token_bytes_[token_id].empty() &&
        candidate.feed(token_bytes_[token_id]) && candidate.viable()) {
      (*mask)[token_id] = 1;
      ++admitted;
    }
  }
  if (prefix.complete()) {
    (*mask)[tokenizer_->eos_token_id()] = 1;
    ++admitted;
  }
  if (admitted == 0) {
    throw std::runtime_error(
        "named VL tool JSON grammar admitted no next token");
  }
}

NativeAssistantOutput NativeNamedToolJsonConstraint::parse_output(
    std::string_view model_output,
    std::string_view call_id_prefix) const {
  NativeFunctionTool tool;
  tool.name = function_name_;
  tool.parameters = {
      {"type", "object"},
      {"properties", {{property_name_, {{"type", "string"}}}}},
      {"required", NativeOrderedJson::array({property_name_})},
      {"additionalProperties", false},
  };
  return parse_native_named_tool_json_output(model_output, tool,
                                              call_id_prefix);
}

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
  parse_media_io_kwargs(request, &prepared);
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
  std::string leading_system_vl;
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
      const ParsedMessageContent content = parse_message_content(
          source, role, message_index, &prepared);
      if (!leading_system.empty() && !content.baseline.empty()) {
        leading_system += '\n';
      }
      if (!leading_system_vl.empty() && !content.vl.empty()) {
        leading_system_vl += '\n';
      }
      leading_system += content.baseline;
      leading_system_vl += content.vl;
      continue;
    }
    left_leading_system = true;
    if (prepared.messages.empty() && !leading_system.empty()) {
      prepared.messages.push_back(
          {"system", leading_system, false, std::string(), {}});
      prepared.vl_prompt_messages.push_back(
          {"system", leading_system_vl, false, std::string(), {}});
    }

    NativeChatMessage message;
    message.role = role;
    const ParsedMessageContent content = parse_message_content(
        source, role, message_index, &prepared);
    message.content = content.baseline;
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
    NativeChatMessage vl_message = message;
    vl_message.content = content.vl;
    prepared.messages.push_back(std::move(message));
    prepared.vl_prompt_messages.push_back(std::move(vl_message));
  }
  if (prepared.messages.empty() && !leading_system.empty()) {
    prepared.messages.push_back(
        {"system", leading_system, false, std::string(), {}});
    prepared.vl_prompt_messages.push_back(
        {"system", leading_system_vl, false, std::string(), {}});
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

  prepared.vl_prompt_tools.reserve(prepared.function_tools.size());
  for (const NativeFunctionTool& tool : prepared.function_tools) {
    prepared.vl_prompt_tools.push_back(
        {render_qwen_json(normalize_vllm_prompt_tool(tool))});
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
