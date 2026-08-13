// SPDX-License-Identifier: Apache-2.0

#include "aima/native_chat_protocol.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_chat_protocol_test: " << message << '\n';
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

}  // namespace

int main() {
  using aima::NativeOrderedJson;

  const NativeOrderedJson ordered = {
      {"type", "function"},
      {"function",
       {{"name", "weather"},
        {"description", "天气"},
        {"parameters",
         {{"type", "object"},
          {"properties",
           {{"city", {{"type", "string"}}},
            {"days", {{"type", "integer"}}}}}}}}}};
  require(
      aima::render_qwen_json(ordered) ==
          R"({"type": "function", "function": {"name": "weather", "description": "天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}})",
      "Qwen JSON serialization changed");

  const NativeOrderedJson request = {
      {"messages",
       NativeOrderedJson::array(
           {{{"role", "system"}, {"content", "Be concise."}},
            {{"role", "user"}, {"content", "Weather in Paris?"}}})},
      {"tools", NativeOrderedJson::array({ordered})},
      {"tool_choice", "auto"},
      {"parallel_tool_calls", false}};
  const aima::NativePreparedChat prepared =
      aima::prepare_native_chat(request);
  require(prepared.messages.size() == 2, "message preparation failed");
  require(prepared.messages[0].role == "system",
          "system message was not retained");
  require(prepared.messages[0].content.find("Call at most one function.") !=
              std::string::npos,
          "parallel-tool directive missing");
  require(prepared.prompt_tools.size() == 1,
          "tool was not admitted to the prompt");
  require(prepared.function_tools[0].name == "weather",
          "function name preparation failed");

  NativeOrderedJson none_request = request;
  none_request["tool_choice"] = "none";
  require(aima::prepare_native_chat(none_request).prompt_tools.empty(),
          "tool_choice none still exposed tools to the model");

  NativeOrderedJson forced_request = request;
  forced_request["tool_choice"] = {
      {"type", "function"}, {"function", {{"name", "weather"}}}};
  const auto forced = aima::prepare_native_chat(forced_request);
  require(forced.prompt_tools.size() == 1 &&
              forced.required_function_name == "weather",
          "specific tool_choice was not prepared");

  NativeOrderedJson invalid_request = request;
  invalid_request["tools"][0]["function"]["name"] = "bad>name";
  require_invalid(
      [&]() { (void)aima::prepare_native_chat(invalid_request); },
      "invalid function name was admitted");

  const std::string raw =
      "I will check.\n<tool_call>\n<function=weather>\n"
      "<parameter=city>\nParis\n</parameter>\n"
      "<parameter=days>\n3\n</parameter>\n"
      "</function>\n</tool_call>";
  const aima::NativeAssistantOutput parsed =
      aima::parse_qwen_tool_output(raw, prepared.function_tools, "call_test_");
  require(parsed.content == "I will check.\n",
          "assistant prefix content changed");
  require(parsed.tool_calls.size() == 1, "tool call was not parsed");
  require(parsed.tool_calls[0].id == "call_test_0",
          "tool call id is not deterministic");
  require(parsed.tool_calls[0].name == "weather",
          "tool call name changed");
  require(parsed.tool_calls[0].arguments["city"] == "Paris",
          "string argument conversion failed");
  require(parsed.tool_calls[0].arguments["days"] == 3,
          "integer argument conversion failed");
  require(parsed.tool_calls[0].serialized_arguments ==
              R"({"city": "Paris", "days": 3})",
          "serialized function arguments changed");

  const std::string two_calls =
      "<tool_call><function=weather><parameter=city>\nParis\n"
      "</parameter></function></tool_call>\n"
      "<tool_call><function=weather><parameter=city>\nTokyo\n"
      "</parameter></function></tool_call>";
  const auto multiple = aima::parse_qwen_tool_output(
      two_calls, prepared.function_tools, "call_multi_");
  require(multiple.tool_calls.size() == 2,
          "parallel tool calls were not parsed");
  require(multiple.tool_calls[1].id == "call_multi_1",
          "parallel tool-call indices changed");

  const std::string malformed =
      "plain <tool_call><function=weather><parameter=city>Paris";
  const auto fallback = aima::parse_qwen_tool_output(
      malformed, prepared.function_tools, "call_bad_");
  require(fallback.tool_calls.empty() && fallback.content == malformed,
          "malformed tool markup did not fall back to text");

  const std::string unknown =
      "<tool_call><function=not_available></function></tool_call>";
  const auto unknown_fallback = aima::parse_qwen_tool_output(
      unknown, prepared.function_tools, "call_unknown_");
  require(unknown_fallback.tool_calls.empty() &&
              unknown_fallback.content == unknown,
          "unknown function was exposed as a tool call");

  aima::NativeIncrementalUtf8Decoder utf8;
  require(utf8.push(std::string("\xe4\xb8", 2)).empty(),
          "incomplete UTF-8 was emitted");
  require(utf8.push(std::string("\xad", 1)) == "中",
          "split UTF-8 code point was not reconstructed");
  require(utf8.finish().empty(), "UTF-8 decoder retained complete bytes");

  aima::NativeIncrementalUtf8Decoder invalid_utf8;
  require(invalid_utf8.push(std::string("\xff", 1)) ==
              std::string("\xef\xbf\xbd", 3),
          "invalid UTF-8 did not become U+FFFD");

  aima::NativeToolStreamGate plain_gate;
  std::string plain_stream;
  for (const char ch : std::string("hello <toolbox>")) {
    plain_stream += plain_gate.push(std::string(1, ch));
  }
  plain_stream += plain_gate.finish(false);
  require(plain_stream == "hello <toolbox>",
          "plain streamed text was not lossless");

  aima::NativeToolStreamGate tool_gate;
  std::string streamed;
  for (const char ch : raw) {
    streamed += tool_gate.push(std::string(1, ch));
  }
  streamed += tool_gate.finish(true);
  require(streamed == "I will check.\n",
          "tool XML leaked into streamed content");
  require(tool_gate.complete_text() == raw,
          "stream gate did not preserve full model output");

  const NativeOrderedJson history_request = {
      {"messages",
       NativeOrderedJson::array(
           {{{"role", "user"}, {"content", "Use it"}},
            {{"role", "assistant"},
             {"content", nullptr},
             {"tool_calls",
              NativeOrderedJson::array(
                  {{{"id", "call_1"},
                    {"type", "function"},
                    {"function",
                     {{"name", "weather"},
                      {"arguments", R"({"city":"Paris","days":2})"}}}}})}},
            {{"role", "tool"},
             {"tool_call_id", "call_1"},
             {"content", R"({"temperature":20})"}},
            {{"role", "user"}, {"content", "Summarize"}}})},
      {"tools", NativeOrderedJson::array({ordered})}};
  const auto history = aima::prepare_native_chat(history_request);
  require(history.messages.size() == 4,
          "assistant/tool history preparation failed");
  require(history.messages[1].tool_calls[0].arguments[0].rendered_value ==
              "Paris",
          "history string argument must remain unquoted");
  require(history.messages[1].tool_calls[0].arguments[1].rendered_value ==
              "2",
          "history numeric argument rendering failed");

  const NativeOrderedJson media_request = {
      {"messages",
       NativeOrderedJson::array(
           {{{"role", "user"},
             {"content",
              NativeOrderedJson::array(
                  {{{"type", "text"}, {"text", "First:"}},
                   {{"type", "image_url"},
                    {"image_url", {{"url", "file:///media/a.png"}}}},
                   {{"type", "text"}, {"text", " then "}},
                   {{"type", "video_url"},
                    {"video_url", "data:video/mp4;base64,AAAA"}},
                   {{"type", "text"}, {"text", " done"}}})}}})}};
  const auto media = aima::prepare_native_chat(media_request);
  require(
      media.messages[0].content ==
          "First:<|vision_start|><|image_pad|><|vision_end|> then "
          "<|vision_start|><|video_pad|><|vision_end|> done",
      "ordered media placeholders changed");
  require(media.media.size() == 2,
          "image/video parts were not retained");
  require(media.media[0].kind == aima::NativeMediaKind::kImage &&
              media.media[0].message_index == 0 &&
              media.media[0].content_part_index == 1 &&
              media.media[0].media_index == 0 &&
              media.media[0].source == "file:///media/a.png",
          "image part metadata changed");
  require(media.media[1].kind == aima::NativeMediaKind::kVideo &&
              media.media[1].content_part_index == 3 &&
              media.media[1].media_index == 1,
          "video part metadata changed");

  NativeOrderedJson assistant_media = media_request;
  assistant_media["messages"][0]["role"] = "assistant";
  require_invalid(
      [&]() { (void)aima::prepare_native_chat(assistant_media); },
      "assistant media part was admitted");

  NativeOrderedJson malformed_media = media_request;
  malformed_media["messages"][0]["content"][1]["image_url"] =
      NativeOrderedJson::object();
  require_invalid(
      [&]() { (void)aima::prepare_native_chat(malformed_media); },
      "media part without a URL was admitted");

  NativeOrderedJson too_many_images = {
      {"messages", NativeOrderedJson::array(
                       {{{"role", "user"},
                         {"content", NativeOrderedJson::array()}}})}};
  for (std::size_t index = 0; index < 17; ++index) {
    too_many_images["messages"][0]["content"].push_back(
        {{"type", "image_url"},
         {"image_url", {{"url", "file:///media/a.png"}}}});
  }
  require_invalid(
      [&]() { (void)aima::prepare_native_chat(too_many_images); },
      "image count above the frozen limit was admitted");

  std::cout << "native_chat_protocol_test: PASS\n";
  return 0;
}
