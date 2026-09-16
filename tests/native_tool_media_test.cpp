// SPDX-License-Identifier: Apache-2.0
#include "aima/native_chat_protocol.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
using Json = aima::NativeOrderedJson;

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_tool_media_test: " << message << '\n';
    std::exit(1);
  }
}

Json image(const char* url = "file:///media/image.png") {
  return {{"type", "image_url"}, {"image_url", {{"url", url}}}};
}

Json video() {
  return {{"type", "video_url"},
          {"video_url", {{"url", "file:///media/clip.mp4"}}}};
}

Json text(const std::string& value) {
  return {{"type", "text"}, {"text", value}};
}

Json request(const Json& content) {
  return {{"messages", Json::array({
      {{"role", "user"}, {"content", "Describe the tool result."}},
      {{"role", "assistant"}, {"content", nullptr},
       {"tool_calls", Json::array({
           {{"id", "call_image"}, {"type", "function"},
            {"function", {{"name", "capture"}, {"arguments", "{}"}}}}})}},
      {{"role", "tool"}, {"tool_call_id", "call_image"},
       {"content", content}}})}};
}

void rejected(const Json& input, const std::string& expected) {
  try {
    (void)aima::prepare_native_chat(input);
  } catch (const std::invalid_argument& error) {
    require(std::string(error.what()).find(expected) != std::string::npos,
            "request failed for an unexpected reason");
    return;
  }
  require(false, "invalid tool media history was admitted");
}

void run() {
  const std::string im = "<|vision_start|><|image_pad|><|vision_end|>";
  const std::string vi = "<|vision_start|><|video_pad|><|vision_end|>";
  const Json single = request(Json::array({image()}));
  const auto prepared = aima::prepare_native_chat(single);
  require(prepared.media.size() == 1 &&
              prepared.media[0].message_index == 2 &&
              prepared.media[0].content_part_index == 0 &&
              prepared.media[0].kind == aima::NativeMediaKind::kImage &&
              prepared.vl_prompt_messages[2].role == "tool" &&
              prepared.vl_prompt_messages[2].content == im,
          "tool image lost its message, role or placeholder association");
  require(prepared.historical_tool_calls[0].result_count == 1 &&
              prepared.historical_tool_calls[0].no_progress_result_count == 0,
          "a media-only tool result must count as payload");

  const Json mixed = request(Json::array({
      text("first"), image(), video(), image("file:///media/second.png"),
      text("last")}));
  const auto media = aima::prepare_native_chat(mixed);
  require(media.media.size() == 3 &&
              media.media[0].content_part_index == 1 &&
              media.media[1].content_part_index == 3 &&
              media.media[2].content_part_index == 2 &&
              media.media[1].source == "file:///media/second.png" &&
              media.media[2].kind == aima::NativeMediaKind::kVideo &&
              media.vl_prompt_messages[2].content ==
                  im + "\n" + im + "\n" + vi + "\nfirst\nlast",
          "tool media must retain the existing modality/placeholder ordering");

  Json parallel = mixed;
  Json other_call = parallel["messages"][1]["tool_calls"][0];
  other_call["id"] = "call_second";
  parallel["messages"][1]["tool_calls"].push_back(other_call);
  parallel["messages"].push_back({{"role", "tool"},
      {"tool_call_id", "call_second"},
      {"content", Json::array({image("file:///media/third.png")})}});
  parallel["messages"][0]["content"] = Json::array({text("Compare"), image()});
  const auto multi = aima::prepare_native_chat(parallel);
  require(multi.media.size() == 5 && multi.media[0].message_index == 0 &&
              multi.media[1].message_index == 2 &&
              multi.media[4].message_index == 3 &&
              multi.media[4].media_index == 4 &&
              multi.media[4].source == "file:///media/third.png",
          "media from different users/tools became associated with another result");

  Json manual = request(Json::array({text(im + "explicit"), image()}));
  require(aima::prepare_native_chat(manual).vl_prompt_messages[2].content ==
              im + "explicit", "an explicit tool media marker was duplicated");
  manual["messages"][2]["content"][0] = text(im + im);
  rejected(manual, "more media placeholders");

  Json invalid = single;
  invalid["messages"][2].erase("tool_call_id");
  rejected(invalid, "require a string tool_call_id");
  invalid = single;
  invalid["messages"][2]["tool_call_id"] = "unknown";
  rejected(invalid, "no preceding assistant tool call");
  invalid = single;
  invalid["messages"].push_back(invalid["messages"][2]);
  rejected(invalid, "must not repeat a tool_call_id");
  invalid = single;
  invalid["messages"].erase(0);
  rejected(invalid, "at least one user message");
  invalid = single;
  invalid["messages"][2]["content"][0]["image_url"]["url"] = "";
  rejected(invalid, "non-empty URL");
  for (const char* role : {"system", "developer", "assistant"}) {
    invalid = single;
    invalid["messages"][0] = {{"role", role},
                             {"content", Json::array({image()})}};
    rejected(invalid, "image and video content parts are supported");
  }
  for (bool images : {true, false}) {
    Json many = single;
    const std::size_t limit = images ? 16 : 21;
    many["messages"][2]["content"] = Json::array();
    for (std::size_t index = 0; index < limit; ++index) {
      many["messages"][2]["content"].push_back(images ? image() : video());
    }
    require(aima::prepare_native_chat(many).media.size() == limit,
            "tool media at the aggregate limit was rejected");
    many["messages"][0]["content"] = Json::array({images ? image() : video()});
    rejected(many, "count exceeds the fixed limit");
  }

  // Neither an inserted marker nor its position may conceal a real failure.
  for (const std::string failure : {
           "Error: capture failed", "Exit code: 1", "failed",
           "Final output: Error: capture failed",
           "{\"error\":\"capture failed\"}", "{\"success\":false}",
           "{\"status\":\"failed\"}", "{\"returncode\":2}",
           "{\"stdout\":\"\",\"result\":{\"error\":\"failed\"}}"}) {
    require(aima::native_tool_result_is_no_progress(failure),
            "text-only explicit failure classification changed");
    for (bool image_first : {true, false}) {
      const Json parts = image_first ? Json::array({image(), text(failure)})
                                    : Json::array({text(failure), image()});
      Json failed = request(parts);
      Json call = failed["messages"][1];
      call["tool_calls"][0]["id"] = "call_retry";
      failed["messages"].push_back(call);
      failed["messages"].push_back({{"role", "tool"},
          {"tool_call_id", "call_retry"}, {"content", parts}});
      const auto history = aima::prepare_native_chat(failed);
      require(history.historical_tool_calls[0].no_progress_result_count == 2 &&
                  history.historical_tool_calls[0].no_progress_streak == 2,
              "media concealed an explicit tool failure or reopened its retry window");
    }
  }
  for (const std::string empty : {"", "no output", "Exit code: 0",
                                  "{\"stdout\":\"\",\"exit_code\":0}"}) {
    require(aima::native_tool_result_is_no_progress(empty),
            "text-only empty output must remain no-progress");
    const auto with_media = aima::prepare_native_chat(
        request(Json::array({text(empty), image()})));
    require(with_media.historical_tool_calls[0].no_progress_result_count == 0,
            "valid media was discarded because its accompanying text is empty");
  }
  const auto split_failure = aima::prepare_native_chat(request(
      Json::array({text("Error:"), image(), text(" capture failed")})));
  require(split_failure.historical_tool_calls[0].no_progress_result_count == 1,
          "splitting text around media concealed an explicit failure");

  Json recovery = request("Error: capture failed");
  // Text and VL have different default thinking modes. This fixture tests
  // tool admission with an answer-only generated call in both cases.
  recovery["thinking"] = {{"type", "disabled"}};
  recovery["tools"] = Json::array({{{"type", "function"}, {"function", {
      {"name", "capture"}, {"parameters", {{"type", "object"},
      {"properties", Json::object()}}}}}}});
  const auto append_result = [](Json& input, const char* id, const Json& value) {
    Json call = input["messages"][1];
    call["tool_calls"][0]["id"] = id;
    input["messages"].push_back(call);
    input["messages"].push_back({{"role", "tool"}, {"tool_call_id", id},
                                {"content", value}});
  };
  append_result(recovery, "retry", "Error: capture failed");
  const std::string generated =
      "<tool_call><function=capture></function></tool_call>";
  const auto blocked = aima::parse_native_assistant_output(
      generated, aima::prepare_native_chat(recovery), "blocked_");
  require(blocked.tool_calls.empty() && blocked.tool_progress.no_progress,
          "repeated failures must suppress the generated call");
  append_result(recovery, "recovered", Json::array({image()}));
  const auto reopened = aima::parse_native_assistant_output(
      generated, aima::prepare_native_chat(recovery), "recovered_");
  require(reopened.tool_calls.size() == 1 && !reopened.tool_progress.no_progress &&
              reopened.tool_progress.history_no_progress_results == 2 &&
              reopened.tool_progress.history_no_progress_streak == 0,
          "a subsequent media result must reopen the retry window");

  Json late = request(Json::array({image()}));
  const Json late_media = late["messages"][2];
  late["messages"].erase(2);
  append_result(late, "new_failure_1", "Error: capture failed");
  append_result(late, "new_failure_2", "Error: capture failed");
  late["messages"].push_back(late_media);
  require(aima::prepare_native_chat(late).historical_tool_calls[0].no_progress_streak == 2,
          "a late media result must not erase failures from newer issuing turns");
}
}  // namespace

int main() {
  try {
    run();
  } catch (const std::exception& error) {
    std::cerr << "native_tool_media_test: " << error.what() << '\n';
    return 1;
  }
  std::cout << "native_tool_media_test: PASS\n";
}
