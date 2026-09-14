// SPDX-License-Identifier: Apache-2.0
// Regression coverage for agent repair/verify loops (September 2026 feedback).
#include "aima/native_chat_protocol.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
using Json = aima::NativeOrderedJson;

void require(bool value, const char* message) {
  if (!value) {
    std::cerr << "native_tool_recovery_test: " << message << '\n';
    std::exit(1);
  }
}

Json request() {
  Json tools = Json::array();
  for (const char* name : {"exec", "edit_file", "read_file"}) {
    tools.push_back({{"type", "function"}, {"function", {
        {"name", name}, {"parameters", {{"type", "object"},
        {"properties", {{"value", {{"type", "string"}}}}}}}}}});
  }
  return {{"tools", tools}, {"messages", Json::array({
      {{"role", "user"}, {"content", "Repair the input and verify the result."}}})}};
}

void append(Json& input, const std::string& name, const std::string& value,
            const std::string& result) {
  const std::string id = "call_" + std::to_string(input["messages"].size());
  input["messages"].push_back({{"role", "assistant"}, {"content", nullptr},
      {"tool_calls", Json::array({{{"id", id}, {"type", "function"},
        {"function", {{"name", name},
          {"arguments", Json({{"value", value}}).dump()}}}}})}});
  input["messages"].push_back({{"role", "tool"},
      {"tool_call_id", id}, {"content", result}});
}

const std::string kRun =
    "I will verify the repair.\n<tool_call><function=exec>"
    "<parameter=value>\npython3 generate_report.py\n</parameter>"
    "</function></tool_call>";

aima::NativeAssistantOutput parse(const Json& input) {
  return aima::parse_native_assistant_output(
      kRun, aima::prepare_native_chat(input), "candidate_");
}

Json twice_failed() {
  Json input = request();
  append(input, "exec", "python3 generate_report.py", "SyntaxError\nExit code: 1");
  append(input, "exec", "python3 generate_report.py", "SyntaxError\nExit code: 1");
  return input;
}
}  // namespace

int main() {
  const Json exhausted = twice_failed();
  require(parse(exhausted).tool_calls.empty() && parse(exhausted).tool_progress.no_progress,
          "unchanged repeated failures must remain bounded");
  const auto mixed = aima::parse_native_assistant_output(
      kRun + "<tool_call><function=exec><parameter=value>"
      "inspect a different input</parameter></function></tool_call>",
      aima::prepare_native_chat(exhausted), "mixed_");
  require(mixed.tool_calls.size() == 1 && mixed.tool_progress.no_progress &&
              mixed.tool_calls[0].arguments["value"] == "inspect a different input",
          "one exhausted call must not discard a different admitted action");

  Json repaired = exhausted;
  append(repaired, "edit_file", "correct the script", "Successfully edited the script");
  const auto recovered = parse(repaired);
  require(recovered.tool_calls.size() == 1 && !recovered.tool_progress.no_progress &&
              recovered.tool_progress.history_no_progress_results == 2 &&
              recovered.tool_progress.history_no_progress_streak == 0,
          "a completed repair must permit verification without losing lifetime metrics");

  Json repeated_repairs = request();
  for (int attempt = 0; attempt != 3; ++attempt) {
    append(repeated_repairs, "exec", "python3 generate_report.py", "IndexError\nExit code: 1");
    append(repeated_repairs, "edit_file", "repair " + std::to_string(attempt),
           "Successfully edited the script");
    require(parse(repeated_repairs).tool_calls.size() == 1,
            "independent repairs must not share a lifetime retry budget");
  }
  append(repeated_repairs, "exec", "python3 generate_report.py", "TypeError\nExit code: 1");
  require(parse(repeated_repairs).tool_calls.size() == 1,
          "the first failure after progress must allow one retry");
  append(repeated_repairs, "exec", "python3 generate_report.py", "TypeError\nExit code: 1");
  require(parse(repeated_repairs).tool_calls.empty(),
          "a new unchanged failure streak must become bounded again");

  Json silent_repair = exhausted;
  append(silent_repair, "exec", "repair the input silently", "\nExit code: 0");
  require(parse(silent_repair).tool_calls.size() == 1,
          "a different successful command may change state without stdout");
  Json json_repair = exhausted;
  append(json_repair, "exec", "repair silently using a JSON wrapper",
         "{\"stdout\":\"\",\"stderr\":\"\",\"exit_code\":0}");
  require(parse(json_repair).tool_calls.size() == 1,
          "explicit JSON exit success must permit verification of a different action");
  Json empty_repeats = request();
  append(empty_repeats, "exec", "python3 generate_report.py", "Exit code: 0");
  append(empty_repeats, "exec", "python3 generate_report.py", "Exit code: 0");
  require(parse(empty_repeats).tool_calls.empty(),
          "identical output-free calls must retain their own no-progress bound");

  for (const std::string result : {"", "Error: edit failed", "Exit code: 1",
                                    "{\"stdout\":\"\",\"exit_code\":1}",
                                    "{\"stdout\":\"\",\"error\":\"failed\",\"exit_code\":0}"}) {
    Json failed_repair = exhausted;
    append(failed_repair, "edit_file", "repair the script", result);
    require(parse(failed_repair).tool_calls.empty(),
            "failed or unconfirmed repairs must not reopen the retry window");
  }
  Json success = exhausted;
  append(success, "exec", "python3 generate_report.py", "Generated and verified\nExit code: 0");
  require(parse(success).tool_calls.size() == 1,
          "a successful retry must clear an obsolete failure streak");
  Json reminder = exhausted;
  reminder["messages"].push_back({{"role", "user"}, {"content", "Continue the task."}});
  reminder["messages"].push_back({{"role", "assistant"}, {"content", "I repaired it."}});
  require(parse(reminder).tool_calls.empty(),
          "plain continuation/claimed progress is not a completed tool result");

  Json late = request();
  append(late, "edit_file", "old repair", "Successfully edited an old input");
  const Json late_result = late["messages"].back();
  late["messages"].erase(late["messages"].size() - 1);
  append(late, "exec", "python3 generate_report.py", "Exit code: 1");
  append(late, "exec", "python3 generate_report.py", "Exit code: 1");
  late["messages"].push_back(late_result);
  require(parse(late).tool_calls.empty(),
          "a delayed older result must not erase newer failures");

  // The observation window must not depend on arrival order within a batch.
  for (const bool repair_first : {false, true}) {
    Json parallel = exhausted;
    Json batch = {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", Json::array()}};
    for (const auto& spec : {std::make_pair("exec", "python3 generate_report.py"),
                             std::make_pair("edit_file", "repair the input")}) {
      batch["tool_calls"].push_back({{"id", std::string("batch_") + spec.first},
          {"type", "function"}, {"function", {{"name", spec.first},
          {"arguments", Json({{"value", spec.second}}).dump()}}}});
    }
    parallel["messages"].push_back(batch);
    const Json bad = {{"role", "tool"}, {"tool_call_id", "batch_exec"}, {"content", "Exit code: 1"}};
    const Json good = {{"role", "tool"}, {"tool_call_id", "batch_edit_file"},
                       {"content", "Successfully edited the input"}};
    parallel["messages"].push_back(repair_first ? good : bad);
    parallel["messages"].push_back(repair_first ? bad : good);
    const auto observed = parse(parallel);
    require(observed.tool_calls.size() == 1 &&
                observed.tool_progress.history_no_progress_results == 3 &&
                observed.tool_progress.history_no_progress_streak == 1,
            "parallel result order changed the active failure window");
  }

  // A tool result may be repeated accidentally in client history. It cannot
  // manufacture a second failure or replay old success as fresh progress.
  Json duplicate = exhausted;
  duplicate["messages"].push_back(duplicate["messages"].back());
  bool duplicate_rejected = false;
  try { (void)aima::prepare_native_chat(duplicate); }
  catch (const std::invalid_argument&) { duplicate_rejected = true; }
  require(duplicate_rejected, "duplicate tool result ids must be rejected");

  std::cout << "native_tool_recovery_test: PASS\n";
}
