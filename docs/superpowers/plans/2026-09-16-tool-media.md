# Native tool media implementation plan

> Execute the approved assessment in this session, with test-first implementation and an independent code review before delivery.

**Goal:** Allow validated tool image/video results through the native pipeline while preserving tool progress and media admission.

**Architecture:** Update native chat parsing and its internal result classifier. Reuse the current template, vision preparation, cache and GPU kernels. Document the already-supported remote-domain configuration.

**Tech Stack:** C++17, existing Python qualification tooling, ROCm/HIP, portable native runtime.

- [ ] Establish a clean baseline in an isolated worktree with `make check`.
- [ ] Add `tests/native_tool_media_test.cpp` and a `check-native-tool-media` Make target, also called from `check-native-syntax`. Build it against unchanged production code and observe the current role restriction.
- [ ] Admit tool media in `native/src/native_chat_protocol.cpp` with `role != "user" && role != "tool"` as the rejection condition. Verify image/video source association, roles and call history.
- [ ] Introduce internal empty/progress/failure classification preserving `native_tool_result_is_no_progress` for existing text callers. Keep media result text separate from rendered visual markers. A result is no-progress when it is an explicit failure, or when it is empty and contains no media. Exercise failure text before/after media, structured error/status/nonzero exit, empty/media-only results, and late/replayed results.
- [ ] Extend `tests/native_chat_template_parity.cpp` with tool-image/video fixtures and compare actual template bytes against model-template reference output. Run `make native-chat-template-parity AIMA_MODEL_DIR="$AIMA_MODEL_DIR"`.
- [ ] Document tool media history and exact remote-domain configuration in `docs/API.md`, adding complete request examples and preserving the release qualification boundary.
- [ ] Run `make check`, `make security-scan`, and `make verify-evidence` on Linux. Build the candidate with the existing pinned `FFMPEG_ROOT`, `CURL_ROOT` and runtime dependencies using `scripts/build-native-runtime.sh`.
- [ ] In an isolated copy of the portable runtime, test the exact candidate on AMD395: image-only and mixed tool results, HTTPS/data/file sources, video, multiple tools/media, image replacement and replay, missing/unknown/replayed IDs, disallowed roles/domains, thinking, SSE and ordinary responses, text controls and existing protocol/control-plane qualification.
- [ ] Ask a code-review subagent to inspect changes while GPU validation proceeds. Resolve important findings and rerun affected checks.
- [ ] Save a reviewable commit, patch, candidate hash and reproducible evidence; verify the installed original release remains unchanged and temporary servers are stopped.

The parser regression starts with a real conversation containing a user
request, a preceding assistant `tool_calls` entry (`id=call_image`), and a tool
result with `tool_call_id=call_image` whose content is an `image_url` array.
Expected admission is one media item at message index 2. The unchanged `.9`
parser instead throws `image and video content parts are supported in user
messages only`. This isolates the requested behavior from malformed history.

The progress regression supplies the same explicit JSON failure both with and
without an image, twice for the same tool signature. Both must keep the retry
window exhausted; media-only responses must count as payload. This prevents
visual marker insertion from turning a structured failure into apparent success.
