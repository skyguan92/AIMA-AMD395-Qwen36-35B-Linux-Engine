# HTTP Timeout and Health Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Remove the 600-second HTTP timeout ceiling, make `0` disable HTTP read/write timeouts, and keep `/health` responsive while chat inference remains strictly serialized.

**Architecture:** The accept thread reads and routes requests and answers control-plane routes immediately. A bounded one-worker executor owns chat sockets and invokes the existing tokenizer/engine path one request at a time; atomics expose server-owned busy and served state without concurrent engine access. Shutdown rejects new work, cancels queued chat sockets with an OpenAI-compatible 503 response, lets the active inference finish, and joins the worker before engine destruction.

**Tech Stack:** C++17, POSIX sockets, `std::thread`/`std::condition_variable`, nlohmann JSON, Python `unittest`, GNU Make, ROCm/hipcc on the AMD Ryzen AI Max+ 395 qualification host.

---

### Task 1: Add failing CPU tests for timeout parsing and serial execution

**Files:**

- Create: `tests/native_http_support_test.cpp`
- Modify: `Makefile:29-35`

- [ ] Add a C++ test executable that includes `aima/native_http_support.h` and checks the public support contract:

```cpp
assert(parse_native_http_timeout_ms("0", "--request-timeout-ms") == 0);
assert(parse_native_http_timeout_ms("600001", "--request-timeout-ms") == 600001);
assert(parse_native_http_timeout_ms("86400000", "--request-timeout-ms") == 86400000);
expect_invalid_timeout("-1");
expect_invalid_timeout("abc");
expect_invalid_timeout("184467440737095516160");
```

- [ ] Add deterministic executor tests using promises and atomics: the first task blocks, `busy()` becomes true, one pending task is accepted at capacity one, a third task is rejected, maximum simultaneous task count stays one, and shutdown invokes the queued task's cancellation callback before joining the active task.
- [ ] Add this test before the HIP syntax-only command in `check-native-syntax`:

```make
	g++ -std=c++17 -O2 -pthread -I native/include tests/native_http_support_test.cpp native/src/native_http_support.cpp -o build/native_http_support_test
	./build/native_http_support_test
```

- [ ] Run `make check-native-syntax` on a host with ROCm headers or compile only the new CPU target locally. Expected result before production code: compilation fails because the support header/source do not exist.
- [ ] Commit the red test with message `test: cover HTTP timeout and serial executor` only after recording the expected failure.

### Task 2: Implement the reusable HTTP support module

**Files:**

- Create: `native/include/aima/native_http_support.h`
- Create: `native/src/native_http_support.cpp`
- Modify: `Makefile:34`
- Modify: `scripts/build-native-runtime.sh:120-165`

- [ ] Declare a non-copyable `NativeSerialExecutor` with `Task { std::function<void()> run; std::function<void()> cancel; }`, `submit`, `busy`, and idempotent `shutdown` methods. Its constructor accepts the maximum pending queue size and starts exactly one worker thread.
- [ ] Implement `parse_native_http_timeout_ms` so `0` is valid, leading signs/non-digits/overflow are rejected, and positive values must fit both `std::chrono::steady_clock::duration` and the `time_t` seconds field used by `timeval`. Do not impose a product policy ceiling.
- [ ] Implement the executor with a mutex-protected deque, condition variable, stopping flag, one worker, and atomic busy flag. Set busy immediately before `run`, clear it after all success/exception paths, catch task exceptions so the worker survives, reject submission while stopping or when the pending queue is full, and never run two tasks simultaneously.
- [ ] In `shutdown`, atomically stop admission, move all pending cancellation callbacks out of the queue, invoke them promptly, notify the worker, and join it after the active task returns. Make repeated calls and destruction safe.
- [ ] Add `native/src/native_http_support.cpp` to the native syntax-only source list and the hipcc link command.
- [ ] Run:

```bash
mkdir -p build
g++ -std=c++17 -O2 -pthread -I native/include \
  tests/native_http_support_test.cpp native/src/native_http_support.cpp \
  -o build/native_http_support_test
./build/native_http_support_test
```

Expected: exit code 0 and a concise pass message.
- [ ] Commit with message `feat: add native HTTP serial executor`.

### Task 3: Add failing server contract tests

**Files:**

- Modify: `tests/test_native_runtime_contract.py:644-662`

- [ ] Extend the native server contract test to require the support header/source in the release build, an atomic served counter, the health `busy` field, queue rejection code `server_busy`, and an executor shutdown before the final stopped event.
- [ ] Assert the obsolete error text `--request-timeout-ms must not exceed 600000` is absent and that option parsing calls `parse_native_http_timeout_ms`.
- [ ] Assert the HTTP status table includes `case 503: return "Service Unavailable";` and the queue error uses OpenAI type `server_error`.
- [ ] Run the focused test:

```bash
python3 -m unittest \
  tests.test_native_runtime_contract.NativeRuntimeContractTest.test_native_release_is_self_contained_and_secure
```

Expected before server changes: failure on the new source-contract assertions.
- [ ] Commit the red contract test with message `test: require responsive native health routing` only after recording the expected failure.

### Task 4: Route chat through the bounded executor and make timeout zero explicit

**Files:**

- Modify: `native/src/native_http_server.cpp:1-1240`

- [ ] Include `aima/native_http_support.h`, `<memory>`, and `<optional>`. Add `503 Service Unavailable` to `status_text` and a constant pending chat capacity of 16.
- [ ] Change `configure_client_timeout` so zero leaves the socket in normal blocking mode. For positive values, construct `timeval` only from the already validated timeout.
- [ ] Change `receive_before` and `read_request` to use an optional absolute deadline. With timeout zero, call blocking `recv` without installing `SO_RCVTIMEO`; with a positive timeout, preserve the existing one-deadline-for-header-and-body behavior. If `recv` is interrupted after a shutdown signal, report cancellation rather than spinning.
- [ ] Replace the inline option parsing and 600,000 check with:

```cpp
options.request_timeout_ms = parse_native_http_timeout_ms(
    next("--request-timeout-ms"), "--request-timeout-ms");
```

- [ ] After model load, create `std::atomic<std::size_t> served{0}` and `NativeSerialExecutor chat_executor(16)`. Health must read `served.load()`, use the post-load constant model readiness snapshot, and add `{"busy", chat_executor.busy()}` without calling mutable engine accessors.
- [ ] Represent queued chat work with a `shared_ptr` that owns the released client descriptor, HTTP start time, and parsed request. Submit a `run` callback that performs JSON parsing, chat validation, streaming/non-streaming inference, OpenAI error mapping, and successful served increments using the unchanged engine/tokenizer objects.
- [ ] Submit a `cancel` callback that sends HTTP 503 with code `server_busy`, type `server_error`, and message `server is shutting down`. If `submit` returns false because the 16-entry pending queue is full or stopping, immediately send HTTP 503 with code `server_busy` and type `server_error`.
- [ ] Keep auth, `/health`, `/v1/models`, method errors, and unknown paths on the accept thread. `POST /shutdown` stops admission and wakes the blocking accept loop with `shutdown(server.get(), SHUT_RDWR)`.
- [ ] Preserve `--max-requests`: after a successful chat response reaches the limit, the worker sets the shutdown flag and wakes accept. When accept fails after shutdown, break cleanly instead of throwing.
- [ ] After leaving the accept loop, call `chat_executor.shutdown()` before emitting the stopped event. The final served value must come from the atomic counter. Executor lifetime must end before tokenizer/engine destruction.
- [ ] Run:

```bash
g++ -std=c++17 -O2 -pthread -I native/include \
  tests/native_http_support_test.cpp native/src/native_http_support.cpp \
  -o build/native_http_support_test && ./build/native_http_support_test
python3 -m unittest \
  tests.test_native_runtime_contract.NativeRuntimeContractTest.test_native_release_is_self_contained_and_secure
```

Expected: both pass.
- [ ] Commit with message `fix: keep health responsive during native inference`.

### Task 5: Document the externally visible behavior

**Files:**

- Modify: `docs/API.md:49-115`
- Modify: `docs/INSTALL.md:128-142`
- Modify: `docs/ARCHITECTURE.md:132-145`
- Modify: `native/README.md:140-150`

- [ ] Change the CLI table to state that positive timeout values are platform-representable milliseconds with no 600-second policy cap and `0` disables request-read/write timeouts; explicitly state it is not an inference deadline and the request-size limit still applies.
- [ ] Document health response field `busy`: `true` only while the serial inference worker is executing a chat, while health remains HTTP 200 after readiness.
- [ ] Replace “one request at a time” wording with the exact concurrency contract: control-plane routes are independent, accepted chat work is bounded, and exactly one chat inference executes at a time.
- [ ] Document 503 `server_busy` behavior for queue saturation and shutdown.
- [ ] Run:

```bash
rg -n "maximum 600000|must not exceed 600000|One process handles one request at a time" \
  docs native native/README.md
```

Expected: no matches. Then run `python3 -m unittest discover -s tests -p 'test_*.py'` with the real-path temporary directory workaround on macOS; expected: all 45 tests pass.
- [ ] Commit with message `docs: explain native HTTP concurrency and timeouts`.

### Task 6: Run complete local verification

**Files:**

- Verify only; do not edit unrelated release evidence.

- [ ] Run `git diff --check` and inspect `git diff origin/main...HEAD` for accidental changes or credentials. Expected: clean whitespace check and only the design, plan, tests, support module, server, build wiring, and four documentation files.
- [ ] Run the new C++ support test under ThreadSanitizer when the local compiler supports it:

```bash
g++ -std=c++17 -O1 -g -pthread -fsanitize=thread -I native/include \
  tests/native_http_support_test.cpp native/src/native_http_support.cpp \
  -o build/native_http_support_tsan && ./build/native_http_support_tsan
```

Expected: exit code 0 with no data-race report. If macOS TSAN cannot initialize, record the tool limitation and rely on the normal C++ test plus Linux syntax/test run.
- [ ] Run `make check-cpu`. On macOS, set `tempfile.tempdir` to the real path of `tempfile.gettempdir()` for the known `/var` versus `/private/var` release-contract issue. Expected: 45 Python tests, release verification, CLI verification, and wheel build all pass.
- [ ] On the AMD395 machine, run `make check-native-syntax`; expected: all three C++ tests pass and HIP server sources compile syntax-only.
- [ ] Commit any verification-driven corrections separately with an accurate message.

### Task 7: Qualify the fix on the AMD395 machine before any push

**Remote paths:** Set `REMOTE_SOURCE_WORKTREE`, `QUALIFIED_RUNTIME`, and `MODEL_DIR` to the machine-specific remote source worktree, validated runtime, and model paths. Supply their values out of band; do not commit them.

- [ ] Over SSH, fetch `origin/main` in the existing clean Git metadata and create a separate detached worktree at the remote source path. Do not modify or remove the existing untracked AIMA/runtime directories.
- [ ] Before pushing to GitHub, transfer only `git diff --name-only origin/main...HEAD` from the local implementation worktree into the matching relative paths of the remote test worktree with `rsync --relative`.
- [ ] Run in the remote worktree:

```bash
: "${REMOTE_SOURCE_WORKTREE:?must be set}"
: "${QUALIFIED_RUNTIME:?must be set}"
: "${MODEL_DIR:?must be set}"

make check-native-syntax
PREFILL_CONTEXTS=8192 OUT_DIR="$PWD/build/native" \
  bash scripts/build-native-runtime.sh
mkdir -p runtime-http-test/bin runtime-http-test/libexec
ln -s "$QUALIFIED_RUNTIME/lib" runtime-http-test/lib
ln -s "$QUALIFIED_RUNTIME/amdgcn" runtime-http-test/amdgcn
cp "$QUALIFIED_RUNTIME/bin/aima-engine" runtime-http-test/bin/aima-engine
cp build/native/aima-engine-native runtime-http-test/libexec/aima-engine.real
```

Expected: native support tests and HIP syntax pass; native runtime links and reports its version; the test bundle uses the already qualified 1.5.1 runtime libraries without overwriting them.
- [ ] Start the test bundle on `127.0.0.1:18425` with the absolute model path, `--context-tokens 8192`, `--cache-capacity 9216`, and `--request-timeout-ms 0`. Wait for ready JSON and verify `/health` returns HTTP 200 with `request_timeout_ms: 0` and `busy: false`.
- [ ] Send a non-streaming chat request large enough to run for several seconds in the background. Poll `/health` every 100 ms while it runs and record status, latency, process PID, and `busy`. Expected: every completed probe is HTTP 200, at least one probe observes `busy: true`, probe latency stays below two seconds, and PID remains stable.
- [ ] Repeat with a streaming chat request and the same health polling assertions. Expected: HTTP 200 health remains responsive until `[DONE]` and returns to `busy: false` after completion.
- [ ] Start two chat requests concurrently and poll health. Expected: both chats complete successfully in admission order, request metrics show consecutive request indices, and server logs/metrics demonstrate serialized inference rather than overlapping `engine.run` calls.
- [ ] Stop through `POST /shutdown`. Expected: the active request is allowed to finish, any queued requests receive 503 `server_busy`, the process exits normally, and the stopped JSON reports the atomic served count.
- [ ] Restart with `--request-timeout-ms 600001`, verify ready JSON and `/health` report `600001`, then shut down. Expected: startup succeeds, proving the old ceiling is removed.

### Task 8: Review, push, and create the pull request

**Files:**

- Review the complete branch; edit only findings required for correctness or clarity.

- [ ] Invoke the `superpowers:requesting-code-review` skill and dispatch its required code-reviewer subagent with the approved design, implementation plan, base `6f3e669`, branch HEAD, and AMD395 evidence. Fix Important/Critical findings, rerun affected local and remote checks, and request follow-up review if needed.
- [ ] Invoke `superpowers:verification-before-completion`, rerun `git diff --check`, the full local suite, native syntax on AMD395, and the final GPU smoke/concurrency test. Record exact pass/fail counts and relevant health latency evidence.
- [ ] Verify `git status --short`, `git log --oneline origin/main..HEAD`, and `git diff --stat origin/main...HEAD`. Confirm no password, token, model data, generated binary, remote log, or unrelated `.DS_Store` is tracked.
- [ ] Push `fix/http-timeout-health-concurrency` to `origin` only after all remote qualification passes.
- [ ] Create a GitHub PR against `main` summarizing root cause, timeout semantics, serialized executor architecture, compatibility of the health schema, shutdown/queue behavior, CPU checks, native syntax, and AMD395 streaming/non-streaming evidence. Do not include machine credentials.
- [ ] Inspect the created PR and checks with `gh pr view` and `gh pr checks`. Return the PR URL, commit list, test evidence, and any still-running CI status to the user.
