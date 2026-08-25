# HTTP Timeout and Health Concurrency Design

## Goal

Allow operators to disable or freely configure native HTTP socket timeouts, and keep `GET /health` responsive while a serialized `POST /v1/chat/completions` request is running.

## Scope

This change covers two HTTP server behaviors:

1. `--request-timeout-ms 0` disables request-read and socket-write timeouts. Positive values are no longer capped at 600,000 ms, but must fit the platform time representations used by the server.
2. Control-plane routes remain responsive while chat inference runs. Chat inference itself remains batch-1 and strictly serialized.

Concurrent `NativeResidentEngine::run()` calls, inference cancellation by elapsed wall time, batching, and a separate health port are outside this change.

## Architecture

The accept thread remains responsible for accepting sockets, configuring their timeouts, reading complete requests, authenticating them, and selecting a route. Lightweight control-plane routes such as `/health`, `/v1/models`, and `/shutdown` are answered directly on that thread.

Validated chat requests transfer ownership of their client socket and parsed HTTP request to a bounded serial executor. The executor owns one worker thread and invokes chat handlers one at a time. It exposes atomic busy state to the server, but never permits concurrent access to tokenizer, engine workspaces, KV state, or prefix-cache state.

This design preserves the existing single-inference contract while removing inference work from the accept loop.

## Components

### Native HTTP support module

A small CPU-only support module provides:

- non-negative timeout parsing with platform-range validation;
- a bounded single-worker task executor;
- observable `busy()` state;
- orderly shutdown that lets the active task finish and cancels queued tasks.

The module has no HIP, tokenizer, or model dependency, so its behavior can be covered by ordinary C++ unit tests.

### Server-owned health state

The HTTP server owns atomic counters and flags used by `/health`:

- `model_loaded`, set after the synchronous model load completes and never cleared while serving;
- `busy`, supplied by the serial executor;
- `served`, incremented only after a successful chat response.

The health handler does not call mutable `NativeResidentEngine` accessors concurrently with inference.

## Request Flow

1. The accept thread accepts and fully reads a request.
2. Authentication runs before route dispatch.
3. `/health`, `/v1/models`, method errors, and unknown paths are handled immediately.
4. A chat request is wrapped in an executor task that owns the accepted file descriptor.
5. If the queue accepts the task, the accept thread continues accepting connections.
6. The worker parses the JSON body and calls the existing streaming or non-streaming chat path.
7. Completion or failure closes the client descriptor and clears busy state before the next task.

The queue capacity is bounded to the existing listen-backlog scale. A full or stopping executor returns HTTP 503 with OpenAI-shaped code `server_busy` instead of accumulating unbounded request bodies.

## Health Contract

`GET /health` continues to return HTTP 200 after model readiness. Its existing fields remain compatible, and it adds:

```json
{
  "status": "ok",
  "model_loaded": true,
  "busy": true
}
```

`busy: true` means one chat request is executing. It is not a liveness failure and does not change the HTTP status.

## Timeout Contract

`--request-timeout-ms` keeps its existing scope: an absolute deadline for reading one complete request plus a per-blocking-write socket timeout. It does not become an inference wall-clock deadline.

- `0`: leave the accepted socket in blocking mode without HTTP read/write deadlines.
- positive value: configure the existing request-read and socket-write behavior.
- non-integer, negative, or unrepresentable value: fail startup with an argument error.

The hard-coded 600,000 ms validation and corresponding documentation are removed.

## Shutdown and Error Handling

`POST /shutdown`, SIGINT, SIGTERM, and `--max-requests` stop new admission. The executor then rejects or cancels queued work with 503, allows the active inference task to finish, and joins its worker before tokenizer and engine destruction.

Stopping admission also wakes a blocking `accept()` call. When timeout `0` leaves request reads unbounded, an interrupted read observes shutdown instead of restarting indefinitely.

Exceptions from one chat request remain contained by the existing request error handlers. They produce the current 400 or 500 response and do not terminate the worker or resident server.

## Security and Resource Bounds

Removing the 600-second cap does not remove request-size limits. Positive read deadlines continue to mitigate slow clients. Operators who explicitly select `0` accept the risk of an indefinitely slow request, so documentation must call out that trade-off.

The in-process chat queue is bounded. File descriptor ownership is explicit: the accept thread owns a socket until successful task admission, after which the queued task owns it through completion or cancellation.

## Tests

CPU-only C++ tests cover:

- timeout values `0`, `600001`, a large representable value, invalid text, and overflow;
- non-blocking task admission while an active task is held;
- strict serialization with no task overlap;
- busy-state transitions;
- queue capacity rejection;
- shutdown cancellation of queued tasks and joining of the active task.

Existing Python contract tests verify that:

- the 600,000 ms rejection is absent;
- zero-timeout behavior is wired into socket and request reading;
- `/health` exposes `busy` from server-owned state;
- chat requests are submitted to the serial executor;
- API and architecture documentation match the implementation.

The full CPU test suite and native syntax check run before the pull request. GPU release qualification must additionally run a long streaming and non-streaming chat request while polling `/health`, verify bounded health latency and stable process identity, and confirm two chat requests never execute the engine concurrently.
