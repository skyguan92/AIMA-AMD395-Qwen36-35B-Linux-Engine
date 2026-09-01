# Native CLI and HTTP API

This page documents the v1.5.1-native-vl.5 native CLI and HTTP API. Earlier
binaries do not include every command, multimodal surface and hardening control
described here.

## CLI

The user-facing `bin/aima-engine` is a fully static launcher. It resolves the
bundle root from `/proc/self/exe` and starts the colocated native payload
through the bundled glibc loader.

### Version

```bash
bin/aima-engine --version
bin/aima-engine --build-info
```

`--build-info` returns machine-readable version and embedded source commit.
The release packager rejects a clean-tree binary whose embedded commit differs
from the checkout being packaged.

### Deployment doctor

```bash
bin/aima-engine doctor --model-dir /srv/models/Qwen3.6-35B-A3B --json
```

`doctor` does not load weights or allocate the inference workspace. It checks
Linux/x86-64, KFD/render access, the HIP-visible `gfx1151` device, the two
required kernel parameters, 512 MiB fixed VRAM, the 96 GiB GTT pool, portable
bundle completeness and—when `--model-dir` is supplied—the four metadata
hashes plus all 26 readable shards. Exit status is zero only when every
required check passes.

### Resident server

```bash
bin/aima-engine serve \
  --model-dir /srv/models/Qwen3.6-35B-A3B \
  --context-tokens 8192 \
  --allowed-local-media-path /srv/aima-media \
  --host 127.0.0.1 \
  --port 8000
```

Options:

| Option | Meaning | Default |
|---|---|---|
| `--model-dir PATH` | qualified model directory | required |
| `--context-tokens N` | preferred qualified AOT prefill context or valid long-window endpoint | 8192 |
| `--cache-capacity N` | maximum prompt-plus-output context capacity | context + 1024 |
| `--host IPv4` | listen address | 127.0.0.1 |
| `--port N` | listen port | 8000 |
| `--workers N` | O_DIRECT checkpoint readers; increase only after measuring the target storage | 1 |
| `--chunk-bytes N` | checkpoint read chunk | 128 MiB |
| `--report PATH` | native weight-load report | working directory |
| `--max-requests N` | stop after N successful chat requests | unlimited |
| `--request-timeout-ms N` | request-read and blocking-write socket timeout; `0` disables both | 15000 |
| `--api-key-file PATH` | read one bearer token from a non-symlink file with mode `0640` or stricter | disabled on loopback |
| `--disable-http-shutdown` | remove the HTTP shutdown route | false |
| `--allow-insecure-remote` | explicitly permit a non-loopback bind without a token | false |
| `--allowed-local-media-path PATH` | repeatable root for descriptor-relative `file:` media | none |
| `--allowed-media-domain HOST` | repeatable exact HTTP/HTTPS media hostname | none |
| `--allowed-private-media-domain HOST` | permit an already-allowlisted hostname to resolve privately | none |
| `--remote-tls-ca-bundle PATH` | explicit CA bundle for remote HTTPS media | bundle CA; libcurl default in source builds |
| `--media-cache-capacity-bytes N` | decoded-media LRU byte capacity, at most 4 GiB | 4 GiB |
| `--disable-media-cache` | disable decoded-media reuse | false |
| `--fmha-provider PATH` | qualification-only provider override | automatic |

The process stays in the foreground and handles `SIGINT` / `SIGTERM`.
Use systemd, a container runtime or another supervisor for detached operation.
A non-loopback `--host` requires `--api-key-file` unless the explicit unsafe
override is present. `/health` is unauthenticated for liveness; the model list,
chat and enabled shutdown routes require `Authorization: Bearer TOKEN` whenever
an API key is configured. The engine never logs the token or its file contents.

Positive `--request-timeout-ms` values are accepted up to the platform's
representable millisecond range; there is no 600-second product cap. It sets an
absolute deadline for reading one HTTP request and bounds blocking socket
writes, not an inference wall-clock deadline. `0` disables the absolute
request-read and per-blocking-write socket timeouts, but the 1 MiB request-body
limit remains. Disable timeouts only when slow clients and indefinitely blocked
writes are operationally acceptable. In particular, request accept/read/routing
is single-threaded: at `0`, an incomplete or slow request can block every HTTP
endpoint, including `/health` and HTTP `/shutdown`, until it completes or the
process receives `SIGINT`/`SIGTERM`. Signals interrupt that blocked read and
begin graceful shutdown, but an HTTP-only operator still needs a finite
positive timeout when liveness matters. A blocked main-thread control write or
queued-chat cancellation write can also delay shutdown.

The optional dependency-free Python client accepts the same protected API:

```bash
export AIMA_API_KEY_FILE=/path/to/client-readable-api-key
aima-engine models --endpoint http://127.0.0.1:8000
aima-engine chat --stream "PROMPT" --endpoint http://127.0.0.1:8000
```

Each client command also accepts `--api-key-file`. The file must be a regular,
non-symlink file with mode `0640` or stricter and one printable-ASCII token;
the token is never placed in the command line.

### Qualification probes

`resident-session-probe` executes raw deterministic token fixtures and emits
the complete load/request/cache/performance record as JSON. It supports
`--max-new-tokens-sequence 512,1024` so output-length decode cells can share
one cold context without reloading the model. `--prompt-tokens N` exercises a
variable request length inside a larger `--context-tokens` resident process.

`tokenizer-probe` and `chat-template-probe` expose the native tokenizer for
fixture preparation. These probes do not import Python or Transformers.

Run `bin/aima-engine --help` for the internal correctness probes shipped with
the release.

## HTTP lifecycle

The server writes one readiness JSON object to stdout only after all 693
language and 333 visual tensors, resident state and both vision-attention
warmups are complete. It writes a final stopped object when exiting.

### `GET /health`

Returns:

- status and model id;
- whether the model is loaded and resident;
- `busy`, a boolean that is true only while the single chat inference worker is
  executing a chat;
- successful request count and uptime;
- total context capacity and selected static AOT prefill specialization;
- selected FMHA provider;
- native-VL readiness, vision warmup, media-cache capacity/residency and visual
  tensor counts;
- command-to-ready time.

### `GET /v1/models`

Returns the single id `aima-amd395-qwen36-35b`.

### `POST /shutdown`

Returns `{"status":"shutting_down"}`, then exits after the response is sent.
It requires the configured bearer token and returns 404 when
`--disable-http-shutdown` is active. The packaged systemd unit disables this
route; use `systemctl stop aima-engine` there. Shutdown stops new chat
admission, cancels queued chats with a `503` OpenAI error (`code` `server_busy`,
`type` `server_error`), lets an active inference finish, then joins its worker
before tearing down the engine. It does not forcibly cancel active inference.

## `POST /v1/chat/completions`

When `--api-key-file` is configured, add this header to every `/v1` example:

```text
Authorization: Bearer YOUR_API_KEY
```

Supported request fields:

- `model`: if present, must be `aima-amd395-qwen36-35b`;
- `messages`: `system`, `developer`, `user`, `assistant` and `tool` history;
  user messages may contain ordered OpenAI `text`, `image_url` and
  `video_url` content parts; assistant `tool_calls` and matching tool
  responses are accepted;
- `max_tokens` or `max_completion_tokens`: positive integer;
- `temperature`: finite number in `[0,2]`; `0` selects certified greedy
  top-1, and a positive value selects stochastic generation;
- `top_p`: exactly `1` for greedy generation, or a finite number in `(0,1]`
  when `temperature` is positive;
- `seed`: optional non-negative integer. It controls the positive-temperature
  PRNG; when omitted the response metrics expose the generated effective seed;
- `thinking`: optional object with `type` exactly `enabled` or `disabled` and
  an optional positive-integer `budget_tokens`. Omission preserves the prior
  product behavior (answer-only text prompts and the frozen VL template
  default). Explicit `enabled` returns Qwen reasoning separately; explicit
  `disabled` selects the answer-only template for both text and VL;
- `n`: exactly `1`;
- `stream`: boolean;
- `stream_options.include_usage`: boolean when `stream` is true;
- `tools`: OpenAI function-tool definitions;
- `tool_choice`: `auto`, `none`, `required`, or a named function object;
- `parallel_tool_calls`: boolean;
- `media_io_kwargs`: an optional request-level object matching the frozen vLLM
  image/video loader surface. RGBA images composite onto white by default;
  `image.rgba_background_color` accepts three integer channels in `[0,255]`.
  `video.fps` and `video.num_frames` may be supplied together and the smaller
  resulting sample count wins; `video.video_backend`, when present, must be
  `opencv`. Request fields shallow-merge within each named modality over the
  launch defaults. An absent modality, `{}`, or `{"video": {}}` is therefore a
  no-op and retains the 32-frame cap and 2 FPS.

Not supported:

- custom stop values;
- sampling/anti-repetition controls other than `temperature`, `top_p`, and
  `seed`, including `frequency_penalty`, `presence_penalty`, `logit_bias`,
  `logprobs`, `top_logprobs`, `repetition_penalty`, `min_p`, and `top_k`;
- audio message parts;
- deprecated `functions` or structured response formats;
- unqualified media-I/O fields such as `video.frame_recovery` and
  `video.max_duration`, or video backends other than OpenCV;
- batching or concurrent inference execution.

Chat execution/inference is serialized while routed control endpoints can be
served during it. Those endpoints are health, model-list and shutdown, after
their requests have been fully read and routed. The accept/read/routing loop
itself is single-threaded, so this does not protect against a slow or incomplete
request occupying that loop when the timeout is disabled. Accepted chats enter
a bounded pending queue of 16; exactly one chat inference runs at a time, so
the engine, tokenizer and cache are never used concurrently. Queue saturation
or shutdown rejects/cancels a chat with HTTP 503 and an OpenAI error whose
`code` is `server_busy` and `type` is `server_error`.

The server applies the model's qualified Qwen tool/chat template. When
`thinking.type` is `enabled`, the assistant suffix remains inside Qwen's
thinking region until the model emits `</think>`; when it is `disabled`, the
renderer closes an empty thinking region before generation. `budget_tokens`
is validated to be no larger than the effective `max_tokens`, but is a budget
declaration rather than a second decoder stop: `max_tokens` remains the hard
combined bound for reasoning plus final content. A disabled request may retain
the budget field so clients can switch only `type`. The omitted-field behavior
is unchanged for compatibility. Explicit thinking is rejected with the raw
`prompt_token_ids` extension because that extension supplies its own complete
prompt. The native renderer is byte-for-byte and token-for-token checked
against the checkpoint template for plain, thinking-enabled, tool,
assistant/tool-history, and multimodal fixtures.

Positive-temperature requests compute a raw-weight BF16 LM-head projection for
all 248,320 tokens, then apply temperature scaling and nucleus sampling with a
stable SplitMix64 stream. This path is separate from the certified greedy
shortlist, so `temperature:0` performs no full-vocabulary transfer or sampling
work. `aima_amd395.sampling` reports the mode, logits source, effective seed,
selection count, transfer bytes and wall time. An explicit seed produces the
same tokens for text/VL and SSE/non-stream execution; it does not promise token
identity with another engine's implementation-specific PRNG.

Media admission is fail-closed. `data:` URIs are bounded by media type. Local
`file:` URLs require one or more `--allowed-local-media-path` roots and are
opened descriptor-relative without following symlinks. HTTP/HTTPS URLs require
exact `--allowed-media-domain` entries; private-address resolution additionally
requires `--allowed-private-media-domain`. A custom trust store can be supplied
with `--remote-tls-ca-bundle`. Redirect, byte, decode, frame, dimension and
deadline limits apply before tensors reach the visual encoder. There is no
separate duration-only cap: the fixed OpenCV reference accepts finite sparse
videos beyond the former 768-second product limit, while byte, source/selected
frame, decoded-pixel and decode-wall bounds remain active.

Decoded processor results use a 4 GiB, 64-entry content-addressed LRU by
default. `--media-cache-capacity-bytes` can reduce the byte bound and
`--disable-media-cache` provides the cold-cache performance surface. The key
binds source-byte SHA-256, media kind and the request-effective image/video
processor identity, so a changed object behind the same URL or pathname, a
changed RGBA background, or a changed `fps`/`num_frames` policy misses while
equivalent local and data-URI bytes may hit.

After tokenization:

- every positive prompt length is admitted when prompt plus requested output
  fits `--cache-capacity`;
- the capacity-bounded LRU cache reuses exact or genuine token-prefix request
  snapshots, restoring state before executing only the suffix (four entries at
  q8192, fewer at very long windows to preserve the 96 GiB memory contract);
- a q8192 process keeps q1024/q2048/q4096/q8192 AOT prefill buckets resident;
- a cold cache miss composes the smallest resident bucket total covering the
  prompt and pads only the final segment when necessary;
- a prompt shorter than q1024 uses a padded q1024 prefill, then repairs its
  convolution and recurrent state at the logical prompt boundary;
- a prefix extension uses the same composed AOT path for its suffix;
- independent and ordinary `user -> assistant -> user` conversations therefore
  fall back to cold execution instead of being rejected;
- the absolute model/runtime window remains 262,144 tokens.

Prefix matching is exact token matching, not a text-prefix heuristic. It only
changes latency. Cache entries are completed request-prefix snapshots, not
arbitrary token checkpoints. For peak cold-prefill throughput, select a
published standard context matching the workload. Non-bucket prompts perform
fixed-shape padding or more than one resident prefill pass, but never ingest
prompt tokens at serial decode throughput.

### Frozen-token eval extension

For reproducible regression evaluation, the native endpoint accepts a
non-standard top-level `prompt_token_ids` array. It replaces chat-template
tokenization for that request and is rejected when combined with tools. The
response reports `aima_amd395.prompt_source = "token_ids"`; ordinary OpenAI
clients should continue to send `messages` without this field.

The repository's resumable `scripts/qualify-native-eval.py` verifies every
frozen prompt-token hash, sends deterministic batch-1 requests and writes a
sanitized scorecard containing no questions or token IDs. Example:

```bash
python3 scripts/qualify-native-eval.py \
  --items /private/eval/items.jsonl \
  --requests-root /private/eval \
  --engine-binary ./bin/aima-engine \
  --output output/native-eval.json \
  --minimum-correct 216
```

The published standard contexts are `1024`, `2048`, `4096`, `8192`, `16384`,
`32768`, `65536` and `131072`. Maximum-window qualifications use
input/output pairs `262143/1`, `261632/512` and `261120/1024`.

### Response

With `stream:false`, the response follows the OpenAI shape: `id`, `object`,
`created`, `model`, `choices` and `usage`. Plain generations return assistant
`content`. With explicit thinking enabled, the same message also returns
`reasoning_content`; `content` contains only bytes after `</think>`. If the
generation limit is reached before that marker, all visible bytes are
reasoning and `content` is empty. Neither thinking marker is returned. A
function generation returns `message.tool_calls` and ends with
`finish_reason: "tool_calls"`:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "chatcmpl-native-3-call-0",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\": \"Paris\"}"
    }
  }]
}
```

Parameter strings, integers, numbers, booleans, objects and arrays are
converted using the function JSON Schema. Unknown or malformed function markup
is not exposed as a valid call. A terminal EOS token counts in
`usage.completion_tokens` but is omitted from visible assistant content.

`aima_amd395` adds:

- runtime and request index;
- the effective thinking mode, declared reasoning budget, and total generation
  bound;
- for tool requests, parsed/suppressed call counts, matching history counts,
  and a machine-readable bounded-retry/no-progress decision;
- `model_loads`;
- prefill/decode throughput and total latency;
- TTFT;
- prompt execution mode and zero-valued legacy cold-tail timing;
- logical AOT-prefill tokens, scheduled bucket tokens, segment count and
  padding tokens;
- output-token SHA-256;
- for VL requests, canonical prompt/output token-array SHA-256 values matching
  the frozen oracle serialization;
- VL media/image/video counts, patch and visual-token counts, M-RoPE delta,
  media/plan cache state, transfer bytes and per-stage media/decode/processor/
  vision/injection timings;
- prefix lookup type, matched/suffix token counts, cumulative hits/misses,
  state-transfer bytes, suffix launch counts and suffix wall time.

### Streaming

`stream:true` uses HTTP/1.1 chunked transfer and
`Content-Type: text/event-stream`. The sequence is:

1. an assistant-role `chat.completion.chunk`;
2. when thinking is explicitly enabled, `delta.reasoning_content` chunks until
   the closing marker, which is withheld even across token boundaries;
3. content deltas as soon as post-thinking token bytes form valid UTF-8;
4. a structured `delta.tool_calls` when Qwen completes a function call;
5. a terminal chunk with `stop`, `length` or `tool_calls`;
6. an optional empty-choices usage chunk;
7. `data: [DONE]`.

This is live decode streaming, not post-generation text splitting. Closing the
connection cancels remaining decode work while keeping the model and valid
prefix state resident.

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aima-amd395-qwen36-35b",
    "messages": [{"role": "user", "content": "Explain prefix caching briefly."}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 512,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

### Function tools

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aima-amd395-qwen36-35b",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 128
  }'
```

Return the assistant call and its result in the next request as standard
assistant and tool messages:

```json
[
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
      "id": "chatcmpl-native-3-call-0",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\":\"Paris\"}"
      }
    }]
  },
  {
    "role": "tool",
    "tool_call_id": "chatcmpl-native-3-call-0",
    "content": "{\"temperature_c\":20,\"condition\":\"sunny\"}"
  }
]
```

Tool definitions and history count toward the cache capacity. Clients do not
pad requests; after tokenization the engine composes resident AOT buckets and
internally pads only the final segment. Required and named `tool_choice`
requests fail if generation does not produce an admitted function call, except
when the bounded no-progress policy deliberately suppresses an exhausted
history signature.

At the protocol boundary, calls are compared by function name plus canonical
JSON arguments. Exact duplicates in one generated response are emitted only
once; the same function with different arguments remains valid. History is
also inspected conservatively: an empty result or explicit error permits one
same-signature retry, while a second no-progress result suppresses another
identical call. `parallel_tool_calls:false` is applied after those checks and
emits at most one call in both response modes.

The terminal `aima_amd395.tool_progress` object reports
`duplicate_calls_suppressed`, `history_signature_occurrences`,
`history_no_progress_results`, `exhausted_history_calls_suppressed`,
`no_progress`, `reason`, and `caller_action`. The native engine prevents exact
duplicate actions and exposes this state; choosing a materially different
strategy, composing a best-effort answer, or returning a domain-specific
blocked result remains the calling agent's responsibility. This boundary is
intentional because the engine cannot determine whether two different tool
calls are semantically equivalent or whether their results add domain-specific
information.

## Errors

Invalid JSON, unsupported fields, model mismatch and prompt-policy violations
return HTTP 400 with an OpenAI-shaped `error` object. Oversized request bodies
return 413. Native execution failures return 500 while the resident process
remains available for the next request unless the underlying device state is
fatal.

## Concurrency

One process owns one model and serializes chat execution/inference. Fully read
and routed control endpoints can be served while a chat runs. This preserves
the measured batch-1 contract and the capacity-bounded exact-token prefix
state. Run separate isolated processes only when the machine has enough memory;
a normal 128 GB AMD395 host does not have room for two copies of this BF16
model.
