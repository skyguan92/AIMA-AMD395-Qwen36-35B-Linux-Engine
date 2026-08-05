# Native CLI and HTTP API

This page documents the v1.5.0 native CLI and HTTP API. Earlier binaries do
not include every command and hardening control described here.

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
| `--chunk-bytes N` | checkpoint read chunk | 512 MiB |
| `--report PATH` | native weight-load report | working directory |
| `--max-requests N` | stop after N successful chat requests | unlimited |
| `--request-timeout-ms N` | absolute request-read deadline and per-write timeout (maximum 600000) | 15000 |
| `--api-key-file PATH` | read one bearer token from a non-symlink file with mode `0640` or stricter | disabled on loopback |
| `--disable-http-shutdown` | remove the HTTP shutdown route | false |
| `--allow-insecure-remote` | explicitly permit a non-loopback bind without a token | false |
| `--fmha-provider PATH` | qualification-only provider override | automatic |

The process stays in the foreground and handles `SIGINT` / `SIGTERM`.
Use systemd, a container runtime or another supervisor for detached operation.
A non-loopback `--host` requires `--api-key-file` unless the explicit unsafe
override is present. `/health` remains available for liveness; the model list,
chat and enabled shutdown routes require `Authorization: Bearer TOKEN` whenever
an API key is configured. The engine never logs the token or its file contents.

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

The server writes one readiness JSON object to stdout after all resident state
is initialized. It writes a final stopped object when exiting.

### `GET /health`

Returns:

- status and model id;
- whether the model is loaded and resident;
- successful request count and uptime;
- total context capacity and selected static AOT prefill specialization;
- selected FMHA provider;
- command-to-ready time.

### `GET /v1/models`

Returns the single id `aima-amd395-qwen36-35b`.

### `POST /shutdown`

Returns `{"status":"shutting_down"}`, then exits after the response is sent.
It requires the configured bearer token and returns 404 when
`--disable-http-shutdown` is active. The packaged systemd unit disables this
route; use `systemctl stop aima-engine` there.

## `POST /v1/chat/completions`

When `--api-key-file` is configured, add this header to every `/v1` example:

```text
Authorization: Bearer YOUR_API_KEY
```

Supported request fields:

- `model`: if present, must be `aima-amd395-qwen36-35b`;
- `messages`: text-only `system`, `developer`, `user`, `assistant` and `tool`
  history; assistant `tool_calls` and matching tool responses are accepted;
- `max_tokens` or `max_completion_tokens`: positive integer;
- `temperature`: exactly `0`;
- `top_p`: exactly `1`;
- `n`: exactly `1`;
- `stream`: boolean;
- `stream_options.include_usage`: boolean when `stream` is true;
- `tools`: OpenAI function-tool definitions;
- `tool_choice`: `auto`, `none`, `required`, or a named function object;
- `parallel_tool_calls`: boolean.

Not supported:

- custom stop values;
- image, audio or video message parts;
- deprecated `functions` or structured response formats;
- stochastic sampling;
- batching or concurrent execution.

The server applies the model's qualified Qwen tool/chat template with thinking
disabled. Its native renderer is byte-for-byte and token-for-token checked
against the checkpoint template for plain, tool and assistant/tool-history
fixtures.

After tokenization:

- every positive prompt length is admitted when prompt plus requested output
  fits `--cache-capacity`;
- the capacity-bounded LRU cache reuses exact or genuine token-prefix request
  snapshots, restoring state before executing only the suffix (four entries at
  q8192, fewer at very long windows to preserve the 96 GiB memory contract);
- a q8192 process keeps q1024/q2048/q4096/q8192 AOT prefill buckets resident;
- a cold cache miss selects the largest resident bucket no longer than the
  prompt and executes only the remaining tail through native decode;
- a prompt shorter than the smallest q1024 bucket starts from empty recurrent/
  KV state and executes through the qualified token path;
- independent and ordinary `user -> assistant -> user` conversations therefore
  fall back to cold execution instead of being rejected;
- the absolute model/runtime window remains 262,144 tokens.

Prefix matching is exact token matching, not a text-prefix heuristic. It only
changes latency. Cache entries are completed request-prefix snapshots, not
arbitrary token checkpoints. For peak cold-prefill throughput, select a
published standard context matching the workload; only the portion after the
largest fitting resident AOT bucket runs at decode throughput.

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
`content`. A function generation returns `message.tool_calls` and ends with
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
- `model_loads`;
- prefill/decode throughput and total latency;
- TTFT;
- prompt execution mode and token-decoded cold-tail timing;
- the selected `aot_prefill_tokens` bucket;
- output-token SHA-256;
- prefix lookup type, matched/suffix token counts, cumulative hits/misses,
  state-transfer bytes, suffix launch counts and suffix wall time.

### Streaming

`stream:true` uses HTTP/1.1 chunked transfer and
`Content-Type: text/event-stream`. The sequence is:

1. an assistant-role `chat.completion.chunk`;
2. content deltas as soon as generated token bytes form valid UTF-8;
3. a structured `delta.tool_calls` when Qwen completes a function call;
4. a terminal chunk with `stop`, `length` or `tool_calls`;
5. an optional empty-choices usage chunk;
6. `data: [DONE]`.

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

Tool definitions and history count toward the cache capacity. No padding is
required; the engine selects the largest fitting resident AOT bucket after the
complete tool template is tokenized. Required and named `tool_choice` requests
fail if generation does not produce an admitted function call.

## Errors

Invalid JSON, unsupported fields, model mismatch and prompt-policy violations
return HTTP 400 with an OpenAI-shaped `error` object. Oversized request bodies
return 413. Native execution failures return 500 while the resident process
remains available for the next request unless the underlying device state is
fatal.

## Concurrency

One process owns one model and serializes requests. This preserves the measured
batch-1 contract and the capacity-bounded exact-token prefix state. Run
separate isolated processes only when the machine has enough memory; a normal
128 GB AMD395 host does not have room for two copies of this BF16 model.
