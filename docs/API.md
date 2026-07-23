# Native CLI and HTTP API

## CLI

The user-facing `bin/aima-engine` is a fully static launcher. It resolves the
bundle root from `/proc/self/exe` and starts the colocated native payload
through the bundled glibc loader.

### Version

```bash
bin/aima-engine --version
```

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
| `--context-tokens N` | qualified static cold context or valid long-window endpoint | 8192 |
| `--cache-capacity N` | prompt plus decode KV capacity | context + 1024 |
| `--host IPv4` | listen address | 127.0.0.1 |
| `--port N` | listen port | 8000 |
| `--workers N` | checkpoint reader workers | 2 |
| `--chunk-bytes N` | checkpoint read chunk | 512 MiB |
| `--report PATH` | native weight-load report | working directory |
| `--max-requests N` | stop after N successful chat requests | unlimited |
| `--fmha-provider PATH` | qualification-only provider override | automatic |

The process stays in the foreground and handles `SIGINT` / `SIGTERM`.
Use systemd, a container runtime or another supervisor for detached operation.

### Qualification probes

`resident-session-probe` executes raw deterministic token fixtures and emits
the complete load/request/cache/performance record as JSON. It supports
`--max-new-tokens-sequence 512,1024` so output-length decode cells can share
one cold context without reloading the model.

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
- admitted static context;
- selected FMHA provider;
- command-to-ready time.

### `GET /v1/models`

Returns the single id `aima-amd395-qwen36-35b`.

### `POST /shutdown`

Returns `{"status":"shutting_down"}`, then exits after the response is sent.
There is no authentication. Keep the server on a trusted interface.

## `POST /v1/chat/completions`

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

- a cold request must contain exactly the process's static context;
- an exact repeat restores the cached state;
- a longer request must begin with the one cached token sequence, then the
  suffix is executed through native decode;
- prompt plus requested output must fit the 262,144-token window;
- all other lengths fail with HTTP 400.

This is an exact token-prefix contract, not a text-prefix heuristic.

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
    "messages": [{"role": "user", "content": "EXACT_LENGTH_PROMPT"}],
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
      {"role": "user", "content": "EXACT_LENGTH_TOOL_PROMPT"}
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

Tool definitions and history count toward the selected static context. Prepare
or pad the prompt after applying the complete tool template. Required and
named `tool_choice` requests fail if generation does not produce an admitted
function call.

## Errors

Invalid JSON, unsupported fields, model mismatch and prompt-policy violations
return HTTP 400 with an OpenAI-shaped `error` object. Oversized request bodies
return 413. Native execution failures return 500 while the resident process
remains available for the next request unless the underlying device state is
fatal.

## Concurrency

One process owns one model and serializes requests. This preserves the measured
batch-1 contract and the single-entry prefix state. Run separate isolated
processes only when the machine has enough memory; a normal 128 GB AMD395 host
does not have room for two copies of this BF16 model.
