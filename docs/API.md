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
- `messages`: leading system strings followed by one or more user strings;
- `max_tokens` or `max_completion_tokens`: positive integer;
- `temperature`: exactly `0`;
- `top_p`: exactly `1`;
- `n`: exactly `1`;
- `stream`: `false`.

Not supported:

- streaming;
- custom stop values;
- assistant/tool messages as input;
- tools, functions or structured response formats;
- stochastic sampling;
- batching or concurrent execution.

The server applies the qualified Qwen chat template with thinking disabled.
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

The response follows the non-streaming OpenAI shape: `id`, `object`,
`created`, `model`, `choices` and `usage`. A terminal EOS token counts in
`usage.completion_tokens` but is omitted from visible assistant content.

`aima_amd395` adds:

- runtime and request index;
- `model_loads`;
- prefill/decode throughput and total latency;
- TTFT;
- output-token SHA-256;
- prefix lookup type, matched/suffix token counts, cumulative hits/misses,
  state-transfer bytes, suffix launch counts and suffix wall time.

Example:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aima-amd395-qwen36-35b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "PROMPT_PREPARED_TO_THE_STATIC_TOKEN_LENGTH"}
    ],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 512
  }'
```

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
