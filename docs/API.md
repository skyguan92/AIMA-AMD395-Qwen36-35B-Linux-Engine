# CLI and HTTP API

## CLI

```text
aima-engine verify
aima-engine doctor [--deep]
aima-engine prepare-images ...
aima-engine register-images ...
aima-engine serve ...
aima-engine status
aima-engine models
aima-engine chat ...
aima-engine shutdown
```

Configuration may be passed explicitly or through:

- `AIMA_RUNTIME_PYTHON`
- `AIMA_MODEL_DIR`
- `AIMA_IMAGE_MANIFEST`
- `AIMA_AOTRITON_LIBRARY` (optional exact-library override)

Explicit CLI flags take precedence over environment variables.

## HTTP endpoints

### `GET /health`

Returns readiness, residency, uptime, served-request count and source commit.
HTTP 200 means the model and resident state are loaded.

### `GET /v1/models`

Returns the single model id `aima-amd395-qwen36-35b`.

### `POST /v1/chat/completions`

Supported request fields:

- `model`
- `messages`: ordered system messages followed by one or more user string messages
- `max_tokens` or `max_completion_tokens`
- `temperature`: must equal `0`
- `top_p`: must equal `1`
- `n`: must equal `1`
- `stream`: must be `false`

Unsupported in v1.0.0:

- streaming;
- custom stop strings;
- assistant/tool messages as input;
- tools, functions, structured response formats;
- stochastic sampling;
- request batching or concurrent execution.

The server serializes inference with a process lock. Model weights, derived
layouts, KV state and the one-entry prefix cache remain resident between
requests.

### Response

The standard response contains `choices`, `finish_reason` and `usage`.
`usage.completion_tokens` counts every sampled token, including a terminal stop
token that is intentionally omitted from assistant-visible content. This
matches the qualified vLLM contract.

An `aima_amd395` extension reports performance and cache metadata, including:

- TTFT, decode throughput and total latency;
- prefix lookup kind and matched/suffix token counts;
- resident cache hits/misses;
- runtime source and artifact paths.

### `POST /shutdown`

Stops the server after returning a confirmation. There is no built-in
authentication, so this endpoint is safe only on a trusted interface.

## Errors

Invalid requests return an OpenAI-shaped `error` object with a 4xx status.
Runtime failures return HTTP 500 and preserve a request-local error artifact
under the configured output directory.

## Examples

```bash
./aima-engine chat \
  --endpoint http://127.0.0.1:8000 \
  --system "Answer concisely." \
  --max-tokens 96 \
  "What is speculative decoding?"
```

Use `--json` to retain the complete response and AIMA metadata.
