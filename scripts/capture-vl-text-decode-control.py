#!/usr/bin/env python3
"""Capture one text-only control at the G4 HTTP/SSE timing boundary."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import socket
from pathlib import Path
import time
from typing import Any, Mapping
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_PATH = Path(__file__).with_name("capture-vl-performance-request.py")
SPEC = importlib.util.spec_from_file_location(
    "vl_performance_request_capture", CAPTURE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the frozen G4 request capture helpers")
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


SCHEMA = "aima-amd395-qwen36/vl-text-decode-control-sample/v1"
TIMING_BOUNDARY = (
    "perf_counter_ns immediately before HTTP open through SSE stream EOF; "
    "TTFT ends at the first semantic delta; decode throughput is "
    "(completion_tokens - 1) / (total_seconds - ttft_seconds)"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-logical-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--text-padding-tokens", type=int, required=True)
    parser.add_argument("--expected-prompt-tokens", type=int, required=True)
    parser.add_argument("--expected-completion-tokens", type=int, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if not args.request.is_file():
        raise SystemExit(f"request file is missing: {args.request}")
    if capture._RUN_ID.fullmatch(args.benchmark_id) is None:
        raise SystemExit("benchmark ID contains unsupported characters")
    if (
        args.request_logical_path.startswith("/")
        or ".." in Path(args.request_logical_path).parts
    ):
        raise SystemExit("request logical path must be a repository-relative path")
    if not capture.pid_is_live(args.server_pid):
        raise SystemExit("server PID is not live")
    if (
        not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
        or args.timeout_seconds > 3600
    ):
        raise SystemExit("timeout must be positive and at most 3600 seconds")
    if (
        args.expected_prompt_tokens <= 0
        or args.expected_prompt_tokens > 262_144
        or args.expected_completion_tokens <= 1
        or args.expected_completion_tokens > 1024
        or args.expected_prompt_tokens + args.expected_completion_tokens
        > 262_144
    ):
        raise SystemExit("expected text control shape is outside the model window")
    try:
        endpoint = capture.loopback_endpoint(args.endpoint)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    request_template = args.request.read_bytes()
    payload = json.loads(request_template)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("messages"), list
    ):
        raise SystemExit("text control request must contain a messages array")
    if capture.count_string(payload, capture.MEDIA_ROOT_PLACEHOLDER) != 0:
        raise SystemExit("text control request cannot contain a media root")
    if capture.count_string(payload, capture.PROMPT_NONCE_PLACEHOLDER) != 0:
        raise SystemExit("text control request cannot contain a prompt nonce")
    if capture.media_components(payload, ROOT):
        raise SystemExit("text control request cannot contain media")
    try:
        capture.append_text_padding(payload, args.text_padding_tokens)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if float(payload.get("temperature", 0)) != 0.0:
        raise SystemExit("G4 text control must use temperature=0")
    payload["temperature"] = 0
    payload["model"] = args.model
    payload["stream"] = True
    stream_options = payload.setdefault("stream_options", {})
    if not isinstance(stream_options, dict):
        raise SystemExit("stream_options must be an object")
    stream_options["include_usage"] = True
    output_limit = payload.get(
        "max_completion_tokens", payload.get("max_tokens")
    )
    if output_limit != args.expected_completion_tokens:
        raise SystemExit("request output limit differs from expected completion tokens")
    request_body = capture.canonical_bytes(payload)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        data=request_body,
        headers={
            "Content-Type": "application/json",
            capture.BENCHMARK_HEADER: args.benchmark_id,
        },
        method="POST",
    )
    event_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    output_bytes = 0
    semantic_chunks = 0
    event_count = 0
    first_semantic_ns: int | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    native_metrics: dict[str, Any] | None = None
    stream_error: dict[str, Any] | None = None
    done = False
    started_ns = time.perf_counter_ns()
    finished_ns = started_ns
    status_code: int | None = None
    capture_error: dict[str, Any] | None = None
    with capture.MemorySampler(args.server_pid) as memory:
        try:
            response = opener.open(request, timeout=args.timeout_seconds)
        except urllib.error.HTTPError as error:
            detail = error.read()
            status_code = error.code
            capture_error = {
                **capture.normalized_capture_error("http_error", error),
                "body_bytes": len(detail),
                "body_sha256": sha256_bytes(detail),
            }
        except Exception as error:
            capture_error = capture.normalized_capture_error(
                "transport_error", error
            )
        else:
            try:
                with response:
                    status_code = response.status
                    if status_code != 200:
                        raise RuntimeError(
                            f"performance endpoint returned HTTP {status_code}"
                        )
                    for raw_line in response:
                        line = raw_line.strip()
                        if not line or not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            done = True
                            continue
                        event = json.loads(data)
                        if not isinstance(event, dict):
                            raise RuntimeError("SSE event is not an object")
                        encoded_event = capture.canonical_bytes(event)
                        event_digest.update(encoded_event)
                        event_digest.update(b"\n")
                        event_count += 1
                        semantic = capture.semantic_delta(event)
                        if semantic:
                            if first_semantic_ns is None:
                                first_semantic_ns = time.perf_counter_ns()
                            encoded = semantic.encode("utf-8")
                            output_digest.update(encoded)
                            output_bytes += len(encoded)
                            semantic_chunks += 1
                        if isinstance(event.get("usage"), dict):
                            usage = dict(event["usage"])
                        if isinstance(event.get("aima_amd395"), dict):
                            native_metrics = dict(event["aima_amd395"])
                        normalized_error = capture.normalized_stream_error(event)
                        if normalized_error is not None and stream_error is None:
                            stream_error = normalized_error
                        choices = event.get("choices")
                        if isinstance(choices, list):
                            for choice in choices:
                                if not isinstance(choice, Mapping):
                                    continue
                                value = choice.get("finish_reason")
                                if isinstance(value, str):
                                    finish_reason = value
            except Exception as error:
                capture_error = capture.normalized_capture_error(
                    "stream_error", error
                )
            finally:
                finished_ns = time.perf_counter_ns()

    total_seconds = (finished_ns - started_ns) / 1e9
    ttft_seconds = (
        (first_semantic_ns - started_ns) / 1e9
        if first_semantic_ns is not None
        else None
    )
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage, dict) else None
    )
    decode_tokens_per_second: float | None = None
    if (
        isinstance(completion_tokens, int)
        and completion_tokens > 1
        and isinstance(ttft_seconds, float)
        and total_seconds > ttft_seconds
    ):
        decode_tokens_per_second = (completion_tokens - 1) / (
            total_seconds - ttft_seconds
        )
    native_shape_exact = (
        isinstance(native_metrics, Mapping)
        and native_metrics.get("prompt_tokens") == args.expected_prompt_tokens
        and native_metrics.get("completion_tokens")
        == args.expected_completion_tokens
    )
    complete = (
        done
        and status_code == 200
        and capture_error is None
        and stream_error is None
        and first_semantic_ns is not None
        and finish_reason in {"stop", "length", "tool_calls"}
        and prompt_tokens == args.expected_prompt_tokens
        and completion_tokens == args.expected_completion_tokens
        and isinstance(decode_tokens_per_second, float)
        and decode_tokens_per_second > 0.0
        and native_shape_exact
    )
    result = {
        "schema": SCHEMA,
        "captured_at": capture.utc_now(),
        "complete": complete,
        "benchmark_id": args.benchmark_id,
        "engine_role": "candidate",
        "endpoint": "${AIMA_LOOPBACK_ENDPOINT}",
        "server_pid": args.server_pid,
        "host": {"hostname": socket.gethostname()},
        "timing_boundary": TIMING_BOUNDARY,
        "request": {
            "path": args.request_logical_path,
            "template_sha256": sha256_bytes(request_template),
            "sha256": sha256_bytes(request_body),
            "bytes": len(request_body),
            "summary": capture.request_summary(payload),
            "media": [],
            "text_padding": {
                "tokens": args.text_padding_tokens,
                "unit_sha256": sha256_bytes(capture.TEXT_PADDING_UNIT.encode("ascii")),
                "frozen_single_token_id": capture.TEXT_PADDING_TOKEN_ID,
            },
            "expected_prompt_tokens": args.expected_prompt_tokens,
            "expected_completion_tokens": args.expected_completion_tokens,
        },
        "response": {
            "status_code": status_code,
            "sse_done": done,
            "event_count": event_count,
            "event_stream_sha256": event_digest.hexdigest(),
            "semantic_chunks": semantic_chunks,
            "content_bytes": output_bytes,
            "content_sha256": output_digest.hexdigest(),
            "finish_reason": finish_reason,
            "usage": usage,
            "error": stream_error,
            "capture_error": capture_error,
        },
        "timings": {
            "ttft_seconds": ttft_seconds,
            "total_seconds": total_seconds,
            "decode_tokens_per_second": decode_tokens_per_second,
        },
        "memory": memory.result(),
        "native_metrics": native_metrics,
    }
    capture.atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "complete": complete},
            sort_keys=True,
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
