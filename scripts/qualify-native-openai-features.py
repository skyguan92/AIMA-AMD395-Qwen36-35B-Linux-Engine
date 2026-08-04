#!/usr/bin/env python3
"""Qualify native SSE streaming and OpenAI function-tool interoperability."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import selectors
import socket
import struct
import subprocess
import time
from typing import Any


MODEL_ID = "aima-amd395-qwen36-35b"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publicize(
    value: Any, *, engine: Path, model_dir: Path, output_dir: Path
) -> Any:
    if isinstance(value, str):
        return (
            value.replace(str(engine), "${AIMA_ENGINE}")
            .replace(str(model_dir), "${AIMA_MODEL_DIR}")
            .replace(str(output_dir), "${AIMA_OUTPUT_DIR}")
            .replace(str(ROOT), "${AIMA_REPO_ROOT}")
        )
    if isinstance(value, list):
        return [
            publicize(
                item,
                engine=engine,
                model_dir=model_dir,
                output_dir=output_dir,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: publicize(
                item,
                engine=engine,
                model_dir=model_dir,
                output_dir=output_dir,
            )
            for key, item in value.items()
        }
    return value


def read_lifecycle(
    process: subprocess.Popen[str], timeout_seconds: float
) -> dict[str, Any]:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    events = selector.select(timeout_seconds)
    selector.close()
    if not events:
        raise RuntimeError("native server lifecycle event timed out")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"native server exited early: {process.poll()}")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError("native server lifecycle event is not an object")
    return value


def probe_prompt(
    engine: Path, model_dir: Path, request: dict[str, Any]
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(engine),
            "chat-template-probe",
            "--model-dir",
            str(model_dir),
            "--request-json",
            json.dumps(request, ensure_ascii=False, separators=(",", ":")),
            "--disable-thinking",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict) or value.get("complete") is not True:
        raise RuntimeError("native chat-template probe failed")
    return value


def exact_request(
    engine: Path,
    model_dir: Path,
    context_tokens: int,
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = copy.deepcopy(request)
    user_indices = [
        index
        for index, message in enumerate(base["messages"])
        if message.get("role") == "user"
    ]
    if not user_indices:
        raise RuntimeError("fixture requires a user message")
    user_index = user_indices[-1]
    initial = probe_prompt(engine, model_dir, base)
    initial_tokens = len(initial["token_ids"])
    if initial_tokens > context_tokens:
        raise RuntimeError(
            f"fixture already has {initial_tokens} tokens, above {context_tokens}"
        )
    estimate = context_tokens - initial_tokens
    for count in range(max(0, estimate - 8), estimate + 17):
        candidate = copy.deepcopy(base)
        candidate["messages"][user_index]["content"] += " test" * count
        probed = probe_prompt(engine, model_dir, candidate)
        tokens = len(probed["token_ids"])
        if tokens == context_tokens:
            return candidate, probed
        if tokens > context_tokens and count >= estimate:
            break
    raise RuntimeError(f"could not construct exact q{context_tokens} fixture")


def request_json(
    port: int,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 300,
) -> tuple[int, dict[str, Any]]:
    body = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        if payload is not None
        else None
    )
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    connection.request(
        method,
        path,
        body=body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    response = connection.getresponse()
    raw = response.read()
    status = response.status
    connection.close()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {path} returned non-object JSON")
    return status, value


def request_stream(
    port: int, payload: dict[str, Any]
) -> dict[str, Any]:
    body = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    started = time.monotonic()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
    )
    response = connection.getresponse()
    content_type = response.getheader("Content-Type")
    transfer_encoding = response.getheader("Transfer-Encoding")
    events: list[dict[str, Any]] = []
    event_times_ms: list[float] = []
    done_ms: float | None = None
    while True:
        line = response.readline()
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        elapsed = (time.monotonic() - started) * 1000
        if data == b"[DONE]":
            done_ms = elapsed
            break
        value = json.loads(data)
        if not isinstance(value, dict):
            raise RuntimeError("SSE data is not a JSON object")
        events.append(value)
        event_times_ms.append(elapsed)
    status = response.status
    connection.close()

    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    first_content_ms: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    role_seen = False
    for event, elapsed in zip(events, event_times_ms):
        if event.get("usage") is not None:
            usage = event["usage"]
        if event.get("aima_amd395") is not None:
            metrics = event["aima_amd395"]
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("role") == "assistant":
                role_seen = True
            if delta.get("content") is not None:
                if first_content_ms is None:
                    first_content_ms = elapsed
                content.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                index = int(call["index"])
                target = calls.setdefault(
                    index,
                    {
                        "id": call.get("id"),
                        "type": call.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    },
                )
                function = call.get("function") or {}
                target["function"]["name"] += function.get("name", "")
                target["function"]["arguments"] += function.get("arguments", "")
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    return {
        "status": status,
        "content_type": content_type,
        "transfer_encoding": transfer_encoding,
        "events": events,
        "role_seen": role_seen,
        "content": "".join(content),
        "tool_calls": [calls[index] for index in sorted(calls)],
        "first_content_ms": first_content_ms,
        "done_ms": done_ms,
        "finish_reason": finish_reason,
        "usage": usage,
        "metrics": metrics,
    }


def disconnect_after_role(port: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200 or not response.readline().startswith(b"data: "):
        raise RuntimeError("disconnect fixture did not receive initial SSE role")
    if connection.sock is not None:
        connection.sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine", type=Path, default=Path("build/native/aima-engine-native")
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18123)
    parser.add_argument("--context-tokens", type=int, default=1024)
    cli = parser.parse_args()

    engine = cli.engine.expanduser().resolve()
    model_dir = cli.model_dir.expanduser().resolve()
    output = cli.output.expanduser().resolve()
    engine_sha256 = sha256(engine)
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
    plain = {
        "messages": [{"role": "user", "content": "Say hello briefly."}]
    }
    plain_probe = probe_prompt(engine, model_dir, plain)
    if len(plain_probe["token_ids"]) >= cli.context_tokens:
        raise RuntimeError("variable-prompt fixture is not below the AOT context")
    tool_request, tool_probe = exact_request(
        engine,
        model_dir,
        cli.context_tokens,
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What is the weather in Paris? "
                        "Use the provided tool."
                    ),
                }
            ],
            "tools": [tool],
            "tool_choice": "auto",
        },
    )
    history, history_probe = exact_request(
        engine,
        model_dir,
        cli.context_tokens,
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What is the weather in Paris? "
                        "Use the provided tool."
                    ),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fixture",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_fixture",
                    "content": (
                        '{"temperature_c":20,"condition":"sunny"}'
                    ),
                },
                {
                    "role": "user",
                    "content": "Summarize the result in one sentence.",
                },
            ],
            "tools": [tool],
            "tool_choice": "auto",
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = output.with_suffix(".stderr.txt")
    load_report = output.with_suffix(".load.json")
    command = [
        str(engine),
        "serve",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        str(cli.context_tokens),
        "--host",
        "127.0.0.1",
        "--port",
        str(cli.port),
        "--report",
        str(load_report),
    ]
    with stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
        try:
            ready = read_lifecycle(process, 180)
            plain_stream_request = {
                "model": MODEL_ID,
                **plain,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 32,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            plain_stream = request_stream(cli.port, plain_stream_request)
            plain_nonstream_request = {
                key: value
                for key, value in plain_stream_request.items()
                if key not in {"stream", "stream_options"}
            }
            plain_status, plain_nonstream = request_json(
                cli.port,
                "POST",
                "/v1/chat/completions",
                plain_nonstream_request,
            )
            ordinary_turn_status, ordinary_turn = request_json(
                cli.port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": MODEL_ID,
                    "messages": [
                        plain["messages"][0],
                        {
                            "role": "assistant",
                            "content": plain_nonstream["choices"][0][
                                "message"
                            ]["content"],
                        },
                        {
                            "role": "user",
                            "content": "Now answer with one word: ready?",
                        },
                    ],
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": 8,
                },
            )
            tool_payload = {
                "model": MODEL_ID,
                **tool_request,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 128,
            }
            tool_status, tool_nonstream = request_json(
                cli.port, "POST", "/v1/chat/completions", tool_payload
            )
            tool_stream = request_stream(
                cli.port,
                {
                    **tool_payload,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            history_status, history_response = request_json(
                cli.port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": MODEL_ID,
                    **history,
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": 64,
                },
            )
            post_long_status, post_long_short = request_json(
                cli.port,
                "POST",
                "/v1/chat/completions",
                plain_nonstream_request,
            )
            invalid_status, invalid_response = request_json(
                cli.port,
                "POST",
                "/v1/chat/completions",
                {
                    "model": MODEL_ID,
                    **tool_request,
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "not_available"},
                    },
                    "max_tokens": 8,
                },
            )
            health_before_status, health_before = request_json(
                cli.port, "GET", "/health"
            )
            disconnect_after_role(
                cli.port,
                {
                    **plain_stream_request,
                    "max_tokens": 512,
                    "stream_options": {"include_usage": False},
                },
            )
            health_after_status = 0
            health_after: dict[str, Any] = {}
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    health_after_status, health_after = request_json(
                        cli.port, "GET", "/health", timeout=2
                    )
                    break
                except (OSError, TimeoutError):
                    time.sleep(0.05)
            shutdown_status, shutdown = request_json(
                cli.port, "POST", "/shutdown"
            )
            stopped = read_lifecycle(process, 30)
            returncode = process.wait(timeout=30)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

    plain_message = plain_nonstream["choices"][0]["message"]
    plain_metrics = plain_nonstream["aima_amd395"]
    plain_pass = bool(
        plain_stream["status"] == 200
        and plain_stream["content_type"].startswith("text/event-stream")
        and plain_stream["transfer_encoding"] == "chunked"
        and plain_stream["role_seen"] is True
        and plain_stream["content"] == plain_message["content"]
        and plain_stream["usage"] == plain_nonstream["usage"]
        and plain_stream["first_content_ms"] is not None
        and plain_stream["done_ms"] is not None
        and plain_stream["first_content_ms"] < plain_stream["done_ms"]
        and plain_stream["metrics"]["output_token_ids_sha256"]
        == plain_metrics["output_token_ids_sha256"]
        and plain_stream["metrics"]["prefix_cache"]["lookup"] == "miss"
        and plain_stream["metrics"]["prompt_execution"]
        == "cold-decode-fallback"
        and plain_metrics["prefix_cache"]["lookup"] == "exact"
        and plain_metrics["prompt_execution"] == "prefix-cache-exact"
    )
    ordinary_turn_metrics = ordinary_turn.get("aima_amd395") or {}
    ordinary_turn_pass = bool(
        ordinary_turn_status == 200
        and ordinary_turn_metrics.get("prefix_cache", {}).get("lookup")
        == "miss"
        and ordinary_turn_metrics.get("prompt_execution")
        == "cold-decode-fallback"
        and ordinary_turn.get("usage", {}).get("prompt_tokens", 0)
        > len(plain_probe["token_ids"])
        and isinstance(
            ordinary_turn["choices"][0]["message"]["content"], str
        )
    )
    post_long_metrics = post_long_short.get("aima_amd395") or {}
    post_long_pass = bool(
        post_long_status == 200
        and post_long_metrics.get("prefix_cache", {}).get("lookup") == "miss"
        and post_long_metrics.get("prompt_execution")
        == "cold-decode-fallback"
        and post_long_metrics.get("output_token_ids_sha256")
        == plain_metrics["output_token_ids_sha256"]
    )
    nonstream_call = tool_nonstream["choices"][0]["message"]["tool_calls"][0]
    stream_call = tool_stream["tool_calls"][0]
    tool_pass = bool(
        tool_status == 200
        and tool_nonstream["choices"][0]["finish_reason"] == "tool_calls"
        and nonstream_call["function"]["name"] == "get_weather"
        and json.loads(nonstream_call["function"]["arguments"])
        == {"city": "Paris"}
        and tool_stream["status"] == 200
        and tool_stream["finish_reason"] == "tool_calls"
        and stream_call["function"]["name"]
        == nonstream_call["function"]["name"]
        and json.loads(stream_call["function"]["arguments"])
        == json.loads(nonstream_call["function"]["arguments"])
        and tool_stream["metrics"]["output_token_ids_sha256"]
        == tool_nonstream["aima_amd395"]["output_token_ids_sha256"]
        and tool_stream["metrics"]["prefix_cache"]["lookup"] == "exact"
    )
    history_pass = bool(
        history_status == 200
        and isinstance(
            history_response["choices"][0]["message"]["content"], str
        )
        and history_response["choices"][0]["message"]["content"]
    )
    disconnect_pass = bool(
        health_before_status == 200
        and health_after_status == 200
        and health_after["status"] == "ok"
        and health_after["served"] == health_before["served"]
    )
    validation_pass = bool(
        invalid_status == 400
        and invalid_response["error"]["code"] == "bad_request"
    )
    lifecycle_pass = bool(
        ready["event"] == "ready"
        and ready["runtime_python"] is False
        and shutdown_status == 200
        and shutdown["status"] == "shutting_down"
        and stopped["event"] == "stopped"
        and stopped["model_loads"] == 1
        and stopped["served"] == 7
        and returncode == 0
    )
    result = {
        "schema": "aima-amd395-qwen36/native-openai-features/v2",
        "complete": True,
        "qualified": all(
            (
                plain_pass,
                ordinary_turn_pass,
                post_long_pass,
                tool_pass,
                history_pass,
                disconnect_pass,
                validation_pass,
                lifecycle_pass,
            )
        ),
        "engine": {
            "path": str(engine),
            "sha256": engine_sha256,
        },
        "model_dir": str(model_dir),
        "host": {
            "hostname": os.uname().nodename,
            "release": os.uname().release,
            "machine": os.uname().machine,
        },
        "context_tokens": cli.context_tokens,
        "fixtures": {
            "plain_tokens": len(plain_probe["token_ids"]),
            "tool_tokens": len(tool_probe["token_ids"]),
            "history_tokens": len(history_probe["token_ids"]),
        },
        "streaming": {
            "content_type": plain_stream["content_type"],
            "transfer_encoding": plain_stream["transfer_encoding"],
            "first_content_ms": plain_stream["first_content_ms"],
            "done_ms": plain_stream["done_ms"],
            "content_matches_nonstream": (
                plain_stream["content"] == plain_message["content"]
            ),
            "token_sha256_matches_nonstream": (
                plain_stream["metrics"]["output_token_ids_sha256"]
                == plain_metrics["output_token_ids_sha256"]
            ),
            "cold_prefix_lookup": plain_stream["metrics"]["prefix_cache"][
                "lookup"
            ],
            "repeat_prefix_lookup": plain_metrics["prefix_cache"]["lookup"],
            "usage": plain_stream["usage"],
            "pass": plain_pass,
        },
        "variable_prompts": {
            "short_cold_tokens": len(plain_probe["token_ids"]),
            "short_cold_execution": plain_stream["metrics"][
                "prompt_execution"
            ],
            "short_exact_execution": plain_metrics["prompt_execution"],
            "ordinary_turn_status": ordinary_turn_status,
            "ordinary_turn_usage": ordinary_turn.get("usage"),
            "ordinary_turn_execution": ordinary_turn_metrics.get(
                "prompt_execution"
            ),
            "ordinary_turn_prefix_lookup": ordinary_turn_metrics.get(
                "prefix_cache", {}
            ).get("lookup"),
            "post_long_status": post_long_status,
            "post_long_execution": post_long_metrics.get(
                "prompt_execution"
            ),
            "post_long_prefix_lookup": post_long_metrics.get(
                "prefix_cache", {}
            ).get("lookup"),
            "post_long_token_sha256_matches": (
                post_long_metrics.get("output_token_ids_sha256")
                == plain_metrics["output_token_ids_sha256"]
            ),
            "pass": ordinary_turn_pass and post_long_pass,
        },
        "tools": {
            "nonstream": nonstream_call,
            "stream": stream_call,
            "finish_reason": tool_stream["finish_reason"],
            "token_sha256_matches_nonstream": (
                tool_stream["metrics"]["output_token_ids_sha256"]
                == tool_nonstream["aima_amd395"][
                    "output_token_ids_sha256"
                ]
            ),
            "history_response": history_response["choices"][0]["message"],
            "pass": tool_pass and history_pass,
        },
        "disconnect": {
            "served_before": health_before["served"],
            "served_after": health_after.get("served"),
            "server_healthy": health_after.get("status") == "ok",
            "pass": disconnect_pass,
        },
        "validation": {
            "unknown_forced_tool_status": invalid_status,
            "error": invalid_response.get("error"),
            "pass": validation_pass,
        },
        "lifecycle": {
            "ready": ready,
            "stopped": stopped,
            "pass": lifecycle_pass,
        },
        "artifacts": {
            "load_report": str(load_report),
            "load_report_sha256": sha256(load_report),
            "stderr": str(stderr_path),
        },
    }
    published = publicize(
        result,
        engine=engine,
        model_dir=model_dir,
        output_dir=output.parent,
    )
    output.write_text(
        json.dumps(published, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": result["qualified"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    if not result["qualified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
