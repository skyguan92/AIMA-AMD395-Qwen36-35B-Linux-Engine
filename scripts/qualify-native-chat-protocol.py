#!/usr/bin/env python3
"""Qualify native thinking and bounded tool-call progress behavior."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import selectors
import subprocess
import sys
import time
from typing import Any


MODEL_ID = "aima-amd395-qwen36-35b"
ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_host_paths(
    value: Any, replacements: list[tuple[Path, str]]
) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_host_paths(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_host_paths(item, replacements) for item in value]
    if not isinstance(value, str):
        return value
    sanitized = value
    for path, placeholder in sorted(
        replacements, key=lambda item: len(str(item[0])), reverse=True
    ):
        root = str(path)
        if sanitized == root:
            return placeholder
        if sanitized.startswith(root + "/"):
            return placeholder + sanitized[len(root):]
    if sanitized.startswith(("/home/", "/Users/", "/data/", "/tmp/")):
        return "${AIMA_HOST_PATH}/" + Path(sanitized).name
    return sanitized


def require_gpu_idle() -> None:
    occupied = subprocess.run(
        ["fuser", "/dev/kfd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if occupied.returncode != 0:
        return
    subprocess.run(["fuser", "-v", "/dev/kfd"], check=False)
    raise RuntimeError("/dev/kfd is owned by another process")


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
    require(isinstance(value, dict), "server lifecycle event is not an object")
    return value


def request_json(
    port: int,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 300,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    status = response.status
    connection.close()
    value = json.loads(raw)
    require(isinstance(value, dict), f"{method} {path} returned non-object JSON")
    return status, value


def request_stream(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    request = dict(payload)
    request["stream"] = True
    request["stream_options"] = {"include_usage": True}
    body = json.dumps(
        request, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
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
    status = response.status
    content_type = response.getheader("Content-Type")
    transfer_encoding = response.getheader("Transfer-Encoding")
    events: list[dict[str, Any]] = []
    done = False
    while True:
        line = response.readline()
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            done = True
            break
        value = json.loads(data)
        require(isinstance(value, dict), "SSE event is not an object")
        events.append(value)
    connection.close()

    reasoning: list[str] = []
    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    delta_order: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    role_seen = False
    for event in events:
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        if isinstance(event.get("aima_amd395"), dict):
            metrics = event["aima_amd395"]
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("role") == "assistant":
                role_seen = True
            if "reasoning_content" in delta:
                delta_order.append("reasoning_content")
                reasoning.append(delta["reasoning_content"])
            if "content" in delta:
                delta_order.append("content")
                content.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                delta_order.append("tool_calls")
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
                target["function"]["arguments"] += function.get(
                    "arguments", ""
                )
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    return {
        "status": status,
        "content_type": content_type,
        "transfer_encoding": transfer_encoding,
        "event_count": len(events),
        "done": done,
        "role_seen": role_seen,
        "delta_order": delta_order,
        "reasoning_content": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": [calls[index] for index in sorted(calls)],
        "finish_reason": finish_reason,
        "usage": usage,
        "metrics": metrics,
    }


def tool_signature(call: dict[str, Any]) -> tuple[str, Any]:
    function = call["function"]
    return function["name"], json.loads(function["arguments"])


def response_summary(response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning_content")
    content = message.get("content")
    metrics = response["aima_amd395"]
    return {
        "finish_reason": choice["finish_reason"],
        "usage": response["usage"],
        "reasoning_content_present": "reasoning_content" in message,
        "reasoning_content_chars": len(reasoning or ""),
        "reasoning_content_sha256": (
            sha256_text(reasoning) if reasoning is not None else None
        ),
        "content": content,
        "tool_calls": message.get("tool_calls", []),
        "output_token_ids_sha256": metrics["output_token_ids_sha256"],
        "thinking": metrics["thinking"],
        "tool_progress": metrics.get("tool_progress"),
    }


def stream_summary(response: dict[str, Any]) -> dict[str, Any]:
    reasoning = response["reasoning_content"]
    metrics = response["metrics"]
    return {
        "status": response["status"],
        "content_type": response["content_type"],
        "transfer_encoding": response["transfer_encoding"],
        "event_count": response["event_count"],
        "done": response["done"],
        "role_seen": response["role_seen"],
        "delta_order": response["delta_order"],
        "finish_reason": response["finish_reason"],
        "usage": response["usage"],
        "reasoning_content_chars": len(reasoning),
        "reasoning_content_sha256": sha256_text(reasoning),
        "content": response["content"],
        "tool_calls": response["tool_calls"],
        "output_token_ids_sha256": metrics["output_token_ids_sha256"],
        "thinking": metrics["thinking"],
        "tool_progress": metrics.get("tool_progress"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        type=Path,
        default=ROOT / "build/native/aima-engine-native",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT
        / Path(
            "benchmarks/fixtures/vl-capability-v0.1.0/image-rgb-256.png"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18127)
    parser.add_argument("--expected-engine-sha256")
    parser.add_argument("--expected-source-commit")
    cli = parser.parse_args()

    engine = cli.engine.expanduser().resolve()
    model_dir = cli.model_dir.expanduser().resolve()
    image = cli.image.expanduser().resolve()
    output = cli.output.expanduser().resolve()
    require(engine.is_file(), "engine binary is missing")
    require(image.is_file(), "qualification image is missing")
    engine_sha256 = sha256_file(engine)
    if cli.expected_engine_sha256:
        require(
            engine_sha256 == cli.expected_engine_sha256,
            "engine SHA-256 does not match the expected binary",
        )
    build_info = json.loads(
        subprocess.run(
            [str(engine), "--build-info"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if cli.expected_source_commit:
        require(
            build_info.get("source_commit") == cli.expected_source_commit,
            "embedded source commit does not match",
        )

    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
    basic = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "你好"}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 64,
    }
    history_thinking = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Get Paris weather."},
            {
                "role": "assistant",
                "reasoning_content": "I should use the weather tool.",
                "content": None,
                "tool_calls": [
                    {
                        "id": "history-weather",
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
                "tool_call_id": "history-weather",
                "content": '{"temperature_c":20,"condition":"sunny"}',
            },
            {"role": "user", "content": "Summarize in one short sentence."},
        ],
        "tools": [tool],
        "tool_choice": "auto",
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 256,
        "thinking": {"type": "enabled", "budget_tokens": 256},
    }
    thinking_tool = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Call get_weather for Paris now."}
        ],
        "tools": [tool],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 512,
        "thinking": {"type": "enabled", "budget_tokens": 512},
    }
    duplicate = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Call get_weather exactly twice for Paris. Both calls "
                    "must use the identical city Paris argument."
                ),
            }
        ],
        "tools": [tool],
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 128,
    }
    different = {
        **duplicate,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Call get_weather twice: once for Paris and once for "
                    "Tokyo. Use two separate calls."
                ),
            }
        ],
    }
    exhausted = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": "Call get_weather for Paris now. Use city exactly Paris.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "hist-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "hist-1", "content": "Exit code: 0"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "hist-2",
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
                "tool_call_id": "hist-2",
                "content": '{"error":"proxy unavailable"}',
            },
            {
                "role": "user",
                "content": "Retry the exact same get_weather action for Paris.",
            },
        ],
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "parallel_tool_calls": True,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 128,
    }
    vl = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the image in one short sentence.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image.as_uri()},
                    },
                ],
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 128,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = output.with_suffix(".stderr.txt")
    load_report = output.with_suffix(".load.json")
    command = [
        str(engine),
        "serve",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        "8192",
        "--host",
        "127.0.0.1",
        "--port",
        str(cli.port),
        "--allowed-local-media-path",
        str(image.parent),
        "--report",
        str(load_report),
    ]
    require_gpu_idle()
    observations: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    with stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=stderr, text=True
        )
        try:
            ready = read_lifecycle(process, 180)
            require(ready.get("event") == "ready", "server did not become ready")

            default_status, default_response = request_json(
                cli.port, "POST", "/v1/chat/completions", basic
            )
            disabled_request = {
                **basic,
                "thinking": {"type": "disabled", "budget_tokens": 64},
            }
            disabled_status, disabled_response = request_json(
                cli.port, "POST", "/v1/chat/completions", disabled_request
            )
            checks["omitted_thinking_is_backward_compatible"] = (
                default_status == 200
                and disabled_status == 200
                and default_response["choices"][0]["message"]
                == disabled_response["choices"][0]["message"]
                and default_response["aima_amd395"]["output_token_ids_sha256"]
                == disabled_response["aima_amd395"]["output_token_ids_sha256"]
                and "reasoning_content"
                not in default_response["choices"][0]["message"]
                and "reasoning_content"
                not in disabled_response["choices"][0]["message"]
            )

            history_status, history_response = request_json(
                cli.port, "POST", "/v1/chat/completions", history_thinking
            )
            history_stream = request_stream(cli.port, history_thinking)
            history_message = history_response["choices"][0]["message"]
            history_reasoning = history_message["reasoning_content"]
            history_content = history_message["content"]
            checks["thinking_stream_nonstream_and_history"] = (
                history_status == 200
                and history_stream["status"] == 200
                and bool(history_reasoning)
                and bool(history_content)
                and history_stream["reasoning_content"] == history_reasoning
                and history_stream["content"] == history_content
                and history_stream["metrics"]["output_token_ids_sha256"]
                == history_response["aima_amd395"]["output_token_ids_sha256"]
                and "<think>" not in history_reasoning + history_content
                and "</think>" not in history_reasoning + history_content
                and (
                    "content" not in history_stream["delta_order"]
                    or history_stream["delta_order"].index("reasoning_content")
                    < history_stream["delta_order"].index("content")
                )
            )

            tool_status, tool_response = request_json(
                cli.port, "POST", "/v1/chat/completions", thinking_tool
            )
            tool_stream = request_stream(cli.port, thinking_tool)
            tool_message = tool_response["choices"][0]["message"]
            checks["thinking_before_tool_call"] = (
                tool_status == 200
                and tool_stream["status"] == 200
                and bool(tool_message["reasoning_content"])
                and len(tool_message.get("tool_calls", [])) == 1
                and tool_stream["reasoning_content"]
                == tool_message["reasoning_content"]
                and tool_signature(tool_stream["tool_calls"][0])
                == tool_signature(tool_message["tool_calls"][0])
                and tool_stream["metrics"]["output_token_ids_sha256"]
                == tool_response["aima_amd395"]["output_token_ids_sha256"]
                and "<think>"
                not in tool_stream["reasoning_content"] + tool_stream["content"]
                and "</think>"
                not in tool_stream["reasoning_content"] + tool_stream["content"]
            )

            invalid_cases = {
                "invalid_type": {**basic, "thinking": {"type": "automatic"}},
                "zero_budget": {
                    **basic,
                    "thinking": {"type": "enabled", "budget_tokens": 0},
                },
                "budget_above_max": {
                    **basic,
                    "thinking": {"type": "enabled", "budget_tokens": 65},
                },
                "unsupported_sampling": {**basic, "frequency_penalty": 0},
                "raw_prompt_thinking": {
                    **basic,
                    "prompt_token_ids": [101],
                    "max_tokens": 1,
                    "thinking": {"type": "enabled", "budget_tokens": 1},
                },
            }
            invalid_results: dict[str, Any] = {}
            for name, payload in invalid_cases.items():
                status, response = request_json(
                    cli.port, "POST", "/v1/chat/completions", payload
                )
                invalid_results[name] = {
                    "status": status,
                    "message": response.get("error", {}).get("message"),
                }
            checks["invalid_and_unsupported_fields_fail_closed"] = all(
                result["status"] == 400 for result in invalid_results.values()
            )

            duplicate_status, duplicate_response = request_json(
                cli.port, "POST", "/v1/chat/completions", duplicate
            )
            duplicate_stream = request_stream(cli.port, duplicate)
            duplicate_progress = duplicate_response["aima_amd395"][
                "tool_progress"
            ]
            checks["same_response_duplicate_suppressed"] = (
                duplicate_status == 200
                and duplicate_stream["status"] == 200
                and len(duplicate_response["choices"][0]["message"]["tool_calls"])
                == 1
                and len(duplicate_stream["tool_calls"]) == 1
                and duplicate_progress["parsed_calls"] == 2
                and duplicate_progress["duplicate_calls_suppressed"] == 1
                and duplicate_stream["metrics"]["tool_progress"]
                == duplicate_progress
            )

            different_status, different_response = request_json(
                cli.port, "POST", "/v1/chat/completions", different
            )
            different_calls = different_response["choices"][0]["message"][
                "tool_calls"
            ]
            checks["different_arguments_remain_parallel"] = (
                different_status == 200
                and len(different_calls) == 2
                and len({json.dumps(tool_signature(call)) for call in different_calls})
                == 2
            )

            serial_request = {**different, "parallel_tool_calls": False}
            serial_status, serial_response = request_json(
                cli.port, "POST", "/v1/chat/completions", serial_request
            )
            serial_progress = serial_response["aima_amd395"]["tool_progress"]
            checks["parallel_false_admits_at_most_one"] = (
                serial_status == 200
                and len(serial_response["choices"][0]["message"]["tool_calls"])
                == 1
                and serial_progress["parallel_calls_suppressed"] == 1
            )

            exhausted_status, exhausted_response = request_json(
                cli.port, "POST", "/v1/chat/completions", exhausted
            )
            exhausted_stream = request_stream(cli.port, exhausted)
            exhausted_message = exhausted_response["choices"][0]["message"]
            exhausted_progress = exhausted_response["aima_amd395"][
                "tool_progress"
            ]
            checks["bounded_history_no_progress_stream_parity"] = (
                exhausted_status == 200
                and exhausted_stream["status"] == 200
                and not exhausted_message.get("tool_calls")
                and not exhausted_stream["tool_calls"]
                and exhausted_progress["history_signature_occurrences"] == 2
                and exhausted_progress["history_no_progress_results"] == 2
                and exhausted_progress["exhausted_history_calls_suppressed"]
                == 1
                and exhausted_progress["no_progress"] is True
                and exhausted_progress["caller_action"]
                == "change_strategy_or_return_blocked"
                and exhausted_stream["metrics"]["tool_progress"]
                == exhausted_progress
                and "<tool_call>" not in exhausted_stream["content"]
            )

            vl_disabled = {
                **vl,
                "thinking": {"type": "disabled", "budget_tokens": 128},
            }
            vl_enabled = {
                **vl,
                "thinking": {"type": "enabled", "budget_tokens": 128},
            }
            vl_disabled_status, vl_disabled_response = request_json(
                cli.port, "POST", "/v1/chat/completions", vl_disabled
            )
            vl_enabled_status, vl_enabled_response = request_json(
                cli.port, "POST", "/v1/chat/completions", vl_enabled
            )
            checks["vl_thinking_prompt_and_response_split"] = (
                vl_disabled_status == 200
                and vl_enabled_status == 200
                and vl_disabled_response["usage"]["prompt_tokens"]
                == vl_enabled_response["usage"]["prompt_tokens"] + 2
                and "reasoning_content"
                not in vl_disabled_response["choices"][0]["message"]
                and "reasoning_content"
                in vl_enabled_response["choices"][0]["message"]
                and vl_disabled_response["aima_amd395"]["vl"]["enabled"]
                is True
                and vl_enabled_response["aima_amd395"]["vl"]["enabled"]
                is True
            )

            observations = {
                "default": response_summary(default_response),
                "disabled": response_summary(disabled_response),
                "thinking_history_nonstream": response_summary(history_response),
                "thinking_history_stream": stream_summary(history_stream),
                "thinking_tool_nonstream": response_summary(tool_response),
                "thinking_tool_stream": stream_summary(tool_stream),
                "invalid_requests": invalid_results,
                "duplicate_nonstream": response_summary(duplicate_response),
                "duplicate_stream": stream_summary(duplicate_stream),
                "different_arguments": response_summary(different_response),
                "parallel_false": response_summary(serial_response),
                "exhausted_nonstream": response_summary(exhausted_response),
                "exhausted_stream": stream_summary(exhausted_stream),
                "vl_disabled": response_summary(vl_disabled_response),
                "vl_enabled": response_summary(vl_enabled_response),
            }
        finally:
            if process.poll() is None:
                try:
                    request_json(cli.port, "POST", "/shutdown")
                except Exception:
                    process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    qualified = bool(checks) and all(checks.values())
    result = {
        "schema": "aima.native-chat-protocol-qualification.v0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qualified": qualified,
        "engine": {
            "path": "${AIMA_ENGINE}",
            "sha256": engine_sha256,
            "build_info": build_info,
        },
        "model": {
            "id": MODEL_ID,
            "path": "${AIMA_MODEL_DIR}",
        },
        "host": {
            "hostname": subprocess.run(
                ["hostname"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "kernel": subprocess.run(
                ["uname", "-r"], capture_output=True, text=True, check=True
            ).stdout.strip(),
        },
        "server": {
            "command": [
                "${AIMA_ENGINE}" if value == str(engine) else
                "${AIMA_MODEL_DIR}" if value == str(model_dir) else
                "${AIMA_IMAGE_DIR}" if value == str(image.parent) else value
                for value in command
            ],
            "ready": ready,
            "load_report": str(load_report.name),
            "stderr": str(stderr_path.name),
        },
        "checks": checks,
        "observations": observations,
    }
    result = sanitize_host_paths(
        result,
        [
            (engine, "${AIMA_ENGINE}"),
            (engine.parent, "${AIMA_ENGINE_DIR}"),
            (model_dir, "${AIMA_MODEL_DIR}"),
            (image.parent, "${AIMA_IMAGE_DIR}"),
            (output.parent, "${AIMA_OUTPUT_DIR}"),
            (ROOT, "${AIMA_SOURCE_ROOT}"),
        ],
    )
    private_prefixes = ("/home/", "/Users/", "/data/", "/tmp/")
    public_payload = json.dumps(result, ensure_ascii=False, sort_keys=True)
    result["checks"]["artifact_paths_sanitized"] = not any(
        prefix in public_payload for prefix in private_prefixes
    )
    checks = result["checks"]
    qualified = bool(checks) and all(checks.values())
    result["qualified"] = qualified
    canonical_payload = json.dumps(
        result,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    result["integrity"] = {
        "algorithm": "sha256",
        "canonical_payload_sha256": hashlib.sha256(canonical_payload).hexdigest(),
    }
    serialized = json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    output.write_text(serialized, encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}  {output.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"qualified": qualified, "checks": checks}, indent=2))
    if not qualified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
