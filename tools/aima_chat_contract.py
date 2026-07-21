#!/usr/bin/env python3
"""Validation and response helpers for the supported Chat Completions subset."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "Answer directly. Preserve exact identifiers, numbers, and code. Do not include analysis."
)
DEFAULT_MODEL_ID = "aima-amd395-qwen36-35b"


def json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_id(request: dict[str, Any], created: int) -> str:
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{created}:{payload}".encode("utf-8")).hexdigest()[:24]
    return "chatcmpl-aima-" + digest


def _string_content(message: dict[str, Any], index: int) -> str:
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise SystemExit(f"messages[{index}].content must be a non-empty string")
    return content


def parse_messages(request: dict[str, Any], default_system_prompt: str) -> tuple[str, str]:
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SystemExit("request.messages must be a non-empty list")
    system_parts: list[str] = []
    user_parts: list[str] = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            raise SystemExit(f"messages[{index}] must be an object")
        role = item.get("role")
        if role == "system":
            if user_parts:
                raise SystemExit("system messages must precede user messages in the supported subset")
            system_parts.append(_string_content(item, index))
        elif role == "user":
            user_parts.append(_string_content(item, index))
        else:
            raise SystemExit(
                "supported subset only accepts system/user string messages; "
                f"got messages[{index}].role={role!r}"
            )
    if not user_parts:
        raise SystemExit("request.messages must include at least one user message")
    system_prompt = "\n\n".join(system_parts) if system_parts else default_system_prompt
    return system_prompt, "\n\n".join(user_parts)


def validate_request(request: dict[str, Any]) -> None:
    model = request.get("model", DEFAULT_MODEL_ID)
    if not isinstance(model, str) or model != DEFAULT_MODEL_ID:
        raise SystemExit(f"model must be {DEFAULT_MODEL_ID!r}")
    stream = request.get("stream", False)
    if not isinstance(stream, bool) or stream:
        raise SystemExit("stream=true is not supported")
    n = request.get("n", 1)
    if not isinstance(n, int) or isinstance(n, bool) or n != 1:
        raise SystemExit("n != 1 is not supported")
    if request.get("stop") not in (None, []):
        raise SystemExit("custom stop strings are not supported; chat EOS stop tokens are always used")
    for unsupported in ("tools", "tool_choice", "functions", "function_call", "response_format"):
        if unsupported in request and request[unsupported] not in (None, "none"):
            raise SystemExit(f"{unsupported!r} is not supported")
    temperature = request.get("temperature", 0)
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or float(temperature) != 0.0
    ):
        raise SystemExit("only deterministic temperature=0 requests are supported")
    top_p = request.get("top_p", 1)
    if (
        not isinstance(top_p, (int, float))
        or isinstance(top_p, bool)
        or float(top_p) != 1.0
    ):
        raise SystemExit("top_p values other than 1 are not supported")
    max_tokens = request.get("max_tokens", request.get("max_completion_tokens", 96))
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise SystemExit("max_tokens/max_completion_tokens must be a positive integer")


def max_tokens_from_request(request: dict[str, Any]) -> int:
    value = request.get("max_tokens", request.get("max_completion_tokens", 96))
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SystemExit("max_tokens/max_completion_tokens must be a positive integer")
    return value


def finish_reason(stop_reason: Any) -> str:
    if stop_reason in {"stop_token", "eos", "stop"}:
        return "stop"
    if stop_reason == "length":
        return "length"
    raise ValueError(f"unsupported or missing stop_reason: {stop_reason!r}")
