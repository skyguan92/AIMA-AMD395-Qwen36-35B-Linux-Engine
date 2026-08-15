"""Frozen mixed-media and conversation cases that extend the VL G1 audit."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import copy
from typing import Any


CASE_ORDER = (
    "mixed_multi_image_video_interleave",
    "mixed_multi_video_image_interleave",
    "conversation_video_reuse_replace",
    "conversation_mixed_prior_turn",
    "stream_mixed_media",
)
SERVED_MODEL_SENTINEL = "${AIMA_SERVED_MODEL}"


def text(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


def request_payload(
    model: str,
    messages: list[dict[str, Any]],
    *,
    stream: bool = False,
    max_tokens: int = 1,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def build_cases(fixtures: Any, model: str) -> list[dict[str, Any]]:
    """Build the exact five requests for both reference and native replay."""

    specs: list[dict[str, Any]] = []

    def add(
        case_id: str,
        surfaces: list[str],
        messages: list[dict[str, Any]],
        replacements: dict[str, Any],
        *,
        stream: bool = False,
        max_tokens: int = 1,
    ) -> None:
        specs.append(
            {
                "case_id": case_id,
                "surfaces": surfaces,
                "expected_accept": True,
                "payload": request_payload(
                    model,
                    messages,
                    stream=stream,
                    max_tokens=max_tokens,
                ),
                "replacements": replacements,
                "require_tool_call": False,
            }
        )

    replacements: dict[str, Any] = {}
    add(
        "mixed_multi_image_video_interleave",
        ["mixed", "conversation", "interleave"],
        [
            {
                "role": "user",
                "content": [
                    text("Image A:"),
                    fixtures.part(
                        "image", "image-rgb-256.png", "local", replacements
                    ),
                    text("Video:"),
                    fixtures.part(
                        "video",
                        "video-8f-4fps-128.mp4",
                        "local",
                        replacements,
                    ),
                    text("Image B:"),
                    fixtures.part(
                        "image",
                        "image-landscape-512x192.jpg",
                        "local",
                        replacements,
                    ),
                    text("Compare all three."),
                ],
            }
        ],
        replacements,
    )

    replacements = {}
    add(
        "mixed_multi_video_image_interleave",
        ["mixed", "conversation", "interleave"],
        [
            {
                "role": "user",
                "content": [
                    text("Video A:"),
                    fixtures.part(
                        "video",
                        "video-8f-4fps-128.mp4",
                        "local",
                        replacements,
                    ),
                    text("Image:"),
                    fixtures.part(
                        "image",
                        "image-transparent-160x320.png",
                        "local",
                        replacements,
                    ),
                    text("Video B:"),
                    fixtures.part(
                        "video",
                        "video-12f-6fps-192x128.avi",
                        "local",
                        replacements,
                    ),
                    text("Compare all three."),
                ],
            }
        ],
        replacements,
    )

    replacements = {}
    add(
        "conversation_video_reuse_replace",
        ["video", "conversation", "reuse", "replace"],
        [
            {"role": "system", "content": "Answer concisely."},
            {
                "role": "user",
                "content": [
                    fixtures.part(
                        "video",
                        "video-8f-4fps-128.mp4",
                        "local",
                        replacements,
                    ),
                    text("Remember this video."),
                ],
            },
            {"role": "assistant", "content": "Acknowledged."},
            {
                "role": "user",
                "content": [
                    fixtures.part(
                        "video",
                        "video-12f-6fps-192x128.avi",
                        "local",
                        replacements,
                    ),
                    text("Compare the current video with the prior video."),
                ],
            },
        ],
        replacements,
    )

    replacements = {}
    add(
        "conversation_mixed_prior_turn",
        ["mixed", "conversation", "prior_turn"],
        [
            {
                "role": "user",
                "content": [
                    fixtures.part(
                        "image", "image-rgb-256.png", "local", replacements
                    ),
                    text("and"),
                    fixtures.part(
                        "video",
                        "video-8f-4fps-128.mp4",
                        "local",
                        replacements,
                    ),
                    text("Remember both media items."),
                ],
            },
            {"role": "assistant", "content": "Acknowledged."},
            {
                "role": "user",
                "content": "Compare the prior image and video.",
            },
        ],
        replacements,
    )

    replacements = {}
    add(
        "stream_mixed_media",
        ["mixed", "stream", "api"],
        [
            {
                "role": "user",
                "content": [
                    fixtures.part(
                        "image",
                        "image-transparent-160x320.png",
                        "local",
                        replacements,
                    ),
                    text("and"),
                    fixtures.part(
                        "video",
                        "video-12f-6fps-192x128.avi",
                        "local",
                        replacements,
                    ),
                    text("Describe briefly."),
                ],
            }
        ],
        replacements,
        stream=True,
        max_tokens=4,
    )

    if tuple(item["case_id"] for item in specs) != CASE_ORDER:
        raise RuntimeError("G1 mixed/conversation case order changed")
    return specs


def normalize_contract_request(request: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(request)
    normalized["model"] = SERVED_MODEL_SENTINEL
    return normalized


def finish_reason(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and len(choices) == 1:
        value = choices[0]
        if isinstance(value, dict) and isinstance(
            value.get("finish_reason"), str
        ):
            return value["finish_reason"]
    events = response.get("events")
    if isinstance(events, list):
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            choices = event.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                continue
            value = choices[0]
            if isinstance(value, dict) and isinstance(
                value.get("finish_reason"), str
            ):
                return value["finish_reason"]
    return None


def response_content(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    aggregate = response.get("aggregate_content")
    if isinstance(aggregate, str):
        return aggregate
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def usage_signature(response: Any) -> tuple[Any, Any, Any] | None:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    return tuple(
        usage.get(name)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    )


def request_media_counts(request: dict[str, Any]) -> dict[str, int]:
    counts = {"image": 0, "video": 0}
    for message in request.get("messages", []):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind == "image_url":
                counts["image"] += 1
            elif kind == "video_url":
                counts["video"] += 1
    return counts
