"""Frozen HTTPS, video-sampling and cache-identity qualification cases."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import copy
import mimetypes
from typing import Any


REFERENCE_CASE_ORDER = (
    "https_image",
    "video_sampling_default",
    "video_sampling_fps_1",
    "video_sampling_num_frames_6",
    "video_content_a",
    "video_content_b",
    "video_content_error",
    "mixed_image_video",
    "mixed_video_image",
    "mixed_mutated_image_video",
)
SERVED_MODEL_SENTINEL = "${AIMA_SERVED_MODEL}"

ENABLED_REPLAY = (
    ("https_image_cold", "https_image", "image"),
    ("https_image_exact", "https_image", "image"),
    ("video_content_a_cold", "video_content_a", "video_a"),
    ("video_content_b_miss", "video_content_b", "video_b"),
    ("video_content_a_restored", "video_content_a", "video_a"),
    ("video_content_error", "video_content_error", "video_error"),
    ("video_content_a_after_error", "video_content_a", "video_a"),
    ("video_sampling_default", "video_sampling_default", "video_b"),
    ("video_sampling_default_exact", "video_sampling_default", "video_b"),
    ("video_sampling_fps_1", "video_sampling_fps_1", "video_b"),
    ("video_sampling_default_restored", "video_sampling_default", "video_b"),
    ("video_sampling_num_frames_6", "video_sampling_num_frames_6", "video_b"),
    (
        "video_sampling_num_frames_6_exact",
        "video_sampling_num_frames_6",
        "video_b",
    ),
    ("mixed_image_video", "mixed_image_video", "video_a"),
    ("mixed_video_image_reordered", "mixed_video_image", "video_a"),
    (
        "mixed_mutated_image_video",
        "mixed_mutated_image_video",
        "video_a",
    ),
    ("mixed_image_video_restored", "mixed_image_video", "video_a"),
)

DISABLED_REPLAY = (
    ("video_content_a_disabled_1", "video_content_a", "video_a"),
    ("video_content_b_disabled", "video_content_b", "video_b"),
    ("video_content_a_disabled_2", "video_content_a", "video_a"),
    ("video_content_error_disabled", "video_content_error", "video_error"),
    ("video_content_a_disabled_3", "video_content_a", "video_a"),
    ("video_sampling_default_disabled_1", "video_sampling_default", "video_b"),
    ("video_sampling_default_disabled_2", "video_sampling_default", "video_b"),
    ("video_sampling_fps_1_disabled_1", "video_sampling_fps_1", "video_b"),
    ("video_sampling_fps_1_disabled_2", "video_sampling_fps_1", "video_b"),
    ("mixed_image_video_disabled_1", "mixed_image_video", "video_a"),
    ("mixed_image_video_disabled_2", "mixed_image_video", "video_a"),
)


def text(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


def request_payload(
    model: str,
    content: list[dict[str, Any]],
    *,
    max_tokens: int = 1,
    media_io_kwargs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if media_io_kwargs is not None:
        payload["media_io_kwargs"] = media_io_kwargs
    return payload


def external_part(
    fixtures: Any,
    modality: str,
    fixture_id: str,
    transport: str,
    url: str,
    replacements: dict[str, Any],
) -> dict[str, Any]:
    record = fixtures.records[fixture_id]
    replacements[url] = {
        "fixture": fixture_id,
        "transport": transport,
        "mime": mimetypes.guess_type(fixture_id)[0]
        or "application/octet-stream",
        "bytes": record["bytes"],
        "sha256": record["sha256"],
    }
    field = f"{modality}_url"
    return {"type": field, field: {"url": url}}


def build_reference_cases(
    fixtures: Any,
    model: str,
    http_base: str,
    https_base: str,
) -> list[dict[str, Any]]:
    """Build requests once for vLLM capture and again for native replay."""

    specs: list[dict[str, Any]] = []

    def add(
        case_id: str,
        surfaces: list[str],
        content: list[dict[str, Any]],
        replacements: dict[str, Any],
        *,
        mutable_mode: str,
        expected_accept: bool = True,
        max_tokens: int = 1,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        specs.append(
            {
                "case_id": case_id,
                "surfaces": surfaces,
                "expected_accept": expected_accept,
                "payload": request_payload(
                    model,
                    content,
                    max_tokens=max_tokens,
                    media_io_kwargs=media_io_kwargs,
                ),
                "replacements": replacements,
                "require_tool_call": False,
                "mutable_mode": mutable_mode,
            }
        )

    replacements: dict[str, Any] = {}
    https_image = external_part(
        fixtures,
        "image",
        "image-rgb-256.png",
        "https",
        f"{https_base}/image-rgb-256.png",
        replacements,
    )
    add(
        "https_image",
        ["transport", "https", "image"],
        [https_image, text("Describe briefly.")],
        replacements,
        mutable_mode="image",
    )

    def sampling_case(
        case_id: str,
        media_io_kwargs: dict[str, dict[str, Any]] | None,
    ) -> None:
        replacements = {}
        video = fixtures.part(
            "video",
            "video-12f-6fps-192x128.avi",
            "local",
            replacements,
        )
        add(
            case_id,
            ["video", "sampling", "cache_identity"],
            [video, text("Describe the sampled video briefly.")],
            replacements,
            mutable_mode="video_b",
            media_io_kwargs=media_io_kwargs,
        )

    sampling_case("video_sampling_default", None)
    sampling_case(
        "video_sampling_fps_1",
        {"video": {"fps": 1.0, "video_backend": "opencv"}},
    )
    sampling_case(
        "video_sampling_num_frames_6",
        {"video": {"num_frames": 6, "video_backend": "opencv"}},
    )

    mutable_url = f"{http_base}/mutable-video"

    def mutable_video_case(
        case_id: str,
        fixture_id: str,
        mutable_mode: str,
        *,
        expected_accept: bool = True,
    ) -> None:
        replacements = {}
        video = external_part(
            fixtures,
            "video",
            fixture_id,
            "http-mutable",
            mutable_url,
            replacements,
        )
        add(
            case_id,
            ["video", "cache_identity", "long_generation"],
            [video, text("Describe the video in one short sentence.")],
            replacements,
            mutable_mode=mutable_mode,
            expected_accept=expected_accept,
            max_tokens=8,
        )

    mutable_video_case(
        "video_content_a", "video-8f-4fps-128.mp4", "video_a"
    )
    mutable_video_case(
        "video_content_b", "video-12f-6fps-192x128.avi", "video_b"
    )
    mutable_video_case(
        "video_content_error",
        "corrupt-video.mp4",
        "video_error",
        expected_accept=False,
    )

    def mixed_case(
        case_id: str,
        first: tuple[str, str],
        second: tuple[str, str],
    ) -> None:
        replacements = {}
        parts = []
        for index, (modality, fixture_id) in enumerate((first, second)):
            parts.append(text(f"Media {index + 1}:"))
            parts.append(
                fixtures.part(modality, fixture_id, "local", replacements)
            )
        parts.append(text("Compare them briefly."))
        add(
            case_id,
            ["mixed", "order", "cache_identity"],
            parts,
            replacements,
            mutable_mode="video_a",
        )

    mixed_case(
        "mixed_image_video",
        ("image", "image-landscape-512x192.jpg"),
        ("video", "video-8f-4fps-128.mp4"),
    )
    mixed_case(
        "mixed_video_image",
        ("video", "video-8f-4fps-128.mp4"),
        ("image", "image-landscape-512x192.jpg"),
    )
    mixed_case(
        "mixed_mutated_image_video",
        ("image", "image-transparent-160x320.png"),
        ("video", "video-8f-4fps-128.mp4"),
    )

    if tuple(item["case_id"] for item in specs) != REFERENCE_CASE_ORDER:
        raise RuntimeError("transport/cache reference case order changed")
    return specs


def normalize_contract_request(request: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(request)
    normalized["model"] = SERVED_MODEL_SENTINEL
    return normalized


def response_content(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def finish_reason(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    reason = choices[0].get("finish_reason")
    return reason if isinstance(reason, str) else None


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


def error_signature(response: Any) -> tuple[Any, Any] | None:
    if not isinstance(response, dict):
        return None
    error = response.get("error")
    if not isinstance(error, dict):
        return None
    return error.get("type"), error.get("code")


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
