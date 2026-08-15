"""Frozen image-IO and error/limit parity cases for native VL."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aima_engine.vl_reference import sha256_file
from aima_engine.vl_transport_cache import request_payload, text


REFERENCE_CASE_ORDER = (
    "rgba_default_white",
    "rgba_background_red",
    "video_sampling_default",
    "video_sampling_empty_mapping",
    "video_long_duration",
    "empty_image_remote",
    "empty_video_remote",
    "unreachable_image_remote",
    "oversize_image_remote",
    "timeout_image_remote",
)

REFERENCE_ERROR_CONTRACT = {
    "empty_image_remote": (400, "BadRequestError", 400, "empty_media"),
    "empty_video_remote": (400, "BadRequestError", 400, "empty_media"),
    "unreachable_image_remote": (
        500,
        "InternalServerError",
        500,
        "media_access",
    ),
    "oversize_image_remote": (400, "BadRequestError", 400, "media_limit"),
    "timeout_image_remote": (
        500,
        "InternalServerError",
        500,
        "media_timeout",
    ),
}
NATIVE_COMPATIBLE_ERROR = (400, "invalid_request_error", "bad_request")

NATIVE_REPLAY = (
    ("rgba_default_cold", "rgba_default_white"),
    ("rgba_red_miss", "rgba_background_red"),
    ("rgba_default_restored", "rgba_default_white"),
    ("video_default_cold", "video_sampling_default"),
    ("video_empty_mapping_exact", "video_sampling_empty_mapping"),
    ("video_default_restored", "video_sampling_default"),
    ("video_long_duration", "video_long_duration"),
    ("empty_image_remote", "empty_image_remote"),
    ("empty_video_remote", "empty_video_remote"),
    ("unreachable_image_remote", "unreachable_image_remote"),
    ("oversize_image_remote", "oversize_image_remote"),
    ("timeout_image_remote", "timeout_image_remote"),
    ("rgba_red_after_errors", "rgba_background_red"),
)


def _external_part(
    modality: str,
    url: str,
    replacements: dict[str, Any],
    *,
    fixture_id: str,
    mime: str,
    bytes_count: int | None,
) -> dict[str, Any]:
    replacement: dict[str, Any] = {
        "fixture": fixture_id,
        "transport": "http-loopback",
        "mime": mime,
    }
    if bytes_count is not None:
        replacement["bytes"] = bytes_count
    replacements[url] = replacement
    field = f"{modality}_url"
    return {"type": field, field: {"url": url}}


def _long_video_part(
    fixture_root: Path, replacements: dict[str, Any]
) -> dict[str, Any]:
    manifest = json.loads(
        (fixture_root / "fixtures-manifest.json").read_text(encoding="utf-8")
    )
    records = {
        item["fixture_id"]: item for item in manifest.get("fixtures", [])
    }
    record = records.get("video_long_duration_low_fps")
    if not isinstance(record, dict):
        raise ValueError("long-duration fixture record is missing")
    path = fixture_root / "video-12f-0.002fps-192x128.avi"
    if (
        path.stat().st_size != record.get("bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError("long-duration fixture differs from its manifest")
    url = path.resolve().as_uri()
    replacements[url] = {
        "fixture": "video_long_duration_low_fps",
        "transport": "local",
        "mime": "video/x-msvideo",
        "bytes": record["bytes"],
        "sha256": record["sha256"],
    }
    return {"type": "video_url", "video_url": {"url": url}}


def build_reference_cases(
    fixtures: Any,
    error_fixture_root: Path,
    model: str,
    media_server: Any,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def add(
        case_id: str,
        surfaces: list[str],
        content: list[dict[str, Any]],
        replacements: dict[str, Any],
        *,
        expected_accept: bool = True,
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
                    media_io_kwargs=media_io_kwargs,
                ),
                "replacements": replacements,
                "require_tool_call": False,
            }
        )

    def rgba_case(
        case_id: str,
        media_io_kwargs: dict[str, dict[str, Any]] | None,
    ) -> None:
        replacements: dict[str, Any] = {}
        image = fixtures.part(
            "image", "image-transparent-160x320.png", "local", replacements
        )
        add(
            case_id,
            ["image", "rgba", "cache_identity"],
            [image, text("Describe the image briefly.")],
            replacements,
            media_io_kwargs=media_io_kwargs,
        )

    rgba_case("rgba_default_white", None)
    rgba_case(
        "rgba_background_red",
        {"image": {"rgba_background_color": [255, 0, 0]}},
    )

    def sampling_case(
        case_id: str,
        media_io_kwargs: dict[str, dict[str, Any]] | None,
    ) -> None:
        replacements: dict[str, Any] = {}
        video = fixtures.part(
            "video", "video-12f-6fps-192x128.avi", "local", replacements
        )
        add(
            case_id,
            ["video", "sampling", "merge_semantics"],
            [video, text("Describe the sampled video briefly.")],
            replacements,
            media_io_kwargs=media_io_kwargs,
        )

    sampling_case("video_sampling_default", None)
    sampling_case("video_sampling_empty_mapping", {"video": {}})

    replacements = {}
    add(
        "video_long_duration",
        ["video", "duration", "limit"],
        [
            _long_video_part(error_fixture_root, replacements),
            text("Describe the sampled video briefly."),
        ],
        replacements,
    )

    error_specs = (
        (
            "empty_image_remote",
            "image",
            media_server.http_base + "/empty-image",
            "image/png",
            0,
        ),
        (
            "empty_video_remote",
            "video",
            media_server.http_base + "/empty-video",
            "video/mp4",
            0,
        ),
        (
            "unreachable_image_remote",
            "image",
            media_server.unreachable_base + "/unreachable-image",
            "image/png",
            None,
        ),
        (
            "oversize_image_remote",
            "image",
            media_server.http_base + "/large-image",
            "image/png",
            media_server.large_image_bytes,
        ),
        (
            "timeout_image_remote",
            "image",
            media_server.http_base + "/slow-image",
            "image/png",
            1,
        ),
    )
    for case_id, modality, url, mime, bytes_count in error_specs:
        replacements = {}
        part = _external_part(
            modality,
            url,
            replacements,
            fixture_id=case_id,
            mime=mime,
            bytes_count=bytes_count,
        )
        add(
            case_id,
            [modality, "transport", "error", "limit"],
            [part, text("Describe briefly.")],
            replacements,
            expected_accept=False,
        )

    if tuple(item["case_id"] for item in specs) != REFERENCE_CASE_ORDER:
        raise RuntimeError("error/limit reference case order changed")
    return specs
