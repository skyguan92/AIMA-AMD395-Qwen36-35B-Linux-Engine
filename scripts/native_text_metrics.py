"""Shared fail-closed checks for text-only native request metrics."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

from typing import Any


MROPE_ZERO_FIELDS = (
    "position_delta",
    "position_upload_bytes",
    "full_attention_launches",
    "unified_attention_launches",
    "decode_steps",
)
VL_ZERO_FIELDS = (
    "media_count",
    "image_count",
    "video_count",
    "source_bytes",
    "vision_patches",
    "visual_tokens",
    "media_cache_hits",
    "media_cache_misses",
    "media_cache_entries",
    "media_cache_resident_bytes",
    "vision_batch_count",
    "vision_max_batch_patches",
    "vision_max_batch_tokens",
    "vision_plan_cache_entries",
    "host_to_device_bytes",
    "media_load_decode_wall_ms",
    "media_load_wall_ms",
    "media_decode_wall_ms",
    "processor_wall_ms",
    "vision_plan_build_wall_ms",
    "vision_input_upload_wall_ms",
    "vision_encode_wall_ms",
    "embedding_injection_wall_ms",
)


def _all_zero(payload: dict[str, Any], fields: tuple[str, ...]) -> bool:
    try:
        return all(float(payload[field]) == 0.0 for field in fields)
    except (KeyError, TypeError, ValueError):
        return False


def text_path_idle_checks(metrics: Any) -> dict[str, bool]:
    """Return auditable checks proving a request did not enter native VL."""

    if not isinstance(metrics, dict):
        return {"metrics_shape_complete": False}
    mrope = metrics.get("mrope")
    vl = metrics.get("vl")
    if not isinstance(mrope, dict) or not isinstance(vl, dict):
        return {"metrics_shape_complete": False}
    return {
        "metrics_shape_complete": True,
        "mrope_disabled": mrope.get("enabled") is False,
        "mrope_vl_state_zero": _all_zero(mrope, MROPE_ZERO_FIELDS),
        "vl_disabled": vl.get("enabled") is False,
        "vl_request_state_zero": _all_zero(vl, VL_ZERO_FIELDS),
        "vision_plan_cache_not_touched": (
            vl.get("vision_plan_cache_hit") is False
        ),
    }


def text_path_is_idle(metrics: Any) -> bool:
    return all(text_path_idle_checks(metrics).values())
