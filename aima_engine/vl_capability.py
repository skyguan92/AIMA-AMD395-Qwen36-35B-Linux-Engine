"""Validation helpers for staged native-VL capability evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aima_engine.vl_reference import (
    CAPABILITY_SCHEMA,
    MODEL_REVISION,
    PINNED_PACKAGES,
)


PROCESSOR_PROBE_SCHEMA = "aima-amd395-qwen36/vl-processor-capability-probe/v1"

EXPECTED_MAX_MODEL_LEN = 262_144
EXPECTED_MAX_TOKENS_PER_ITEM = {"image": 16_384, "video": 12_288}
EXPECTED_MAX_ITEMS_PER_PROMPT = {"image": 16, "video": 21}

REQUIRED_IMAGE_CASES = {
    "minimum_source",
    "aspect_ratio_200",
    "aspect_ratio_over_200",
    "factor_minus_one",
    "factor_exact",
    "factor_plus_one",
    "maximum_pixels",
    "above_maximum_pixels",
    "portrait",
    "landscape",
}
REQUIRED_VIDEO_RESIZE_CASES = {
    "below_temporal_factor",
    "temporal_factor",
    "below_spatial_factor",
    "spatial_factor",
    "aspect_ratio_200",
    "aspect_ratio_over_200",
    "typical",
    "maximum_feature_shape",
}
REQUIRED_VIDEO_SAMPLING_CASES = {
    "below_min_frames",
    "minimum_frames",
    "typical_fps",
    "maximum_frames",
    "above_maximum_frames",
    "explicit_num_frames",
    "fps_num_frames_conflict",
}

REQUIRED_API_CASES = {
    "residency_text_before": True,
    "image_local_png": True,
    "image_data_jpeg": True,
    "image_http_webp": True,
    "image_transparent_png": True,
    "multi_image_interleaved": True,
    "video_local_mp4": True,
    "video_data_mp4": True,
    "video_http_avi": True,
    "multi_video": True,
    "mixed_image_then_video": True,
    "mixed_video_then_image": True,
    "conversation_prior_image": True,
    "conversation_media_replace": True,
    "tool_history_with_image": True,
    "tool_forced_image": True,
    "tool_auto_image": True,
    "stream_image": True,
    "stream_video": True,
    "residency_text_after": True,
    "error_corrupt_image": False,
    "error_corrupt_video": False,
    "error_aspect_ratio": False,
    "error_outside_local_root": False,
    "error_disallowed_domain": False,
    "error_image_count_over_limit": False,
    "error_video_count_over_limit": False,
    "error_malformed_data_uri": False,
    "error_type_mismatch": False,
    "error_audio_out_of_scope": False,
}

REQUIRED_API_SURFACES = {
    "api",
    "conversation",
    "error",
    "generation",
    "image",
    "mixed",
    "residency",
    "stream",
    "tool",
    "transport",
    "video",
}


def _case_ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        item["case_id"]
        for item in value
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }


def validate_processor_probe(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != PROCESSOR_PROBE_SCHEMA:
        errors.append(f"processor probe schema must be {PROCESSOR_PROBE_SCHEMA}")
    if payload.get("complete") is not True:
        errors.append("processor probe is not complete")
    if payload.get("qualified") is not True:
        errors.append("processor probe is not qualified")

    model = payload.get("model")
    if not isinstance(model, dict) or model.get("revision") != MODEL_REVISION:
        errors.append("processor probe model revision is not frozen")

    runtime = payload.get("runtime")
    versions = runtime.get("packages") if isinstance(runtime, dict) else None
    if not isinstance(versions, dict):
        errors.append("processor probe has no package versions")
    else:
        for name, expected in PINNED_PACKAGES.items():
            actual = versions.get(name)
            if not isinstance(actual, str) or not (
                actual == expected or actual.startswith(expected + ".")
            ):
                errors.append(f"processor probe {name} pin mismatch: {actual!r}")

    budget = payload.get("vllm_budget")
    if not isinstance(budget, dict):
        errors.append("processor probe has no vLLM budget")
    else:
        if budget.get("max_model_len") != EXPECTED_MAX_MODEL_LEN:
            errors.append("vLLM budget max_model_len must be 262144")
        if budget.get("supported_limits") != {"image": None, "video": None}:
            errors.append("fixed model must expose unlimited image/video item counts")
        if budget.get("max_tokens_per_item") != EXPECTED_MAX_TOKENS_PER_ITEM:
            errors.append("vLLM maximum tokens per media item drifted")
        if budget.get("max_items_per_prompt") != EXPECTED_MAX_ITEMS_PER_PROMPT:
            errors.append("vLLM derived media-count boundary drifted")
        if budget.get("max_items_per_batch") != EXPECTED_MAX_ITEMS_PER_PROMPT:
            errors.append("batch-1 media-count boundary drifted")
        if budget.get("encoder_budget_tokens") != EXPECTED_MAX_MODEL_LEN:
            errors.append("multimodal encoder budget must cover the full window")

    missing_image = REQUIRED_IMAGE_CASES - _case_ids(payload.get("image_resize_cases"))
    if missing_image:
        errors.append("missing image resize cases: " + ", ".join(sorted(missing_image)))
    missing_video = REQUIRED_VIDEO_RESIZE_CASES - _case_ids(
        payload.get("video_resize_cases")
    )
    if missing_video:
        errors.append("missing video resize cases: " + ", ".join(sorted(missing_video)))
    missing_sampling = REQUIRED_VIDEO_SAMPLING_CASES - _case_ids(
        payload.get("video_sampling_cases")
    )
    if missing_sampling:
        errors.append(
            "missing video sampling cases: " + ", ".join(sorted(missing_sampling))
        )

    fixtures = payload.get("deterministic_processor_fixtures")
    if not isinstance(fixtures, list):
        errors.append("deterministic processor fixtures must be an array")
    else:
        modalities = {
            item.get("modality") for item in fixtures if isinstance(item, dict)
        }
        if modalities != {"image", "video"}:
            errors.append("deterministic processor fixtures must cover image and video")
        for fixture in fixtures:
            if not isinstance(fixture, dict):
                errors.append("malformed deterministic processor fixture")
                continue
            outputs = fixture.get("outputs")
            if not isinstance(outputs, dict):
                errors.append("processor fixture has no outputs")
                continue
            for name, record in outputs.items():
                if not isinstance(record, dict) or not record.get("sha256"):
                    errors.append(f"processor fixture output is not hash-bound: {name}")
    return errors


def validate_capability_manifest(payload: Mapping[str, Any]) -> list[str]:
    """Validate that API discovery covers the complete frozen VL surface."""

    errors: list[str] = []
    if payload.get("schema") != CAPABILITY_SCHEMA:
        errors.append(f"capability schema must be {CAPABILITY_SCHEMA}")
    if payload.get("complete") is not True:
        errors.append("capability manifest is not complete")
    if payload.get("qualified") is not True:
        errors.append("capability manifest is not qualified")

    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        errors.append("capability manifest has no bindings")
    else:
        for name in ("processor_probe", "fixture_manifest"):
            binding = bindings.get(name)
            if not isinstance(binding, dict):
                errors.append(f"capability binding is missing: {name}")
                continue
            digest = binding.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"capability binding SHA-256 is invalid: {name}")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        return errors + ["capability cases must be an array"]
    by_id = {
        item.get("case_id"): item
        for item in cases
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    missing = set(REQUIRED_API_CASES) - set(by_id)
    if missing:
        errors.append("missing API capability cases: " + ", ".join(sorted(missing)))
    duplicates = len(cases) != len(by_id)
    if duplicates:
        errors.append("API capability case IDs must be unique")

    observed_surfaces: set[str] = set()
    for case_id, expected_accept in REQUIRED_API_CASES.items():
        case = by_id.get(case_id)
        if not isinstance(case, dict):
            continue
        surfaces = case.get("surfaces")
        if isinstance(surfaces, list):
            observed_surfaces.update(
                item for item in surfaces if isinstance(item, str)
            )
        if case.get("expected_accept") is not expected_accept:
            errors.append(f"capability expectation drifted: {case_id}")
        if case.get("passed") is not True:
            errors.append(f"capability case did not pass: {case_id}")
        status = case.get("status_code")
        if not isinstance(status, int) or isinstance(status, bool):
            errors.append(f"capability status is missing: {case_id}")
        elif expected_accept != (200 <= status < 300):
            errors.append(f"capability accept/reject mismatch: {case_id}")
        for field in ("request_sha256", "response_sha256"):
            digest = case.get(field)
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"capability {field} is invalid: {case_id}")

    missing_surfaces = REQUIRED_API_SURFACES - observed_surfaces
    if missing_surfaces:
        errors.append(
            "missing API capability surfaces: " + ", ".join(sorted(missing_surfaces))
        )
    return errors
