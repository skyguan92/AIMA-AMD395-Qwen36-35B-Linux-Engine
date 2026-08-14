"""Validation helpers for staged native-VL capability evidence."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from aima_engine.vl_reference import (
    CAPABILITY_SCHEMA,
    MODEL_REVISION,
    PINNED_PACKAGES,
    canonical_json_sha256,
    verify_manifest_integrity,
)


PROCESSOR_PROBE_SCHEMA = "aima-amd395-qwen36/vl-processor-capability-probe/v1"
API_RENDER_SCHEMA = "aima-amd395-qwen36/vl-api-render-manifest/v1"

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
REQUIRED_API_RENDER_CASES = tuple(
    case_id for case_id, expected in REQUIRED_API_CASES.items() if expected
)
API_RENDER_TOOL_CASES = frozenset(
    {"tool_history_with_image", "tool_forced_image", "tool_auto_image"}
)
API_RENDER_USAGELESS_CASES = frozenset({"stream_image", "stream_video"})
API_RENDER_MEDIA_COUNTS = {
    "residency_text_before": {},
    "image_local_png": {"image": 1},
    "image_data_jpeg": {"image": 1},
    "image_http_webp": {"image": 1},
    "image_transparent_png": {"image": 1},
    "multi_image_interleaved": {"image": 2},
    "video_local_mp4": {"video": 1},
    "video_data_mp4": {"video": 1},
    "video_http_avi": {"video": 1},
    "multi_video": {"video": 2},
    "mixed_image_then_video": {"image": 1, "video": 1},
    "mixed_video_then_image": {"image": 1, "video": 1},
    "conversation_prior_image": {"image": 1},
    "conversation_media_replace": {"image": 2},
    "tool_history_with_image": {"image": 1},
    "tool_forced_image": {"image": 1},
    "tool_auto_image": {"image": 1},
    "stream_image": {"image": 1},
    "stream_video": {"video": 1},
    "residency_text_after": {},
}
API_RENDER_MAX_TOKENS = {
    case_id: (
        64
        if case_id == "tool_forced_image"
        else 192
        if case_id == "tool_auto_image"
        else 4
        if case_id in {"stream_image", "stream_video"}
        else 1
    )
    for case_id in REQUIRED_API_RENDER_CASES
}
EXPECTED_TOOL_JSON_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}

_API_RENDER_BINDING_PATHS = {
    "capability_manifest": "benchmarks/results/vl-capability-manifest.json",
    "fixture_manifest": (
        "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json"
    ),
    "reference_launch": "benchmarks/results/vl-reference-launch.json",
    "reference_manifest": "benchmarks/results/vl-reference-manifest.json",
}
_API_RENDER_SOURCE_PATHS = (
    "aima_engine/vl_capability.py",
    "aima_engine/vl_reference.py",
    "scripts/probe-vllm-vl-api-capabilities.py",
    "scripts/capture-vllm-vl-api-render.py",
)
_API_RENDER_TRUE_DECISIONS = (
    "success_render_cases_20_of_20",
    "non_tool_non_stream_render_matches_full_usage",
    "tool_full_server_usage_offset_one",
    "named_tool_json_schema_bound",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


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


def validate_api_render_manifest(payload: Mapping[str, Any]) -> list[str]:
    """Validate exact prompts emitted by the frozen vLLM render boundary."""

    errors = verify_manifest_integrity(payload)
    if payload.get("schema") != API_RENDER_SCHEMA:
        errors.append(f"API render schema must be {API_RENDER_SCHEMA}")
    if payload.get("complete") is not True:
        errors.append("API render manifest is not complete")
    if payload.get("qualified") is not True:
        errors.append("API render manifest is not qualified")
    if payload.get("scope") != "fixed-vllm-openai-gpu-less-render-token-boundary":
        errors.append("API render scope changed")

    host = payload.get("host")
    if not isinstance(host, dict) or host.get("label") != "amd395":
        errors.append("API render host is not the frozen amd395 target")
    if not isinstance(host, dict) or not isinstance(host.get("hostname"), str):
        errors.append("API render hostname is missing")

    source = payload.get("source")
    if not isinstance(source, dict):
        errors.append("API render source identity is missing")
    else:
        if not isinstance(source.get("commit"), str) or not _GIT_COMMIT.fullmatch(
            source.get("commit", "")
        ):
            errors.append("API render source commit is invalid")
        if source.get("dirty") is not False:
            errors.append("API render source must be clean")
        if not isinstance(source.get("status_sha256"), str) or not _SHA256.fullmatch(
            source.get("status_sha256", "")
        ):
            errors.append("API render source status hash is invalid")
        source_files = source.get("files")
        paths = (
            tuple(
                item.get("path") if isinstance(item, dict) else None
                for item in source_files
            )
            if isinstance(source_files, list)
            else ()
        )
        if paths != _API_RENDER_SOURCE_PATHS:
            errors.append("API render source file set changed")
        if isinstance(source_files, list):
            for item in source_files:
                if not isinstance(item, dict):
                    continue
                if (
                    not isinstance(item.get("bytes"), int)
                    or isinstance(item.get("bytes"), bool)
                    or item["bytes"] <= 0
                    or not isinstance(item.get("sha256"), str)
                    or not _SHA256.fullmatch(item.get("sha256", ""))
                ):
                    errors.append(
                        f"API render source component is invalid: {item.get('path')}"
                    )

    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        errors.append("API render manifest has no bindings")
    else:
        if set(bindings) != set(_API_RENDER_BINDING_PATHS):
            errors.append("API render binding set changed")
        for name, expected_path in _API_RENDER_BINDING_PATHS.items():
            binding = bindings.get(name)
            digest = binding.get("sha256") if isinstance(binding, dict) else None
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                errors.append(f"API render binding SHA-256 is invalid: {name}")
            if not isinstance(binding, dict) or binding.get("path") != expected_path:
                errors.append(f"API render binding path changed: {name}")
            if (
                not isinstance(binding, dict)
                or not isinstance(binding.get("bytes"), int)
                or isinstance(binding.get("bytes"), bool)
                or binding["bytes"] <= 0
            ):
                errors.append(f"API render binding size is invalid: {name}")

    runtime = payload.get("runtime")
    version = runtime.get("vllm") if isinstance(runtime, dict) else None
    expected_version = PINNED_PACKAGES["vllm"]
    if not isinstance(version, str) or not (
        version == expected_version or version.startswith(expected_version + ".")
    ):
        errors.append(f"API render vLLM pin mismatch: {version!r}")
    endpoint = runtime.get("endpoint") if isinstance(runtime, dict) else None
    if (
        not isinstance(endpoint, dict)
        or endpoint.get("scheme") != "http"
        or endpoint.get("host") != "127.0.0.1"
        or not isinstance(endpoint.get("port"), int)
        or isinstance(endpoint.get("port"), bool)
        or not 1 <= endpoint["port"] <= 65_535
    ):
        errors.append("API render endpoint is not an explicit loopback HTTP port")

    contract = payload.get("contract")
    if contract != {
        "content_format": "auto-resolved-string",
        "request_identity": "fixture-normalized-reference-request",
        "tool_normalization": "ChatCompletionRequest-Pydantic-model_dump",
        "render_runtime_uses_gpu": False,
    }:
        errors.append("API render contract changed")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        return errors + ["API render cases must be an array"]
    case_ids = [
        case.get("case_id") if isinstance(case, dict) else None for case in cases
    ]
    if tuple(case_ids) != REQUIRED_API_RENDER_CASES:
        errors.append("API render case order or membership changed")

    for case in cases:
        if not isinstance(case, dict):
            errors.append("API render case must be an object")
            continue
        case_id = case.get("case_id")
        surfaces = case.get("surfaces")
        if (
            not isinstance(surfaces, list)
            or not surfaces
            or not all(isinstance(value, str) and value for value in surfaces)
            or ((case_id in API_RENDER_TOOL_CASES) != ("tool" in surfaces))
        ):
            errors.append(f"API render surfaces are invalid: {case_id}")
        token_ids = case.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in token_ids
        ):
            errors.append(f"API render prompt token IDs are invalid: {case_id}")
            continue
        if case.get("prompt_tokens") != len(token_ids):
            errors.append(f"API render prompt length changed: {case_id}")
        if case.get("prompt_token_ids_sha256") != canonical_json_sha256(
            token_ids
        ):
            errors.append(f"API render prompt hash changed: {case_id}")
        request = case.get("request")
        request_digest = case.get("request_sha256")
        if not isinstance(request, dict):
            errors.append(f"API render normalized request is invalid: {case_id}")
        elif request_digest != canonical_json_sha256(request):
            errors.append(f"API render normalized request hash changed: {case_id}")
        if isinstance(request, dict):
            if (
                request.get("model") != "qwen36-vl-reference"
                or not isinstance(request.get("messages"), list)
                or not request["messages"]
                or request.get("temperature") != 0
                or request.get("max_tokens") != API_RENDER_MAX_TOKENS.get(case_id)
                or request.get("stream")
                != (case_id in {"stream_image", "stream_video"})
            ):
                errors.append(f"API render normalized request changed: {case_id}")
        for field in (
            "reference_transport_request_sha256",
            "render_transport_request_sha256",
        ):
            transport_digest = case.get(field)
            if not isinstance(transport_digest, str) or not _SHA256.fullmatch(
                transport_digest
            ):
                errors.append(f"API render {field} is invalid: {case_id}")
        reference_prompt_tokens = case.get("reference_usage_prompt_tokens")
        usage_delta = case.get("reference_usage_delta")
        if case_id in API_RENDER_USAGELESS_CASES:
            if reference_prompt_tokens is not None or usage_delta is not None:
                errors.append(f"API render stream usage must be absent: {case_id}")
        else:
            if (
                not isinstance(reference_prompt_tokens, int)
                or isinstance(reference_prompt_tokens, bool)
                or reference_prompt_tokens <= 0
            ):
                errors.append(f"API render reference usage is invalid: {case_id}")
            elif (
                not isinstance(usage_delta, int)
                or isinstance(usage_delta, bool)
                or usage_delta != reference_prompt_tokens - len(token_ids)
            ):
                errors.append(
                    f"API render reference usage delta is invalid: {case_id}"
                )
            expected_delta = 1 if case_id in API_RENDER_TOOL_CASES else 0
            if usage_delta != expected_delta:
                errors.append(
                    f"API render full-server usage offset changed: {case_id}"
                )
        if case.get("max_tokens") != API_RENDER_MAX_TOKENS.get(case_id):
            errors.append(f"API render max_tokens changed: {case_id}")

        placeholders = case.get("mm_placeholders")
        if not isinstance(placeholders, dict):
            errors.append(f"API render placeholders are invalid: {case_id}")
            continue
        occupied: list[tuple[int, int]] = []
        for modality, ranges in placeholders.items():
            if modality not in {"image", "video"} or not isinstance(ranges, list):
                errors.append(f"API render placeholder modality is invalid: {case_id}")
                continue
            pad_token = 248056 if modality == "image" else 248057
            for value in ranges:
                if not isinstance(value, dict):
                    errors.append(f"API render placeholder is invalid: {case_id}")
                    continue
                offset = value.get("offset")
                length = value.get("length")
                if (
                    not isinstance(offset, int)
                    or isinstance(offset, bool)
                    or not isinstance(length, int)
                    or isinstance(length, bool)
                    or offset < 0
                    or length <= 0
                    or offset + length > len(token_ids)
                    or token_ids[offset : offset + length] != [pad_token] * length
                ):
                    errors.append(
                        f"API render placeholder span is invalid: {case_id}"
                    )
                    continue
                occupied.append((offset, offset + length))
        occupied.sort()
        if any(left[1] > right[0] for left, right in zip(occupied, occupied[1:])):
            errors.append(f"API render placeholder spans overlap: {case_id}")
        observed_counts = {
            modality: len(ranges)
            for modality, ranges in placeholders.items()
            if isinstance(ranges, list)
        }
        if observed_counts != API_RENDER_MEDIA_COUNTS.get(case_id):
            errors.append(f"API render media placeholder counts changed: {case_id}")

        structured_outputs = case.get("structured_outputs")
        expected_structured = (
            {"json": EXPECTED_TOOL_JSON_SCHEMA}
            if case_id == "tool_forced_image"
            else None
        )
        if structured_outputs != expected_structured:
            errors.append(f"API render structured output changed: {case_id}")

    forced = next(
        (
            case
            for case in cases
            if isinstance(case, dict)
            and case.get("case_id") == "tool_forced_image"
        ),
        None,
    )
    structured = forced.get("structured_outputs") if isinstance(forced, dict) else None
    schema = structured.get("json") if isinstance(structured, dict) else None
    if schema != EXPECTED_TOOL_JSON_SCHEMA:
        errors.append("named tool render is not bound to a JSON schema")

    decision = payload.get("decision")
    by_id = {
        case.get("case_id"): case
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    recomputed = {
        "success_render_cases_20_of_20": tuple(case_ids)
        == REQUIRED_API_RENDER_CASES,
        "non_tool_non_stream_render_matches_full_usage": all(
            by_id.get(case_id, {}).get("reference_usage_delta") == 0
            for case_id in REQUIRED_API_RENDER_CASES
            if case_id not in API_RENDER_TOOL_CASES
            and case_id not in API_RENDER_USAGELESS_CASES
        ),
        "tool_full_server_usage_offset_one": all(
            by_id.get(case_id, {}).get("reference_usage_delta") == 1
            for case_id in API_RENDER_TOOL_CASES
        ),
        "named_tool_json_schema_bound": schema == EXPECTED_TOOL_JSON_SCHEMA,
    }
    if not isinstance(decision, dict):
        errors.append("API render decision is missing")
    else:
        if set(decision) != {*_API_RENDER_TRUE_DECISIONS, "g1_passed", "g2_passed"}:
            errors.append("API render decision set changed")
        for name, expected in recomputed.items():
            if decision.get(name) is not expected:
                errors.append(f"API render decision is inconsistent: {name}")
        for name in ("g1_passed", "g2_passed"):
            if decision.get(name) is not False:
                errors.append(f"API render evidence must not claim {name[:2].upper()}")
    if payload.get("qualified") is not all(recomputed.values()):
        errors.append("API render qualification is inconsistent with its decisions")
    return errors
