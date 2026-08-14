"""Derive the executable VL boundary envelope from frozen reference evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from aima_engine.vl_capability import (
    EXPECTED_MAX_ITEMS_PER_PROMPT,
    EXPECTED_MAX_MODEL_LEN,
    EXPECTED_MAX_TOKENS_PER_ITEM,
    validate_capability_manifest,
    validate_processor_probe,
)
from aima_engine.vl_reference import (
    ReferenceManifestError,
    canonical_json_sha256,
    seal_manifest,
    verify_manifest_integrity,
)


ENVELOPE_SCHEMA = "aima-amd395-qwen36/vl-capability-envelope/v1"
VISION_BATCH_TOKEN_LIMIT = 16_384
VIDEO_TEMPORAL_FACTOR = 2
VIDEO_SPATIAL_FACTOR = 32
VIDEO_PATCH_SIZE = 16
VIDEO_MERGE_SIZE = 2
VIDEO_MINIMUM_PIXELS = 4_096
VIDEO_MAXIMUM_PIXELS = 25_165_824
VIDEO_SAMPLING_HEIGHT = 256
VIDEO_SAMPLING_WIDTH = 256

IMAGE_BOUNDARY_ROLES = {
    "minimum_source": "minimum",
    "aspect_ratio_200": "maximum-aspect-ratio",
    "aspect_ratio_over_200": "first-rejected-aspect-ratio",
    "factor_minus_one": "discrete-factor-minus-one",
    "factor_exact": "discrete-factor-exact",
    "factor_plus_one": "discrete-factor-plus-one",
    "portrait": "typical-portrait",
    "landscape": "typical-landscape",
    "maximum_pixels": "maximum-output-pixels",
    "above_maximum_pixels": "maximum-output-pixels-clamp",
}

VIDEO_RESIZE_BOUNDARY_ROLES = {
    "below_temporal_factor": "first-rejected-temporal-factor",
    "temporal_factor": "minimum-temporal-factor",
    "below_spatial_factor": "first-rejected-spatial-factor",
    "spatial_factor": "minimum-spatial-factor",
    "aspect_ratio_200": "maximum-aspect-ratio",
    "aspect_ratio_over_200": "first-rejected-aspect-ratio",
    "typical": "typical",
    "maximum_feature_shape": "maximum-feature-shape",
}

VIDEO_SAMPLING_BOUNDARY_ROLES = {
    "below_min_frames": "source-below-minimum",
    "minimum_frames": "minimum-sampled-frames",
    "typical_fps": "typical-fps",
    "maximum_frames": "maximum-sampled-frames",
    "above_maximum_frames": "maximum-sampled-frames-clamp",
    "explicit_num_frames": "explicit-frame-count",
    "fps_num_frames_conflict": "mutually-exclusive-options",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _indexed(records: Any, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(records, list):
        raise ReferenceManifestError(f"{label} must be an array")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ReferenceManifestError(f"{label} contains a malformed record")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in by_id:
            raise ReferenceManifestError(f"{label} case IDs must be unique")
        by_id[case_id] = record
    return by_id


def _boundary_records(
    records: Any, roles: Mapping[str, str], modality: str, stage: str
) -> list[dict[str, Any]]:
    by_id = _indexed(records, f"{modality} {stage} boundaries")
    if set(by_id) != set(roles):
        missing = sorted(set(roles) - set(by_id))
        extra = sorted(set(by_id) - set(roles))
        raise ReferenceManifestError(
            f"{modality} {stage} boundary membership drifted: "
            f"missing={missing}, extra={extra}"
        )
    result: list[dict[str, Any]] = []
    for case_id in roles:
        source = by_id[case_id]
        item = {
            "boundary_id": f"{modality}.{stage}.{case_id}",
            "modality": modality,
            "stage": stage,
            "role": roles[case_id],
            "outcome": source.get("outcome"),
        }
        for name in ("source", "result", "error_class", "error_message"):
            if name in source:
                value = source[name]
                if (
                    name == "result"
                    and stage == "sampling"
                    and isinstance(value, Mapping)
                    and isinstance(value.get("indices"), list)
                ):
                    indices = value["indices"]
                    value = {
                        key: child
                        for key, child in value.items()
                        if key != "indices"
                    } | {
                        "first_index": indices[0] if indices else None,
                        "last_index": indices[-1] if indices else None,
                        "indices_sha256": canonical_json_sha256(indices),
                    }
                item[name] = value
        result.append(item)
    return result


def _vision_batch_count(token_counts: Sequence[int]) -> int:
    batches = 0
    current = 0
    for count in token_counts:
        if count <= 0 or count > VISION_BATCH_TOKEN_LIMIT:
            raise ReferenceManifestError(
                "execution cell contains an invalid per-item token count"
            )
        if current and current + count > VISION_BATCH_TOKEN_LIMIT:
            batches += 1
            current = 0
        current += count
    return batches + int(current != 0)


def _video_sampling_visual_tokens(sampled_frames: int) -> int:
    """Project a frozen 256x256 sampling probe through video smart-resize.

    The sampling probe records frame indices independently from resize.  An
    execution cell must combine the sampled-frame count with the probe's
    256x256 metadata before claiming a visual-token count; borrowing a token
    count from an unrelated resize case produces an impossible request.
    """

    if sampled_frames < VIDEO_TEMPORAL_FACTOR:
        raise ReferenceManifestError(
            "video sampling execution cell is below the temporal factor"
        )
    height = VIDEO_SAMPLING_HEIGHT
    width = VIDEO_SAMPLING_WIDTH
    temporal_rounded = round(sampled_frames / VIDEO_TEMPORAL_FACTOR)
    temporal_rounded *= VIDEO_TEMPORAL_FACTOR
    rounded_height = round(height / VIDEO_SPATIAL_FACTOR) * VIDEO_SPATIAL_FACTOR
    rounded_width = round(width / VIDEO_SPATIAL_FACTOR) * VIDEO_SPATIAL_FACTOR
    rounded_pixels = temporal_rounded * rounded_height * rounded_width
    source_pixels = sampled_frames * height * width
    if rounded_pixels > VIDEO_MAXIMUM_PIXELS:
        beta = math.sqrt(source_pixels / VIDEO_MAXIMUM_PIXELS)
        rounded_height = max(
            VIDEO_SPATIAL_FACTOR,
            math.floor(height / beta / VIDEO_SPATIAL_FACTOR)
            * VIDEO_SPATIAL_FACTOR,
        )
        rounded_width = max(
            VIDEO_SPATIAL_FACTOR,
            math.floor(width / beta / VIDEO_SPATIAL_FACTOR)
            * VIDEO_SPATIAL_FACTOR,
        )
    elif rounded_pixels < VIDEO_MINIMUM_PIXELS:
        beta = math.sqrt(VIDEO_MINIMUM_PIXELS / source_pixels)
        rounded_height = (
            math.ceil(height * beta / VIDEO_SPATIAL_FACTOR)
            * VIDEO_SPATIAL_FACTOR
        )
        rounded_width = (
            math.ceil(width * beta / VIDEO_SPATIAL_FACTOR)
            * VIDEO_SPATIAL_FACTOR
        )
    temporal_grid = (sampled_frames + VIDEO_TEMPORAL_FACTOR - 1) // (
        VIDEO_TEMPORAL_FACTOR
    )
    return (
        temporal_grid
        * (rounded_height // VIDEO_PATCH_SIZE)
        * (rounded_width // VIDEO_PATCH_SIZE)
        // (VIDEO_MERGE_SIZE * VIDEO_MERGE_SIZE)
    )


def _execution_cell(
    cell_id: str,
    boundary_ids: Sequence[str],
    media_token_counts: Sequence[int],
    *,
    expected_outcome: str = "accepted",
    qualification_layers: Sequence[str] = (
        "processor",
        "http",
        "vision",
        "language",
        "generation",
    ),
    note: str | None = None,
) -> dict[str, Any]:
    total = sum(media_token_counts)
    if total > EXPECTED_MAX_MODEL_LEN:
        raise ReferenceManifestError(
            f"execution cell exceeds the frozen encoder budget: {cell_id}"
        )
    result: dict[str, Any] = {
        "cell_id": cell_id,
        "boundary_ids": list(boundary_ids),
        "expected_outcome": expected_outcome,
        "media_items": len(media_token_counts),
        "media_token_counts": list(media_token_counts),
        "aggregate_visual_tokens": total,
        "vision_batch_count": _vision_batch_count(media_token_counts),
        "qualification_layers": list(qualification_layers),
    }
    if note is not None:
        result["note"] = note
    return result


def derive_execution_cells(
    processor_probe: Mapping[str, Any],
) -> list[dict[str, Any]]:
    image = _indexed(processor_probe["image_resize_cases"], "image resize")
    video = _indexed(processor_probe["video_resize_cases"], "video resize")
    sampling = _indexed(
        processor_probe["video_sampling_cases"], "video sampling"
    )

    def tokens(records: Mapping[str, Mapping[str, Any]], case_id: str) -> int:
        result = records[case_id].get("result")
        if not isinstance(result, Mapping) or not isinstance(
            result.get("vision_tokens"), int
        ):
            raise ReferenceManifestError(
                f"accepted processor boundary has no token count: {case_id}"
            )
        return int(result["vision_tokens"])

    image_min = tokens(image, "minimum_source")
    image_landscape = tokens(image, "landscape")
    image_portrait = tokens(image, "portrait")
    image_max = tokens(image, "maximum_pixels")
    video_min = tokens(video, "temporal_factor")
    video_typical = tokens(video, "typical")
    video_max = tokens(video, "maximum_feature_shape")

    def sampled_count(case_id: str) -> int:
        result = sampling[case_id].get("result")
        if not isinstance(result, Mapping) or not isinstance(
            result.get("count"), int
        ):
            raise ReferenceManifestError(
                f"accepted video sampling boundary has no count: {case_id}"
            )
        return int(result["count"])

    video_sampling_min = _video_sampling_visual_tokens(
        sampled_count("minimum_frames")
    )
    video_sampling_typical = _video_sampling_visual_tokens(
        sampled_count("typical_fps")
    )
    video_sampling_max = _video_sampling_visual_tokens(
        sampled_count("maximum_frames")
    )
    if video_sampling_max != _video_sampling_visual_tokens(
        sampled_count("above_maximum_frames")
    ):
        raise ReferenceManifestError(
            "maximum video sampling boundaries disagree after smart-resize"
        )
    max_images = int(processor_probe["vllm_budget"]["max_items_per_prompt"]["image"])
    max_videos = int(processor_probe["vllm_budget"]["max_items_per_prompt"]["video"])

    cells = [
        _execution_cell(
            "image_minimum",
            ["image.resize.minimum_source"],
            [image_min],
        ),
        _execution_cell(
            "image_typical_portrait",
            ["image.resize.portrait"],
            [image_portrait],
        ),
        _execution_cell(
            "image_typical_landscape",
            ["image.resize.landscape"],
            [image_landscape],
        ),
        _execution_cell(
            "image_maximum_pixels",
            ["image.resize.maximum_pixels"],
            [image_max],
        ),
        _execution_cell(
            "image_above_maximum_clamp",
            ["image.resize.above_maximum_pixels"],
            [image_max],
        ),
        _execution_cell(
            "image_aspect_rejection",
            ["image.resize.aspect_ratio_over_200"],
            [],
            expected_outcome="rejected",
            qualification_layers=("processor", "http"),
        ),
        _execution_cell(
            "video_minimum",
            ["video.resize.temporal_factor", "video.resize.spatial_factor"],
            [video_min],
        ),
        _execution_cell(
            "video_typical",
            ["video.resize.typical"],
            [video_typical],
        ),
        _execution_cell(
            "video_maximum_feature_shape",
            ["video.resize.maximum_feature_shape"],
            [video_max],
        ),
        _execution_cell(
            "video_minimum_rejections",
            [
                "video.resize.below_temporal_factor",
                "video.resize.below_spatial_factor",
            ],
            [],
            expected_outcome="rejected",
            qualification_layers=("processor", "http"),
        ),
        _execution_cell(
            "video_aspect_rejection",
            ["video.resize.aspect_ratio_over_200"],
            [],
            expected_outcome="rejected",
            qualification_layers=("processor", "http"),
        ),
        _execution_cell(
            "video_sampling_minimum",
            ["video.sampling.minimum_frames"],
            [video_sampling_min],
        ),
        _execution_cell(
            "video_sampling_typical",
            ["video.sampling.typical_fps"],
            [video_sampling_typical],
        ),
        _execution_cell(
            "video_sampling_maximum",
            [
                "video.sampling.maximum_frames",
                "video.sampling.above_maximum_frames",
            ],
            [video_sampling_max],
        ),
        _execution_cell(
            "video_sampling_option_conflict",
            ["video.sampling.fps_num_frames_conflict"],
            [],
            expected_outcome="rejected",
            qualification_layers=("processor",),
            note=(
                "the frozen evidence is a direct processor option conflict, "
                "not an OpenAI HTTP content-part contract"
            ),
        ),
        _execution_cell(
            "image_count_maximum_small",
            ["image.count.maximum"],
            [image_min] * max_images,
        ),
        _execution_cell(
            "video_count_maximum_small",
            ["video.count.maximum"],
            [video_min] * max_videos,
        ),
        _execution_cell(
            "image_count_over_limit",
            ["image.count.first_rejected"],
            [],
            expected_outcome="rejected",
            qualification_layers=("http",),
        ),
        _execution_cell(
            "video_count_over_limit",
            ["video.count.first_rejected"],
            [],
            expected_outcome="rejected",
            qualification_layers=("http",),
        ),
        _execution_cell(
            "mixed_cross_batch_boundary",
            ["image.resize.maximum_pixels", "video.resize.temporal_factor"],
            [image_max, video_min],
            note="first mixed-media cell above one bounded 16,384-token vision batch",
        ),
        _execution_cell(
            "image_near_window_maximum",
            ["image.resize.maximum_pixels", "image.count.maximum"],
            [image_max] * (max_images - 1),
            note="leaves one maximum-image token block for text and wrapper headroom",
        ),
        _execution_cell(
            "image_full_encoder_budget",
            ["image.resize.maximum_pixels", "image.count.maximum"],
            [image_max] * max_images,
            qualification_layers=("processor", "vision"),
            note=(
                "encoder-budget boundary only; an HTTP prompt also needs "
                "wrapper/text tokens"
            ),
        ),
        _execution_cell(
            "video_full_item_budget",
            ["video.resize.maximum_feature_shape", "video.count.maximum"],
            [video_max] * max_videos,
            note="21 maximum-token videos consume 258,048 of 262,144 encoder tokens",
        ),
    ]
    return cells


def build_envelope(
    processor_probe: Mapping[str, Any],
    capability_manifest: Mapping[str, Any],
    bindings: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    errors = validate_processor_probe(processor_probe)
    errors.extend(verify_manifest_integrity(processor_probe))
    errors.extend(validate_capability_manifest(capability_manifest))
    errors.extend(verify_manifest_integrity(capability_manifest))
    if errors:
        raise ReferenceManifestError(
            "cannot derive VL envelope from invalid evidence:\n- "
            + "\n- ".join(errors)
        )
    budget = processor_probe["vllm_budget"]
    if budget["max_model_len"] != EXPECTED_MAX_MODEL_LEN or budget[
        "max_tokens_per_item"
    ] != EXPECTED_MAX_TOKENS_PER_ITEM or budget[
        "max_items_per_prompt"
    ] != EXPECTED_MAX_ITEMS_PER_PROMPT:
        raise ReferenceManifestError("frozen VL budget drifted")

    image_boundaries = _boundary_records(
        processor_probe["image_resize_cases"],
        IMAGE_BOUNDARY_ROLES,
        "image",
        "resize",
    )
    video_resize_boundaries = _boundary_records(
        processor_probe["video_resize_cases"],
        VIDEO_RESIZE_BOUNDARY_ROLES,
        "video",
        "resize",
    )
    video_sampling_boundaries = _boundary_records(
        processor_probe["video_sampling_cases"],
        VIDEO_SAMPLING_BOUNDARY_ROLES,
        "video",
        "sampling",
    )
    count_boundaries = []
    for modality in ("image", "video"):
        maximum = int(budget["max_items_per_prompt"][modality])
        count_boundaries.extend(
            [
                {
                    "boundary_id": f"{modality}.count.maximum",
                    "modality": modality,
                    "stage": "server-admission",
                    "role": "maximum-item-count",
                    "count": maximum,
                    "outcome": "accepted",
                },
                {
                    "boundary_id": f"{modality}.count.first_rejected",
                    "modality": modality,
                    "stage": "server-admission",
                    "role": "first-rejected-item-count",
                    "count": maximum + 1,
                    "outcome": "rejected",
                },
            ]
        )

    payload = {
        "schema": ENVELOPE_SCHEMA,
        "complete": True,
        "generated_at": generated_at,
        "derivation": {
            "method": (
                "deterministic projection of frozen processor and API "
                "capability evidence"
            ),
            "hand_authored_boundary_values": False,
            "cartesian_product": False,
            "coverage_strategy": (
                "all discrete boundaries plus min/typical/max and pairwise "
                "cross-batch cells"
            ),
        },
        "bindings": dict(bindings),
        "limits": {
            "model_tokens": EXPECTED_MAX_MODEL_LEN,
            "encoder_tokens": int(budget["encoder_budget_tokens"]),
            "vision_batch_tokens": VISION_BATCH_TOKEN_LIMIT,
            "tokens_per_item": dict(budget["max_tokens_per_item"]),
            "items_per_prompt": dict(budget["max_items_per_prompt"]),
        },
        "processor_boundaries": {
            "image_resize": image_boundaries,
            "video_resize": video_resize_boundaries,
            "video_sampling": video_sampling_boundaries,
            "media_count": count_boundaries,
        },
        "execution_cells": derive_execution_cells(processor_probe),
        "decision": {
            "processor_boundary_manifest_complete": True,
            "native_execution_qualification_complete": False,
            "task_quality_qualification_complete": False,
            "g1_passed": False,
            "g2_passed": False,
        },
    }
    return seal_manifest(payload)


def validate_envelope(payload: Mapping[str, Any]) -> list[str]:
    errors = verify_manifest_integrity(payload)
    if payload.get("schema") != ENVELOPE_SCHEMA:
        errors.append(f"VL envelope schema must be {ENVELOPE_SCHEMA}")
    if payload.get("complete") is not True:
        errors.append("VL envelope is incomplete")
    limits = payload.get("limits")
    if not isinstance(limits, Mapping):
        errors.append("VL envelope limits are missing")
    else:
        if limits.get("model_tokens") != EXPECTED_MAX_MODEL_LEN:
            errors.append("VL envelope model-token limit drifted")
        if limits.get("encoder_tokens") != EXPECTED_MAX_MODEL_LEN:
            errors.append("VL envelope encoder-token limit drifted")
        if limits.get("vision_batch_tokens") != VISION_BATCH_TOKEN_LIMIT:
            errors.append("VL envelope vision batch limit drifted")
        if limits.get("tokens_per_item") != EXPECTED_MAX_TOKENS_PER_ITEM:
            errors.append("VL envelope per-item token limits drifted")
        if limits.get("items_per_prompt") != EXPECTED_MAX_ITEMS_PER_PROMPT:
            errors.append("VL envelope media-count limits drifted")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        errors.append("VL envelope bindings are missing")
    else:
        for name in (
            "processor_probe",
            "api_capability_manifest",
            "derivation_module",
            "generator",
        ):
            component = bindings.get(name)
            if not isinstance(component, Mapping) or not _SHA256.fullmatch(
                str(component.get("sha256", ""))
            ):
                errors.append(f"VL envelope binding is invalid: {name}")
    cells = payload.get("execution_cells")
    if not isinstance(cells, list) or len(cells) != 23:
        errors.append("VL envelope must contain exactly 23 execution cells")
    elif len(
        {
            cell.get("cell_id")
            for cell in cells
            if isinstance(cell, Mapping)
        }
    ) != len(cells):
        errors.append("VL envelope execution cell IDs must be unique")
    else:
        by_id = {
            cell["cell_id"]: cell
            for cell in cells
            if isinstance(cell, Mapping) and isinstance(cell.get("cell_id"), str)
        }
        expected = {
            "video_sampling_minimum": (128, 1),
            "video_sampling_typical": (640, 1),
            "video_sampling_maximum": (9_600, 1),
            "mixed_cross_batch_boundary": (16_388, 2),
            "image_near_window_maximum": (245_760, 15),
            "image_full_encoder_budget": (262_144, 16),
            "video_full_item_budget": (258_048, 21),
        }
        for cell_id, (tokens, batches) in expected.items():
            cell = by_id.get(cell_id)
            if not isinstance(cell, Mapping):
                errors.append(f"VL envelope execution cell is missing: {cell_id}")
            elif cell.get("aggregate_visual_tokens") != tokens or cell.get(
                "vision_batch_count"
            ) != batches:
                errors.append(f"VL envelope execution cell drifted: {cell_id}")
    return errors
