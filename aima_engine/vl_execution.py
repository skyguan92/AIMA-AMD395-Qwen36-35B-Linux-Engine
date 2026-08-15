"""Executable request plan and validators for the frozen native VL envelope."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

from aima_engine.vl_envelope import VISION_BATCH_TOKEN_LIMIT, validate_envelope
from aima_engine.vl_reference import sha256_file, verify_manifest_integrity


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCHEMA = "aima-amd395-qwen36/vl-envelope-fixtures/v1"
QUALIFICATION_SCHEMA = (
    "aima-amd395-qwen36/native-vl-envelope-qualification/v1"
)
MODEL_ID = "aima-amd395-qwen36-35b"
MAX_MODEL_TOKENS = 262_144
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


EXPECTED_FIXTURES: dict[str, dict[str, Any]] = {
    "image-minimum-1x1.png": {
        "modality": "image",
        "width": 1,
        "height": 1,
    },
    "image-portrait-256x1024.png": {
        "modality": "image",
        "width": 256,
        "height": 1024,
    },
    "image-landscape-1024x256.png": {
        "modality": "image",
        "width": 1024,
        "height": 256,
    },
    "image-maximum-4096x4096.png": {
        "modality": "image",
        "width": 4096,
        "height": 4096,
    },
    "image-above-maximum-8192x8192.png": {
        "modality": "image",
        "width": 8192,
        "height": 8192,
    },
    "image-rejected-aspect-201x1.png": {
        "modality": "image",
        "width": 201,
        "height": 1,
    },
    "video-minimum-2f-2fps-32x32.mp4": {
        "modality": "video",
        "width": 32,
        "height": 32,
        "frame_count": 2,
        "fps": 2,
    },
    "video-typical-4f-2fps-256x256.mp4": {
        "modality": "video",
        "width": 256,
        "height": 256,
        "frame_count": 4,
        "fps": 2,
    },
    "video-maximum-2f-2fps-4096x3072.mp4": {
        "modality": "video",
        "width": 4096,
        "height": 3072,
        "frame_count": 2,
        "fps": 2,
    },
    "video-rejected-temporal-1f-2fps-32x32.mp4": {
        "modality": "video",
        "width": 32,
        "height": 32,
        "frame_count": 1,
        "fps": 2,
    },
    "video-rejected-spatial-2f-2fps-32x31.avi": {
        "modality": "video",
        "width": 32,
        "height": 31,
        "frame_count": 2,
        "fps": 2,
    },
    "video-rejected-aspect-2f-2fps-6432x32.mp4": {
        "modality": "video",
        "width": 6432,
        "height": 32,
        "frame_count": 2,
        "fps": 2,
    },
    "video-sampling-minimum-48f-24fps-256x256.mp4": {
        "modality": "video",
        "width": 256,
        "height": 256,
        "frame_count": 48,
        "fps": 24,
    },
    "video-sampling-typical-240f-24fps-256x256.mp4": {
        "modality": "video",
        "width": 256,
        "height": 256,
        "frame_count": 240,
        "fps": 24,
    },
    "video-sampling-maximum-9216f-24fps-256x256.mp4": {
        "modality": "video",
        "width": 256,
        "height": 256,
        "frame_count": 9216,
        "fps": 24,
    },
    "video-sampling-above-maximum-18432f-24fps-256x256.mp4": {
        "modality": "video",
        "width": 256,
        "height": 256,
        "frame_count": 18432,
        "fps": 24,
    },
}


# (probe_id, envelope cell_id, ((fixture_id, repeat), ...))
HTTP_PROBE_RECIPES: tuple[
    tuple[str, str, tuple[tuple[str, int], ...]], ...
] = (
    ("image_minimum", "image_minimum", (("image-minimum-1x1.png", 1),)),
    (
        "image_typical_portrait",
        "image_typical_portrait",
        (("image-portrait-256x1024.png", 1),),
    ),
    (
        "image_typical_landscape",
        "image_typical_landscape",
        (("image-landscape-1024x256.png", 1),),
    ),
    (
        "image_maximum_pixels",
        "image_maximum_pixels",
        (("image-maximum-4096x4096.png", 1),),
    ),
    (
        "image_above_maximum_clamp",
        "image_above_maximum_clamp",
        (("image-above-maximum-8192x8192.png", 1),),
    ),
    (
        "image_aspect_rejection",
        "image_aspect_rejection",
        (("image-rejected-aspect-201x1.png", 1),),
    ),
    (
        "video_minimum",
        "video_minimum",
        (("video-minimum-2f-2fps-32x32.mp4", 1),),
    ),
    (
        "video_typical",
        "video_typical",
        (("video-typical-4f-2fps-256x256.mp4", 1),),
    ),
    (
        "video_maximum_feature_shape",
        "video_maximum_feature_shape",
        (("video-maximum-2f-2fps-4096x3072.mp4", 1),),
    ),
    (
        "video_below_temporal_rejection",
        "video_minimum_rejections",
        (("video-rejected-temporal-1f-2fps-32x32.mp4", 1),),
    ),
    (
        "video_below_spatial_rejection",
        "video_minimum_rejections",
        (("video-rejected-spatial-2f-2fps-32x31.avi", 1),),
    ),
    (
        "video_aspect_rejection",
        "video_aspect_rejection",
        (("video-rejected-aspect-2f-2fps-6432x32.mp4", 1),),
    ),
    (
        "video_sampling_minimum",
        "video_sampling_minimum",
        (("video-sampling-minimum-48f-24fps-256x256.mp4", 1),),
    ),
    (
        "video_sampling_typical",
        "video_sampling_typical",
        (("video-sampling-typical-240f-24fps-256x256.mp4", 1),),
    ),
    (
        "video_sampling_maximum",
        "video_sampling_maximum",
        (("video-sampling-maximum-9216f-24fps-256x256.mp4", 1),),
    ),
    (
        "video_sampling_above_maximum",
        "video_sampling_maximum",
        (("video-sampling-above-maximum-18432f-24fps-256x256.mp4", 1),),
    ),
    (
        "image_count_maximum_small",
        "image_count_maximum_small",
        (("image-minimum-1x1.png", 16),),
    ),
    (
        "video_count_maximum_small",
        "video_count_maximum_small",
        (("video-minimum-2f-2fps-32x32.mp4", 21),),
    ),
    (
        "image_count_over_limit",
        "image_count_over_limit",
        (("image-minimum-1x1.png", 17),),
    ),
    (
        "video_count_over_limit",
        "video_count_over_limit",
        (("video-minimum-2f-2fps-32x32.mp4", 22),),
    ),
    (
        "mixed_cross_batch_boundary",
        "mixed_cross_batch_boundary",
        (
            ("image-maximum-4096x4096.png", 1),
            ("video-minimum-2f-2fps-32x32.mp4", 1),
        ),
    ),
    (
        "image_near_window_maximum",
        "image_near_window_maximum",
        (("image-maximum-4096x4096.png", 15),),
    ),
    (
        "video_full_item_budget",
        "video_full_item_budget",
        (("video-maximum-2f-2fps-4096x3072.mp4", 21),),
    ),
)


NON_HTTP_CELL_MODES = {
    "video_sampling_option_conflict": "native-processor-probe",
    "image_full_encoder_budget": "native-vision-probe",
}


def fixture_records(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = payload.get("fixtures")
    if not isinstance(records, list):
        return {}
    return {
        str(record["fixture_id"]): record
        for record in records
        if isinstance(record, Mapping)
        and isinstance(record.get("fixture_id"), str)
    }


def validate_fixture_manifest(
    payload: Mapping[str, Any], fixture_root: Path,
) -> list[str]:
    errors = verify_manifest_integrity(payload)
    if payload.get("schema") != FIXTURE_SCHEMA:
        errors.append(f"VL envelope fixture schema must be {FIXTURE_SCHEMA}")
    if payload.get("complete") is not True:
        errors.append("VL envelope fixture manifest is incomplete")
    generator = payload.get("generator")
    if not isinstance(generator, Mapping):
        errors.append("VL envelope fixture generator binding is missing")
    else:
        expected_path = "scripts/generate-vl-envelope-fixtures.py"
        path = ROOT / expected_path
        if generator.get("path") != expected_path or not path.is_file():
            errors.append("VL envelope fixture generator path drifted")
        elif (
            generator.get("bytes") != path.stat().st_size
            or generator.get("sha256") != sha256_file(path)
        ):
            errors.append("VL envelope fixture generator binding drifted")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append("VL envelope fixture runtime is missing")
    else:
        for name, expected in (
            ("numpy", "2.1.3"),
            ("pillow", "12.2.0"),
            ("imageio_ffmpeg", "0.6.0"),
        ):
            if runtime.get(name) != expected:
                errors.append(f"VL envelope fixture {name} pin drifted")
    records = fixture_records(payload)
    if set(records) != set(EXPECTED_FIXTURES):
        errors.append("VL envelope fixture set drifted")
        return errors
    root = fixture_root.resolve()
    for fixture_id, expected in EXPECTED_FIXTURES.items():
        record = records[fixture_id]
        if any(record.get(key) != value for key, value in expected.items()):
            errors.append(f"VL envelope fixture metadata drifted: {fixture_id}")
        relative = record.get("path")
        if relative != fixture_id or Path(str(relative)).name != relative:
            errors.append(f"VL envelope fixture path is unsafe: {fixture_id}")
            continue
        path = root / str(relative)
        digest = record.get("sha256")
        if not path.is_file() or not _SHA256.fullmatch(str(digest)):
            errors.append(f"VL envelope fixture is missing: {fixture_id}")
        elif record.get("bytes") != path.stat().st_size:
            errors.append(f"VL envelope fixture size drifted: {fixture_id}")
        elif sha256_file(path) != digest:
            errors.append(f"VL envelope fixture hash drifted: {fixture_id}")
    return errors


def _batch_metrics(token_counts: Sequence[int]) -> tuple[int, int]:
    batches = 0
    current = 0
    maximum = 0
    for count in token_counts:
        if current and current + count > VISION_BATCH_TOKEN_LIMIT:
            maximum = max(maximum, current)
            batches += 1
            current = 0
        current += count
    if current:
        maximum = max(maximum, current)
        batches += 1
    return batches, maximum


def _media_part(modality: str, uri: str) -> dict[str, Any]:
    field = f"{modality}_url"
    return {"type": field, field: {"url": uri}}


def build_http_probe_specs(
    envelope: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
    fixture_root: Path,
) -> list[dict[str, Any]]:
    envelope_errors = validate_envelope(envelope)
    fixture_errors = validate_fixture_manifest(fixture_manifest, fixture_root)
    if envelope_errors or fixture_errors:
        raise ValueError(
            "cannot build native VL envelope probes:\n- "
            + "\n- ".join((*envelope_errors, *fixture_errors))
        )
    cells = {cell["cell_id"]: cell for cell in envelope["execution_cells"]}
    records = fixture_records(fixture_manifest)
    probes: list[dict[str, Any]] = []
    for probe_id, cell_id, uses in HTTP_PROBE_RECIPES:
        cell = cells[cell_id]
        content: list[dict[str, Any]] = []
        replacements: dict[str, Any] = {}
        modality_counts: Counter[str] = Counter()
        for fixture_id, repeat in uses:
            record = records[fixture_id]
            modality = str(record["modality"])
            uri = (fixture_root.resolve() / fixture_id).as_uri()
            replacements[uri] = {
                "fixture": fixture_id,
                "transport": "local",
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for _ in range(repeat):
                content.append(_media_part(modality, uri))
            modality_counts[modality] += repeat
        content.append({"type": "text", "text": "Reply with one token."})
        expected_accept = cell["expected_outcome"] == "accepted"
        expected: dict[str, Any] = {
            "cell_id": cell_id,
            "outcome": cell["expected_outcome"],
            "status_code": 200 if expected_accept else 400,
            "media_counts": dict(sorted(modality_counts.items())),
        }
        if expected_accept:
            batch_count, maximum_batch_tokens = _batch_metrics(
                cell["media_token_counts"]
            )
            expected.update(
                {
                    "visual_tokens": cell["aggregate_visual_tokens"],
                    "vision_patches": cell["aggregate_visual_tokens"] * 4,
                    "vision_batch_count": batch_count,
                    "vision_max_batch_tokens": maximum_batch_tokens,
                    "vision_max_batch_patches": maximum_batch_tokens * 4,
                }
            )
        surfaces = sorted(
            set(modality_counts)
            | ({"error"} if not expected_accept else {"generation"})
        )
        probes.append(
            {
                "probe_id": probe_id,
                "cell_id": cell_id,
                "surfaces": surfaces,
                "expected_accept": expected_accept,
                "expected": expected,
                "payload": {
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                    "max_tokens": 1,
                    "stream": False,
                },
                "replacements": replacements,
            }
        )
    return probes


def validate_http_observation(
    observation: Mapping[str, Any], expected: Mapping[str, Any],
) -> dict[str, bool]:
    status = observation.get("status_code")
    response = observation.get("response")
    checks = {
        "status_exact": status == expected.get("status_code"),
        "acceptance_exact": observation.get("accepted")
        == (expected.get("outcome") == "accepted"),
    }
    if expected.get("outcome") != "accepted":
        error = response.get("error") if isinstance(response, Mapping) else None
        checks.update(
            {
                "compatible_error_type": isinstance(error, Mapping)
                and error.get("type") == "invalid_request_error",
                "compatible_error_code": isinstance(error, Mapping)
                and error.get("code") == "bad_request",
                "nonempty_error_message": isinstance(error, Mapping)
                and isinstance(error.get("message"), str)
                and bool(error["message"]),
            }
        )
        return checks
    native = (
        response.get("aima_amd395") if isinstance(response, Mapping) else None
    )
    vl = native.get("vl") if isinstance(native, Mapping) else None
    mrope = native.get("mrope") if isinstance(native, Mapping) else None
    usage = response.get("usage") if isinstance(response, Mapping) else None
    choices = response.get("choices") if isinstance(response, Mapping) else None
    media_counts = expected.get("media_counts")
    prompt_tokens = (
        usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    )
    completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    )
    aot_segments = (
        native.get("aot_prefill_segments")
        if isinstance(native, Mapping)
        else None
    )
    padded_tokens = (
        native.get("padded_prefill_tokens")
        if isinstance(native, Mapping)
        else None
    )
    expected_unified_launches = 0
    if (
        aot_segments == 1
        and isinstance(padded_tokens, int)
        and padded_tokens > 0
    ):
        expected_unified_launches = 10
    checks.update(
        {
            "native_runtime": isinstance(native, Mapping)
            and str(native.get("runtime", "")).startswith("native-resident-"),
            "no_oracle_reads": isinstance(native, Mapping)
            and native.get("oracle_tensor_reads") == 0,
            "one_token_generation": isinstance(usage, Mapping)
            and completion_tokens == 1
            and isinstance(choices, list)
            and len(choices) == 1
            and choices[0].get("finish_reason") == "length",
            "model_window_respected": isinstance(prompt_tokens, int)
            and isinstance(completion_tokens, int)
            and prompt_tokens + completion_tokens <= MAX_MODEL_TOKENS,
            "mrope_dispatch_accounted": isinstance(mrope, Mapping)
            and isinstance(aot_segments, int)
            and mrope.get("full_attention_launches") == aot_segments * 10
            and mrope.get("full_attention_launches")
            == mrope.get("fmha_launches", -1)
            + mrope.get("unified_attention_launches", -1),
            "mrope_initial_padding_only": isinstance(mrope, Mapping)
            and mrope.get("unified_attention_launches")
            == expected_unified_launches,
            "vl_enabled": isinstance(vl, Mapping) and vl.get("enabled") is True,
            "media_count_exact": isinstance(vl, Mapping)
            and vl.get("media_count")
            == sum(media_counts.values())
            if isinstance(media_counts, Mapping)
            else False,
            "image_count_exact": isinstance(vl, Mapping)
            and vl.get("image_count")
            == (
                media_counts.get("image", 0)
                if isinstance(media_counts, Mapping)
                else -1
            ),
            "video_count_exact": isinstance(vl, Mapping)
            and vl.get("video_count")
            == (
                media_counts.get("video", 0)
                if isinstance(media_counts, Mapping)
                else -1
            ),
            "visual_tokens_exact": isinstance(vl, Mapping)
            and vl.get("visual_tokens") == expected.get("visual_tokens"),
            "vision_patches_exact": isinstance(vl, Mapping)
            and vl.get("vision_patches") == expected.get("vision_patches"),
            "vision_batch_count_exact": isinstance(vl, Mapping)
            and vl.get("vision_batch_count")
            == expected.get("vision_batch_count"),
            "vision_max_batch_tokens_exact": isinstance(vl, Mapping)
            and vl.get("vision_max_batch_tokens")
            == expected.get("vision_max_batch_tokens"),
            "vision_max_batch_patches_exact": isinstance(vl, Mapping)
            and vl.get("vision_max_batch_patches")
            == expected.get("vision_max_batch_patches"),
            "vision_executed": isinstance(vl, Mapping)
            and isinstance(vl.get("vision_encode_wall_ms"), (int, float))
            and vl["vision_encode_wall_ms"] > 0,
        }
    )
    return checks


def execution_cell_coverage(
    envelope: Mapping[str, Any], probes: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    cells = {cell["cell_id"] for cell in envelope["execution_cells"]}
    modes = {str(probe["cell_id"]): "http" for probe in probes}
    modes.update(NON_HTTP_CELL_MODES)
    return {cell_id: modes.get(cell_id, "missing") for cell_id in sorted(cells)}


def validate_processor_probe_observation(
    return_code: int, stdout: str, stderr: str,
) -> dict[str, bool]:
    return {
        "exit_zero": return_code == 0,
        "exact_pass_marker": stdout == "native_vl_processor_test: PASS\n",
        "empty_stderr": stderr == "",
    }


def validate_vision_probe_observation(
    observation: Mapping[str, Any], envelope_cell: Mapping[str, Any],
) -> dict[str, bool]:
    output_elements = observation.get("output_elements_per_batch")
    expected_finite = observation.get("expected_finite_output_elements")
    return {
        "schema_exact": observation.get("schema")
        == "aima-amd395-qwen36/native-vl-envelope-vision-probe/v1",
        "complete": observation.get("complete") is True,
        "cell_exact": observation.get("cell_id")
        == envelope_cell.get("cell_id")
        == "image_full_encoder_budget",
        "media_items_exact": observation.get("media_items")
        == envelope_cell.get("media_items")
        == 16,
        "visual_tokens_exact": observation.get("visual_tokens")
        == envelope_cell.get("aggregate_visual_tokens")
        == 262_144,
        "vision_patches_exact": observation.get("vision_patches")
        == 1_048_576,
        "vision_batch_count_exact": observation.get("vision_batch_count")
        == envelope_cell.get("vision_batch_count")
        == 16,
        "vision_max_batch_tokens_exact": observation.get(
            "vision_max_batch_tokens"
        )
        == 16_384,
        "vision_max_batch_patches_exact": observation.get(
            "vision_max_batch_patches"
        )
        == 65_536,
        "all_batches_executed": observation.get("executed_batches") == 16,
        "all_outputs_finite": isinstance(output_elements, int)
        and output_elements > 0
        and expected_finite == output_elements * 16
        and observation.get("finite_output_elements") == expected_finite,
        "repeat_deterministic": observation.get("repeat_deterministic") is True
        and _SHA256.fullmatch(
            str(observation.get("repeat_output_sha256", ""))
        )
        is not None,
        "visual_weights_complete": observation.get("weight_payload_bytes")
        == 893_142_496,
        "attention_image_bound": _SHA256.fullmatch(
            str(observation.get("attention_image_sha256", ""))
        )
        is not None,
        "positive_vision_time": isinstance(
            observation.get("total_vision_wall_ms"), (int, float)
        )
        and observation["total_vision_wall_ms"] > 0,
    }
