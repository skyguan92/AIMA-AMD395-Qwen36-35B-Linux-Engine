#!/usr/bin/env python3
"""Generate the frozen pairwise G4 request matrix from VL reference evidence."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    ReferenceManifestError,
    atomic_json,
    file_component,
    load_json_object,
    seal_manifest,
)


SCHEMA = "aima-amd395-qwen36/vl-performance-matrix/v1"
MEDIA_ROOT = "${AIMA_VL_MEDIA_ROOT}"
PROMPT_NONCE = "${AIMA_VL_PROMPT_NONCE}"
CONTEXT_BUCKETS = {
    "short": [1, 1_023],
    "1k": [960, 1_088],
    "8k": [8_128, 8_256],
    "32k": [32_704, 32_832],
    "128k": [131_008, 131_136],
    "near_262144": [261_120, 262_143],
}
REQUIRED_COVERAGE = {
    "image": [
        "single_minimum",
        "single_typical",
        "single_maximum",
        "multi_typical",
        "multi_maximum_count",
        "multi_near_window_maximum",
        "portrait",
        "landscape",
        "square",
    ],
    "video": [
        "single_minimum",
        "single_typical",
        "single_maximum_shape",
        "sampling_minimum",
        "sampling_typical",
        "sampling_maximum",
        "sampling_above_maximum_clamp",
        "multi_typical",
        "multi_maximum_count",
    ],
    "mixed": ["image_video", "text_media_interleave", "multi_turn"],
    "context": list(CONTEXT_BUCKETS),
    "output": ["1", "512", "1024"],
    "cache": [
        "disabled",
        "cold_media",
        "warm_media",
        "media_exact_hit",
        "a_b_a",
    ],
}


def indexed(records: Any, key: str, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(records, list):
        raise ReferenceManifestError(f"{label} must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ReferenceManifestError(f"{label} contains a malformed item")
        value = record.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ReferenceManifestError(f"{label} IDs must be unique strings")
        result[value] = record
    return result


def media_part(modality: str, fixture_path: str) -> dict[str, Any]:
    url = f"file://{MEDIA_ROOT}/vl-envelope-v0.1.0/{fixture_path}"
    if modality == "image":
        return {"type": "image_url", "image_url": {"url": url}}
    if modality == "video":
        return {"type": "video_url", "video_url": {"url": url}}
    raise ReferenceManifestError(f"unsupported performance modality: {modality}")


def text_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def instruction(output_tokens: int) -> str:
    prefix = f"Benchmark {PROMPT_NONCE}. Inspect every supplied media item. "
    if output_tokens == 1:
        return prefix + "Answer with one token."
    return (
        prefix
        + "For deterministic decode measurement, emit the digit 1 followed by "
        "one space repeatedly. Do not explain, conclude, or stop; continue until "
        "the server output limit."
    )


def request(messages: list[dict[str, Any]], output_tokens: int) -> dict[str, Any]:
    return {
        "max_tokens": output_tokens,
        "messages": messages,
        "temperature": 0,
    }


def one_turn(
    fixtures: Sequence[tuple[str, str]], output_tokens: int, *, interleave: bool = False
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [text_part(instruction(output_tokens))]
    for index, (modality, fixture_path) in enumerate(fixtures):
        if interleave and index:
            content.append(text_part(f"Media item {index + 1} follows."))
        content.append(media_part(modality, fixture_path))
    return request([{"role": "user", "content": content}], output_tokens)


def mixed_multi_turn(
    image_path: str, video_path: str, output_tokens: int
) -> dict[str, Any]:
    return request(
        [
            {"role": "system", "content": "Follow the benchmark exactly."},
            {
                "role": "user",
                "content": [
                    text_part(f"Benchmark {PROMPT_NONCE}. Remember this image."),
                    media_part("image", image_path),
                ],
            },
            {"role": "assistant", "content": "Acknowledged."},
            {
                "role": "user",
                "content": [
                    text_part(
                        "Now compare it with this video. For deterministic decode "
                        "measurement, emit the digit 1 followed by one space "
                        "repeatedly. Do not explain, conclude, or stop; continue "
                        "until the server output limit."
                    ),
                    media_part("video", video_path),
                ],
            },
        ],
        output_tokens,
    )


def dump_request(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_matrix(
    capability: Mapping[str, Any],
    envelope: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
    *,
    requests_dir: Path,
    logical_requests_dir: str,
    bindings: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    capability_cases = indexed(capability.get("cases"), "case_id", "capability cases")
    execution_cells = indexed(
        envelope.get("execution_cells"), "cell_id", "envelope execution cells"
    )
    fixtures = indexed(
        fixture_manifest.get("fixtures"), "fixture_id", "envelope fixtures"
    )

    required_capability = {
        "multi_image_interleaved",
        "multi_video",
        "mixed_image_then_video",
        "conversation_prior_image",
    }
    required_execution = {
        "image_minimum",
        "image_typical_portrait",
        "image_typical_landscape",
        "image_maximum_pixels",
        "image_above_maximum_clamp",
        "video_minimum",
        "video_typical",
        "video_maximum_feature_shape",
        "video_sampling_minimum",
        "video_sampling_typical",
        "video_sampling_maximum",
        "image_count_maximum_small",
        "video_count_maximum_small",
        "mixed_cross_batch_boundary",
        "image_near_window_maximum",
    }
    missing_capability = sorted(required_capability - set(capability_cases))
    missing_execution = sorted(required_execution - set(execution_cells))
    if missing_capability or missing_execution:
        raise ReferenceManifestError(
            "frozen matrix sources drifted: "
            f"capability={missing_capability}, envelope={missing_execution}"
        )

    fixture_ids = {
        "image_min": "image-minimum-1x1.png",
        "image_portrait": "image-portrait-256x1024.png",
        "image_landscape": "image-landscape-1024x256.png",
        "image_max": "image-maximum-4096x4096.png",
        "image_above_max": "image-above-maximum-8192x8192.png",
        "video_min": "video-minimum-2f-2fps-32x32.mp4",
        "video_typical": "video-typical-4f-2fps-256x256.mp4",
        "video_max": "video-maximum-2f-2fps-4096x3072.mp4",
        "video_sampling_min": "video-sampling-minimum-48f-24fps-256x256.mp4",
        "video_sampling_typical": "video-sampling-typical-240f-24fps-256x256.mp4",
        "video_sampling_max": "video-sampling-maximum-9216f-24fps-256x256.mp4",
        "video_sampling_above": "video-sampling-above-maximum-18432f-24fps-256x256.mp4",
    }
    missing_fixtures = sorted(set(fixture_ids.values()) - set(fixtures))
    if missing_fixtures:
        raise ReferenceManifestError(
            f"frozen performance fixtures are missing: {missing_fixtures}"
        )

    def fixture(name: str) -> tuple[str, str]:
        record = fixtures[fixture_ids[name]]
        modality = record.get("modality")
        path = record.get("path")
        if modality not in {"image", "video"} or not isinstance(path, str):
            raise ReferenceManifestError(f"malformed frozen fixture: {name}")
        return str(modality), path

    def cell(
        cell_id: str,
        source_cells: Sequence[str],
        source_cases: Sequence[str],
        fixture_names: Sequence[str],
        context: str,
        padding: int,
        output: int,
        coverage: Mapping[str, Sequence[str]],
        *,
        cache_process: str = "disabled",
        cache_expectation: str = "disabled",
        payload: Mapping[str, Any] | None = None,
        interleave: bool = False,
        sequence: str | None = None,
    ) -> dict[str, Any]:
        for source in source_cells:
            if source not in execution_cells:
                raise ReferenceManifestError(f"unknown execution cell: {source}")
        for source in source_cases:
            if source not in capability_cases:
                raise ReferenceManifestError(f"unknown capability case: {source}")
        selected = [fixture(name) for name in fixture_names]
        if payload is None:
            payload = one_turn(selected, output, interleave=interleave)
        request_name = f"{cell_id}.json"
        request_path = requests_dir / request_name
        dump_request(request_path, payload)
        media = []
        for index, name in enumerate(fixture_names):
            record = fixtures[fixture_ids[name]]
            media.append(
                {
                    "index": index,
                    "fixture_id": record["fixture_id"],
                    "modality": record["modality"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
            )
        visual_tokens = sum(
            int(execution_cells[source]["aggregate_visual_tokens"])
            for source in source_cells
            if source not in {
                "image_above_maximum_clamp",
                "image_near_window_maximum",
            }
        )
        if cell_id == "image_multi_max_near_window_output1":
            visual_tokens = 245_760
        elif cell_id.startswith("cache_"):
            visual_tokens = 256
        elif cell_id == "image_multi_typical_q128k_output1":
            visual_tokens = 512
        elif cell_id == "video_multi_typical_q32k_output1":
            visual_tokens = 256
        elif cell_id == "mixed_multi_turn_q8k_output512":
            visual_tokens = 384
        result = {
            "cell_id": cell_id,
            "request": file_component(
                request_path, f"{logical_requests_dir}/{request_name}"
            ),
            "source_execution_cells": list(source_cells),
            "source_capability_cases": list(source_cases),
            "media": media,
            "aggregate_visual_tokens": visual_tokens,
            "context_bucket": context,
            "expected_prompt_tokens_range": CONTEXT_BUCKETS[context],
            "text_padding_tokens": padding,
            "output_tokens": output,
            "cache_process": cache_process,
            "media_cache_expectation": cache_expectation,
            "prefix_cache_expectation": "disabled",
            "coverage": {key: list(values) for key, values in coverage.items()},
        }
        if sequence is not None:
            result["cache_sequence"] = sequence
        return result

    cells = [
        cell("image_min_short_output1", ["image_minimum"], [], ["image_min"], "short", 0, 1, {"image": ["single_minimum", "square"], "context": ["short"], "output": ["1"], "cache": ["disabled"]}),
        cell("image_portrait_q1k_output512", ["image_typical_portrait"], [], ["image_portrait"], "1k", 704, 512, {"image": ["single_typical", "portrait"], "context": ["1k"], "output": ["512"], "cache": ["disabled"]}),
        cell("image_landscape_q8k_output1", ["image_typical_landscape"], [], ["image_landscape"], "8k", 7_872, 1, {"image": ["single_typical", "landscape"], "context": ["8k"], "output": ["1"], "cache": ["disabled"]}),
        cell("image_max_q32k_output1", ["image_maximum_pixels"], [], ["image_max"], "32k", 16_320, 1, {"image": ["single_maximum", "square"], "context": ["32k"], "output": ["1"], "cache": ["disabled"]}),
        cell("image_multi_typical_q128k_output1", ["image_typical_portrait", "image_typical_landscape"], ["multi_image_interleaved"], ["image_portrait", "image_landscape"], "128k", 130_496, 1, {"image": ["multi_typical", "portrait", "landscape"], "context": ["128k"], "output": ["1"], "cache": ["disabled"]}, interleave=True),
        cell("image_count_max_q8k_output1", ["image_count_maximum_small"], ["multi_image_interleaved"], ["image_min"] * 16, "8k", 7_088, 1, {"image": ["multi_maximum_count", "square"], "context": ["8k"], "output": ["1"], "cache": ["disabled"]}, interleave=True),
        cell("image_multi_max_near_window_output1", ["image_near_window_maximum", "image_above_maximum_clamp"], ["multi_image_interleaved"], ["image_max"] * 14 + ["image_above_max"], "near_262144", 15_700, 1, {"image": ["multi_near_window_maximum", "single_maximum", "square"], "context": ["near_262144"], "output": ["1"], "cache": ["disabled"]}, interleave=True),
        cell("video_min_short_output1", ["video_minimum"], [], ["video_min"], "short", 0, 1, {"video": ["single_minimum"], "context": ["short"], "output": ["1"], "cache": ["disabled"]}),
        cell("video_typical_q1k_output1024", ["video_typical"], [], ["video_typical"], "1k", 832, 1024, {"video": ["single_typical"], "context": ["1k"], "output": ["1024"], "cache": ["disabled"]}),
        cell("video_max_shape_q32k_output1", ["video_maximum_feature_shape"], [], ["video_max"], "32k", 20_416, 1, {"video": ["single_maximum_shape"], "context": ["32k"], "output": ["1"], "cache": ["disabled"]}),
        cell("video_sampling_min_q32k_output1", ["video_sampling_minimum"], [], ["video_sampling_min"], "32k", 32_576, 1, {"video": ["sampling_minimum"], "context": ["32k"], "output": ["1"], "cache": ["disabled"]}),
        cell("video_sampling_typical_q128k_output1", ["video_sampling_typical"], [], ["video_sampling_typical"], "128k", 130_368, 1, {"video": ["sampling_typical"], "context": ["128k"], "output": ["1"], "cache": ["disabled"]}),
        cell("video_sampling_max_q32k_output1", ["video_sampling_maximum"], [], ["video_sampling_max"], "32k", 23_104, 1, {"video": ["sampling_maximum"], "context": ["32k"], "output": ["1"], "cache": ["disabled"]}),
        cell("video_sampling_clamp_q32k_output1", ["video_sampling_maximum"], [], ["video_sampling_above"], "32k", 23_104, 1, {"video": ["sampling_above_maximum_clamp"], "context": ["32k"], "output": ["1"], "cache": ["disabled"]}),
        cell("video_multi_typical_q32k_output1", ["video_typical", "video_typical"], ["multi_video"], ["video_typical", "video_typical"], "32k", 32_448, 1, {"video": ["multi_typical"], "context": ["32k"], "output": ["1"], "cache": ["disabled"]}, interleave=True),
        cell("video_count_max_q1k_output1", ["video_count_maximum_small"], ["multi_video"], ["video_min"] * 21, "1k", 832, 1, {"video": ["multi_maximum_count"], "context": ["1k"], "output": ["1"], "cache": ["disabled"]}, interleave=True),
        cell("mixed_cross_batch_q32k_output1", ["mixed_cross_batch_boundary"], ["mixed_image_then_video"], ["image_max", "video_min"], "32k", 16_304, 1, {"mixed": ["image_video", "text_media_interleave"], "context": ["32k"], "output": ["1"], "cache": ["disabled"]}, interleave=True),
        cell("mixed_multi_turn_q8k_output512", ["image_typical_portrait", "video_typical"], ["mixed_image_then_video", "conversation_prior_image"], ["image_portrait", "video_typical"], "8k", 7_744, 512, {"mixed": ["image_video", "multi_turn"], "context": ["8k"], "output": ["512"], "cache": ["disabled"]}, payload=mixed_multi_turn(fixture("image_portrait")[1], fixture("video_typical")[1], 512)),
        cell("cache_a_disabled_output1", ["image_typical_portrait"], [], ["image_portrait"], "short", 0, 1, {"image": ["single_typical", "portrait"], "context": ["short"], "output": ["1"], "cache": ["disabled"]}),
        cell("cache_a_cold_output1", ["image_typical_portrait"], [], ["image_portrait"], "short", 0, 1, {"cache": ["cold_media", "a_b_a"], "context": ["short"], "output": ["1"]}, cache_process="enabled", cache_expectation="cold", sequence="A1"),
        cell("cache_a_exact_output1", ["image_typical_portrait"], [], ["image_portrait"], "short", 0, 1, {"cache": ["warm_media", "media_exact_hit", "a_b_a"], "context": ["short"], "output": ["1"]}, cache_process="enabled", cache_expectation="exact", sequence="A2"),
        cell("cache_b_cold_output1", ["image_typical_landscape"], [], ["image_landscape"], "short", 0, 1, {"cache": ["cold_media", "a_b_a"], "context": ["short"], "output": ["1"]}, cache_process="enabled", cache_expectation="cold", sequence="B"),
        cell("cache_a_restored_output1", ["image_typical_portrait"], [], ["image_portrait"], "short", 0, 1, {"cache": ["warm_media", "media_exact_hit", "a_b_a"], "context": ["short"], "output": ["1"]}, cache_process="enabled", cache_expectation="exact", sequence="A3"),
    ]

    observed: dict[str, set[str]] = {key: set() for key in REQUIRED_COVERAGE}
    for current in cells:
        for surface, values in current["coverage"].items():
            observed.setdefault(surface, set()).update(values)
    missing_coverage = {
        surface: sorted(set(required) - observed.get(surface, set()))
        for surface, required in REQUIRED_COVERAGE.items()
        if set(required) - observed.get(surface, set())
    }
    if missing_coverage:
        raise ReferenceManifestError(
            f"generated G4 matrix lacks required coverage: {missing_coverage}"
        )

    disabled = [item["cell_id"] for item in cells if item["cache_process"] == "disabled"]
    enabled = [item["cell_id"] for item in cells if item["cache_process"] == "enabled"]
    payload = {
        "schema": SCHEMA,
        "complete": True,
        "generated_at": generated_at,
        "bindings": dict(bindings),
        "derivation": {
            "cartesian_product": False,
            "strategy": (
                "all accepted min/typical/max and discrete sampling/count "
                "boundaries plus pairwise context/output/media/cache combinations"
            ),
            "rejected_boundary_performance": False,
            "minimum_alternating_pairs_per_cell": 5,
        },
        "context_buckets": CONTEXT_BUCKETS,
        "output_lengths": [1, 512, 1024],
        "process_groups": [
            {
                "process_group": "disabled",
                "media_cache": "disabled",
                "prefix_cache": "disabled",
                "balanced_orders": [disabled, list(reversed(disabled))],
            },
            {
                "process_group": "enabled",
                "media_cache": "enabled",
                "prefix_cache": "disabled",
                "balanced_orders": [enabled, enabled],
                "ordered_cache_sequence": ["A1", "A2", "B", "A3"],
            },
        ],
        "required_coverage": REQUIRED_COVERAGE,
        "observed_coverage": {
            key: sorted(values) for key, values in observed.items()
        },
        "cells": cells,
    }
    return seal_manifest(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capability-manifest",
        type=Path,
        default=ROOT / "benchmarks/results/vl-capability-manifest.json",
    )
    parser.add_argument(
        "--envelope-manifest",
        type=Path,
        default=ROOT / "benchmarks/results/vl-capability-envelope-v0.1.0.json",
    )
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        default=ROOT / "benchmarks/fixtures/vl-envelope-v0.1.0/fixtures-manifest.json",
    )
    parser.add_argument(
        "--requests-dir",
        type=Path,
        default=ROOT / "benchmarks/fixtures/vl-performance-v0.1.0/requests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/fixtures/vl-performance-v0.1.0/matrix.json",
    )
    parser.add_argument("--generated-at")
    args = parser.parse_args()

    capability_path = args.capability_manifest.resolve()
    envelope_path = args.envelope_manifest.resolve()
    fixture_path = args.fixture_manifest.resolve()
    requests_dir = args.requests_dir.resolve()
    output = args.output.resolve()
    script = Path(__file__).resolve()
    bindings = {
        "capability_manifest": file_component(
            capability_path, "benchmarks/results/vl-capability-manifest.json"
        ),
        "capability_envelope": file_component(
            envelope_path,
            "benchmarks/results/vl-capability-envelope-v0.1.0.json",
        ),
        "fixture_manifest": file_component(
            fixture_path,
            "benchmarks/fixtures/vl-envelope-v0.1.0/fixtures-manifest.json",
        ),
        "generator": file_component(
            script, "scripts/generate-vl-performance-matrix.py"
        ),
    }
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    try:
        result = build_matrix(
            load_json_object(capability_path),
            load_json_object(envelope_path),
            load_json_object(fixture_path),
            requests_dir=requests_dir,
            logical_requests_dir=(
                "benchmarks/fixtures/vl-performance-v0.1.0/requests"
            ),
            bindings=bindings,
            generated_at=generated_at,
        )
    except ReferenceManifestError as exc:
        parser.error(str(exc))
    print(atomic_json(output, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
