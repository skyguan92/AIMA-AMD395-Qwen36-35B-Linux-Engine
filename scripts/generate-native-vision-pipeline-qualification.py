#!/usr/bin/env python3
"""Generate an exact-source full native vision-pipeline qualification."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import date
import hashlib
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    load_json_object,
    seal_manifest,
)


CASE_ORDER = (
    "image_local_png",
    "video_local_mp4",
    "multi_image",
    "multi_video",
    "mixed_image_video",
)
EXPECTED_SHAPES = {
    "image_local_png": (256, 64, 1),
    "video_local_mp4": (128, 32, 1),
    "multi_image": (640, 160, 1),
    "multi_video": (320, 80, 1),
    "mixed_image_video": (384, 96, 2),
}
BOUNDARY_ORDER = (
    "vision_block_0",
    "vision_block_13",
    "vision_block_26",
    "vision_merger",
)
CANDIDATE_SOURCE_FILES = (
    ROOT / "native/include/aima/native_vision_pipeline.h",
    ROOT / "native/src/native_vision_pipeline.hip.cpp",
)
QUALIFICATION_FILES = (
    ROOT / "native/tools/vision_pipeline_oracle_probe.hip.cpp",
    ROOT / "scripts/build-native-vision-pipeline-probe.sh",
    Path(__file__).resolve(),
)
DEPENDENCIES = {
    "encoder_qualification": (
        ROOT / "benchmarks/results/native-vision-aot-encoder-v0.1.0.json"
    ),
    "merger_qualification": (
        ROOT / "benchmarks/results/native-vision-merger-v0.1.0.json"
    ),
    "multimedia_block_oracle": (
        ROOT
        / "benchmarks/results/native-vision-multimedia-block-oracle-v0.1.0.json"
    ),
    "full_model_oracle": ROOT / "benchmarks/results/vl-oracle-manifest.json",
}
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object")
    return value


def validate_identity(
    source_commit: str,
    candidate_binary_sha256: str,
    probe_binary_sha256: str,
    attention_image_sha256: str,
    qualification_source_commit: str,
) -> None:
    for label, value in (
        ("source", source_commit),
        ("qualification source", qualification_source_commit),
    ):
        if SHA1.fullmatch(value) is None:
            raise ValueError(f"{label} commit must be a lowercase SHA-1")
    for label, value in (
        ("candidate binary", candidate_binary_sha256),
        ("probe binary", probe_binary_sha256),
        ("attention image", attention_image_sha256),
    ):
        if SHA256.fullmatch(value) is None:
            raise ValueError(f"{label} hash must be a lowercase SHA-256")


def summarize_case(
    case_id: str,
    payload: dict[str, Any],
    attention_image_sha256: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    comparisons = require_mapping(payload.get("comparisons"), f"{case_id}.comparisons")
    expected_patches, expected_tokens, expected_groups = EXPECTED_SHAPES[case_id]
    comparison_values = [
        require_mapping(comparisons.get(name), f"{case_id}.{name}")
        for name in BOUNDARY_ORDER
    ]
    checks = {
        "schema_exact": payload.get("schema")
        == "aima-amd395-qwen36/native-vision-pipeline-oracle/v2",
        "complete": payload.get("complete") is True,
        "shape_exact": (
            payload.get("patches") == expected_patches
            and payload.get("merged_tokens") == expected_tokens
            and payload.get("group_count") == expected_groups
            and isinstance(payload.get("groups"), list)
            and len(payload["groups"]) == expected_groups
        ),
        "attention_image_exact": payload.get("attention_image_sha256")
        == attention_image_sha256,
        "boundary_set_exact": set(comparisons) == set(BOUNDARY_ORDER),
        "all_boundaries_bit_exact": all(
            item.get("passed") is True
            and item.get("bit_exact") is True
            and int(item.get("elements", 0)) > 0
            and item.get("exact_elements") == item.get("elements")
            and item.get("finite_elements") == item.get("elements")
            and item.get("expected_sha256") == item.get("actual_sha256")
            and float(item.get("relative_l2_error", 1.0)) == 0.0
            and float(item.get("cosine_similarity", 0.0)) == 1.0
            for item in comparison_values
        ),
        "repeat_deterministic": (
            payload.get("repeat_deterministic") is True
            and payload.get("repeat_actual_sha256")
            == comparison_values[-1].get("actual_sha256")
        ),
    }
    boundaries = [
        {
            "name": name,
            "elements": item.get("elements"),
            "exact_elements": item.get("exact_elements"),
            "sha256": item.get("actual_sha256"),
        }
        for name, item in zip(BOUNDARY_ORDER, comparison_values, strict=True)
    ]
    record = {
        "case_id": case_id,
        "patches": payload.get("patches"),
        "merged_tokens": payload.get("merged_tokens"),
        "group_count": payload.get("group_count"),
        "groups": payload.get("groups"),
        "temporary_bytes": payload.get("temporary_bytes"),
        "metadata_resident_bytes": payload.get("metadata_resident_bytes"),
        "library_workspace_bytes": payload.get("library_workspace_bytes"),
        "median_ms": payload.get("median_ms"),
        "boundaries": boundaries,
        "repeat_actual_sha256": payload.get("repeat_actual_sha256"),
        "relative_l2_error": max(
            float(item.get("relative_l2_error", 1.0))
            for item in comparison_values
        ),
        "cosine_similarity": min(
            float(item.get("cosine_similarity", 0.0))
            for item in comparison_values
        ),
        "passed": all(checks.values()),
    }
    return record, checks


def build_payload(
    *,
    run_paths: dict[str, Path],
    source_commit: str,
    candidate_binary_sha256: str,
    probe_binary_sha256: str,
    probe_binary_bytes: int,
    attention_image_sha256: str,
    qualification_source_commit: str,
    recorded_on: str,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    case_checks: dict[str, dict[str, bool]] = {}
    for case_id in CASE_ORDER:
        record, checks = summarize_case(
            case_id,
            load_json_object(run_paths[case_id]),
            attention_image_sha256,
        )
        cases.append(record)
        case_checks[case_id] = checks

    boundary_elements = sum(
        int(boundary["elements"])
        for case in cases
        for boundary in case["boundaries"]
    )
    exact_boundary_elements = sum(
        int(boundary["exact_elements"])
        for case in cases
        for boundary in case["boundaries"]
    )
    merger_elements = sum(
        int(case["boundaries"][-1]["elements"]) for case in cases
    )
    aggregate_checks = {
        "five_case_order_exact": tuple(case["case_id"] for case in cases)
        == CASE_ORDER,
        "all_case_checks_pass": all(
            all(checks.values()) for checks in case_checks.values()
        ),
        "twenty_boundaries": sum(len(case["boundaries"]) for case in cases)
        == 20,
        "boundary_elements_exact": (
            boundary_elements == 6_856_704
            and exact_boundary_elements == boundary_elements
        ),
        "visual_output_elements_exact": merger_elements == 884_736,
    }
    qualified = all(aggregate_checks.values())
    return {
        "schema": "aima-amd395-qwen36/native-vision-pipeline-qualification/v2",
        "recorded_on": recorded_on,
        "complete": qualified,
        "qualified": qualified,
        "source": {
            "commit": source_commit,
            "clean": True,
            "files": [
                file_component(path, str(path.relative_to(ROOT)))
                for path in CANDIDATE_SOURCE_FILES
            ],
        },
        "qualification_tool": {
            "commit": qualification_source_commit,
            "files": [
                file_component(path, str(path.relative_to(ROOT)))
                for path in QUALIFICATION_FILES
            ],
        },
        "candidate_binary_sha256": candidate_binary_sha256,
        "probe_binary": {
            "path": "${AIMA_BUILD_DIR}/native-vision-pipeline-probe",
            "sha256": probe_binary_sha256,
            "bytes": probe_binary_bytes,
            "runtime_python": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
        },
        "attention_image_sha256": attention_image_sha256,
        "dependencies": {
            name: file_component(path, str(path.relative_to(ROOT)))
            for name, path in DEPENDENCIES.items()
        },
        "inputs": {
            case_id: file_component(path, str(path.relative_to(ROOT)))
            for case_id, path in run_paths.items()
        },
        "cases": cases,
        "checks": {"cases": case_checks, "aggregate": aggregate_checks},
        "decision": {
            "case_count": len(cases),
            "passed_cases": sum(case["passed"] is True for case in cases),
            "boundary_comparison_count": sum(
                len(case["boundaries"]) for case in cases
            ),
            "total_boundary_elements": boundary_elements,
            "exact_boundary_elements": exact_boundary_elements,
            "total_visual_output_elements": merger_elements,
            "all_finite": qualified,
            "all_repeats_deterministic": qualified,
            "all_boundaries_bit_exact": qualified,
            "all_27_blocks_executed": qualified,
            "full_visual_pipeline_qualified": qualified,
            "visual_tower_output_qualified": qualified,
        },
    }


def verify_exact(path: Path, expected: dict[str, Any]) -> None:
    if load_json_object(path) != expected:
        raise SystemExit(f"vision-pipeline qualification is stale: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{digest}  {path.name}\n":
        raise SystemExit(f"vision-pipeline sidecar is stale: {sidecar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--qualification-source-commit", required=True)
    parser.add_argument("--candidate-binary-sha256", required=True)
    parser.add_argument("--probe-binary-sha256", required=True)
    parser.add_argument("--probe-binary-bytes", type=int, required=True)
    parser.add_argument("--attention-image-sha256", required=True)
    parser.add_argument("--recorded-on", default="2026-08-21")
    for case_id in CASE_ORDER:
        parser.add_argument(
            "--" + case_id.replace("_", "-"),
            dest=case_id,
            type=Path,
            required=True,
        )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        validate_identity(
            args.source_commit,
            args.candidate_binary_sha256,
            args.probe_binary_sha256,
            args.attention_image_sha256,
            args.qualification_source_commit,
        )
        date.fromisoformat(args.recorded_on)
    except ValueError as exc:
        parser.error(str(exc))
    if args.probe_binary_bytes <= 0:
        parser.error("probe binary bytes must be positive")
    run_paths = {case_id: getattr(args, case_id).resolve() for case_id in CASE_ORDER}
    try:
        for path in (*run_paths.values(), args.output.resolve()):
            path.relative_to(ROOT)
    except ValueError:
        parser.error("qualification output and run inputs must be inside the repository")
    sealed = seal_manifest(
        build_payload(
            run_paths=run_paths,
            source_commit=args.source_commit,
            candidate_binary_sha256=args.candidate_binary_sha256,
            probe_binary_sha256=args.probe_binary_sha256,
            probe_binary_bytes=args.probe_binary_bytes,
            attention_image_sha256=args.attention_image_sha256,
            qualification_source_commit=args.qualification_source_commit,
            recorded_on=args.recorded_on,
        )
    )
    output = args.output.resolve()
    if args.check:
        verify_exact(output, sealed)
        print(f"native vision-pipeline qualification: PASS ({output})")
        return 0
    digest = atomic_json(output, sealed)
    print(
        f"native vision-pipeline qualification: "
        f"{'PASS' if sealed['qualified'] else 'FAIL'} ({digest})"
    )
    return 0 if sealed["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
