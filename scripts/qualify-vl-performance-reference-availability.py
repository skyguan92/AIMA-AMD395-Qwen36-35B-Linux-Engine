#!/usr/bin/env python3
"""Classify fixed-reference failures and derive the comparable G4 matrix."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
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
    verify_manifest_integrity,
)


SPEC_SCHEMA = (
    "aima-amd395-qwen36/vl-performance-reference-availability-spec/v1"
)
AVAILABILITY_SCHEMA = (
    "aima-amd395-qwen36/vl-performance-reference-availability/v1"
)
COMPARABLE_MATRIX_SCHEMA = (
    "aima-amd395-qwen36/vl-performance-comparable-matrix/v1"
)
BENCHMARK_ID = re.compile(
    r"^(.+)\.(?:pair|isolated)-([1-9][0-9]*)$"
)
COMPUTED_TOKENS = re.compile(r"num_computed_tokens=\[([0-9]+)\]")
SCHEDULED_TOKENS = re.compile(
    r"num_scheduled_tokens=\{[^}\n]*:\s*([0-9]+)\}"
)
FROZEN_MODEL_INDEX_SHA256 = (
    "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
)
FROZEN_VLLM_VERSION = "0.19.1rc1.dev300+g29e5d1020"


def finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def benchmark_base(value: object) -> str | None:
    match = BENCHMARK_ID.fullmatch(value) if isinstance(value, str) else None
    return match.group(1) if match is not None else None


def resolve_evidence_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReferenceManifestError(f"{label} must be a relative path")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ReferenceManifestError(f"{label} escapes the evidence root") from exc
    if not candidate.is_file():
        raise ReferenceManifestError(f"{label} is missing: {candidate}")
    return candidate


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def bound_file(path: Path) -> dict[str, Any]:
    return file_component(path, logical_path(path))


def matrix_index(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cells = matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ReferenceManifestError("performance matrix has no cells")
    result: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ReferenceManifestError("performance matrix cell is malformed")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in result:
            raise ReferenceManifestError("performance matrix cell IDs drifted")
        result[cell_id] = cell
    return result


def candidate_record(
    cell: Mapping[str, Any], request_path: Path, health_path: Path
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    request = load_json_object(request_path)
    health = load_json_object(health_path)
    response = mapping(request.get("response"))
    usage = mapping(response.get("usage"))
    captured_request = mapping(request.get("request"))
    native = mapping(request.get("native_metrics"))
    native_vl = mapping(native.get("vl"))
    prefix = mapping(native.get("prefix_cache"))
    prompt_range = cell.get("expected_prompt_tokens_range")
    prompt_tokens = usage.get("prompt_tokens")
    media = captured_request.get("media")
    expected_media = cell.get("media")
    media_hashes = (
        [mapping(item).get("sha256") for item in media]
        if isinstance(media, list)
        else None
    )
    expected_hashes = (
        [mapping(item).get("sha256") for item in expected_media]
        if isinstance(expected_media, list)
        else None
    )
    checks = {
        "candidate_complete": request.get("complete") is True,
        "benchmark_cell_exact": benchmark_base(request.get("benchmark_id"))
        == cell_id,
        "request_template_exact": captured_request.get("template_sha256")
        == mapping(cell.get("request")).get("sha256"),
        "media_order_and_content_exact": media_hashes == expected_hashes,
        "text_padding_exact": mapping(captured_request.get("text_padding")).get(
            "tokens"
        )
        == cell.get("text_padding_tokens"),
        "expected_completion_exact": captured_request.get(
            "expected_completion_tokens"
        )
        == cell.get("output_tokens"),
        "response_completion_exact": usage.get("completion_tokens")
        == cell.get("output_tokens"),
        "prompt_context_exact": (
            isinstance(prompt_range, list)
            and len(prompt_range) == 2
            and all(isinstance(item, int) for item in prompt_range)
            and isinstance(prompt_tokens, int)
            and prompt_range[0] <= prompt_tokens <= prompt_range[1]
        ),
        "native_runtime_exact": str(native.get("runtime", "")).startswith(
            "native-resident-q"
        ),
        "native_visual_tokens_exact": native_vl.get("visual_tokens")
        == cell.get("aggregate_visual_tokens"),
        "native_media_cache_disabled": (
            native_vl.get("media_cache_hits") == 0
            and native_vl.get("media_cache_entries") == 0
        ),
        "native_prefix_cache_disabled": prefix.get("lookup") == "disabled",
        "ready_includes_vision": (
            health.get("vision_warmup_completed") is True
            and health.get("vision_warmup_patches") == 1024
            and health.get("vision_warmup_visual_tokens") == 256
        ),
    }
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "request": bound_file(request_path),
        "health": bound_file(health_path),
        "benchmark_id": request.get("benchmark_id"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_seconds": mapping(request.get("timings")).get("ttft_seconds"),
        "total_seconds": mapping(request.get("timings")).get("total_seconds"),
        "peak_host_rss_bytes": mapping(request.get("memory")).get(
            "peak_host_rss_bytes"
        ),
        "peak_gtt_used_bytes": mapping(request.get("memory")).get(
            "peak_gtt_used_bytes"
        ),
        "hostname": mapping(request.get("host")).get("hostname"),
    }


def stage_record(path: Path, cell_id: str) -> Mapping[str, Any] | None:
    matches: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None
        if (
            isinstance(value, Mapping)
            and benchmark_base(value.get("benchmark_id")) == cell_id
        ):
            matches.append(value)
    return matches[0] if len(matches) == 1 else None


def reference_attempt_record(
    cell: Mapping[str, Any], root: Path, spec: Mapping[str, Any], index: int
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    stage_path = resolve_evidence_path(
        root, spec.get("stage_log"), f"{cell_id} attempt {index} stage log"
    )
    server_log_path = resolve_evidence_path(
        root, spec.get("server_log"), f"{cell_id} attempt {index} server log"
    )
    stage = stage_record(stage_path, cell_id)
    server_log = server_log_path.read_text(encoding="utf-8", errors="replace")
    computed = [int(item) for item in COMPUTED_TOKENS.findall(server_log)]
    scheduled = [int(item) for item in SCHEDULED_TOKENS.findall(server_log)]
    request_path: Path | None = None
    request: Mapping[str, Any] = {}
    if spec.get("request") is not None:
        request_path = resolve_evidence_path(
            root, spec.get("request"), f"{cell_id} attempt {index} request"
        )
        request = load_json_object(request_path)
    captured_request = mapping(request.get("request"))
    response = mapping(request.get("response"))
    checks = {
        "one_matching_stage_record": stage is not None,
        "stage_reports_engine_dead": mapping(stage).get("stats_error")
        == "EngineDeadError",
        "stage_has_no_request_error": mapping(stage).get("request_error") is None,
        "stage_total_observed": finite_positive(
            mapping(mapping(stage).get("timings")).get("asgi_total_secs")
        ),
        "engine_core_fatal_logged": "EngineCore encountered a fatal error"
        in server_log,
        "hip_launch_failure_logged": (
            "HIP error: unspecified launch failure" in server_log
            and "Error code 719" in server_log
        ),
        "request_absent_or_incomplete": not request
        or request.get("complete") is False,
        "request_absent_or_same_cell": not request
        or benchmark_base(request.get("benchmark_id")) == cell_id,
        "request_absent_or_template_exact": not request
        or captured_request.get("template_sha256")
        == mapping(cell.get("request")).get("sha256"),
        "request_absent_or_zero_output": not request
        or (
            mapping(response.get("usage")).get("completion_tokens") in {None, 0}
            and response.get("content_bytes") == 0
        ),
    }
    evidence = {
        "stage_log": bound_file(stage_path),
        "server_log": bound_file(server_log_path),
    }
    if request_path is not None:
        evidence["request"] = bound_file(request_path)
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "benchmark_id": mapping(stage).get("benchmark_id"),
        "asgi_total_seconds": mapping(mapping(stage).get("timings")).get(
            "asgi_total_secs"
        ),
        "computed_tokens_before_failure": computed[-1] if computed else None,
        "scheduled_tokens_at_failure": scheduled[-1] if scheduled else None,
        "request_total_seconds": mapping(request.get("timings")).get(
            "total_seconds"
        ),
        "peak_host_rss_bytes": mapping(request.get("memory")).get(
            "peak_host_rss_bytes"
        ),
        "peak_gtt_used_bytes": mapping(request.get("memory")).get(
            "peak_gtt_used_bytes"
        ),
        "evidence": evidence,
    }


def qualify(
    matrix: Mapping[str, Any],
    matrix_path: Path,
    spec: Mapping[str, Any],
    spec_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    if spec.get("schema") != SPEC_SCHEMA:
        raise ReferenceManifestError("availability evidence spec schema drifted")
    cells = matrix_index(matrix)
    entries = spec.get("cells")
    if not isinstance(entries, list) or not entries:
        raise ReferenceManifestError("availability evidence spec has no cells")
    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    root = spec_path.resolve().parent
    identity_path = resolve_evidence_path(
        root, spec.get("artifact_identity"), "artifact identity"
    )
    identity = load_json_object(identity_path)
    candidate_identity = mapping(identity.get("candidate"))
    reference_identity = mapping(identity.get("reference"))
    model_identity = mapping(identity.get("model"))
    host_identity = mapping(identity.get("host"))
    candidate_files = candidate_identity.get("files")
    candidate_binary = (
        next(
            (
                item
                for item in candidate_files
                if isinstance(item, Mapping)
                and item.get("path") == "aima-engine-native"
            ),
            None,
        )
        if isinstance(candidate_files, list)
        else None
    )
    identity_checks = {
        "identity_schema_exact": identity.get("schema")
        == "aima-amd395-qwen36/vl-performance-artifact-identity/v1",
        "candidate_source_identity_present": isinstance(
            candidate_identity.get("source_commit"), str
        )
        and bool(candidate_identity.get("source_commit")),
        "candidate_binary_sha256_present": isinstance(candidate_binary, Mapping)
        and isinstance(candidate_binary.get("sha256"), str)
        and len(candidate_binary["sha256"]) == 64,
        "candidate_closure_complete": isinstance(candidate_files, list)
        and len(candidate_files) == 7,
        "reference_vllm_exact": str(
            reference_identity.get("vllm_version", "")
        ).startswith(FROZEN_VLLM_VERSION),
        "model_checkpoint_index_exact": model_identity.get(
            "checkpoint_index_sha256"
        )
        == FROZEN_MODEL_INDEX_SHA256,
        "host_identity_present": isinstance(host_identity.get("hostname"), str)
        and bool(host_identity.get("hostname")),
    }
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ReferenceManifestError("availability evidence cell is malformed")
        cell_id = entry.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in cells or cell_id in seen:
            raise ReferenceManifestError("availability cell IDs are invalid")
        seen.add(cell_id)
        candidate_spec = mapping(entry.get("candidate"))
        request_path = resolve_evidence_path(
            root, candidate_spec.get("request"), f"{cell_id} candidate request"
        )
        health_path = resolve_evidence_path(
            root, candidate_spec.get("health"), f"{cell_id} candidate health"
        )
        candidate = candidate_record(cells[cell_id], request_path, health_path)
        attempts_spec = entry.get("reference_attempts")
        if not isinstance(attempts_spec, list) or not attempts_spec:
            raise ReferenceManifestError(f"{cell_id} has no reference attempts")
        attempts = [
            reference_attempt_record(cells[cell_id], root, item, index)
            for index, item in enumerate(attempts_spec, start=1)
            if isinstance(item, Mapping)
        ]
        checks = {
            "candidate_exact_capability_complete": candidate["complete"],
            "reference_attempts_present": len(attempts) == len(attempts_spec),
            "every_reference_attempt_fatal": bool(attempts)
            and all(item["complete"] for item in attempts),
        }
        observed.append(
            {
                "cell_id": cell_id,
                "status": "reference_unavailable",
                "performance_decision": "not_comparable_not_candidate_pass",
                "complete": all(checks.values()),
                "checks": checks,
                "reason": (
                    "fixed vLLM reference fatally exits with HIP launch failure "
                    "719 before producing a token"
                ),
                "candidate_capability": candidate,
                "reference_attempts": attempts,
                "coverage": cells[cell_id].get("coverage"),
            }
        )
    checks = {
        "frozen_matrix_complete": matrix.get("complete") is True,
        "frozen_matrix_integrity_exact": not verify_manifest_integrity(matrix),
        "unavailable_cells_unique": len(seen) == len(observed),
        "every_candidate_capability_complete": all(
            item["candidate_capability"]["complete"] for item in observed
        ),
        "every_reference_attempt_fatal": all(
            item["checks"]["every_reference_attempt_fatal"] for item in observed
        ),
        "unavailable_cells_not_counted_as_candidate_pass": all(
            item["performance_decision"]
            == "not_comparable_not_candidate_pass"
            for item in observed
        ),
        "artifact_identity_complete": all(identity_checks.values()),
        "candidate_host_exact": all(
            item["candidate_capability"].get("hostname")
            == host_identity.get("hostname")
            for item in observed
        ),
    }
    payload = {
        "schema": AVAILABILITY_SCHEMA,
        "complete": all(checks.values()),
        "generated_at": generated_at,
        "scope": (
            "exact frozen G4 cells which the fixed vLLM reference cannot "
            "complete; candidate capability is preserved without awarding a "
            "performance pass"
        ),
        "bindings": {
            "matrix": bound_file(matrix_path),
            "evidence_spec": bound_file(spec_path),
            "artifact_identity": bound_file(identity_path),
            "qualifier": bound_file(Path(__file__)),
        },
        "artifact_identity": {
            "checks": identity_checks,
            "candidate": candidate_identity,
            "reference": {
                "python_sha256": reference_identity.get("python_sha256"),
                "vllm_version": reference_identity.get("vllm_version"),
                "vllm_metadata_sha256": reference_identity.get(
                    "vllm_metadata_sha256"
                ),
            },
            "model": {
                "checkpoint_index_sha256": model_identity.get(
                    "checkpoint_index_sha256"
                ),
            },
            "host": host_identity,
        },
        "checks": checks,
        "cell_count": len(observed),
        "cells": observed,
    }
    return seal_manifest(payload)


def observed_coverage(cells: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, set[str]] = {}
    for cell in cells:
        coverage = cell.get("coverage")
        if not isinstance(coverage, Mapping):
            continue
        for surface, values in coverage.items():
            if isinstance(values, list) and all(isinstance(item, str) for item in values):
                result.setdefault(str(surface), set()).update(values)
    return {key: sorted(values) for key, values in sorted(result.items())}


def derive_comparable_matrix(
    matrix: Mapping[str, Any],
    availability: Mapping[str, Any],
    bindings: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    if availability.get("complete") is not True:
        raise ReferenceManifestError("reference availability evidence is incomplete")
    unavailable_records = availability.get("cells")
    if not isinstance(unavailable_records, list) or not unavailable_records:
        raise ReferenceManifestError("reference availability contains no cells")
    unavailable = {
        item.get("cell_id")
        for item in unavailable_records
        if isinstance(item, Mapping)
        and item.get("status") == "reference_unavailable"
        and item.get("performance_decision")
        == "not_comparable_not_candidate_pass"
        and item.get("complete") is True
    }
    cells = list(matrix_index(matrix).values())
    matrix_ids = {str(cell["cell_id"]) for cell in cells}
    if None in unavailable or not unavailable or not unavailable < matrix_ids:
        raise ReferenceManifestError("reference-unavailable partition is invalid")
    comparable = [cell for cell in cells if cell["cell_id"] not in unavailable]
    groups: list[dict[str, Any]] = []
    for raw_group in matrix.get("process_groups", []):
        if not isinstance(raw_group, Mapping):
            raise ReferenceManifestError("matrix process group is malformed")
        group = dict(raw_group)
        orders = group.get("balanced_orders")
        if not isinstance(orders, list) or len(orders) != 2:
            raise ReferenceManifestError("matrix balanced orders are malformed")
        if group.get("ordered_cache_sequence") is not None:
            original = {item for order in orders for item in order}
            if unavailable & original:
                raise ReferenceManifestError(
                    "reference-unavailable cells cannot truncate cache sequences"
                )
        group["balanced_orders"] = [
            [cell_id for cell_id in order if cell_id not in unavailable]
            for order in orders
            if isinstance(order, list)
        ]
        if len(group["balanced_orders"]) != 2 or not all(
            group["balanced_orders"]
        ):
            raise ReferenceManifestError("comparable process group became empty")
        groups.append(group)
    comparable_ids = {str(cell["cell_id"]) for cell in comparable}
    checks = {
        "full_matrix_complete": matrix.get("complete") is True,
        "full_matrix_integrity_exact": not verify_manifest_integrity(matrix),
        "availability_integrity_exact": not verify_manifest_integrity(availability),
        "partition_covers_every_frozen_cell": (
            comparable_ids | unavailable == matrix_ids
            and not (comparable_ids & unavailable)
        ),
        "unavailable_cells_retained_as_nonpassing_records": len(unavailable)
        == len(unavailable_records),
        "both_cache_process_groups_retained": {
            group.get("process_group") for group in groups
        }
        == {"disabled", "enabled"},
    }
    coverage = observed_coverage(comparable)
    payload = {
        "schema": COMPARABLE_MATRIX_SCHEMA,
        "complete": all(checks.values()),
        "generated_at": generated_at,
        "bindings": dict(bindings),
        "derivation": {
            "strategy": (
                "retain the frozen 23-cell coverage ledger; measure every fixed-"
                "reference-available cell and carry unavailable cells as explicit "
                "non-passing records"
            ),
            "minimum_alternating_pairs_per_comparable_cell": 5,
            "full_cell_count": len(cells),
            "comparable_cell_count": len(comparable),
            "reference_unavailable_cell_count": len(unavailable),
        },
        "checks": checks,
        "full_required_coverage": matrix.get("required_coverage"),
        "full_observed_coverage": matrix.get("observed_coverage"),
        "required_coverage": coverage,
        "observed_coverage": coverage,
        "full_cell_status": [
            {
                "cell_id": cell["cell_id"],
                "status": (
                    "reference_unavailable"
                    if cell["cell_id"] in unavailable
                    else "comparable"
                ),
            }
            for cell in cells
        ],
        "reference_unavailable": [
            {
                "cell_id": item.get("cell_id"),
                "status": item.get("status"),
                "performance_decision": item.get("performance_decision"),
                "coverage": item.get("coverage"),
            }
            for item in unavailable_records
            if isinstance(item, Mapping)
        ],
        "process_groups": groups,
        "cells": comparable,
    }
    return seal_manifest(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--evidence-spec", type=Path, required=True)
    parser.add_argument("--availability-output", type=Path, required=True)
    parser.add_argument("--comparable-matrix-output", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args()
    for output in (args.availability_output, args.comparable_matrix_output):
        if output.exists() or output.with_name(output.name + ".sha256").exists():
            parser.error(f"refusing to overwrite output: {output}")
    matrix_path = args.matrix.resolve()
    spec_path = args.evidence_spec.resolve()
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    try:
        matrix = load_json_object(matrix_path)
        spec = load_json_object(spec_path)
        availability = qualify(
            matrix, matrix_path, spec, spec_path, generated_at
        )
        atomic_json(args.availability_output.resolve(), availability)
        comparable = derive_comparable_matrix(
            matrix,
            availability,
            {
                "full_matrix": bound_file(matrix_path),
                "reference_availability": bound_file(
                    args.availability_output.resolve()
                ),
                "deriver": bound_file(Path(__file__)),
            },
            generated_at,
        )
        atomic_json(args.comparable_matrix_output.resolve(), comparable)
    except ReferenceManifestError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "availability": str(args.availability_output.resolve()),
                "comparable_matrix": str(args.comparable_matrix_output.resolve()),
                "unavailable_cells": availability["cell_count"],
                "comparable_cells": len(comparable["cells"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
