#!/usr/bin/env python3
"""Qualify the complete G4 matrix from at least five alternating pairs."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
    verify_manifest_integrity,
)


SCHEMA = "aima-amd395-qwen36/vl-performance/v1"
COMPARABLE_MATRIX_SCHEMA = (
    "aima-amd395-qwen36/vl-performance-comparable-matrix/v1"
)
STARTUP_LIMIT_MS = 44_900.0
LATENCY_RATIOS = (
    "ttft_candidate_over_reference",
    "total_candidate_over_reference",
)
THROUGHPUT_RATIOS = (
    "prefill_tps_candidate_over_reference",
    "vision_tps_candidate_over_reference",
)
AVAILABILITY_SCHEMA = "aima-amd395-qwen36/vl-performance-reference-availability/v1"
TEXT_MATRIX_SCHEMA = (
    "aima-amd395-qwen36/native-v151-paired-text-matrix/v1"
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def paired_identity_module():
    path = Path(__file__).with_name("summarize-vl-performance-pairs.py")
    spec = importlib.util.spec_from_file_location("vl_paired_identity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load paired VL identity helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_record(pair_dir: Path) -> dict[str, Any]:
    summary_path = pair_dir / "summary.json"
    summary = load_object(summary_path)
    return {
        "pair_dir": logical_path(pair_dir),
        "summary_sha256": sha256_file(summary_path),
        "pair_index": summary.get("pair_index"),
        "execution_order": summary.get("execution_order"),
        "complete": summary.get("complete") is True,
        "matrix": summary.get("matrix"),
        "checks": summary.get("checks"),
        "process_groups": summary.get("process_groups"),
        "cells": summary.get("cells"),
    }


def identity_from_availability(
    path: Path,
    matrix: Mapping[str, Any],
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    availability = load_object(path)
    if (
        availability.get("schema") != AVAILABILITY_SCHEMA
        or availability.get("complete") is not True
        or verify_manifest_integrity(availability)
    ):
        raise ValueError("reference availability manifest is incomplete or corrupt")
    identity = availability.get("artifact_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("reference availability artifact identity is missing")
    identity_checks = identity.get("checks")
    if (
        not isinstance(identity_checks, Mapping)
        or not identity_checks
        or not all(value is True for value in identity_checks.values())
    ):
        raise ValueError("reference availability artifact identity is incomplete")
    binding = matrix.get("bindings", {}).get("reference_availability", {})
    if not isinstance(binding, Mapping) or binding.get("sha256") != sha256_file(path):
        raise ValueError("comparable matrix does not bind reference availability")
    result = dict(identity)
    if candidate is not None:
        probe_candidate = result.get("candidate")
        if not isinstance(probe_candidate, Mapping):
            raise ValueError(
                "reference availability probe candidate identity is missing"
            )
        result["reference_availability_probe_candidate"] = dict(
            probe_candidate
        )
        result["candidate"] = dict(candidate)
    return {
        **result,
        "reference_availability": file_component(path, logical_path(path)),
    }


def aggregate_cell(
    cell: Mapping[str, Any], pair_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    records: list[Mapping[str, Any]] = []
    for pair in pair_records:
        matches = [
            item
            for item in pair.get("cells", [])
            if isinstance(item, Mapping) and item.get("cell_id") == cell_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"pair {pair.get('pair_index')} does not contain one {cell_id}"
            )
        records.append(matches[0])
    ratio_names = tuple(records[0].get("comparisons", {})) if records else ()
    if not ratio_names or any(
        tuple(record.get("comparisons", {})) != ratio_names for record in records
    ):
        raise ValueError(f"comparison membership drifted for {cell_id}")
    medians = {
        name: float(
            statistics.median(
                finite_float(
                    record["comparisons"].get(name),
                    f"{cell_id} pair comparison {name}",
                )
                for record in records
            )
        )
        for name in ratio_names
    }
    exact_media_cache = cell.get("media_cache_expectation") == "exact"
    invariant_names = ("contract", "process_group")
    checks = {
        "all_pairs_complete": all(record.get("complete") is True for record in records),
        "all_pair_checks_pass": all(
            all(record.get("checks", {}).values())
            and all(record.get("diagnostic_checks", {}).values())
            for record in records
        ),
        "cross_pair_contract_exact": all(
            all(record.get(name) == records[0].get(name) for record in records)
            for name in invariant_names
        ),
        "pair_indices_exact": [record.get("pair_index") for record in records]
        == list(range(1, len(records) + 1)),
    }
    gates = {
        "ttft_paired_median_lte_reference": (
            medians.get("ttft_candidate_over_reference", math.inf) <= 1.0
        ),
        "total_paired_median_lte_reference": (
            medians.get("total_candidate_over_reference", math.inf) <= 1.0
        ),
        "prefill_paired_median_gte_reference": (
            medians.get("prefill_tps_candidate_over_reference", -math.inf)
            >= 1.0
        ),
    }
    if exact_media_cache:
        gates["vision_cache_hit_candidate_median_not_executed"] = (
            medians.get("vision_cache_hit_candidate_seconds", math.inf) == 0.0
        )
    else:
        gates["vision_paired_median_gte_reference"] = (
            medians.get("vision_tps_candidate_over_reference", -math.inf)
            >= 1.0
        )
    if int(cell.get("output_tokens", 0)) > 1:
        gates["decode_paired_median_gte_reference"] = (
            medians.get("decode_tps_candidate_over_reference", -math.inf)
            >= 1.0
        )
    complete = all(checks.values())
    return {
        "cell_id": cell_id,
        "source_execution_cells": cell.get("source_execution_cells"),
        "source_capability_cases": cell.get("source_capability_cases"),
        "coverage": cell.get("coverage"),
        "context_bucket": cell.get("context_bucket"),
        "output_tokens": cell.get("output_tokens"),
        "cache_process": cell.get("cache_process"),
        "media_cache_expectation": cell.get("media_cache_expectation"),
        "metric_applicability": {
            "vision_throughput": not exact_media_cache,
            "vision_cache_hit_execution": exact_media_cache,
        },
        "complete": complete,
        "qualified": complete and all(gates.values()),
        "checks": checks,
        "gates": gates,
        "paired_medians": medians,
        "pairs": list(records),
    }


def text_decode_curve(
    text_matrix: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[tuple[int, int], float]:
    """Return exact-candidate text decode medians keyed by effective shape."""
    candidate = identity.get("candidate")
    text_candidate = text_matrix.get("engines", {}).get("candidate")
    host = identity.get("host")
    text_host = text_matrix.get("host")
    if not isinstance(candidate, Mapping) or not isinstance(
        text_candidate, Mapping
    ):
        raise ValueError("candidate identity is missing from text/G4 evidence")
    files = candidate.get("files")
    if not isinstance(files, list):
        raise ValueError("G4 candidate closure is missing")
    engine_files = [
        item
        for item in files
        if isinstance(item, Mapping)
        and item.get("path") == "aima-engine-native"
    ]
    if len(engine_files) != 1:
        raise ValueError("G4 candidate closure must contain one native engine")
    if (
        text_matrix.get("schema") != TEXT_MATRIX_SCHEMA
        or text_matrix.get("complete") is not True
        or text_matrix.get("qualified") is not True
        or text_matrix.get("all_cells_pass") is not True
        or text_matrix.get("text_request_path_idle") is not True
        or candidate.get("source_commit")
        != text_candidate.get("build_info", {}).get("source_commit")
        or engine_files[0].get("sha256") != text_candidate.get("sha256")
        or not isinstance(host, Mapping)
        or not isinstance(text_host, Mapping)
        or host.get("hostname") != text_host.get("hostname")
    ):
        raise ValueError("text decode curve does not bind the G4 candidate/host")
    cells = text_matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("text decode curve has no cells")
    curve: dict[tuple[int, int], float] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("text decode curve contains a malformed cell")
        input_tokens = cell.get("input_tokens")
        output_tokens = cell.get("output_tokens")
        if (
            not isinstance(input_tokens, int)
            or isinstance(input_tokens, bool)
            or not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens <= 1
        ):
            continue
        candidate_medians = cell.get("candidate_medians")
        if not isinstance(candidate_medians, Mapping):
            raise ValueError("text decode candidate medians are missing")
        decode_tps = finite_float(
            candidate_medians.get("decode_tps"),
            f"text q{input_tokens}/o{output_tokens} decode throughput",
        )
        if decode_tps <= 0.0 or (input_tokens, output_tokens) in curve:
            raise ValueError("text decode curve is duplicate or nonpositive")
        curve[(input_tokens, output_tokens)] = decode_tps
    if not curve:
        raise ValueError("text decode curve contains no positive-output cells")
    return curve


def apply_text_decode_retention(
    cells: Sequence[dict[str, Any]], curve: Mapping[tuple[int, int], float]
) -> list[dict[str, Any]]:
    """Gate VL engine decode against the same binary's exact text curve."""
    records: list[dict[str, Any]] = []
    for cell in cells:
        output_tokens = int(cell.get("output_tokens", 0))
        if output_tokens <= 1:
            continue
        prompt_values: list[int] = []
        decode_values: list[float] = []
        for pair in cell.get("pairs", []):
            measurements = pair.get("measurements")
            candidate = (
                measurements.get("candidate")
                if isinstance(measurements, Mapping)
                else None
            )
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"candidate measurements are missing for {cell['cell_id']}"
                )
            prompt_tokens = candidate.get("prompt_tokens")
            completion_tokens = candidate.get("completion_tokens")
            decode_tokens = candidate.get("engine_decode_tokens_executed")
            if (
                not isinstance(prompt_tokens, int)
                or isinstance(prompt_tokens, bool)
                or completion_tokens != output_tokens
                or decode_tokens != output_tokens - 1
            ):
                raise ValueError(
                    f"engine decode shape is invalid for {cell['cell_id']}"
                )
            prompt_values.append(prompt_tokens)
            decode_values.append(
                finite_float(
                    candidate.get("engine_decode_tokens_per_second"),
                    f"{cell['cell_id']} engine decode throughput",
                )
            )
        if len(set(prompt_values)) != 1 or not decode_values:
            raise ValueError(
                f"effective prompt shape drifted for {cell['cell_id']}"
            )
        input_tokens = prompt_values[0]
        key = (input_tokens, output_tokens)
        if key not in curve:
            raise ValueError(
                f"text decode baseline is missing for q{input_tokens}/o{output_tokens}"
            )
        candidate_median = float(statistics.median(decode_values))
        text_median = float(curve[key])
        retention = candidate_median / text_median
        cell["paired_medians"][
            "decode_tps_candidate_over_text_product"
        ] = retention
        cell["gates"]["decode_paired_median_gte_text_product"] = (
            retention >= 1.0
        )
        cell["qualified"] = cell["complete"] and all(cell["gates"].values())
        records.append(
            {
                "cell_id": cell["cell_id"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "candidate_measurements": decode_values,
                "candidate_median": candidate_median,
                "text_product_median": text_median,
                "candidate_over_text_product": retention,
                "qualified": retention >= 1.0,
            }
        )
    return records


def aggregate(
    pair_records: Sequence[Mapping[str, Any]],
    matrix: Mapping[str, Any],
    identity: Mapping[str, Any],
    minimum_pairs: int = 5,
    text_matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ordered = sorted(pair_records, key=lambda item: int(item["pair_index"]))
    comparable_partition = matrix.get("schema") == COMPARABLE_MATRIX_SCHEMA
    unavailable = matrix.get("reference_unavailable", [])
    if not isinstance(unavailable, list):
        raise ValueError("matrix reference-unavailable ledger is malformed")
    cells = matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("matrix has no cells")
    cell_results = [aggregate_cell(cell, ordered) for cell in cells]
    text_decode_retention = (
        apply_text_decode_retention(
            cell_results, text_decode_curve(text_matrix, identity)
        )
        if text_matrix is not None
        else []
    )
    startup_by_group: dict[str, list[float]] = {"disabled": [], "enabled": []}
    for pair in ordered:
        groups = pair.get("process_groups")
        if not isinstance(groups, list):
            raise ValueError("pair process groups are missing")
        for group in groups:
            if not isinstance(group, Mapping):
                raise ValueError("pair process group is malformed")
            name = group.get("process_group")
            if name not in startup_by_group:
                raise ValueError(f"unknown process group in pair: {name}")
            value = group.get("candidate_health", {}).get(
                "command_to_ready_wall_ms"
            )
            startup_by_group[str(name)].append(
                finite_float(value, f"candidate {name} startup")
            )
    startup = {
        name: {
            "measurements_ms": values,
            "median_ms": float(statistics.median(values)),
            "limit_ms": STARTUP_LIMIT_MS,
            "qualified": statistics.median(values) <= STARTUP_LIMIT_MS,
        }
        for name, values in startup_by_group.items()
    }
    indices = [int(pair["pair_index"]) for pair in ordered]
    matrix_sha = identity.get("matrix", {}).get("sha256")
    full_status = matrix.get("full_cell_status", [])
    full_cell_count = (
        matrix.get("derivation", {}).get("full_cell_count")
        if comparable_partition
        else len(cell_results)
    )
    checks = {
        "minimum_five_pairs": len(ordered) >= minimum_pairs,
        "pair_indices_consecutive": indices == list(range(1, len(ordered) + 1)),
        "execution_order_alternates": all(
            pair.get("execution_order")
            == (
                "reference candidate"
                if int(pair["pair_index"]) % 2
                else "candidate reference"
            )
            for pair in ordered
        ),
        "all_pair_summaries_complete": all(pair.get("complete") is True for pair in ordered),
        "matrix_exact_across_pairs": all(
            pair.get("matrix", {}).get("sha256") == matrix_sha for pair in ordered
        ),
        "all_cells_complete": all(cell["complete"] for cell in cell_results),
        "coverage_exact": matrix.get("complete") is True
        and {
            key: sorted(values)
            for key, values in matrix.get("required_coverage", {}).items()
        }
        == {
            key: sorted(values)
            for key, values in matrix.get("observed_coverage", {}).items()
        },
        "artifact_identity_complete": bool(identity),
        "reference_availability_partition_complete": (
            not comparable_partition
            or (
                matrix.get("checks", {}).get(
                    "partition_covers_every_frozen_cell"
                )
                is True
                and isinstance(full_status, list)
                and len(full_status) == full_cell_count
                and len(cell_results) + len(unavailable) == full_cell_count
            )
        ),
        "unavailable_cells_not_counted_as_candidate_pass": all(
            isinstance(item, Mapping)
            and item.get("status") == "reference_unavailable"
            and item.get("performance_decision")
            == "not_comparable_not_candidate_pass"
            for item in unavailable
        ),
    }
    if text_matrix is not None:
        checks["every_decode_cell_bound_to_text_product_curve"] = (
            len(text_decode_retention)
            == sum(int(cell.get("output_tokens", 0)) > 1 for cell in cells)
        )
    paired_gate_name = (
        "every_comparable_cell_paired_median_qualified"
        if comparable_partition
        else "every_cell_paired_median_qualified"
    )
    gates = {
        paired_gate_name: all(cell["qualified"] for cell in cell_results),
        "candidate_startup_disabled_median_lte_44_9_seconds": startup[
            "disabled"
        ]["qualified"],
        "candidate_startup_enabled_median_lte_44_9_seconds": startup[
            "enabled"
        ]["qualified"],
    }
    if text_matrix is not None:
        gates["every_decode_cell_gte_text_product_curve"] = all(
            item["qualified"] for item in text_decode_retention
        )
    complete = all(checks.values())
    qualified = complete and all(gates.values())
    if qualified and unavailable:
        decision = "qualified_on_all_reference_available_cells"
    elif qualified:
        decision = "qualified_on_all_frozen_cells"
    else:
        decision = "not_qualified"
    return {
        "schema": SCHEMA,
        "complete": complete,
        "qualified": qualified,
        "decision": decision,
        "qualification_scope": (
            "all fixed-reference-available cells; reference-unavailable cells "
            "remain explicit non-passing capability records"
            if comparable_partition
            else "all frozen G4 cells"
        ),
        "scope": (
            "complete frozen G4 coverage accounting with paired native-versus-"
            "vLLM qualification for every comparable cell"
            if comparable_partition
            else "complete frozen G4 native-versus-vLLM performance matrix"
        ),
        "pair_count": len(ordered),
        "cell_count": len(cell_results),
        "full_cell_count": full_cell_count,
        "comparable_cell_count": len(cell_results),
        "reference_unavailable_cell_count": len(unavailable),
        "all_frozen_cells_performance_compared": not unavailable,
        "checks": checks,
        "gates": gates,
        "candidate_startup": startup,
        "text_product_decode_retention": text_decode_retention,
        "artifact_identity": dict(identity),
        "matrix_derivation": matrix.get("derivation"),
        "required_coverage": matrix.get("required_coverage"),
        "full_required_coverage": matrix.get(
            "full_required_coverage", matrix.get("required_coverage")
        ),
        "full_cell_status": full_status if comparable_partition else None,
        "reference_unavailable": unavailable,
        "cells": cell_results,
        "raw_pairs": [
            {
                "pair_dir": pair["pair_dir"],
                "summary_sha256": pair["summary_sha256"],
                "pair_index": pair["pair_index"],
                "execution_order": pair["execution_order"],
            }
            for pair in ordered
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", action="append", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--text-matrix", type=Path)
    parser.add_argument("--reference-availability", type=Path)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--candidate-source-commit")
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_name(args.output.name + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(
            f"refusing to overwrite G4 matrix summary or sidecar: {args.output}"
        )
    matrix_path = args.matrix.resolve()
    matrix = load_object(matrix_path)
    if matrix.get("complete") is not True or verify_manifest_integrity(matrix):
        raise SystemExit("performance matrix is incomplete or corrupt")
    text_matrix_path = (
        args.text_matrix.resolve() if args.text_matrix is not None else None
    )
    if (
        matrix.get("schema") == COMPARABLE_MATRIX_SCHEMA
        and text_matrix_path is None
    ):
        parser.error("the comparable G4 partition requires --text-matrix")
    text_matrix = (
        load_object(text_matrix_path)
        if text_matrix_path is not None
        else None
    )
    if args.reference_availability is not None:
        if args.reference_python is not None or args.model_dir is not None:
            parser.error(
                "--reference-availability cannot be combined with "
                "--reference-python or --model-dir"
            )
        if (
            args.candidate_binary is None
            or args.candidate_source_commit is None
        ):
            parser.error(
                "--reference-availability requires --candidate-binary and "
                "--candidate-source-commit"
            )
        availability_path = args.reference_availability.resolve()
        helper = paired_identity_module()
        current_candidate = helper.candidate_identity(
            args.candidate_binary, args.candidate_source_commit
        )
        identity = identity_from_availability(
            availability_path, matrix, current_candidate
        )
    else:
        legacy_args = (
            args.candidate_binary,
            args.candidate_source_commit,
            args.reference_python,
            args.model_dir,
        )
        if any(value is None for value in legacy_args):
            parser.error(
                "provide --reference-availability or every runtime identity argument"
            )
        assert args.candidate_binary is not None
        assert args.candidate_source_commit is not None
        assert args.reference_python is not None
        assert args.model_dir is not None
        helper = paired_identity_module()
        model_index = args.model_dir.resolve() / "model.safetensors.index.json"
        if not model_index.is_file():
            raise SystemExit("model checkpoint index is missing")
        identity = {
            "candidate": helper.candidate_identity(
                args.candidate_binary, args.candidate_source_commit
            ),
            "reference": helper.reference_identity(args.reference_python),
            "model": {
                "checkpoint_index_path": str(model_index),
                "checkpoint_index_sha256": sha256_file(model_index),
            },
        }
    identity["matrix"] = {
        "path": logical_path(matrix_path),
        "bytes": matrix_path.stat().st_size,
        "sha256": sha256_file(matrix_path),
        "integrity": matrix.get("integrity"),
    }
    records = [pair_record(path.resolve()) for path in args.pair_dir]
    result = aggregate(
        records, matrix, identity, text_matrix=text_matrix
    )
    binding_paths = {
        "matrix": matrix_path,
        "pair_runner": ROOT / "scripts/run-vl-performance-matrix-pair.sh",
        "candidate_launcher": (
            ROOT / "scripts/run-native-vl-performance-candidate.sh"
        ),
        "reference_launcher": (
            ROOT / "scripts/run-vllm-vl-performance-reference.sh"
        ),
        "request_capture": ROOT / "scripts/capture-vl-performance-request.py",
        "raw_log_sanitizer": (
            ROOT / "scripts/sanitize-vl-performance-reference-log.py"
        ),
        "reference_entrypoint": (
            ROOT / "scripts/aima_vllm_vl_performance_server.py"
        ),
        "reference_middleware": (
            ROOT / "scripts/vllm_vl_benchmark_middleware.py"
        ),
        "pair_summarizer": (
            ROOT / "scripts/summarize-vl-performance-matrix-pair.py"
        ),
        "diagnostic_summarizer": (
            ROOT / "scripts/summarize-vl-performance-diagnostic.py"
        ),
        "aggregator": Path(__file__).resolve(),
    }
    if text_matrix_path is not None:
        binding_paths["text_product_matrix"] = text_matrix_path
    result["bindings"] = {
        name: file_component(path, logical_path(path))
        for name, path in binding_paths.items()
    }
    result["command_templates"] = {
        "pair": (
            "AIMA_VL_PAIR_INDEX=<1..5> AIMA_VL_MATRIX_DIR=<fresh-pair-dir> "
            "AIMA_VL_MATRIX_PATH=<bound-comparable-matrix> "
            "scripts/run-vl-performance-matrix-pair.sh"
        ),
        "aggregate": (
            "python3 scripts/summarize-vl-performance-matrix.py "
            "--pair-dir <pair-1> ... --pair-dir <pair-5> "
            "--matrix <bound-comparable-matrix> "
            "--text-matrix <bound-exact-candidate-text-matrix> "
            "--reference-availability <bound-availability> "
            "--candidate-binary <exact-native-binary> "
            "--candidate-source-commit <embedded-source-commit> "
            "--output <result>"
        ),
    }
    result = seal_manifest(result)
    atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "complete": result["complete"],
                "qualified": result["qualified"],
                "cells": result["cell_count"],
                "pairs": result["pair_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
