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
from typing import Any, Mapping, Sequence


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
        "pair_dir": str(pair_dir),
        "summary_sha256": sha256_file(summary_path),
        "pair_index": summary.get("pair_index"),
        "execution_order": summary.get("execution_order"),
        "complete": summary.get("complete") is True,
        "matrix": summary.get("matrix"),
        "checks": summary.get("checks"),
        "process_groups": summary.get("process_groups"),
        "cells": summary.get("cells"),
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
        "vision_paired_median_gte_reference": (
            medians.get("vision_tps_candidate_over_reference", -math.inf)
            >= 1.0
        ),
    }
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
        "complete": complete,
        "qualified": complete and all(gates.values()),
        "checks": checks,
        "gates": gates,
        "paired_medians": medians,
        "pairs": list(records),
    }


def aggregate(
    pair_records: Sequence[Mapping[str, Any]],
    matrix: Mapping[str, Any],
    identity: Mapping[str, Any],
    minimum_pairs: int = 5,
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
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--candidate-source-commit", required=True)
    parser.add_argument("--reference-python", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite G4 matrix summary: {args.output}")
    matrix_path = args.matrix.resolve()
    matrix = load_object(matrix_path)
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
        "matrix": {
            "path": str(matrix_path),
            "bytes": matrix_path.stat().st_size,
            "sha256": sha256_file(matrix_path),
            "integrity": matrix.get("integrity"),
        },
    }
    records = [pair_record(path.resolve()) for path in args.pair_dir]
    result = aggregate(records, matrix, identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
