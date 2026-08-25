#!/usr/bin/env python3
"""Generate the exact-binary G3 text no-regression qualification record."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
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


OUTPUT = ROOT / "benchmarks/results/text-v151-nonregression-v0.1.0.json"
ARTIFACT_PATHS = {
    "goal": ROOT / "docs/NATIVE_VL_GOAL.md",
    "correctness": (
        ROOT
        / "benchmarks/runs/native-correctness-20260824-bd01287-final/correctness.json"
    ),
    "doctor": (
        ROOT
        / "benchmarks/runs/native-doctor-20260824-bd01287-final/doctor.json"
    ),
    "openai_features": (
        ROOT
        / (
            "benchmarks/runs/native-openai-features-20260824-bd01287-final/"
            "features.json"
        )
    ),
    "mmlu256": (
        ROOT
        / "benchmarks/runs/native-mmlu256-eval-20260824-bd01287-final/mmlu256.json"
    ),
    "product_surfaces": (
        ROOT
        / (
            "benchmarks/runs/native-product-surfaces-20260824-bd01287-final/"
            "surfaces.json"
        )
    ),
    "paired_text_matrix": (
        ROOT
        / (
            "benchmarks/runs/"
            "native-paired-text-matrix-20260824-bd01287-final-balanced6/"
            "matrix.json"
        )
    ),
    "paired_text_runner": (
        ROOT / "scripts/qualify-native-paired-text-matrix.py"
    ),
    "generator": Path(__file__).resolve(),
}

EXPECTED_CONTEXTS = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    261632,
)
MINIMUM_PAIRED_MATRIX_PAIRS = 6
EXPECTED_BINARY_SHA256 = (
    "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
)
EXPECTED_SOURCE_COMMIT = "bd012874027defa528279a357609b713e9069df4"
EXPECTED_BASELINE_SHA256 = (
    "a9f18771175757af080c8a1d8d7e3fb3906c9aa41b43a496686103b626f80262"
)
EXPECTED_BASELINE_SOURCE_COMMIT = "65c198415709dad6d046c247acab3dc9df2a95a0"
EXPECTED_EXACT_OUTPUT_SHA256 = (
    "aa910692fd03ed4a8e89c04497751e3a28eee36c6148237f7e97c74a6dd68201"
)


def configure_candidate_identity(source_commit: str, binary_sha256: str) -> None:
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("candidate source commit must be a lowercase SHA-1")
    if len(binary_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in binary_sha256
    ):
        raise ValueError("candidate binary hash must be a lowercase SHA-256")
    global EXPECTED_SOURCE_COMMIT, EXPECTED_BINARY_SHA256
    EXPECTED_SOURCE_COMMIT = source_commit
    EXPECTED_BINARY_SHA256 = binary_sha256


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object")
    return value


def require_sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SystemExit(f"{label} must be an array")
    return value


def all_boolean_values_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value) and all(
            all_boolean_values_true(item) for item in value.values()
        )
    if isinstance(value, list):
        return bool(value) and all(all_boolean_values_true(item) for item in value)
    return False


def engine_sha256(payload: dict[str, Any]) -> str | None:
    engine = payload.get("engine")
    return engine.get("sha256") if isinstance(engine, dict) else None


def check_correctness(
    payload: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    cases = require_sequence(payload.get("cases"), "correctness.cases")
    contexts = tuple(case.get("context_tokens") for case in cases)
    exact = require_mapping(
        payload.get("exact_completion"), "correctness.exact_completion"
    )
    maximum_kld = max(
        (float(case.get("kl_divergence", 1.0)) for case in cases),
        default=1.0,
    )
    checks = {
        "complete": payload.get("complete") is True,
        "qualified": payload.get("qualified") is True,
        "candidate_identity": engine_sha256(payload) == EXPECTED_BINARY_SHA256,
        "nine_contexts_exact": contexts == EXPECTED_CONTEXTS,
        "all_top1_match": all(case.get("top1_match") is True for case in cases),
        "all_kld_strictly_below_0_005": all(
            float(case.get("kl_divergence", 1.0)) < 0.005 for case in cases
        ),
        "q8192_exact_128_qualified": (
            exact.get("context_tokens") == 8192
            and exact.get("completion_tokens") == 128
            and exact.get("expected_tokens_match") is True
            and exact.get("qualified") is True
            and exact.get("output_token_ids_sha256")
            == EXPECTED_EXACT_OUTPUT_SHA256
        ),
    }
    summary = {
        "context_tokens": list(contexts),
        "case_count": len(cases),
        "maximum_kl_divergence": maximum_kld,
        "top1_matches": sum(case.get("top1_match") is True for case in cases),
        "exact_completion_tokens": exact.get("completion_tokens"),
        "exact_output_token_ids_sha256": exact.get("output_token_ids_sha256"),
    }
    return checks, summary


def check_mmlu(payload: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    score = require_mapping(payload.get("score"), "mmlu256.score")
    gate = require_mapping(payload.get("gate"), "mmlu256.gate")
    progress = require_mapping(payload.get("progress"), "mmlu256.progress")
    comparison = require_mapping(
        payload.get("reference_comparison"), "mmlu256.reference_comparison"
    )
    checks = {
        "complete": payload.get("complete") is True,
        "qualified": payload.get("qualified") is True,
        "candidate_identity": engine_sha256(payload) == EXPECTED_BINARY_SHA256,
        "all_256_items_completed": (
            progress.get("completed") == 256 and progress.get("items") == 256
        ),
        "score_at_least_218": int(score.get("correct", -1)) >= 218,
        "invalid_answers_zero": score.get("invalid_answers") == 0,
        "prompt_hashes_256_of_256": comparison.get("prompt_token_hash_matches")
        == 256,
        "reference_nonregression": (
            comparison.get("score_nonregression_pass") is True
            and gate.get("reference_score_nonregression") is True
        ),
        "all_text_paths_idle": (
            gate.get("all_text_paths_idle") is True
            and gate.get("text_path_idle_records") == 256
        ),
    }
    summary = {
        "items": score.get("items"),
        "correct": score.get("correct"),
        "invalid_answers": score.get("invalid_answers"),
        "prompt_token_hash_matches": comparison.get("prompt_token_hash_matches"),
        "reference_correct": comparison.get("reference_correct"),
        "correct_delta": comparison.get("correct_delta"),
    }
    return checks, summary


def check_openai_features(
    payload: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    lifecycle = require_mapping(payload.get("lifecycle"), "openai_features.lifecycle")
    stopped = require_mapping(lifecycle.get("stopped"), "openai_features.stopped")
    text_idle = require_mapping(
        payload.get("text_path_idle"), "openai_features.text_path_idle"
    )
    surface_names = (
        "streaming",
        "variable_prompts",
        "resident_prefill_dispatch",
        "prefix_lru",
        "tools",
        "disconnect",
        "validation",
    )
    checks = {
        "complete": payload.get("complete") is True,
        "qualified": payload.get("qualified") is True,
        "candidate_identity": engine_sha256(payload) == EXPECTED_BINARY_SHA256,
        "all_api_surfaces_pass": all(
            require_mapping(payload.get(name), f"openai_features.{name}").get("pass")
            is True
            for name in surface_names
        ),
        "single_resident_model_load": (
            lifecycle.get("pass") is True
            and stopped.get("model_loads") == 1
            and stopped.get("served") == 14
        ),
        "all_fourteen_text_paths_idle": (
            text_idle.get("pass") is True
            and text_idle.get("request_count") == 14
            and text_idle.get("expected_request_count") == 14
            and all_boolean_values_true(text_idle.get("checks"))
        ),
    }
    summary = {
        "surface_count": len(surface_names),
        "served": stopped.get("served"),
        "model_loads": stopped.get("model_loads"),
        "text_path_idle_requests": text_idle.get("request_count"),
    }
    return checks, summary


def check_product_surfaces(
    payload: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    engine = require_mapping(payload.get("engine"), "product_surfaces.engine")
    build_info = require_mapping(
        engine.get("build_info"), "product_surfaces.build_info"
    )
    http = require_mapping(payload.get("http"), "product_surfaces.http")
    prefix = require_mapping(
        payload.get("prefix_cache"), "product_surfaces.prefix_cache"
    )
    baseline = require_mapping(
        prefix.get("baseline_engine"), "product_surfaces.baseline"
    )
    startup = require_mapping(payload.get("startup"), "product_surfaces.startup")
    paired = require_mapping(
        prefix.get("paired_candidate_over_baseline_medians"),
        "product_surfaces.paired_prefix",
    )
    checks = {
        "complete": payload.get("complete") is True,
        "qualified": payload.get("qualified") is True,
        "candidate_identity": (
            engine.get("sha256") == EXPECTED_BINARY_SHA256
            and build_info.get("source_commit") == EXPECTED_SOURCE_COMMIT
        ),
        "baseline_identity": (
            baseline.get("sha256") == EXPECTED_BASELINE_SHA256
            and require_mapping(
                baseline.get("build_info"), "product_surfaces.baseline.build_info"
            ).get("source_commit")
            == EXPECTED_BASELINE_SOURCE_COMMIT
        ),
        "resident_http_and_text_idle": (
            http.get("pass") is True
            and http.get("resident") is True
            and http.get("model_loads") == 1
            and http.get("text_path_idle") is True
        ),
        "prefix_five_pair_no_regression": (
            prefix.get("complete") is True
            and prefix.get("qualified") is True
            and prefix.get("pair_count", 0) >= 5
            and prefix.get("pass") is True
            and float(paired.get("ttft_speedup", 0.0)) >= 1.0
            and float(paired.get("decode_retention", 0.0)) >= 1.0
            and all(require_mapping(prefix.get("checks"), "prefix.checks").values())
        ),
        "startup_at_most_44_90_seconds": (
            startup.get("pass") is True
            and float(startup.get("command_to_ready_median_ms", float("inf")))
            <= 44_900.0
        ),
    }
    summary = {
        "prefix_pair_count": prefix.get("pair_count"),
        "prefix_ttft_speedup_candidate_over_baseline": paired.get("ttft_speedup"),
        "prefix_decode_retention_candidate_over_baseline": paired.get(
            "decode_retention"
        ),
        "startup_command_to_ready_median_ms": startup.get(
            "command_to_ready_median_ms"
        ),
    }
    return checks, summary


def paired_matrix_cell_is_strict(cell: Mapping[str, Any]) -> bool:
    pair_count = cell.get("pair_count")
    required_pair_count = cell.get("required_pair_count")
    pairs = cell.get("pairs")
    if (
        isinstance(pair_count, bool)
        or not isinstance(pair_count, int)
        or isinstance(required_pair_count, bool)
        or not isinstance(required_pair_count, int)
        or pair_count < MINIMUM_PAIRED_MATRIX_PAIRS
        or required_pair_count != pair_count
        or not isinstance(pairs, list)
        or len(pairs) != pair_count
    ):
        return False
    expected_execution_orders = [
        ["baseline", "candidate"]
        if pair_index % 2
        else ["candidate", "baseline"]
        for pair_index in range(1, pair_count + 1)
    ]
    return (
        cell.get("complete") is True
        and cell.get("qualified") is True
        and [pair.get("execution_order") for pair in pairs]
        == expected_execution_orders
        and all(
            require_mapping(
                cell.get("paired_checks"), "matrix.paired_checks"
            ).values()
        )
        and all(
            require_mapping(
                cell.get("legacy_floor_checks"), "matrix.legacy_checks"
            ).values()
        )
    )


def check_paired_matrix(
    payload: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    engines = require_mapping(payload.get("engines"), "paired_text_matrix.engines")
    candidate = require_mapping(
        engines.get("candidate"), "paired_text_matrix.candidate"
    )
    baseline = require_mapping(
        engines.get("baseline"), "paired_text_matrix.baseline"
    )
    cells = require_sequence(payload.get("cells"), "paired_text_matrix.cells")
    startup = require_mapping(
        payload.get("q8192_startup"), "paired_text_matrix.q8192_startup"
    )
    protocol = require_mapping(
        payload.get("protocol", {}), "paired_text_matrix.protocol"
    )
    expected_cells = {
        (context, output)
        for context in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
        for output in (512, 1024)
    } | {(262143, 1), (261632, 512), (261120, 1024)}
    observed_cells = {
        (cell.get("input_tokens"), cell.get("output_tokens")) for cell in cells
    }
    all_cells_strict = (
        len(cells) == len(expected_cells)
        and observed_cells == expected_cells
        and all(paired_matrix_cell_is_strict(cell) for cell in cells)
    )
    candidate_build = require_mapping(
        candidate.get("build_info"), "paired_text_matrix.candidate.build_info"
    )
    baseline_build = require_mapping(
        baseline.get("build_info"), "paired_text_matrix.baseline.build_info"
    )
    checks = {
        "complete": payload.get("complete") is True,
        "qualified": payload.get("qualified") is True,
        "candidate_identity": (
            candidate.get("sha256") == EXPECTED_BINARY_SHA256
            and candidate_build.get("source_commit") == EXPECTED_SOURCE_COMMIT
        ),
        "baseline_identity": (
            baseline.get("sha256") == EXPECTED_BASELINE_SHA256
            and baseline_build.get("source_commit")
            == EXPECTED_BASELINE_SOURCE_COMMIT
        ),
        "exactly_nineteen_frozen_cells": (
            payload.get("expected_cell_count") == 19
            and payload.get("observed_cell_count") == 19
            and len(cells) == 19
            and observed_cells == expected_cells
        ),
        "all_cells_minimum_six_pair_order_balanced_strict_no_regression": (
            payload.get("all_cells_pass") is True
            and protocol.get("pair_count") == MINIMUM_PAIRED_MATRIX_PAIRS
            and protocol.get("minimum_observed_pair_count")
            == MINIMUM_PAIRED_MATRIX_PAIRS
            and all_cells_strict
        ),
        "q8192_startup_qualified": (
            startup.get("complete") is True
            and startup.get("qualified") is True
            and isinstance(startup.get("candidate_runs_ms"), list)
            and len(startup["candidate_runs_ms"]) == 6
            and isinstance(startup.get("baseline_runs_ms"), list)
            and len(startup["baseline_runs_ms"]) == 6
            and require_mapping(startup.get("checks"), "matrix.startup.checks").get(
                "candidate_at_most_44_90_seconds"
            )
            is True
        ),
        "all_candidate_text_paths_idle": payload.get("text_request_path_idle")
        is True,
    }
    minimum_prefill = min(
        (
            float(
                require_mapping(
                    cell.get("paired_medians"), "matrix.paired_medians"
                ).get("prefill_tps_candidate_over_baseline", float("inf"))
            )
            for cell in cells
            if cell.get("output_tokens") != 1
        ),
        default=None,
    )
    minimum_decode = min(
        (
            float(
                require_mapping(
                    cell.get("paired_medians"), "matrix.paired_medians"
                ).get("decode_tps_candidate_over_baseline", float("inf"))
            )
            for cell in cells
            if cell.get("output_tokens") != 1
        ),
        default=None,
    )
    maximum_total_latency = max(
        (
            float(
                require_mapping(
                    cell.get("paired_medians"), "matrix.paired_medians"
                ).get("total_wall_candidate_over_baseline", 0.0)
            )
            for cell in cells
        ),
        default=None,
    )
    summary = {
        "cell_count": len(cells),
        "minimum_pair_count": min(
            (int(cell.get("pair_count", 0)) for cell in cells), default=0
        ),
        "minimum_prefill_tps_candidate_over_baseline": minimum_prefill,
        "minimum_decode_tps_candidate_over_baseline": minimum_decode,
        "maximum_total_wall_candidate_over_baseline": maximum_total_latency,
        "q8192_candidate_command_to_ready_median_ms": startup.get(
            "candidate_median_ms"
        ),
    }
    return checks, summary


def check_doctor(payload: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    records = require_sequence(payload.get("checks"), "doctor.checks")
    by_id = {record.get("id"): record for record in records}
    gtt = require_mapping(by_id.get("memory.gtt"), "doctor.memory.gtt")
    vram = require_mapping(by_id.get("memory.vram"), "doctor.memory.vram")
    shards = require_mapping(by_id.get("model.shards"), "doctor.model.shards")
    checks = {
        "complete": payload.get("complete") is True,
        "qualified": payload.get("qualified") is True,
        "source_identity": payload.get("source_commit") == EXPECTED_SOURCE_COMMIT,
        "all_required_checks_pass": all(
            record.get("passed") is True
            for record in records
            if record.get("required") is True
        ),
        "gfx1151": require_mapping(
            by_id.get("gpu.architecture"), "doctor.gpu.architecture"
        ).get("actual", {}).get("architecture")
        == "gfx1151",
        "vram_512_mib": vram.get("actual") == 512 * 1024 * 1024,
        "gtt_at_least_96_gib": int(gtt.get("actual", 0)) >= 96 * 1024**3,
        "all_26_model_shards": shards.get("actual", {}).get("readable") == 26,
    }
    summary = {
        "required_checks": sum(record.get("required") is True for record in records),
        "required_checks_passed": sum(
            record.get("required") is True and record.get("passed") is True
            for record in records
        ),
        "vram_bytes": vram.get("actual"),
        "gtt_bytes": gtt.get("actual"),
        "model_shards": shards.get("actual", {}).get("readable"),
    }
    return checks, summary


def build_payload(
    artifact_paths: dict[str, Path] | None = None,
    recorded_on: str = "2026-08-24",
) -> dict[str, Any]:
    paths = ARTIFACT_PATHS if artifact_paths is None else artifact_paths
    payloads = {
        name: load_json_object(path)
        for name, path in paths.items()
        if name not in {"goal", "generator", "paired_text_runner"}
    }
    validators = {
        "correctness": check_correctness,
        "doctor": check_doctor,
        "openai_features": check_openai_features,
        "mmlu256": check_mmlu,
        "product_surfaces": check_product_surfaces,
        "paired_text_matrix": check_paired_matrix,
    }
    checks: dict[str, dict[str, bool]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, validator in validators.items():
        checks[name], summaries[name] = validator(payloads[name])

    candidate_hashes = {
        engine_sha256(payloads["correctness"]),
        engine_sha256(payloads["openai_features"]),
        engine_sha256(payloads["mmlu256"]),
        engine_sha256(payloads["product_surfaces"]),
        payloads["paired_text_matrix"]["engines"]["candidate"]["sha256"],
    }
    hostnames = {
        payloads["correctness"].get("host", {}).get("hostname"),
        payloads["openai_features"].get("host", {}).get("hostname"),
        payloads["product_surfaces"].get("host", {}).get("hostname"),
        payloads["paired_text_matrix"].get("host", {}).get("hostname"),
    }
    cross_checks = {
        "one_exact_candidate_binary": candidate_hashes == {EXPECTED_BINARY_SHA256},
        "one_exact_candidate_source": (
            payloads["doctor"].get("source_commit") == EXPECTED_SOURCE_COMMIT
            and payloads["mmlu256"]["engine"]["build_info"].get("source_commit")
            == EXPECTED_SOURCE_COMMIT
            and payloads["product_surfaces"]["engine"]["build_info"].get(
                "source_commit"
            )
            == EXPECTED_SOURCE_COMMIT
            and payloads["paired_text_matrix"]["engines"]["candidate"][
                "build_info"
            ].get("source_commit")
            == EXPECTED_SOURCE_COMMIT
        ),
        "one_exact_release_baseline": (
            payloads["product_surfaces"]["prefix_cache"]["baseline_engine"].get(
                "sha256"
            )
            == EXPECTED_BASELINE_SHA256
            and payloads["paired_text_matrix"]["engines"]["baseline"].get(
                "sha256"
            )
            == EXPECTED_BASELINE_SHA256
        ),
        "same_host": len(hostnames) == 1 and None not in hostnames,
    }
    complete = all(payload.get("complete") is True for payload in payloads.values())
    qualified = complete and all(
        all(group.values()) for group in (*checks.values(), cross_checks)
    )
    return {
        "schema": "aima-amd395-qwen36/text-v151-nonregression/v1",
        "recorded_on": recorded_on,
        "complete": complete,
        "qualified": qualified,
        "scope": "NATIVE_VL_GOAL.md G3 exact-binary text product no-regression",
        "candidate": {
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "binary_sha256": EXPECTED_BINARY_SHA256,
        },
        "baseline": {
            "release": "v1.5.1",
            "source_commit": EXPECTED_BASELINE_SOURCE_COMMIT,
            "binary_sha256": EXPECTED_BASELINE_SHA256,
        },
        "inputs": {
            name: file_component(path, str(path.relative_to(ROOT)))
            for name, path in paths.items()
        },
        "checks": checks,
        "cross_evidence_checks": cross_checks,
        "summaries": summaries,
        "decision": {
            "g3_text_product_no_regression": qualified,
            "g1_full_vl_functional_parity": False,
            "g2_vl_correctness_parity": False,
            "g4_native_vl_performance": False,
            "g5_native_release_product": False,
            "next_blocking_boundary": (
                "run and pass the frozen G4 paired VL performance protocol "
                "against the fixed vLLM reference, preserving this exact-source "
                "G1/G2/G3 evidence for G5 release qualification"
                if qualified
                else "complete every blocking G3 check"
            ),
        },
    }


def verify_exact(path: Path, expected: dict[str, Any]) -> None:
    actual = load_json_object(path)
    if actual != expected:
        raise SystemExit(f"G3 qualification is stale: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{digest}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise SystemExit(f"G3 qualification sidecar is stale: {sidecar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--candidate-source-commit", default=EXPECTED_SOURCE_COMMIT
    )
    parser.add_argument(
        "--candidate-binary-sha256", default=EXPECTED_BINARY_SHA256
    )
    parser.add_argument("--recorded-on", default="2026-08-24")
    for name in (
        "correctness",
        "doctor",
        "openai_features",
        "mmlu256",
        "product_surfaces",
        "paired_text_matrix",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            dest=name,
            type=Path,
            default=ARTIFACT_PATHS[name],
        )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        configure_candidate_identity(
            args.candidate_source_commit, args.candidate_binary_sha256
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        date.fromisoformat(args.recorded_on)
    except ValueError:
        parser.error("recorded-on must be an ISO calendar date")
    artifact_paths = {
        **ARTIFACT_PATHS,
        **{
            name: getattr(args, name).resolve()
            for name in (
                "correctness",
                "doctor",
                "openai_features",
                "mmlu256",
                "product_surfaces",
                "paired_text_matrix",
            )
        },
    }
    try:
        for path in artifact_paths.values():
            path.relative_to(ROOT)
    except ValueError:
        parser.error("G3 evidence inputs must be inside the repository")
    output = args.output.resolve()
    sealed = seal_manifest(build_payload(artifact_paths, args.recorded_on))
    if args.check:
        verify_exact(output, sealed)
        print(f"native VL G3 qualification: PASS ({output})")
        return 0
    digest = atomic_json(output, sealed)
    print(
        json.dumps(
            {
                "output": str(output),
                "qualified": sealed["qualified"],
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
