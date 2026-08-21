#!/usr/bin/env python3
"""Generate the exact-binary native VL G2 correctness qualification."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
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


OUTPUT = ROOT / "benchmarks/results/vl-correctness-v0.1.0.json"
CASE_ORDER = (
    "image_local_png",
    "video_local_mp4",
    "multi_image",
    "multi_video",
    "mixed_image_video",
)
VISION_BOUNDARY_ORDER = (
    "vision_block_0",
    "vision_block_13",
    "vision_block_26",
    "vision_merger",
)
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object")
    return value


def require_sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SystemExit(f"{label} must be an array")
    return value


def all_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value) and all(all_true(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(all_true(item) for item in value)
    return False


def source_commit(payload: dict[str, Any]) -> str | None:
    source = payload.get("source")
    if isinstance(source, dict):
        value = source.get("commit")
        return value if isinstance(value, str) else None
    value = payload.get("source_commit")
    return value if isinstance(value, str) else None


def binary_sha256(payload: dict[str, Any]) -> str | None:
    binary = payload.get("binary")
    if isinstance(binary, dict):
        value = binary.get("sha256")
        return value if isinstance(value, str) else None
    return None


def exact_case_order(payload: dict[str, Any], expected: tuple[str, ...]) -> bool:
    cases = require_sequence(payload.get("cases"), "cases")
    return tuple(case.get("case_id") for case in cases) == expected


def validate_high_level(
    label: str,
    payload: dict[str, Any],
    candidate_source_commit: str,
    candidate_binary_sha256: str,
) -> dict[str, bool]:
    return {
        "complete": payload.get("complete") is True,
        "qualified": payload.get("qualified") is True,
        "source_commit_exact": source_commit(payload) == candidate_source_commit,
        "binary_sha256_exact": binary_sha256(payload) == candidate_binary_sha256,
    }


def validate_vision_pipeline(
    payload: dict[str, Any],
    candidate_source_commit: str,
    candidate_binary_sha256: str,
) -> dict[str, bool]:
    decision = require_mapping(payload.get("decision"), "vision_pipeline.decision")
    cases = require_sequence(payload.get("cases"), "vision_pipeline.cases")
    boundary_names_exact = len(cases) == len(CASE_ORDER) and all(
        isinstance(case, dict)
        and tuple(
            boundary.get("name")
            for boundary in require_sequence(
                case.get("boundaries"), "vision_pipeline.case.boundaries"
            )
            if isinstance(boundary, dict)
        )
        == VISION_BOUNDARY_ORDER
        for case in cases
    )
    return {
        "schema_exact": payload.get("schema")
        == "aima-amd395-qwen36/native-vision-pipeline-qualification/v2",
        "complete": payload.get("complete") is True,
        "qualified": payload.get("qualified") is True,
        "source_commit_exact": source_commit(payload) == candidate_source_commit,
        "candidate_binary_sha256_exact": payload.get("candidate_binary_sha256")
        == candidate_binary_sha256,
        "five_case_order_exact": exact_case_order(payload, CASE_ORDER),
        "all_internal_checks": all_true(payload.get("checks")),
        "required_boundary_order_exact": boundary_names_exact,
        "twenty_boundaries_bit_exact": (
            decision.get("boundary_comparison_count") == 20
            and decision.get("total_boundary_elements") == 6_856_704
            and decision.get("exact_boundary_elements") == 6_856_704
            and decision.get("all_boundaries_bit_exact") is True
        ),
        "all_visual_outputs_exact": (
            decision.get("total_visual_output_elements") == 884_736
            and decision.get("all_repeats_deterministic") is True
            and decision.get("all_27_blocks_executed") is True
            and decision.get("full_visual_pipeline_qualified") is True
        ),
    }


def summarize_layer0(
    payload: dict[str, Any], candidate_source_commit: str
) -> tuple[dict[str, Any], dict[str, bool]]:
    cases = require_sequence(payload.get("cases"), "layer0.cases")
    comparison_sets = [
        require_sequence(
            case.get("diagnostic_comparisons"),
            "layer0.diagnostic_comparisons",
        )
        + require_sequence(
            case.get("seeded_moe_diagnostic_comparisons"),
            "layer0.seeded_moe_diagnostic_comparisons",
        )
        for case in cases
    ]
    diagnostic_count = sum(len(comparisons) for comparisons in comparison_sets)
    all_comparisons = [
        comparison for comparisons in comparison_sets for comparison in comparisons
    ]
    checks = {
        "schema_exact": payload.get("schema")
        == "aima-amd395-qwen36/native-vl-language-layer0-qualification-run/v1",
        "complete": payload.get("complete") is True,
        "source_commit_exact": payload.get("source_commit")
        == candidate_source_commit,
        "five_case_order_exact": exact_case_order(payload, CASE_ORDER),
        "single_resident_weight_load": payload.get("single_resident_weight_load")
        is True,
        "all_output_elements_bit_exact": (
            payload.get("all_bit_exact") is True
            and payload.get("total_elements") == payload.get("total_exact_elements")
            and int(payload.get("total_elements", 0)) > 0
        ),
        "all_outputs_finite": all(
            case.get("finite_elements") == case.get("elements") for case in cases
        ),
        "all_outputs_repeat_deterministic": all(
            case.get("repeat_deterministic") is True for case in cases
        ),
        "all_diagnostics_pass": all(
            case.get("diagnostic_complete") is True
            and case.get("seeded_moe_diagnostic_complete") is True
            and len(comparisons) == 33
            and all(
                comparison.get("passed") is True
                and comparison.get("finite_elements")
                == comparison.get("elements")
                and float(comparison.get("relative_l2_error", 1.0)) <= 0.002
                and float(comparison.get("cosine_similarity", 0.0)) >= 0.999
                for comparison in comparisons
            )
            and any(
                comparison.get("label") == "input_norm_full_sequence"
                and comparison.get("exact_elements") == comparison.get("elements")
                for comparison in comparisons
            )
            and case.get("router_expert_set_rows_exact")
            == case.get("router_expert_set_rows")
            and case.get("seeded_router_expert_set_rows_exact")
            == case.get("seeded_router_expert_set_rows")
            for case, comparisons in zip(cases, comparison_sets, strict=True)
        ),
    }
    summary = {
        "case_count": len(cases),
        "prompt_tokens": sum(int(case.get("prompt_tokens", 0)) for case in cases),
        "output_elements": payload.get("total_elements"),
        "exact_output_elements": payload.get("total_exact_elements"),
        "diagnostic_comparisons": diagnostic_count,
        "maximum_diagnostic_relative_l2_error": max(
            (float(item.get("relative_l2_error", 1.0)) for item in all_comparisons),
            default=1.0,
        ),
        "minimum_diagnostic_cosine_similarity": min(
            (float(item.get("cosine_similarity", 0.0)) for item in all_comparisons),
            default=0.0,
        ),
    }
    return summary, checks


def summarize_layer3(
    payload: dict[str, Any], candidate_source_commit: str
) -> tuple[dict[str, Any], dict[str, bool]]:
    cases = require_sequence(payload.get("cases"), "layer3.cases")
    checks = {
        "schema_exact": payload.get("schema")
        == "aima-amd395-qwen36/native-vl-language-layer3-mrope-qualification-run/v1",
        "complete": payload.get("complete") is True,
        "source_commit_exact": payload.get("source_commit")
        == candidate_source_commit,
        "five_case_order_exact": exact_case_order(payload, CASE_ORDER),
        "mrope_contract_exact": (
            payload.get("mrope_section") == [11, 11, 10]
            and payload.get("mrope_interleaved") is True
            and payload.get("rotary_dimension") == 64
        ),
        "all_elements_bit_exact": (
            payload.get("all_bit_exact") is True
            and payload.get("total_elements") == payload.get("total_exact_elements")
            and int(payload.get("total_elements", 0)) > 0
        ),
        "runtime_native_only": all(
            payload.get(key) is False
            for key in (
                "runtime_python",
                "runtime_numpy",
                "runtime_torch",
                "runtime_vllm",
                "runtime_triton",
            )
        ),
    }
    return {
        "capture_source_commit": payload.get("capture_source_commit"),
        "capture_manifest_sha256": payload.get("capture_manifest_sha256"),
        "case_count": len(cases),
        "elements": payload.get("total_elements"),
        "exact_elements": payload.get("total_exact_elements"),
    }, checks


def summarize_full_language(
    payload: dict[str, Any],
    candidate_source_commit: str,
    expected_cases: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, bool]]:
    cases = require_sequence(payload.get("cases"), "full_language.cases")
    logits_rows = [
        row
        for case in cases
        for row in require_sequence(
            require_mapping(
                case.get("full_vocabulary_logits"),
                "full_language.full_vocabulary_logits",
            ).get("rows"),
            "full_language.full_vocabulary_logits.rows",
        )
    ]
    final_norm_elements = sum(
        int(require_mapping(case.get("final_norm"), "final_norm").get("elements", 0))
        for case in cases
    )
    final_norm_exact = sum(
        int(
            require_mapping(case.get("final_norm"), "final_norm").get(
                "exact_elements", 0
            )
        )
        for case in cases
    )
    logits_elements = sum(
        int(require_mapping(row.get("tensor"), "logits.tensor").get("elements", 0))
        for row in logits_rows
    )
    logits_exact = sum(
        int(
            require_mapping(row.get("tensor"), "logits.tensor").get(
                "exact_elements", 0
            )
        )
        for row in logits_rows
    )
    checks = {
        "schema_exact": payload.get("schema")
        == "aima-amd395-qwen36/native-vl-language-full-qualification-run/v1",
        "complete": payload.get("complete") is True,
        "source_commit_exact": payload.get("source_commit")
        == candidate_source_commit,
        "case_order_exact": exact_case_order(payload, expected_cases),
        "single_resident_weight_load": payload.get("single_resident_weight_load")
        is True,
        "all_cases_complete": all(case.get("complete") is True for case in cases),
        "all_final_norms_pass": all(
            require_mapping(case.get("final_norm"), "final_norm").get("passed")
            is True
            for case in cases
        ),
        "all_final_norms_bit_exact": (
            final_norm_elements > 0 and final_norm_elements == final_norm_exact
        ),
        "all_logits_rows_pass": bool(logits_rows)
        and all(
            row.get("passed") is True
            and row.get("top1_match") is True
            and float(row.get("kl_divergence", 1.0)) < 0.005
            for row in logits_rows
        ),
        "all_selected_logits_bit_exact": (
            logits_elements > 0 and logits_elements == logits_exact
        ),
        "all_outputs_repeat_deterministic": all(
            case.get("repeat_deterministic") is True for case in cases
        ),
        "all_cases_production_operation_shape": all(
            case.get("production_operation_shape") is True for case in cases
        ),
    }
    summary = {
        "case_count": len(cases),
        "logical_prompt_tokens": sum(
            int(case.get("prompt_tokens", 0)) for case in cases
        ),
        "final_norm_elements": final_norm_elements,
        "final_norm_exact_elements": final_norm_exact,
        "selected_logits_rows": len(logits_rows),
        "selected_logits_elements": logits_elements,
        "selected_logits_exact_elements": logits_exact,
        "maximum_logits_kl_divergence": max(
            (float(row.get("kl_divergence", 1.0)) for row in logits_rows),
            default=1.0,
        ),
        "top1_matches": sum(row.get("top1_match") is True for row in logits_rows),
    }
    return summary, checks


def summarize_deep_language(
    payload: dict[str, Any], candidate_source_commit: str
) -> tuple[dict[str, Any], dict[str, bool]]:
    cases = require_sequence(payload.get("cases"), "deep_language.cases")
    case = require_mapping(cases[0], "deep_language.cases[0]") if cases else {}
    diagnostics = require_mapping(
        case.get("layer_diagnostics"), "deep_language.layer_diagnostics"
    )
    comparisons = require_sequence(
        diagnostics.get("comparisons"), "deep_language.comparisons"
    )
    router_sets = require_sequence(
        diagnostics.get("router_expert_sets"), "deep_language.router_expert_sets"
    )
    router_rows = sum(int(item.get("rows", 0)) for item in router_sets)
    exact_router_rows = sum(int(item.get("exact_rows", 0)) for item in router_sets)
    checks = {
        "complete": payload.get("complete") is True and case.get("complete") is True,
        "source_commit_exact": payload.get("source_commit")
        == candidate_source_commit,
        "selector_exact": payload.get("case_selector") == "multi_video",
        "one_case_exact": len(cases) == 1 and case.get("case_id") == "multi_video",
        "diagnostics_provided": diagnostics.get("provided") is True,
        "all_tensor_comparisons_pass": len(comparisons) >= 40
        and all(item.get("passed") is True for item in comparisons),
        "forty_layer_router_sets": len(router_sets) == 40,
        "all_router_rows_exact": (
            diagnostics.get("all_router_expert_sets_exact") is True
            and router_rows > 0
            and router_rows == exact_router_rows
        ),
    }
    return {
        "tensor_comparisons": len(comparisons),
        "router_layer_sets": len(router_sets),
        "router_rows": router_rows,
        "exact_router_rows": exact_router_rows,
    }, checks


def build_payload(
    *,
    candidate_source_commit: str,
    candidate_binary_sha256: str,
    recorded_on: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    payloads = {name: load_json_object(path) for name, path in paths.items()}
    envelope = payloads["envelope"]
    vision_pipeline = payloads["vision_pipeline"]
    task_quality = payloads["task_quality"]
    generation = payloads["generation"]
    error_limits = payloads["error_limits"]

    layer0_summary, layer0_checks = summarize_layer0(
        payloads["layer0"], candidate_source_commit
    )
    layer3_a_summary, layer3_a_checks = summarize_layer3(
        payloads["layer3_a"], candidate_source_commit
    )
    layer3_b_summary, layer3_b_checks = summarize_layer3(
        payloads["layer3_b"], candidate_source_commit
    )
    private_summary, private_checks = summarize_full_language(
        payloads["full_private"], candidate_source_commit, CASE_ORDER
    )
    http_summary, http_checks = summarize_full_language(
        payloads["full_http"], candidate_source_commit, CASE_ORDER
    )
    deep_summary, deep_checks = summarize_deep_language(
        payloads["full_http_deep"], candidate_source_commit
    )

    envelope_checks = validate_high_level(
        "envelope",
        envelope,
        candidate_source_commit,
        candidate_binary_sha256,
    )
    envelope_checks.update(
        {
            "processor_probe_exact": all_true(
                require_mapping(envelope.get("processor_probe"), "processor_probe").get(
                    "checks"
                )
            ),
            "vision_envelope_exact": all_true(
                require_mapping(envelope.get("vision_probe"), "vision_probe").get(
                    "checks"
                )
            ),
            "all_envelope_observations_qualified": all(
                item.get("qualified") is True
                for item in require_sequence(
                    require_mapping(envelope.get("matrix"), "envelope.matrix").get(
                        "observations"
                    ),
                    "envelope.matrix.observations",
                )
            ),
            "twenty_three_envelope_observations": len(
                require_sequence(
                    require_mapping(envelope.get("matrix"), "envelope.matrix").get(
                        "observations"
                    ),
                    "envelope.matrix.observations",
                )
            )
            == 23,
        }
    )
    vision_checks = validate_vision_pipeline(
        vision_pipeline,
        candidate_source_commit,
        candidate_binary_sha256,
    )
    task_checks = validate_high_level(
        "task_quality",
        task_quality,
        candidate_source_commit,
        candidate_binary_sha256,
    )
    task_checks["all_decisions_qualified"] = all(
        require_mapping(task_quality.get("decision"), "task_quality.decision").get(
            key
        )
        is True
        for key in (
            "twelve_task_quality_cases_qualified",
            "twelve_long_greedy_cases_reference_exact",
            "twelve_output_token_vectors_exact",
            "twelve_render_prompt_vectors_exact",
            "image_task_quality_not_below_reference",
            "video_task_quality_not_below_reference",
            "single_resident_model_load",
        )
    )
    generation_checks = validate_high_level(
        "generation",
        generation,
        candidate_source_commit,
        candidate_binary_sha256,
    )
    generation_checks["all_internal_checks"] = all_true(generation.get("checks"))
    generation_checks["g1_generation_closed"] = require_mapping(
        generation.get("decision"), "generation.decision"
    ).get("g1_generation_closed") is True
    error_checks = validate_high_level(
        "error_limits",
        error_limits,
        candidate_source_commit,
        candidate_binary_sha256,
    )
    error_checks["all_qualification_checks"] = all_true(
        error_limits.get("qualification_checks")
    )
    error_checks["all_error_observations_qualified"] = all(
        case.get("qualified") is True
        and all_true(case.get("qualification_checks"))
        for case in require_sequence(
            require_mapping(error_limits.get("run"), "error_limits.run").get("cases"),
            "error_limits.run.cases",
        )
    )
    error_checks["thirteen_error_observations"] = len(
        require_sequence(
            require_mapping(error_limits.get("run"), "error_limits.run").get("cases"),
            "error_limits.run.cases",
        )
    ) == 13

    cross_evidence_checks = {
        "independent_layer3_capture_manifests": (
            layer3_a_summary["capture_manifest_sha256"]
            != layer3_b_summary["capture_manifest_sha256"]
        ),
        "independent_private_and_http_prompt_manifests": (
            payloads["full_private"].get("vl_oracle_manifest_sha256")
            != payloads["full_http"].get("vl_oracle_manifest_sha256")
        ),
        "ten_full_language_cases": (
            int(private_summary["case_count"]) + int(http_summary["case_count"])
            == 10
        ),
        "two_million_four_hundred_twenty_thousand_seven_hundred_thirty_six_final_norm_elements": (
            int(private_summary["final_norm_elements"])
            + int(http_summary["final_norm_elements"])
            == 2_420_736
        ),
        "eighty_four_full_vocabulary_rows": (
            int(private_summary["selected_logits_rows"])
            + int(http_summary["selected_logits_rows"])
            == 84
        ),
        "twenty_million_eight_hundred_fifty_eight_thousand_eight_hundred_eighty_selected_logits": (
            int(private_summary["selected_logits_elements"])
            + int(http_summary["selected_logits_elements"])
            == 20_858_880
        ),
    }

    gates = {
        "processor_and_execution_envelope": {
            "checks": envelope_checks,
        },
        "vision_boundary": {
            "checks": vision_checks,
        },
        "language_layer0": {
            "summary": layer0_summary,
            "checks": layer0_checks,
        },
        "language_layer3_mrope_capture_a": {
            "summary": layer3_a_summary,
            "checks": layer3_a_checks,
        },
        "language_layer3_mrope_capture_b": {
            "summary": layer3_b_summary,
            "checks": layer3_b_checks,
        },
        "full_language_private_processor_prompt": {
            "summary": private_summary,
            "checks": private_checks,
        },
        "full_language_http_rendered_prompt": {
            "summary": http_summary,
            "checks": http_checks,
        },
        "deep_http_multi_video_language": {
            "summary": deep_summary,
            "checks": deep_checks,
        },
        "deterministic_generation": {
            "checks": generation_checks,
        },
        "task_quality": {
            "checks": task_checks,
        },
        "error_parity": {
            "checks": error_checks,
        },
        "cross_evidence": {
            "checks": cross_evidence_checks,
        },
    }
    qualified = all(
        all(checks.values())
        for gate in gates.values()
        for checks in (require_mapping(gate.get("checks"), "gate.checks"),)
    )
    total_logits_rows = int(private_summary["selected_logits_rows"]) + int(
        http_summary["selected_logits_rows"]
    )
    return {
        "schema": "aima-amd395-qwen36/native-vl-g2-qualification/v1",
        "recorded_on": recorded_on,
        "complete": qualified,
        "qualified": qualified,
        "scope": "processor-through-task-quality-exact-binary-vl-correctness",
        "candidate": {
            "source_commit": candidate_source_commit,
            "binary_sha256": candidate_binary_sha256,
        },
        "thresholds": {
            "minimum_cosine_similarity": 0.999,
            "maximum_relative_l2_error": 0.002,
            "maximum_logits_kl_divergence_exclusive": 0.005,
            "top1_match": True,
            "discrete_state_exact": True,
        },
        "inputs": {
            name: file_component(path, str(path.relative_to(ROOT)))
            for name, path in {
                "goal": ROOT / "docs/NATIVE_VL_GOAL.md",
                "generator": Path(__file__).resolve(),
                **paths,
            }.items()
        },
        "gates": gates,
        "aggregate": {
            "prompt_identities": 2,
            "full_language_cases": int(private_summary["case_count"])
            + int(http_summary["case_count"]),
            "full_vocabulary_rows": total_logits_rows,
            "full_vocabulary_top1_matches": int(private_summary["top1_matches"])
            + int(http_summary["top1_matches"]),
            "maximum_logits_kl_divergence": max(
                float(private_summary["maximum_logits_kl_divergence"]),
                float(http_summary["maximum_logits_kl_divergence"]),
            ),
            "all_gates_passed": qualified,
        },
        "decision": {
            "g2_passed": qualified,
            "exact_binary_bound": qualified,
            "no_threshold_widening": True,
        },
    }


def verify_exact(path: Path, expected: dict[str, Any]) -> None:
    actual = load_json_object(path)
    if actual != expected:
        raise SystemExit(f"native VL G2 qualification is stale: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_sidecar = f"{digest}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise SystemExit(f"native VL G2 sidecar is stale: {sidecar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source-commit", required=True)
    parser.add_argument("--candidate-binary-sha256", required=True)
    parser.add_argument("--recorded-on", type=date.fromisoformat, required=True)
    for name in (
        "envelope",
        "vision_pipeline",
        "task_quality",
        "generation",
        "error_limits",
        "layer0",
        "layer3_a",
        "layer3_b",
        "full_private",
        "full_http",
        "full_http_deep",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not SHA1.fullmatch(args.candidate_source_commit):
        parser.error("--candidate-source-commit must be a lowercase SHA-1")
    if not SHA256.fullmatch(args.candidate_binary_sha256):
        parser.error("--candidate-binary-sha256 must be a lowercase SHA-256")
    paths = {
        name: getattr(args, name).resolve()
        for name in (
            "envelope",
            "vision_pipeline",
            "task_quality",
            "generation",
            "error_limits",
            "layer0",
            "layer3_a",
            "layer3_b",
            "full_private",
            "full_http",
            "full_http_deep",
        )
    }
    for name, path in paths.items():
        try:
            path.relative_to(ROOT)
        except ValueError:
            parser.error(f"--{name.replace('_', '-')} must be inside the repository")
    payload = seal_manifest(
        build_payload(
            candidate_source_commit=args.candidate_source_commit,
            candidate_binary_sha256=args.candidate_binary_sha256,
            recorded_on=args.recorded_on.isoformat(),
            paths=paths,
        )
    )
    output = args.output.resolve()
    if args.check:
        verify_exact(output, payload)
        print(f"native VL G2 qualification: PASS ({output})")
        return 0
    digest = atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "qualified": payload["qualified"],
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
