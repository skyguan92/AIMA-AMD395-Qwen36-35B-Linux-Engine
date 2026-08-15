#!/usr/bin/env python3
"""Generate the requirement-to-evidence audit for the native VL G1 gate."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
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


OUTPUT = ROOT / "benchmarks/results/native-vl-g1-coverage-audit-v0.1.0.json"
ARTIFACT_PATHS = {
    "goal": ROOT / "docs/NATIVE_VL_GOAL.md",
    "reference_capability": (
        ROOT / "benchmarks/results/vl-capability-manifest.json"
    ),
    "native_capability": (
        ROOT / "benchmarks/results/native-vl-capability-v0.1.0.json"
    ),
    "execution_envelope": (
        ROOT / "benchmarks/results/native-vl-envelope-v0.1.0.json"
    ),
    "resident_serving": (
        ROOT / "benchmarks/results/native-vl-serving-v0.1.0.json"
    ),
    "mixed_conversation_reference": (
        ROOT
        / "benchmarks/results/vl-g1-mixed-conversation-reference-v0.1.0.json"
    ),
    "mixed_conversation_native": (
        ROOT / "benchmarks/results/native-vl-g1-extension-v0.1.0.json"
    ),
    "transport_cache_reference": (
        ROOT
        / "benchmarks/results/vl-transport-cache-reference-v0.1.0.json"
    ),
    "transport_cache_native": (
        ROOT / "benchmarks/results/native-vl-transport-cache-v0.1.0.json"
    ),
    "vision_pipeline": (
        ROOT / "benchmarks/results/native-vision-pipeline-v0.1.0.json"
    ),
    "language_boundary": (
        ROOT / "benchmarks/results/native-vl-language-full-v0.2.0.json"
    ),
    "cache_identity_unit": ROOT / "tests/native_multimodal_cache_test.cpp",
    "generator": Path(__file__).resolve(),
}


def evidence(artifact: str, *case_ids: str, note: str) -> dict[str, Any]:
    record: dict[str, Any] = {"artifact": artifact, "note": note}
    if case_ids:
        record["case_ids"] = list(case_ids)
    return record


def requirement(
    requirement_id: str,
    title: str,
    status: str,
    evidence_records: list[dict[str, Any]],
    gaps: list[str] | None = None,
) -> dict[str, Any]:
    if status not in {"covered", "partial", "missing"}:
        raise ValueError(f"invalid coverage status: {status}")
    gap_list = gaps or []
    if (status == "covered") == bool(gap_list):
        raise ValueError(
            f"{requirement_id}: covered requirements cannot have gaps and "
            "partial/missing requirements must have gaps"
        )
    return {
        "requirement_id": requirement_id,
        "title": title,
        "status": status,
        "evidence": evidence_records,
        "gaps": gap_list,
    }


def build_requirements() -> list[dict[str, Any]]:
    return [
        requirement(
            "G1.2.1.image",
            "Image formats, geometry, dynamic resolution and count boundaries",
            "covered",
            [
                evidence(
                    "native_capability",
                    "image_local_png",
                    "image_data_jpeg",
                    "image_http_webp",
                    "image_transparent_png",
                    "multi_image_interleaved",
                    note="PNG, JPEG, WebP, alpha, local/data/HTTP and multi-image",
                ),
                evidence(
                    "execution_envelope",
                    "image_minimum",
                    "image_typical_portrait",
                    "image_typical_landscape",
                    "image_maximum_pixels",
                    "image_above_maximum_clamp",
                    "image_aspect_rejection",
                    "image_count_maximum_small",
                    "image_count_over_limit",
                    "image_near_window_maximum",
                    note="min/typical/max, aspect, clamp, count and near-window execution",
                ),
            ],
        ),
        requirement(
            "G1.2.1.video",
            "Video formats, sampling, geometry and count boundaries",
            "covered",
            [
                evidence(
                    "native_capability",
                    "video_local_mp4",
                    "video_data_mp4",
                    "video_http_avi",
                    "multi_video",
                    note="MP4/AVI, local/data/HTTP and multi-video serving",
                ),
                evidence(
                    "execution_envelope",
                    "video_minimum",
                    "video_typical",
                    "video_maximum_feature_shape",
                    "video_sampling_minimum",
                    "video_sampling_typical",
                    "video_sampling_maximum",
                    "video_sampling_above_maximum",
                    "video_count_maximum_small",
                    "video_count_over_limit",
                    "video_full_item_budget",
                    note="sampling, feature, count and full-budget execution",
                ),
            ],
        ),
        requirement(
            "G1.2.1.mixed",
            "Mixed image/video ordering and interleaving",
            "covered",
            [
                evidence(
                    "native_capability",
                    "mixed_image_then_video",
                    "mixed_video_then_image",
                    note="both one-image/one-video orders",
                ),
                evidence(
                    "execution_envelope",
                    "mixed_cross_batch_boundary",
                    note="mixed request crossing the vision batch boundary",
                ),
                evidence(
                    "mixed_conversation_reference",
                    note="five new mixed/conversation requests frozen on fixed vLLM",
                ),
                evidence(
                    "mixed_conversation_native",
                    "mixed_multi_image_video_interleave",
                    "mixed_multi_video_image_interleave",
                    note="both three-media interleave directions are reference-exact",
                ),
            ],
        ),
        requirement(
            "G1.2.1.conversation",
            "System, assistant, tool and multi-turn media history",
            "covered",
            [
                evidence(
                    "native_capability",
                    "conversation_prior_image",
                    "conversation_media_replace",
                    "tool_history_with_image",
                    note="system/assistant/tool history and image replacement",
                ),
                evidence(
                    "resident_serving",
                    note="same-process image A/B/A and prompt variants",
                ),
                evidence(
                    "mixed_conversation_native",
                    "conversation_video_reuse_replace",
                    "conversation_mixed_prior_turn",
                    note="video replacement and mixed prior-turn history match vLLM",
                ),
            ],
        ),
        requirement(
            "G1.2.1.openai_api",
            "Streaming and non-stream OpenAI content parts",
            "covered",
            [
                evidence(
                    "native_capability",
                    "stream_image",
                    "stream_video",
                    "image_local_png",
                    "video_local_mp4",
                    "mixed_image_then_video",
                    note="complete image/video SSE and non-stream media requests",
                ),
                evidence(
                    "mixed_conversation_native",
                    "stream_mixed_media",
                    note="mixed-media SSE aggregate and finish reason match vLLM",
                ),
            ],
        ),
        requirement(
            "G1.2.1.generation",
            "Greedy tokens, finish reason, usage and response parity",
            "partial",
            [
                evidence(
                    "native_capability",
                    note="20/20 finish-reason parity but only 14/18 exact usage",
                ),
                evidence(
                    "resident_serving",
                    note="five frozen private-oracle 8-token generations preserved",
                ),
                evidence(
                    "transport_cache_native",
                    "video_content_a_cold",
                    "video_content_b_miss",
                    "video_content_a_disabled_1",
                    "video_content_b_disabled",
                    note=(
                        "8-token video outputs and usage are reference-exact with "
                        "cache enabled and disabled"
                    ),
                ),
            ],
            [
                "four capability cases still differ in completion or usage",
                "task-level longer greedy image/video corpus is not qualified",
            ],
        ),
        requirement(
            "G1.2.1.tools",
            "VL tools, tool choice and assistant/tool history",
            "covered",
            [
                evidence(
                    "native_capability",
                    "tool_history_with_image",
                    "tool_forced_image",
                    "tool_auto_image",
                    note="history, none, forced named and auto tool behavior",
                )
            ],
        ),
        requirement(
            "G1.2.1.transport",
            "URL, data URI/base64, local file and media-format transport",
            "covered",
            [
                evidence(
                    "native_capability",
                    "image_local_png",
                    "image_data_jpeg",
                    "image_http_webp",
                    "video_local_mp4",
                    "video_data_mp4",
                    "video_http_avi",
                    note="local, data and HTTP across supported formats",
                ),
                evidence(
                    "resident_serving",
                    note=(
                        "same HTTP URL A/B/A and video local/data content "
                        "equivalence are live-qualified"
                    ),
                ),
                evidence(
                    "transport_cache_reference",
                    "https_image",
                    note="verified loopback HTTPS is frozen on fixed vLLM",
                ),
                evidence(
                    "transport_cache_native",
                    "https_image_cold",
                    "https_image_exact",
                    note=(
                        "native verified-CA HTTPS is reference-exact on cold and "
                        "resident reuse"
                    ),
                ),
            ],
        ),
        requirement(
            "G1.2.1.residency",
            "Text, image, video and mixed requests in one resident process",
            "covered",
            [
                evidence(
                    "native_capability",
                    "residency_text_before",
                    "image_local_png",
                    "video_local_mp4",
                    "mixed_image_then_video",
                    "residency_text_after",
                    note="one ordered 20-success native run with text before and after VL",
                ),
                evidence(
                    "execution_envelope",
                    note="17 accepted boundary requests in one additional model load",
                ),
            ],
        ),
        requirement(
            "G1.2.2.model_semantics",
            "Processor, vision, injection, M-RoPE and language semantics",
            "partial",
            [
                evidence(
                    "vision_pipeline",
                    note="five-case full visual pipeline boundary evidence",
                ),
                evidence(
                    "language_boundary",
                    note="84/84 full-vocabulary rows bit-exact",
                ),
                evidence(
                    "resident_serving",
                    note="five real-HTTP render vectors and deterministic outputs",
                ),
                evidence(
                    "execution_envelope",
                    note="current long-window M-RoPE execution and launch accounting",
                ),
                evidence(
                    "transport_cache_native",
                    note=(
                        "current runtime binds vLLM-exact sampling, mixed ordering, "
                        "prompt tokens, outputs and usage"
                    ),
                ),
            ],
            [
                "historical numerical evidence is source-hash bound to earlier commits",
                "current HEAD still requires consolidated processor-to-output requalification",
            ],
        ),
        requirement(
            "G1.2.2.error_parity",
            "Invalid media status and error-category parity",
            "partial",
            [
                evidence(
                    "native_capability",
                    "error_corrupt_image",
                    "error_corrupt_video",
                    "error_aspect_ratio",
                    "error_outside_local_root",
                    "error_disallowed_domain",
                    "error_image_count_over_limit",
                    "error_video_count_over_limit",
                    "error_malformed_data_uri",
                    "error_type_mismatch",
                    "error_audio_out_of_scope",
                    note="ten frozen API rejection cases",
                ),
                evidence(
                    "execution_envelope",
                    "image_aspect_rejection",
                    "video_below_temporal_rejection",
                    "video_below_spatial_rejection",
                    "video_aspect_rejection",
                    "image_count_over_limit",
                    "video_count_over_limit",
                    note="six min/aspect/count envelope rejections",
                ),
                evidence(
                    "transport_cache_native",
                    "video_content_error",
                    "video_content_error_disabled",
                    note=(
                        "corrupt-video status is reference-exact and invariant to "
                        "cache mode"
                    ),
                ),
            ],
            [
                "empty image and empty video categories are not frozen",
                "remote timeout and unreachable-media categories are not frozen",
                "compressed-byte, decoded-pixel and duration over-limit categories are not frozen",
                "error-category compatibility is not sealed separately from HTTP status",
            ],
        ),
        requirement(
            "G1.2.3.product_preservation",
            "Existing text, API, security, startup and cache product behavior",
            "partial",
            [
                evidence(
                    "native_capability",
                    "residency_text_before",
                    "residency_text_after",
                    note="text remains serviceable around a VL workload",
                ),
                evidence(
                    "resident_serving",
                    note="native-only launch, one model load and prefix behavior",
                ),
            ],
            ["the complete G3 text and release no-regression protocol has not run"],
        ),
        requirement(
            "G1.2.4.cache_identity",
            "Content-addressed media identity and conservative cache reuse",
            "covered",
            [
                evidence(
                    "resident_serving",
                    note=(
                        "same-path and same-HTTP-URL A/B/A, image/video "
                        "data/local equivalence and prompt variant checks"
                    ),
                ),
                evidence(
                    "cache_identity_unit",
                    note="content, order, processor identity and token span affect namespace",
                ),
                evidence(
                    "transport_cache_native",
                    "video_content_a_cold",
                    "video_content_b_miss",
                    "video_content_a_restored",
                    "video_sampling_default",
                    "video_sampling_fps_1",
                    "video_sampling_default_restored",
                    "video_sampling_num_frames_6",
                    "video_sampling_num_frames_6_exact",
                    "mixed_image_video",
                    "mixed_video_image_reordered",
                    "mixed_mutated_image_video",
                    "mixed_image_video_restored",
                    note=(
                        "live A/B/A covers video bytes, sampling policy, mixed order "
                        "and media mutation"
                    ),
                ),
            ],
        ),
        requirement(
            "G1.2.4.cache_invariance",
            "Cache changes latency only, never outputs or request semantics",
            "covered",
            [
                evidence(
                    "resident_serving",
                    note=(
                        "image A/B/A plus video and mixed cold/hit outputs "
                        "remain token-exact"
                    ),
                ),
                evidence(
                    "transport_cache_native",
                    "video_content_error",
                    "video_content_error_disabled",
                    "video_content_a_disabled_1",
                    "video_content_a_disabled_2",
                    "video_sampling_default_disabled_1",
                    "video_sampling_default_disabled_2",
                    "mixed_image_video_disabled_1",
                    "mixed_image_video_disabled_2",
                    note=(
                        "8-token output and usage equality, all-miss disabled mode, "
                        "and error status/payload invariance are live-qualified"
                    ),
                ),
            ],
        ),
    ]


def case_map(
    payload: dict[str, Any],
    path: tuple[str, ...],
    id_field: str = "case_id",
) -> dict[str, Any]:
    value: Any = payload
    for key in path:
        value = value[key]
    return {item[id_field]: item for item in value}


def validate_inputs(payloads: dict[str, dict[str, Any]]) -> None:
    native = payloads["native_capability"]
    execution = payloads["execution_envelope"]
    serving = payloads["resident_serving"]
    extension_reference = payloads["mixed_conversation_reference"]
    extension_native = payloads["mixed_conversation_native"]
    transport_reference = payloads["transport_cache_reference"]
    transport_native = payloads["transport_cache_native"]
    vision = payloads["vision_pipeline"]
    language = payloads["language_boundary"]
    for name, payload in (
        ("native capability", native),
        ("execution envelope", execution),
        ("resident serving", serving),
        ("mixed/conversation reference", extension_reference),
        ("mixed/conversation native", extension_native),
        ("transport/cache reference", transport_reference),
        ("transport/cache native", transport_native),
    ):
        if payload.get("complete") is not True or payload.get("qualified") is not True:
            raise SystemExit(f"{name} evidence is not complete and qualified")
    if vision.get("complete") is not True or not vision["decision"].get(
        "full_visual_pipeline_qualified"
    ):
        raise SystemExit("vision pipeline evidence is incomplete")
    if language.get("complete") is not True or language["decision"].get(
        "teacher_forced_full_vocabulary_logits_gate"
    ) != "passed-84-of-84-rows-bit-exact":
        raise SystemExit("language boundary evidence is incomplete")
    if native["matrix"].get("reference_status_exact") != "30/30":
        raise SystemExit("native capability status parity is incomplete")
    if native["matrix"].get("reference_finish_reason_exact") != "20/20":
        raise SystemExit("native capability finish-reason parity is incomplete")
    if native["matrix"].get("reference_usage_exact") != "14/18":
        raise SystemExit("native capability usage gap changed")
    if not all(serving["cache_correctness"]["checks"].values()):
        raise SystemExit("resident cache evidence contains a failed check")
    for decision in (
        "same_http_url_content_mutation_qualified",
        "video_transport_cache_equivalence_qualified",
        "mixed_cache_invariance_qualified",
    ):
        if serving["decision"].get(decision) is not True:
            raise SystemExit(
                f"resident cache evidence is missing decision: {decision}"
            )
    if extension_reference["source"].get("commit") != extension_native[
        "source"
    ].get("commit"):
        raise SystemExit("mixed/conversation reference and native commits differ")
    reference_binding = extension_native["dependencies"].get("reference", {})
    expected_reference = file_component(
        ARTIFACT_PATHS["mixed_conversation_reference"],
        "benchmarks/results/vl-g1-mixed-conversation-reference-v0.1.0.json",
    )
    if reference_binding != expected_reference:
        raise SystemExit("mixed/conversation native reference binding changed")
    for decision in (
        "mixed_multi_item_orders_qualified",
        "video_and_mixed_history_qualified",
        "mixed_sse_qualified",
    ):
        if extension_native["decision"].get(decision) is not True:
            raise SystemExit(
                f"mixed/conversation native evidence is missing: {decision}"
            )
    if transport_reference["source"].get("commit") != transport_native[
        "source"
    ].get("commit"):
        raise SystemExit("transport/cache reference and native commits differ")
    transport_binding = transport_native["dependencies"].get("reference", {})
    expected_transport_reference = file_component(
        ARTIFACT_PATHS["transport_cache_reference"],
        "benchmarks/results/vl-transport-cache-reference-v0.1.0.json",
    )
    if transport_binding != expected_transport_reference:
        raise SystemExit("transport/cache native reference binding changed")
    if not all(transport_reference["qualification_checks"].values()):
        raise SystemExit("transport/cache reference contains a failed check")
    for checks in transport_native["qualification_checks"].values():
        if not all(checks.values()):
            raise SystemExit("transport/cache native contains a failed check")
    for decision in (
        "all_observations_reference_exact",
        "verified_https_qualified",
        "video_sampling_cache_identity_qualified",
        "video_content_a_b_a_qualified",
        "mixed_order_mutation_qualified",
        "long_generation_usage_qualified",
        "cache_disabled_and_error_invariance_qualified",
        "two_resident_model_loads",
    ):
        if transport_native["decision"].get(decision) is not True:
            raise SystemExit(
                f"transport/cache native evidence is missing: {decision}"
            )


def build_payload() -> dict[str, Any]:
    payloads = {
        name: load_json_object(path)
        for name, path in ARTIFACT_PATHS.items()
        if path.suffix == ".json"
    }
    validate_inputs(payloads)
    requirements = build_requirements()
    native_cases = case_map(payloads["native_capability"], ("matrix", "cases"))
    execution_cases = case_map(
        payloads["execution_envelope"], ("matrix", "observations")
    )
    extension_cases = case_map(payloads["mixed_conversation_native"], ("cases",))
    transport_reference_cases = case_map(
        payloads["transport_cache_reference"], ("cases",)
    )
    transport_native_cases = {
        **case_map(
            payloads["transport_cache_native"],
            ("runs", "cache_enabled", "cases"),
            "observation_id",
        ),
        **case_map(
            payloads["transport_cache_native"],
            ("runs", "cache_disabled", "cases"),
            "observation_id",
        ),
    }
    for item in requirements:
        for record in item["evidence"]:
            artifact = record["artifact"]
            cases = record.get("case_ids", [])
            if not cases:
                continue
            available = (
                native_cases
                if artifact == "native_capability"
                else execution_cases
                if artifact == "execution_envelope"
                else extension_cases
                if artifact == "mixed_conversation_native"
                else transport_reference_cases
                if artifact == "transport_cache_reference"
                else transport_native_cases
                if artifact == "transport_cache_native"
                else None
            )
            if available is None:
                raise SystemExit(
                    f"{item['requirement_id']}: case ids cannot reference {artifact}"
                )
            for case_id in cases:
                if case_id not in available:
                    raise SystemExit(
                        f"{item['requirement_id']}: missing evidence case {case_id}"
                    )
                if available[case_id].get("qualified") is not True:
                    raise SystemExit(
                        f"{item['requirement_id']}: evidence case {case_id} failed"
                    )
    counts = {
        status: sum(item["status"] == status for item in requirements)
        for status in ("covered", "partial", "missing")
    }
    blockers = [
        {
            "requirement_id": item["requirement_id"],
            "gaps": item["gaps"],
        }
        for item in requirements
        if item["status"] != "covered"
    ]
    return {
        "schema": "aima-amd395-qwen36/native-vl-g1-coverage-audit/v1",
        "audited_on": "2026-08-15",
        "complete": True,
        "qualified": False,
        "scope": "goal-sections-2.1-through-2.4-requirement-to-evidence",
        "inputs": {
            name: file_component(path, str(path.relative_to(ROOT)))
            for name, path in ARTIFACT_PATHS.items()
        },
        "coverage": {
            "requirements": len(requirements),
            "counts": counts,
            "items": requirements,
        },
        "blocking_gaps": blockers,
        "next_evidence": [
            {
                "evidence_id": "g1-error-extension",
                "requirement_ids": ["G1.2.2.error_parity"],
                "cases": [
                    "error_empty_image",
                    "error_empty_video",
                    "error_remote_timeout",
                    "error_remote_unreachable",
                    "error_media_bytes_over_limit",
                    "error_video_duration_over_limit",
                ],
            },
            {
                "evidence_id": "g1-generation-and-current-head-requalification",
                "requirement_ids": [
                    "G1.2.1.generation",
                    "G1.2.2.model_semantics",
                    "G1.2.3.product_preservation",
                ],
                "cases": [
                    "resolve_four_usage_completion_differences",
                    "current_head_processor_to_output_requalification",
                    "complete_g3_text_nonregression",
                ],
            },
        ],
        "decision": {
            "audit_complete": True,
            "all_referenced_cases_qualified": True,
            "coverage_complete": counts["partial"] == 0
            and counts["missing"] == 0,
            "new_evidence_required": counts["partial"] > 0
            or counts["missing"] > 0,
            "g1_passed": False,
            "g2_passed": False,
            "g3_passed": False,
            "g4_passed": False,
            "g5_passed": False,
        },
    }


def verify_exact(path: Path, expected: dict[str, Any]) -> None:
    actual = load_json_object(path)
    if actual != expected:
        raise SystemExit(f"G1 coverage audit is stale: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected_sidecar = f"{digest}  {path.name}\n"
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise SystemExit(f"G1 coverage audit sidecar is stale: {sidecar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    sealed = seal_manifest(build_payload())
    if args.check:
        verify_exact(output, sealed)
        print(f"native VL G1 coverage audit: PASS ({output})")
        return 0
    digest = atomic_json(output, sealed)
    print(
        json.dumps(
            {
                "output": str(output),
                "qualified": False,
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
