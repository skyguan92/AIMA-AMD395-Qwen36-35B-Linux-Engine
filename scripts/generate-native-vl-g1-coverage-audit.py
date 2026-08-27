#!/usr/bin/env python3
"""Generate the requirement-to-evidence audit for the native VL G1 gate."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_error_limits import (  # noqa: E402
    NATIVE_REPLAY as ERROR_LIMITS_NATIVE_REPLAY,
    REFERENCE_CASE_ORDER as ERROR_LIMITS_REFERENCE_CASE_ORDER,
)
from aima_engine.vl_generation_layer_oracle import (  # noqa: E402
    validate_generation_layer_oracle_manifest,
)
from aima_engine.vl_generation_oracle import (  # noqa: E402
    validate_generation_oracle_manifest,
)
from aima_engine.vl_prefill_state_oracle import (  # noqa: E402
    validate_vl_prefill_state_oracle_manifest,
)
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    load_json_object,
    seal_manifest,
    verify_manifest_integrity,
)
from aima_engine.vl_task_quality import (  # noqa: E402
    CASE_ORDER as TASK_QUALITY_CASE_ORDER,
    validate_reference_manifest as validate_task_quality_reference_manifest,
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
    "media_io_reference": (
        ROOT / "benchmarks/results/vl-media-io-reference-v0.1.0.json"
    ),
    "error_limits_reference": (
        ROOT / "benchmarks/results/vl-error-limits-reference-v0.1.0.json"
    ),
    "error_limits_native": (
        ROOT / "benchmarks/results/native-vl-error-limits-v0.1.0.json"
    ),
    "vision_pipeline": (
        ROOT
        / "benchmarks/results/native-vision-pipeline-current-head-v0.2.0.json"
    ),
    "language_boundary": (
        ROOT / "benchmarks/results/vl-correctness-v0.1.0.json"
    ),
    "task_quality_reference": (
        ROOT / "benchmarks/results/vl-task-quality-reference-v0.1.0.json"
    ),
    "task_quality_native": (
        ROOT / "benchmarks/results/native-vl-task-quality-v0.1.0.json"
    ),
    "generation_oracle": (
        ROOT / "benchmarks/results/vl-generation-oracle-v0.1.0.json"
    ),
    "generation_layer_oracle": (
        ROOT
        / "benchmarks/results/vl-generation-layer-oracle-v0.1.0.json"
    ),
    "prefill_state_oracle": (
        ROOT / "benchmarks/results/vl-prefill-state-oracle-v0.1.0.json"
    ),
    "generation_native": (
        ROOT
        / "benchmarks/results/native-vl-generation-current-head-v0.1.0.json"
    ),
    "text_v151_nonregression": (
        ROOT / "benchmarks/results/text-v151-nonregression-v0.1.0.json"
    ),
    "error_limits_contract": ROOT / "aima_engine/vl_error_limits.py",
    "cache_identity_unit": ROOT / "tests/native_multimodal_cache_test.cpp",
    "generator": Path(__file__).resolve(),
}

GENERATION_ORACLE_ROOT = ROOT / "benchmarks/oracles/vl-generation-v0.1.0"
GENERATION_LAYER_ORACLE_ROOT = (
    ROOT / "benchmarks/oracles/vl-generation-layer-v0.1.0"
)
PREFILL_STATE_ORACLE_ROOT = (
    ROOT / "benchmarks/oracles/vl-prefill-state-v0.1.0"
)
EXPECTED_CANDIDATE = {
    "source_commit": "bd012874027defa528279a357609b713e9069df4",
    "binary_sha256": (
        "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
    ),
}
EXPECTED_BASELINE = {
    "release": "v1.5.1",
    "source_commit": "65c198415709dad6d046c247acab3dc9df2a95a0",
    "binary_sha256": (
        "a9f18771175757af080c8a1d8d7e3fb3906c9aa41b43a496686103b626f80262"
    ),
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
                evidence(
                    "media_io_reference",
                    "default_white",
                    "red",
                    note="fixed vLLM white/default and request-red RGBA compositing",
                ),
                evidence(
                    "error_limits_native",
                    "rgba_default_cold",
                    "rgba_red_miss",
                    "rgba_default_restored",
                    "rgba_red_after_errors",
                    note="resident native RGBA request semantics and A/B/A identity",
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
                evidence(
                    "error_limits_reference",
                    "video_sampling_default",
                    "video_sampling_empty_mapping",
                    "video_long_duration",
                    note="fixed vLLM merge semantics and 6000-second sparse video acceptance",
                ),
                evidence(
                    "error_limits_native",
                    "video_default_cold",
                    "video_empty_mapping_exact",
                    "video_default_restored",
                    "video_long_duration",
                    note="resident native replay is reference-exact across merge and duration",
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
            "covered",
            [
                evidence(
                    "native_capability",
                    note=(
                        "20/20 finish-reason parity and 16/16 VL usage parity; "
                        "the two text-only diagnostics remain owned by G3"
                    ),
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
                evidence(
                    "generation_native",
                    note=(
                        "current-HEAD tool prefixes, selected tokens, full-vocabulary "
                        "top-1 and internal decode boundaries are qualified"
                    ),
                ),
                evidence(
                    "task_quality_native",
                    note=(
                        "12/12 long image/video tasks are reference-exact for "
                        "generated content, output tokens, usage, finish reason "
                        "and rendered prompt vectors"
                    ),
                ),
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
            "covered",
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
                evidence(
                    "generation_native",
                    note=(
                        "current-HEAD processor-to-output replay binds exact prompts, "
                        "prefill states, decode boundaries, full-attention internals, "
                        "top-1 logits and selected tokens"
                    ),
                ),
            ],
        ),
        requirement(
            "G1.2.2.error_parity",
            "Invalid media status and error-category parity",
            "covered",
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
                evidence(
                    "error_limits_reference",
                    "empty_image_remote",
                    "empty_video_remote",
                    "unreachable_image_remote",
                    "oversize_image_remote",
                    "timeout_image_remote",
                    note=(
                        "fixed vLLM freezes empty, inaccessible, byte-limit and "
                        "timeout status/type contracts"
                    ),
                ),
                evidence(
                    "error_limits_native",
                    "empty_image_remote",
                    "empty_video_remote",
                    "unreachable_image_remote",
                    "oversize_image_remote",
                    "timeout_image_remote",
                    note=(
                        "native preserves the fail-closed product error shape with "
                        "explicit compatible-category checks"
                    ),
                ),
            ],
        ),
        requirement(
            "G1.2.3.product_preservation",
            "Existing text, API, security, startup and cache product behavior",
            "covered",
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
                evidence(
                    "text_v151_nonregression",
                    note=(
                        "exact-candidate G3 requalification covers text correctness, "
                        "MMLU-256, OpenAI surfaces, startup, prefix cache, doctor, "
                        "memory and the frozen 19-cell release comparison"
                    ),
                ),
            ],
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
                evidence(
                    "error_limits_native",
                    "rgba_default_cold",
                    "rgba_red_miss",
                    "rgba_default_restored",
                    "rgba_red_after_errors",
                    "video_default_cold",
                    "video_empty_mapping_exact",
                    "video_default_restored",
                    note=(
                        "processor identity v3 binds RGBA background and shallow-merged "
                        "video sampling policy"
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
                evidence(
                    "error_limits_native",
                    "empty_image_remote",
                    "empty_video_remote",
                    "unreachable_image_remote",
                    "oversize_image_remote",
                    "timeout_image_remote",
                    "rgba_red_after_errors",
                    note="five errors do not pollute media identity or later resident reuse",
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


def validate_source_components(
    label: str, commit: str, components: list[dict[str, Any]]
) -> None:
    for component in components:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{commit}:{component['path']}",
            ],
            capture_output=True,
            check=False,
        )
        payload = completed.stdout
        if (
            completed.returncode != 0
            or len(payload) != component["bytes"]
            or hashlib.sha256(payload).hexdigest() != component["sha256"]
        ):
            raise SystemExit(
                f"{label} component is missing or stale: {component['path']}"
            )


def validate_historical_reference(
    label: str, payload: dict[str, Any]
) -> None:
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("dirty") is not False:
        raise SystemExit(f"{label} capture identity is not clean")
    components = source.get("files")
    commit = source.get("commit")
    if not isinstance(commit, str) or not isinstance(components, list):
        raise SystemExit(f"{label} source components are missing")
    validate_source_components(label, commit, components)


def validate_inputs(payloads: dict[str, dict[str, Any]]) -> None:
    native = payloads["native_capability"]
    execution = payloads["execution_envelope"]
    serving = payloads["resident_serving"]
    extension_reference = payloads["mixed_conversation_reference"]
    extension_native = payloads["mixed_conversation_native"]
    transport_reference = payloads["transport_cache_reference"]
    transport_native = payloads["transport_cache_native"]
    media_io_reference = payloads["media_io_reference"]
    error_limits_reference = payloads["error_limits_reference"]
    error_limits_native = payloads["error_limits_native"]
    vision = payloads["vision_pipeline"]
    language = payloads["language_boundary"]
    task_reference = payloads["task_quality_reference"]
    task_native = payloads["task_quality_native"]
    generation_oracle = payloads["generation_oracle"]
    generation_layer = payloads["generation_layer_oracle"]
    prefill_state = payloads["prefill_state_oracle"]
    generation_native = payloads["generation_native"]
    text_nonregression = payloads["text_v151_nonregression"]
    for name, payload in (
        ("native capability", native),
        ("execution envelope", execution),
        ("resident serving", serving),
        ("mixed/conversation reference", extension_reference),
        ("mixed/conversation native", extension_native),
        ("transport/cache reference", transport_reference),
        ("transport/cache native", transport_native),
        ("media IO reference", media_io_reference),
        ("error/limit reference", error_limits_reference),
        ("error/limit native", error_limits_native),
    ):
        if payload.get("complete") is not True or payload.get("qualified") is not True:
            raise SystemExit(f"{name} evidence is not complete and qualified")
    if vision.get("complete") is not True or not vision["decision"].get(
        "full_visual_pipeline_qualified"
    ):
        raise SystemExit("vision pipeline evidence is incomplete")
    if (
        language.get("complete") is not True
        or language.get("qualified") is not True
        or language["decision"].get("g2_passed") is not True
        or language["aggregate"].get("full_vocabulary_rows") != 84
        or language["aggregate"].get("full_vocabulary_top1_matches") != 84
    ):
        raise SystemExit("language boundary evidence is incomplete")

    candidate_commit = execution["source"]["commit"]
    candidate_binary = execution["binary"]["sha256"]
    for name, payload in (
        ("native capability", native),
        ("resident serving", serving),
        ("mixed/conversation native", extension_native),
        ("transport/cache native", transport_native),
        ("error/limit native", error_limits_native),
        ("task-quality native", task_native),
        ("generation native", generation_native),
    ):
        if (
            payload["source"]["commit"] != candidate_commit
            or payload["binary"]["sha256"] != candidate_binary
        ):
            raise SystemExit(f"{name} candidate identity differs")
    if (
        vision["source"]["commit"] != candidate_commit
        or vision["candidate_binary_sha256"] != candidate_binary
        or language["candidate"]["source_commit"] != candidate_commit
        or language["candidate"]["binary_sha256"] != candidate_binary
    ):
        raise SystemExit("vision/language candidate identity differs")
    for name, payload in (
        ("task-quality reference", task_reference),
        ("task-quality native", task_native),
        ("generation oracle", generation_oracle),
        ("generation layer oracle", generation_layer),
        ("prefill-state oracle", prefill_state),
        ("generation native", generation_native),
    ):
        integrity_errors = verify_manifest_integrity(payload)
        if integrity_errors:
            raise SystemExit(
                f"{name} integrity failed: " + "; ".join(integrity_errors)
            )
    task_reference_errors = validate_task_quality_reference_manifest(
        task_reference
    )
    if task_reference_errors:
        raise SystemExit(
            "task-quality reference is invalid: "
            + "; ".join(task_reference_errors)
        )
    if (
        task_native.get("complete") is not True
        or task_native.get("qualified") is not True
        or tuple(
            case.get("case_id")
            for case in task_native.get("matrix", {}).get("cases", [])
        )
        != TASK_QUALITY_CASE_ORDER
        or not all(
            case.get("qualified") is True
            and all(case.get("qualification_checks", {}).values())
            for case in task_native["matrix"]["cases"]
        )
    ):
        raise SystemExit("native task-quality evidence is incomplete")
    if task_native["dependencies"].get("reference") != file_component(
        ARTIFACT_PATHS["task_quality_reference"],
        "benchmarks/results/vl-task-quality-reference-v0.1.0.json",
    ):
        raise SystemExit("native task-quality reference binding changed")
    for decision in (
        "twelve_task_quality_cases_qualified",
        "twelve_render_prompt_vectors_exact",
        "image_task_quality_not_below_reference",
        "video_task_quality_not_below_reference",
        "single_resident_model_load",
    ):
        if task_native["decision"].get(decision) is not True:
            raise SystemExit(
                f"native task-quality evidence is missing: {decision}"
            )
    if (
        task_native["decision"].get(
            "twelve_long_greedy_cases_reference_exact"
        )
        is not True
        or task_native["decision"].get(
            "twelve_output_token_vectors_exact"
        )
        is not True
        or task_native["matrix"].get("exact_output_token_vectors")
        != "12/12"
        or task_native["matrix"].get("exact_generated_content") != "12/12"
        or task_native["matrix"].get("exact_reference_usage") != "12/12"
        or task_native["matrix"].get("exact_reference_finish_reason")
        != "12/12"
    ):
        raise SystemExit("native task-quality generation exactness changed")
    generation_errors = validate_generation_oracle_manifest(
        generation_oracle, oracle_root=GENERATION_ORACLE_ROOT
    )
    layer_errors = validate_generation_layer_oracle_manifest(
        generation_layer, oracle_root=GENERATION_LAYER_ORACLE_ROOT
    )
    state_errors = validate_vl_prefill_state_oracle_manifest(
        prefill_state, oracle_root=PREFILL_STATE_ORACLE_ROOT
    )
    if generation_errors or layer_errors or state_errors:
        raise SystemExit(
            "generation oracle closure is invalid: "
            f"generation={len(generation_errors)}, "
            f"layer={len(layer_errors)}, state={len(state_errors)}"
        )
    if (
        generation_native.get("complete") is not True
        or generation_native.get("qualified") is not True
        or not all(generation_native.get("checks", {}).values())
    ):
        raise SystemExit("current-HEAD generation evidence is incomplete")
    generation_dependencies = {
        "generation_oracle": (
            "generation_oracle",
            "benchmarks/results/vl-generation-oracle-v0.1.0.json",
        ),
        "generation_layer_oracle": (
            "generation_layer_oracle",
            "benchmarks/results/vl-generation-layer-oracle-v0.1.0.json",
        ),
        "vl_prefill_state_oracle": (
            "prefill_state_oracle",
            "benchmarks/results/vl-prefill-state-oracle-v0.1.0.json",
        ),
    }
    for dependency, (artifact, relative) in generation_dependencies.items():
        if generation_native["dependencies"].get(dependency) != file_component(
            ARTIFACT_PATHS[artifact], relative
        ):
            raise SystemExit(
                f"current-HEAD generation binding changed: {dependency}"
            )
    for decision in (
        "two_shared_prefixes_exact",
        "two_native_full_vocab_finite",
        "two_decode_boundary_sets_bit_exact",
        "two_prefill_state_sets_bit_exact",
        "two_native_generation_top1_exact",
        "two_generation_logits_kld_under_0_005",
        "g1_generation_closed",
    ):
        if generation_native["decision"].get(decision) is not True:
            raise SystemExit(
                f"current-HEAD generation evidence is missing: {decision}"
            )
    if (
        text_nonregression.get("schema")
        != "aima-amd395-qwen36/text-v151-nonregression/v1"
        or text_nonregression.get("complete") is not True
        or text_nonregression.get("qualified") is not True
        or text_nonregression.get("candidate") != EXPECTED_CANDIDATE
        or text_nonregression.get("baseline") != EXPECTED_BASELINE
        or text_nonregression.get("decision", {}).get(
            "g3_text_product_no_regression"
        )
        is not True
    ):
        raise SystemExit("exact-candidate G3 text non-regression evidence is invalid")
    cross_checks = text_nonregression.get("cross_evidence_checks")
    check_groups = text_nonregression.get("checks")
    if (
        not isinstance(cross_checks, dict)
        or not cross_checks
        or not all(value is True for value in cross_checks.values())
        or not isinstance(check_groups, dict)
        or not check_groups
        or not all(
            isinstance(group, dict)
            and group
            and all(value is True for value in group.values())
            for group in check_groups.values()
        )
    ):
        raise SystemExit("G3 text non-regression contains a failed check")
    validate_source_components(
        "native task-quality",
        task_native["source"]["commit"],
        task_native["source"]["files"],
    )
    validate_source_components(
        "qualified generation",
        generation_native["source"]["commit"],
        generation_native["source"]["files"],
    )
    if native["matrix"].get("reference_status_exact") != "30/30":
        raise SystemExit("native capability status parity is incomplete")
    if native["matrix"].get("reference_finish_reason_exact") != "20/20":
        raise SystemExit("native capability finish-reason parity is incomplete")
    if (
        native["matrix"].get("reference_usage_exact") != "16/18"
        or native["matrix"].get("vl_reference_usage_exact") != "16/16"
        or native["matrix"].get("text_vllm_usage_diagnostic") != "0/2"
        or native["decision"].get("deterministic_vl_reference_usage_exact")
        is not True
        or native["decision"].get("text_usage_boundary_owned_by_g3_v151")
        is not True
    ):
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
    validate_historical_reference(
        "mixed/conversation reference", extension_reference
    )
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
    validate_historical_reference("transport/cache reference", transport_reference)
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
    if tuple(
        case["case_id"] for case in error_limits_reference["cases"]
    ) != ERROR_LIMITS_REFERENCE_CASE_ORDER:
        raise SystemExit("error/limit reference case order changed")
    if not all(
        case.get("qualified") is True
        and all(case.get("qualification_checks", {}).values())
        for case in error_limits_reference["cases"]
    ):
        raise SystemExit("error/limit reference contains a failed case")
    if error_limits_native["dependencies"].get("media_io_oracle") != file_component(
        ARTIFACT_PATHS["media_io_reference"],
        "benchmarks/results/vl-media-io-reference-v0.1.0.json",
    ):
        raise SystemExit("error/limit native media IO binding changed")
    if error_limits_native["dependencies"].get("reference") != file_component(
        ARTIFACT_PATHS["error_limits_reference"],
        "benchmarks/results/vl-error-limits-reference-v0.1.0.json",
    ):
        raise SystemExit("error/limit native reference binding changed")
    if tuple(
        (case["observation_id"], case["reference_case_id"])
        for case in error_limits_native["run"]["cases"]
    ) != ERROR_LIMITS_NATIVE_REPLAY:
        raise SystemExit("error/limit native replay order changed")
    if not all(
        case.get("qualified") is True
        and all(case.get("qualification_checks", {}).values())
        for case in error_limits_native["run"]["cases"]
    ):
        raise SystemExit("error/limit native contains a failed case")
    for checks in error_limits_native["qualification_checks"].values():
        if not all(checks.values()):
            raise SystemExit("error/limit native contains a failed qualification")
    for decision in (
        "empty_video_mapping_qualified",
        "error_cache_non_pollution_qualified",
        "error_limit_categories_qualified",
        "long_duration_video_qualified",
        "one_resident_model_load",
        "rgba_background_cache_identity_qualified",
        "thirteen_observations_reference_exact",
    ):
        if error_limits_native["decision"].get(decision) is not True:
            raise SystemExit(
                f"error/limit native evidence is missing: {decision}"
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
    media_io_reference_cases = case_map(
        payloads["media_io_reference"], ("cases",)
    )
    error_limits_reference_cases = case_map(
        payloads["error_limits_reference"], ("cases",)
    )
    error_limits_native_cases = case_map(
        payloads["error_limits_native"], ("run", "cases"), "observation_id"
    )
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
                else media_io_reference_cases
                if artifact == "media_io_reference"
                else error_limits_reference_cases
                if artifact == "error_limits_reference"
                else error_limits_native_cases
                if artifact == "error_limits_native"
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
    coverage_complete = counts["partial"] == 0 and counts["missing"] == 0
    g2_passed = payloads["language_boundary"]["decision"]["g2_passed"]
    g3_passed = payloads["text_v151_nonregression"]["decision"][
        "g3_text_product_no_regression"
    ]
    qualified = coverage_complete and g2_passed and g3_passed
    return {
        "schema": "aima-amd395-qwen36/native-vl-g1-coverage-audit/v1",
        "audited_on": "2026-08-21",
        "complete": True,
        "qualified": qualified,
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
        "next_evidence": [],
        "decision": {
            "audit_complete": True,
            "all_referenced_cases_qualified": True,
            "current_head_processor_to_output_qualified": True,
            "twelve_task_quality_cases_qualified": True,
            "twelve_long_greedy_cases_reference_exact": True,
            "coverage_complete": coverage_complete,
            "new_evidence_required": not coverage_complete,
            "g1_passed": qualified,
            "g2_passed": g2_passed,
            "g3_passed": g3_passed,
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
                "qualified": sealed["qualified"],
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
