#!/usr/bin/env python3
"""Validate and summarize one fresh-process native/vLLM G4 diagnostic pair."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "aima-amd395-qwen36/vl-performance-diagnostic-pair/v1"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON input must be an object: {path}")
    return value


def positive_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result) and result > 0:
            return result
    return None


def ratio(numerator: Any, denominator: Any) -> float | None:
    left = positive_number(numerator)
    right = positive_number(denominator)
    return left / right if left is not None and right is not None else None


def seconds_from_milliseconds(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result) and result >= 0:
            return result / 1000.0
    return None


def nonnegative_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result) and result >= 0:
            return result
    return None


def add_nonnegative(*values: Any) -> float | None:
    converted = [nonnegative_number(value) for value in values]
    if any(value is None for value in converted):
        return None
    return sum(value for value in converted if value is not None)


def subtract_stage(total: Any, component: Any) -> float | None:
    total_value = positive_number(total)
    component_value = nonnegative_number(component)
    if (
        total_value is None
        or component_value is None
        or total_value <= component_value
    ):
        return None
    return total_value - component_value


def one_mm_record(stage: Mapping[str, Any]) -> dict[str, Any] | None:
    multimodal = stage.get("multimodal")
    if not isinstance(multimodal, Mapping):
        return None
    merged = multimodal.get("merged")
    if not isinstance(merged, Mapping) or not merged:
        return None
    records = [value for value in merged.values() if isinstance(value, Mapping)]
    if len(records) != len(merged):
        return None
    processor_records = [
        value
        for value in records
        if nonnegative_number(value.get("preprocessor_total_secs")) is not None
    ]
    encoder_records = [
        value
        for value in records
        if positive_number(value.get("encoder_forward_secs")) is not None
    ]
    # Processor and encoder timings can be published under different
    # request-local identifiers.  An exact multimodal processor-cache hit
    # legitimately publishes one processor interval and no encoder interval.
    # More than one interval of either kind, or no recognized interval at all,
    # is ambiguous and must fail closed.
    if (
        len(processor_records) > 1
        or len(encoder_records) > 1
        or not processor_records and not encoder_records
    ):
        return None
    return {
        "preprocessor_total_secs": (
            processor_records[0]["preprocessor_total_secs"]
            if processor_records
            else 0.0
        ),
        "encoder_forward_secs": (
            encoder_records[0]["encoder_forward_secs"]
            if encoder_records
            else 0.0
        ),
        "processor_record_count": len(processor_records),
        "encoder_record_count": len(encoder_records),
    }


def stage_record(
    lines: list[str], benchmark_id: object
) -> dict[str, Any] | None:
    """Select exactly one middleware record for a benchmark request."""

    if not isinstance(benchmark_id, str) or not benchmark_id:
        return None
    matches: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            return None
        if value.get("benchmark_id") == benchmark_id:
            matches.append(value)
    return matches[0] if len(matches) == 1 else None


def prometheus_delta(sample: Mapping[str, Any]) -> dict[str, float]:
    prometheus = sample.get("prometheus")
    if not isinstance(prometheus, Mapping):
        return {}
    delta = prometheus.get("delta")
    if not isinstance(delta, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in delta.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def warmup_contract(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    usage = payload.get("usage")
    choices = payload.get("choices")
    if not isinstance(usage, Mapping) or not isinstance(choices, list):
        return None
    if len(choices) != 1 or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return None
    token_counts = {
        key: usage.get(key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in token_counts.values()
    ):
        return None
    return {
        **token_counts,
        "finish_reason": choices[0].get("finish_reason"),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def valid_media_components(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            return False
        digest = item.get("sha256")
        if (
            item.get("index") != index
            or item.get("modality") not in {"image", "video"}
            or not isinstance(item.get("path"), str)
            or not item["path"].startswith("${AIMA_VL_MEDIA_ROOT}/")
            or not isinstance(item.get("bytes"), int)
            or isinstance(item.get("bytes"), bool)
            or item["bytes"] <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return False
    return True


def build_summary(
    root: Path,
    *,
    request_relative_path: Path = Path("request.json"),
    expectations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expectations = expectations or {}
    reference = load_object(root / "reference" / request_relative_path)
    candidate = load_object(root / "candidate" / request_relative_path)
    reference_warmup = warmup_contract(
        load_object(root / "reference/text-warmup.json")
    )
    candidate_warmup = warmup_contract(
        load_object(root / "candidate/text-warmup.json")
    )
    stage_lines = (root / "reference/vllm-vl-stages.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    stage = stage_record(stage_lines, reference.get("benchmark_id"))
    if stage is None:
        raise SystemExit(
            "reference stage log must contain exactly one matching record"
        )

    reference_response = reference.get("response", {})
    candidate_response = candidate.get("response", {})
    reference_timings = reference.get("timings", {})
    candidate_timings = candidate.get("timings", {})
    native = candidate.get("native_metrics")
    native_vl = native.get("vl") if isinstance(native, Mapping) else None
    mm = one_mm_record(stage)
    prom = prometheus_delta(reference)
    reference_media = reference.get("request", {}).get("media")
    candidate_media = candidate.get("request", {}).get("media")

    reference_prefill_path_seconds = prom.get(
        "vllm:request_prefill_time_seconds_sum"
    )
    reference_decode_seconds = prom.get(
        "vllm:request_decode_time_seconds_sum"
    )
    prompt_tokens = (
        reference_response.get("usage", {}).get("prompt_tokens")
        if isinstance(reference_response, Mapping)
        and isinstance(reference_response.get("usage"), Mapping)
        else None
    )
    completion_tokens = (
        reference_response.get("usage", {}).get("completion_tokens")
        if isinstance(reference_response, Mapping)
        and isinstance(reference_response.get("usage"), Mapping)
        else None
    )
    expected_output_tokens = expectations.get("output_tokens")
    if expected_output_tokens is None:
        expected_output_tokens = completion_tokens
    exact_media_cache = expectations.get("media_cache_mode") == "exact"
    visual_tokens = (
        native_vl.get("visual_tokens")
        if isinstance(native_vl, Mapping)
        else None
    )
    reference_encoder_seconds = (
        mm.get("encoder_forward_secs") if isinstance(mm, Mapping) else None
    )
    candidate_encoder_seconds = seconds_from_milliseconds(
        native_vl.get("vision_encode_wall_ms")
        if isinstance(native_vl, Mapping)
        else None
    )
    candidate_plan_seconds = seconds_from_milliseconds(
        native_vl.get("vision_plan_build_wall_ms")
        if isinstance(native_vl, Mapping)
        else None
    )
    candidate_cold_vision_seconds = add_nonnegative(
        candidate_plan_seconds, candidate_encoder_seconds
    )
    reference_encoder_executed = (
        positive_number(reference_encoder_seconds) is not None
    )
    candidate_encoder_executed = (
        positive_number(candidate_encoder_seconds) is not None
    )
    candidate_prefill_path_seconds = seconds_from_milliseconds(
        native.get("ttft_ms") if isinstance(native, Mapping) else None
    )
    reference_llm_prefill_seconds = subtract_stage(
        reference_prefill_path_seconds, reference_encoder_seconds
    )
    candidate_llm_prefill_seconds = subtract_stage(
        candidate_prefill_path_seconds, candidate_cold_vision_seconds
    )

    checks = {
        "both_samples_complete": (
            reference.get("complete") is True
            and candidate.get("complete") is True
        ),
        "roles_exact": (
            reference.get("engine_role") == "reference"
            and candidate.get("engine_role") == "candidate"
        ),
        "benchmark_id_exact": (
            reference.get("benchmark_id") == candidate.get("benchmark_id")
            == stage.get("benchmark_id")
        ),
        "request_template_exact": (
            reference.get("request", {}).get("template_sha256")
            == candidate.get("request", {}).get("template_sha256")
            and reference.get("request", {}).get("summary")
            == candidate.get("request", {}).get("summary")
        ),
        "ordered_media_content_exact": (
            valid_media_components(reference_media)
            and reference_media == candidate_media
        ),
        "text_padding_contract_exact": (
            reference.get("request", {}).get("text_padding")
            == candidate.get("request", {}).get("text_padding")
            and reference.get("request", {})
            .get("text_padding", {})
            .get("frozen_single_token_id")
            == 830
        ),
        "symmetric_text_warmup": (
            reference_warmup is not None
            and candidate_warmup is not None
            and all(
                warmup.get("completion_tokens") == 1
                and warmup.get("total_tokens")
                == warmup.get("prompt_tokens", 0) + 1
                and warmup.get("finish_reason") == "length"
                for warmup in (reference_warmup, candidate_warmup)
            )
        ),
        "response_shape_and_length_exact": (
            isinstance(reference_response.get("finish_reason"), str)
            and reference_response.get("finish_reason")
            == candidate_response.get("finish_reason")
            and isinstance(reference_response.get("usage"), Mapping)
            and reference_response.get("usage")
            == candidate_response.get("usage")
        ),
        "completion_tokens_exact": (
            isinstance(expected_output_tokens, int)
            and not isinstance(expected_output_tokens, bool)
            and expected_output_tokens > 0
            and completion_tokens == expected_output_tokens
            and candidate_response.get("usage", {}).get(
                "completion_tokens"
            )
            == expected_output_tokens
        ),
        "one_reference_stage_record": mm is not None,
        "reference_encoder_execution_matches_cache_mode": (
            not reference_encoder_executed
            if exact_media_cache
            else reference_encoder_executed
        ),
        "reference_stage_clean": (
            stage.get("status_code") == 200
            and stage.get("request_error") is None
            and stage.get("stats_error") is None
        ),
        "reference_prefill_observed_once": prom.get(
            "vllm:request_prefill_time_seconds_count"
        )
        == 1.0,
        "reference_ttft_observed_once": prom.get(
            "vllm:time_to_first_token_seconds_count"
        )
        == 1.0,
        "native_vl_metrics_present": isinstance(native_vl, Mapping),
        "prompt_token_count_exact": (
            isinstance(native, Mapping)
            and native.get("prompt_tokens") == prompt_tokens
            and candidate_response.get("usage", {}).get("prompt_tokens")
            == prompt_tokens
        ),
        "stage_boundaries_valid": (
            reference_llm_prefill_seconds is not None
            and candidate_llm_prefill_seconds is not None
            and candidate_cold_vision_seconds is not None
            and (
                exact_media_cache
                or candidate_cold_vision_seconds > 0
            )
        ),
        "candidate_prefix_cache_expected": (
            isinstance(native, Mapping)
            and isinstance(native.get("prefix_cache"), Mapping)
            and native["prefix_cache"].get("lookup")
            == expectations.get("prefix_cache_lookup", "miss")
        ),
    }

    media_cache_mode = expectations.get("media_cache_mode")
    if media_cache_mode is not None:
        media_count = len(candidate_media) if isinstance(candidate_media, list) else -1
        hits = native_vl.get("media_cache_hits") if isinstance(native_vl, Mapping) else None
        misses = native_vl.get("media_cache_misses") if isinstance(native_vl, Mapping) else None
        entries = native_vl.get("media_cache_entries") if isinstance(native_vl, Mapping) else None
        vision_embedding_hit = (
            native_vl.get("vision_embedding_cache_hit")
            if isinstance(native_vl, Mapping)
            else None
        )
        if media_cache_mode == "disabled":
            cache_ok = hits == 0 and misses == media_count and entries == 0
            vision_embedding_cache_ok = vision_embedding_hit is False
        elif media_cache_mode == "cold":
            cache_ok = hits == 0 and misses == media_count and isinstance(entries, int) and entries > 0
            vision_embedding_cache_ok = vision_embedding_hit is False
        elif media_cache_mode == "exact":
            cache_ok = hits == media_count and misses == 0 and isinstance(entries, int) and entries > 0
            vision_embedding_cache_ok = vision_embedding_hit is True
        else:
            cache_ok = False
            vision_embedding_cache_ok = False
        checks["candidate_media_cache_expected"] = cache_ok
        checks["candidate_vision_embedding_cache_expected"] = (
            vision_embedding_cache_ok
        )

    reference_prefill_path_tps = ratio(
        prompt_tokens, reference_prefill_path_seconds
    )
    candidate_prefill_path_tps = ratio(
        prompt_tokens, candidate_prefill_path_seconds
    )
    reference_llm_prefill_tps = ratio(
        prompt_tokens, reference_llm_prefill_seconds
    )
    candidate_llm_prefill_tps = ratio(
        prompt_tokens, candidate_llm_prefill_seconds
    )
    reference_vision_tps = ratio(visual_tokens, reference_encoder_seconds)
    candidate_vision_tps = ratio(visual_tokens, candidate_encoder_seconds)
    candidate_cold_vision_path_tps = ratio(
        visual_tokens, candidate_cold_vision_seconds
    )
    comparisons = {
        "ttft_candidate_over_reference": ratio(
            candidate_timings.get("ttft_seconds"),
            reference_timings.get("ttft_seconds"),
        ),
        "total_candidate_over_reference": ratio(
            candidate_timings.get("total_seconds"),
            reference_timings.get("total_seconds"),
        ),
        "prefill_tps_candidate_over_reference": ratio(
            candidate_llm_prefill_tps, reference_llm_prefill_tps
        ),
        "prefill_path_tps_candidate_over_reference": ratio(
            candidate_prefill_path_tps, reference_prefill_path_tps
        ),
    }
    if exact_media_cache:
        # vLLM's exact multimodal cache hit reuses the encoder output and has
        # no encoder interval.  A throughput ratio against zero time is
        # undefined, so retain an explicit candidate execution-time gate.
        comparisons["vision_cache_hit_candidate_seconds"] = (
            candidate_cold_vision_seconds
        )
    else:
        comparisons["vision_tps_candidate_over_reference"] = ratio(
            candidate_vision_tps, reference_vision_tps
        )
        comparisons["cold_vision_path_tps_candidate_over_reference"] = ratio(
            candidate_cold_vision_path_tps, reference_vision_tps
        )
    decode_required = (
        isinstance(expected_output_tokens, int)
        and not isinstance(expected_output_tokens, bool)
        and expected_output_tokens > 1
    )
    if decode_required:
        comparisons["decode_tps_candidate_over_reference"] = ratio(
            candidate_timings.get("decode_tokens_per_second"),
            reference_timings.get("decode_tokens_per_second"),
        )
    diagnostic_thresholds = {
        "ttft": comparisons["ttft_candidate_over_reference"] is not None
        and comparisons["ttft_candidate_over_reference"] <= 1.0,
        "total": comparisons["total_candidate_over_reference"] is not None
        and comparisons["total_candidate_over_reference"] <= 1.0,
        "prefill": comparisons[
            "prefill_tps_candidate_over_reference"
        ]
        is not None
        and comparisons["prefill_tps_candidate_over_reference"] >= 1.0,
        "vision": (
            comparisons.get("vision_cache_hit_candidate_seconds") == 0.0
            if exact_media_cache
            else (
                comparisons.get("vision_tps_candidate_over_reference")
                is not None
                and comparisons["vision_tps_candidate_over_reference"] >= 1.0
            )
        ),
        "decode": (
            not decode_required
            or (
                comparisons.get("decode_tps_candidate_over_reference")
                is not None
                and comparisons["decode_tps_candidate_over_reference"] >= 1.0
            )
        ),
    }
    complete = all(checks.values()) and all(
        value is not None for value in comparisons.values()
    )
    return {
        "schema": SCHEMA,
        "complete": complete,
        "qualified": complete and all(diagnostic_thresholds.values()),
        "scope": "one-pair diagnosis only; not final G4 evidence",
        "stage_accounting": {
            "reference_llm_prefill": (
                "vLLM scheduled-to-first-token prefill minus the official "
                "multimodal encoder-forward interval"
            ),
            "candidate_llm_prefill": (
                "native request-to-first-token prefill minus cold vision "
                "plan construction and vision encoder execution"
            ),
            "candidate_cold_vision": (
                "vision plan construction plus vision encoder execution; "
                "the reference encoder interval retains any analogous "
                "first-execution setup"
            ),
        },
        "metric_applicability": {
            "vision_throughput": {
                "applicable": not exact_media_cache,
                "reason": (
                    None
                    if not exact_media_cache
                    else "reference exact media-cache hit did not execute the encoder"
                ),
            },
            "vision_cache_hit_execution": {
                "applicable": exact_media_cache,
                "required_candidate_seconds": 0.0 if exact_media_cache else None,
            },
        },
        "response_audit": {
            "reference": {
                "content_sha256": reference_response.get("content_sha256"),
                "content_bytes": reference_response.get("content_bytes"),
                "semantic_chunks": reference_response.get("semantic_chunks"),
            },
            "candidate": {
                "content_sha256": candidate_response.get("content_sha256"),
                "content_bytes": candidate_response.get("content_bytes"),
                "semantic_chunks": candidate_response.get("semantic_chunks"),
            },
            "content_equality_required": False,
        },
        "checks": checks,
        "measurements": {
            "text_warmup": {
                "request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Reply with one token for symmetric warmup."
                            ),
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 1,
                    "stream": False,
                },
                "reference_response": reference_warmup,
                "candidate_response": candidate_warmup,
            },
            "reference": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "client_ttft_seconds": reference_timings.get(
                    "ttft_seconds"
                ),
                "client_total_seconds": reference_timings.get(
                    "total_seconds"
                ),
                "media_load_decode_seconds": stage.get("media", {}).get(
                    "media_load_decode_secs"
                ),
                "processor_seconds": (
                    mm.get("preprocessor_total_secs")
                    if isinstance(mm, Mapping)
                    else None
                ),
                "vision_encode_seconds": reference_encoder_seconds,
                "vision_encoder_executed": reference_encoder_executed,
                "prefill_path_seconds": reference_prefill_path_seconds,
                "llm_prefill_seconds": reference_llm_prefill_seconds,
                "llm_decode_seconds": reference_decode_seconds,
                "client_decode_tokens_per_second": reference_timings.get(
                    "decode_tokens_per_second"
                ),
                "prefill_path_tokens_per_second": reference_prefill_path_tps,
                "llm_prefill_tokens_per_second": reference_llm_prefill_tps,
                "vision_tokens_per_second": reference_vision_tps,
                "memory": reference.get("memory"),
            },
            "candidate": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "client_ttft_seconds": candidate_timings.get(
                    "ttft_seconds"
                ),
                "client_total_seconds": candidate_timings.get(
                    "total_seconds"
                ),
                "media_load_decode_seconds": seconds_from_milliseconds(
                    native_vl.get("media_load_decode_wall_ms")
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "processor_seconds": seconds_from_milliseconds(
                    native_vl.get("processor_wall_ms")
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "vision_plan_build_seconds": candidate_plan_seconds,
                "vision_encode_seconds": candidate_encoder_seconds,
                "vision_encoder_executed": candidate_encoder_executed,
                "vision_embedding_cache_hit": (
                    native_vl.get("vision_embedding_cache_hit")
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "vision_embedding_cache_entries": (
                    native_vl.get("vision_embedding_cache_entries")
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "vision_embedding_cache_resident_bytes": (
                    native_vl.get("vision_embedding_cache_resident_bytes")
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "vision_embedding_cache_capacity_bytes": (
                    native_vl.get("vision_embedding_cache_capacity_bytes")
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "cold_vision_seconds": candidate_cold_vision_seconds,
                "logical_projection_tokens": (
                    native_vl.get("logical_projection_tokens")
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "logical_projection_plan_count": (
                    native_vl.get("logical_projection_plan_count")
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "logical_projection_workspace_bytes": (
                    native_vl.get("logical_projection_workspace_bytes")
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "logical_projection_plan_build_seconds": (
                    seconds_from_milliseconds(
                        native_vl.get(
                            "logical_projection_plan_build_wall_ms"
                        )
                    )
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "logical_projection_plan_reused": (
                    native_vl.get("logical_projection_plan_reused")
                    if isinstance(native_vl, Mapping)
                    else None
                ),
                "prefill_path_seconds": candidate_prefill_path_seconds,
                "llm_prefill_seconds": candidate_llm_prefill_seconds,
                "prefill_path_tokens_per_second": candidate_prefill_path_tps,
                "llm_prefill_tokens_per_second": candidate_llm_prefill_tps,
                "vision_tokens_per_second": candidate_vision_tps,
                "cold_vision_path_tokens_per_second": (
                    candidate_cold_vision_path_tps
                ),
                "client_decode_tokens_per_second": candidate_timings.get(
                    "decode_tokens_per_second"
                ),
                "engine_decode_tokens_executed": (
                    native.get("decode_tokens_executed")
                    if isinstance(native, Mapping)
                    else None
                ),
                "engine_decode_wall_seconds": (
                    seconds_from_milliseconds(native.get("decode_wall_ms"))
                    if isinstance(native, Mapping)
                    else None
                ),
                "engine_decode_tokens_per_second": (
                    native.get("decode_tokens_per_second")
                    if isinstance(native, Mapping)
                    else None
                ),
                "memory": candidate.get("memory"),
            },
        },
        "comparisons": comparisons,
        "diagnostic_thresholds": diagnostic_thresholds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-output-tokens", type=int)
    parser.add_argument(
        "--expected-prefix-cache-lookup",
        choices=("disabled", "miss", "exact", "append"),
    )
    parser.add_argument(
        "--expected-media-cache-mode",
        choices=("disabled", "cold", "exact"),
    )
    args = parser.parse_args()
    root = args.run_dir.resolve()
    output = (args.output or (root / "summary.json")).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite diagnostic summary: {output}")
    expectations = {
        key: value
        for key, value in {
            "output_tokens": args.expected_output_tokens,
            "prefix_cache_lookup": args.expected_prefix_cache_lookup,
            "media_cache_mode": args.expected_media_cache_mode,
        }.items()
        if value is not None
    }
    summary = build_summary(root, expectations=expectations)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "complete": summary["complete"],
                "qualified": summary["qualified"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
