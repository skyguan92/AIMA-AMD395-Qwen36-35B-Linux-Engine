"""Fail-closed contract for real-HTTP-rendered VL numerical oracles."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from aima_engine.vl_oracle import (
    canonical_int_list_sha256,
    validate_oracle_manifest,
)
from aima_engine.vl_serving_render import (
    SERVING_RENDER_CASES,
    validate_serving_render_manifest,
)


HTTP_ORACLE_SCOPE = (
    "fixed-vllm-real-http-rendered-five-case-numerical-oracle"
)
HTTP_ORACLE_ROOT = "benchmarks/oracles/vl-http-v0.1.0"
HTTP_RENDER_BINDING_PATH = (
    "benchmarks/results/vl-serving-render-manifest-v0.1.0.json"
)
HTTP_ORACLE_VARIANT = {
    "content_format": "auto-resolved-string",
    "prompt_identity": "fixed-vllm-openai-http-render",
    "private_processor_prompt_reused": False,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def http_render_case_checks(
    case: Mapping[str, Any], render_case: Mapping[str, Any]
) -> dict[str, bool]:
    """Return exact prompt/request/placeholder binding checks for one case."""

    processor = case.get("processor")
    generation = case.get("generation")
    http_render = case.get("http_render")
    if not isinstance(processor, Mapping):
        processor = {}
    if not isinstance(generation, Mapping):
        generation = {}
    if not isinstance(http_render, Mapping):
        http_render = {}
    prompt_ids = processor.get("prompt_token_ids")
    render_ids = render_case.get("prompt_token_ids")
    prompt_hash = (
        canonical_int_list_sha256(prompt_ids)
        if isinstance(prompt_ids, list)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in prompt_ids)
        else None
    )

    processor_placeholders = processor.get("placeholders")
    render_placeholders = render_case.get("mm_placeholders")
    placeholder_geometry_exact = isinstance(processor_placeholders, Mapping) and (
        isinstance(render_placeholders, Mapping)
    )
    if placeholder_geometry_exact:
        processor_geometry = {
            str(modality): [
                {"offset": item.get("offset"), "length": item.get("length")}
                for item in spans
                if isinstance(item, Mapping)
            ]
            for modality, spans in processor_placeholders.items()
            if isinstance(spans, list)
        }
        render_geometry = {
            str(modality): [
                {"offset": item.get("offset"), "length": item.get("length")}
                for item in spans
                if isinstance(item, Mapping)
            ]
            for modality, spans in render_placeholders.items()
            if isinstance(spans, list)
        }
        placeholder_geometry_exact = processor_geometry == render_geometry

    return {
        "case_id_exact": case.get("case_id") == render_case.get("case_id"),
        "request_identity_exact": case.get("request_sha256")
        == render_case.get("oracle_request_sha256"),
        "prompt_token_ids_exact": isinstance(prompt_ids, list)
        and prompt_ids == render_ids,
        "prompt_token_ids_sha256_exact": prompt_hash is not None
        and prompt_hash == render_case.get("prompt_token_ids_sha256")
        and processor.get("prompt_token_ids_sha256") == prompt_hash,
        "prompt_tokens_exact": isinstance(prompt_ids, list)
        and len(prompt_ids) == render_case.get("prompt_tokens")
        and generation.get("prompt_tokens") == len(prompt_ids),
        "generation_prompt_identity_exact": generation.get(
            "prompt_token_ids_sha256"
        )
        == render_case.get("prompt_token_ids_sha256"),
        "placeholder_geometry_exact": placeholder_geometry_exact,
        "render_record_exact": http_render
        == {
            "prompt_tokens": render_case.get("prompt_tokens"),
            "prompt_token_ids_sha256": render_case.get(
                "prompt_token_ids_sha256"
            ),
            "render_transport_request_sha256": render_case.get(
                "render_transport_request_sha256"
            ),
            "private_prompt_token_ids_sha256": render_case.get(
                "private_prompt_token_ids_sha256"
            ),
            "private_prompt_matches_real_http": render_case.get(
                "private_prompt_matches_real_http"
            ),
        },
    }


def validate_http_oracle_manifest(
    manifest: Mapping[str, Any],
    *,
    render_manifest: Mapping[str, Any],
    render_manifest_sha256: str,
    oracle_root: Any | None = None,
) -> list[str]:
    """Validate an oracle captured from the exact fixed HTTP render prompts."""

    errors = validate_oracle_manifest(manifest, oracle_root=oracle_root)
    render_errors = validate_serving_render_manifest(render_manifest)
    errors.extend(f"render manifest: {error}" for error in render_errors)
    if not isinstance(render_manifest_sha256, str) or not _SHA256.fullmatch(
        render_manifest_sha256
    ):
        errors.append("render manifest SHA-256 is invalid")
    if manifest.get("scope") != HTTP_ORACLE_SCOPE:
        errors.append("HTTP oracle scope changed")
    if manifest.get("oracle_root") != HTTP_ORACLE_ROOT:
        errors.append("HTTP oracle root changed")
    if manifest.get("oracle_variant") != HTTP_ORACLE_VARIANT:
        errors.append("HTTP oracle variant changed")

    bindings = manifest.get("bindings")
    render_binding = (
        bindings.get("serving_render_manifest")
        if isinstance(bindings, Mapping)
        else None
    )
    if (
        not isinstance(render_binding, Mapping)
        or render_binding.get("path") != HTTP_RENDER_BINDING_PATH
        or render_binding.get("sha256") != render_manifest_sha256
        or not isinstance(render_binding.get("bytes"), int)
        or isinstance(render_binding.get("bytes"), bool)
        or render_binding["bytes"] <= 0
    ):
        errors.append("HTTP oracle serving-render binding is invalid")

    cases = manifest.get("cases")
    render_cases = render_manifest.get("cases")
    if not isinstance(cases, list) or not isinstance(render_cases, list):
        return errors + ["HTTP oracle cases are missing"]
    case_ids = tuple(
        item.get("case_id") if isinstance(item, Mapping) else None
        for item in cases
    )
    render_ids = tuple(
        item.get("case_id") if isinstance(item, Mapping) else None
        for item in render_cases
    )
    if case_ids != SERVING_RENDER_CASES or render_ids != SERVING_RENDER_CASES:
        errors.append("HTTP oracle case order or membership changed")

    all_checks: list[bool] = []
    for case, render_case in zip(cases, render_cases, strict=False):
        if not isinstance(case, Mapping) or not isinstance(render_case, Mapping):
            errors.append("HTTP oracle case is malformed")
            continue
        checks = http_render_case_checks(case, render_case)
        all_checks.extend(checks.values())
        for name, passed in checks.items():
            if not passed:
                errors.append(
                    f"HTTP oracle render binding failed: {case.get('case_id')}/{name}"
                )

    decision = manifest.get("decision")
    expected_exact = len(all_checks) > 0 and all(all_checks)
    if not isinstance(decision, Mapping):
        errors.append("HTTP oracle decision is missing")
    else:
        if decision.get("five_real_http_numerical_oracles_exact") is not expected_exact:
            errors.append("HTTP oracle decision is inconsistent")
        if decision.get("g2_passed") is not False:
            errors.append("HTTP oracle capture cannot close G2")
    return errors
