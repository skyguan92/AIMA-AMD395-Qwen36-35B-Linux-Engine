"""Validation contract for fixed-vLLM rendering of serving-oracle requests."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from aima_engine.vl_reference import (
    PINNED_PACKAGES,
    canonical_json_sha256,
    verify_manifest_integrity,
)


SERVING_RENDER_SCHEMA = (
    "aima-amd395-qwen36/vl-serving-render-manifest/v1"
)
SERVING_RENDER_CASES = (
    "image_local_png",
    "video_local_mp4",
    "multi_image",
    "multi_video",
    "mixed_image_video",
)
SERVING_RENDER_SCOPE = (
    "fixed-vllm-openai-gpu-less-five-serving-oracle-render-boundary"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_PATHS = (
    "aima_engine/vl_reference.py",
    "aima_engine/vl_serving_render.py",
    "scripts/capture-vllm-vl-serving-render.py",
    "scripts/qualify-native-vl-serving.py",
)
_BINDING_PATHS = {
    "fixture_manifest": (
        "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json"
    ),
    "oracle_manifest": "benchmarks/results/vl-oracle-manifest.json",
}
_CONTRACT = {
    "content_format": "auto-resolved-string",
    "request_identity": "oracle-request-with-model-id-only-translation",
    "request_serialization": "native-serving-client-json-separators",
    "render_runtime_uses_gpu": False,
}


def _valid_component(value: Any, path: str) -> bool:
    return (
        isinstance(value, dict)
        and value.get("path") == path
        and isinstance(value.get("bytes"), int)
        and not isinstance(value.get("bytes"), bool)
        and value["bytes"] > 0
        and isinstance(value.get("sha256"), str)
        and _SHA256.fullmatch(value["sha256"]) is not None
    )


def validate_serving_render_manifest(payload: Mapping[str, Any]) -> list[str]:
    """Validate exact prompts for the five frozen native-serving requests."""

    errors = verify_manifest_integrity(payload)
    if payload.get("schema") != SERVING_RENDER_SCHEMA:
        errors.append(f"serving render schema must be {SERVING_RENDER_SCHEMA}")
    if payload.get("complete") is not True:
        errors.append("serving render manifest is not complete")
    if payload.get("qualified") is not True:
        errors.append("serving render manifest is not qualified")
    if payload.get("scope") != SERVING_RENDER_SCOPE:
        errors.append("serving render scope changed")

    host = payload.get("host")
    if not isinstance(host, dict) or host.get("label") != "amd395":
        errors.append("serving render host is not the frozen amd395 target")
    if not isinstance(host, dict) or not isinstance(host.get("hostname"), str):
        errors.append("serving render hostname is missing")

    source = payload.get("source")
    if not isinstance(source, dict):
        errors.append("serving render source identity is missing")
    else:
        if not isinstance(source.get("commit"), str) or not _GIT_COMMIT.fullmatch(
            source.get("commit", "")
        ):
            errors.append("serving render source commit is invalid")
        if source.get("dirty") is not False:
            errors.append("serving render source must be clean")
        if not isinstance(source.get("status_sha256"), str) or not _SHA256.fullmatch(
            source.get("status_sha256", "")
        ):
            errors.append("serving render source status hash is invalid")
        source_files = source.get("files")
        paths = (
            tuple(
                item.get("path") if isinstance(item, dict) else None
                for item in source_files
            )
            if isinstance(source_files, list)
            else ()
        )
        if paths != _SOURCE_PATHS:
            errors.append("serving render source file set changed")
        if isinstance(source_files, list):
            for item in source_files:
                path = item.get("path") if isinstance(item, dict) else ""
                if not _valid_component(item, path):
                    errors.append(
                        f"serving render source component is invalid: {path}"
                    )

    bindings = payload.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(_BINDING_PATHS):
        errors.append("serving render binding set changed")
    else:
        for name, path in _BINDING_PATHS.items():
            if not _valid_component(bindings.get(name), path):
                errors.append(f"serving render binding is invalid: {name}")

    runtime = payload.get("runtime")
    version = runtime.get("vllm") if isinstance(runtime, dict) else None
    expected_version = PINNED_PACKAGES["vllm"]
    if not isinstance(version, str) or not (
        version == expected_version or version.startswith(expected_version + ".")
    ):
        errors.append(f"serving render vLLM pin mismatch: {version!r}")
    endpoint = runtime.get("endpoint") if isinstance(runtime, dict) else None
    if (
        not isinstance(endpoint, dict)
        or endpoint.get("scheme") != "http"
        or endpoint.get("host") != "127.0.0.1"
        or not isinstance(endpoint.get("port"), int)
        or isinstance(endpoint.get("port"), bool)
        or not 1 <= endpoint["port"] <= 65_535
    ):
        errors.append("serving render endpoint is not explicit loopback HTTP")
    if payload.get("contract") != _CONTRACT:
        errors.append("serving render contract changed")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        return errors + ["serving render cases must be an array"]
    case_ids = [
        item.get("case_id") if isinstance(item, dict) else None
        for item in cases
    ]
    if tuple(case_ids) != SERVING_RENDER_CASES:
        errors.append("serving render case order or membership changed")
    private_prompt_matches: list[bool] = []
    for case in cases:
        if not isinstance(case, dict):
            errors.append("serving render case must be an object")
            continue
        case_id = case.get("case_id")
        token_ids = case.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids or not all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and 0 <= item < 248_320
            for item in token_ids
        ):
            errors.append(f"serving render token IDs are malformed: {case_id}")
            continue
        if case.get("prompt_tokens") != len(token_ids):
            errors.append(f"serving render prompt count changed: {case_id}")
        if case.get("prompt_token_ids_sha256") != canonical_json_sha256(
            token_ids
        ):
            errors.append(f"serving render prompt digest mismatch: {case_id}")
        for field in (
            "oracle_request_sha256",
            "render_transport_request_sha256",
            "private_prompt_token_ids_sha256",
        ):
            value = case.get(field)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                errors.append(f"serving render digest is invalid: {case_id}/{field}")
        private_tokens = case.get("private_prompt_tokens")
        if (
            not isinstance(private_tokens, int)
            or isinstance(private_tokens, bool)
            or private_tokens <= 0
        ):
            errors.append(f"serving private prompt count is invalid: {case_id}")
        matches_private = (
            private_tokens == len(token_ids)
            and case.get("private_prompt_token_ids_sha256")
            == case.get("prompt_token_ids_sha256")
        )
        private_prompt_matches.append(matches_private)
        if case.get("private_prompt_matches_real_http") is not matches_private:
            errors.append(f"serving private prompt comparison drifted: {case_id}")
        if case.get("max_tokens") != 8:
            errors.append(f"serving render max_tokens changed: {case_id}")
        placeholders = case.get("mm_placeholders")
        if not isinstance(placeholders, dict) or not placeholders:
            errors.append(f"serving render placeholders are missing: {case_id}")
        else:
            for modality, spans in placeholders.items():
                if modality not in {"image", "video"} or not isinstance(
                    spans, list
                ):
                    errors.append(
                        f"serving render placeholders are malformed: {case_id}"
                    )
                    continue
                for span in spans:
                    offset = span.get("offset") if isinstance(span, dict) else None
                    length = span.get("length") if isinstance(span, dict) else None
                    if (
                        not isinstance(offset, int)
                        or isinstance(offset, bool)
                        or not isinstance(length, int)
                        or isinstance(length, bool)
                        or offset < 0
                        or length <= 0
                        or offset + length > len(token_ids)
                    ):
                        errors.append(
                            f"serving render placeholder range is invalid: {case_id}"
                        )

    decision = payload.get("decision")
    expected_decisions = {
        "five_serving_render_cases_5_of_5": tuple(case_ids)
        == SERVING_RENDER_CASES,
        "private_preprocessor_boundary_distinguished_5_of_5": (
            len(private_prompt_matches) == len(SERVING_RENDER_CASES)
            and not any(private_prompt_matches)
        ),
    }
    if not isinstance(decision, dict):
        errors.append("serving render decision is missing")
    else:
        for name, expected in expected_decisions.items():
            if decision.get(name) is not expected:
                errors.append(f"serving render decision is inconsistent: {name}")
        if decision.get("g1_passed") is not False or decision.get(
            "g2_passed"
        ) is not False:
            errors.append("serving render reference cannot close G1 or G2")
    if payload.get("qualified") is not all(expected_decisions.values()):
        errors.append("serving render qualification is inconsistent")
    return errors
