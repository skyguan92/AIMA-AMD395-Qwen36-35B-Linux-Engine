#!/usr/bin/env python3
"""Capture fixed-vLLM HTTP render prompts for five serving-oracle requests."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import ModuleType
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_oracle import validate_oracle_manifest  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    PINNED_PACKAGES,
    atomic_json,
    canonical_json_sha256,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_bytes,
)
from aima_engine.vl_serving_render import (  # noqa: E402
    SERVING_RENDER_CASES,
    SERVING_RENDER_SCHEMA,
    SERVING_RENDER_SCOPE,
    validate_serving_render_manifest,
)


ORACLE_MANIFEST = ROOT / "benchmarks/results/vl-oracle-manifest.json"
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-capability-v0.1.0"
QUALIFIER_SCRIPT = ROOT / "scripts/qualify-native-vl-serving.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_qualifier_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aima_vl_serving_qualifier_for_render", QUALIFIER_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the native VL serving qualifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def loopback_endpoint(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("render endpoint must be an explicit loopback HTTP port")
    return value.rstrip("/")


def request_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    payload_bytes: bytes | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"} if payload_bytes else {},
        method="POST" if payload_bytes else "GET",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"render endpoint returned HTTP {error.code}: "
            f"{error.read().decode('utf-8', errors='replace')}"
        ) from error
    if status != 200:
        raise RuntimeError(f"render endpoint returned HTTP {status}")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError("render endpoint response must be an object")
    return value


def normalized_placeholders(
    value: Any, token_ids: list[int]
) -> dict[str, list[dict[str, int | str]]]:
    if not isinstance(value, dict):
        raise RuntimeError("render features.mm_placeholders must be an object")
    result: dict[str, list[dict[str, int | str]]] = {}
    for modality, ranges in value.items():
        if modality not in {"image", "video"} or not isinstance(ranges, list):
            raise RuntimeError("render placeholder collection is malformed")
        result[modality] = []
        for item in ranges:
            if not isinstance(item, dict):
                raise RuntimeError("render placeholder range is malformed")
            offset = item.get("offset")
            length = item.get("length")
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or not isinstance(length, int)
                or isinstance(length, bool)
                or offset < 0
                or length <= 0
                or offset + length > len(token_ids)
            ):
                raise RuntimeError("render placeholder range is malformed")
            span = token_ids[offset : offset + length]
            pad_token = 248056 if modality == "image" else 248057
            result[modality].append(
                {
                    "offset": offset,
                    "length": length,
                    "pad_token_count": span.count(pad_token),
                    "token_ids_sha256": canonical_json_sha256(span),
                }
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--oracle-manifest", type=Path, default=ORACLE_MANIFEST
    )
    parser.add_argument("--fixture-root", type=Path, default=FIXTURE_ROOT)
    args = parser.parse_args()

    endpoint = loopback_endpoint(args.endpoint)
    endpoint_port = urllib.parse.urlparse(endpoint).port
    output = args.output.resolve()
    oracle_path = args.oracle_manifest.resolve()
    fixture_root = args.fixture_root.resolve()
    for path in (
        oracle_path,
        fixture_root / "fixtures-manifest.json",
        QUALIFIER_SCRIPT,
    ):
        if not path.exists():
            raise SystemExit(f"serving render capture input is missing: {path}")
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise SystemExit("serving render output and sidecar must not exist")

    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("serving render capture requires clean source")
    oracle = load_json_object(oracle_path)
    errors = validate_oracle_manifest(oracle)
    if errors:
        raise SystemExit("invalid VL oracle manifest:\n- " + "\n- ".join(errors))
    cases_by_id = {
        case["case_id"]: case
        for case in oracle["cases"]
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    if tuple(case["case_id"] for case in oracle["cases"]) != (
        SERVING_RENDER_CASES
    ):
        raise SystemExit("serving oracle case order changed")

    qualifier = load_qualifier_module()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    version = request_json(opener, endpoint + "/version").get("version")
    expected_version = PINNED_PACKAGES["vllm"]
    if not isinstance(version, str) or not (
        version == expected_version or version.startswith(expected_version + ".")
    ):
        raise SystemExit(f"render vLLM pin mismatch: {version!r}")

    cases: list[dict[str, Any]] = []
    for case_id in SERVING_RENDER_CASES:
        oracle_case = cases_by_id[case_id]
        render_request = qualifier.build_request(oracle_case, fixture_root)
        render_request["model"] = "qwen36-vl-reference"
        transport_body = qualifier.request_bytes(render_request)
        rendered = request_json(
            opener,
            endpoint + "/v1/chat/completions/render",
            payload_bytes=transport_body,
        )
        token_ids = rendered.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise RuntimeError(f"render returned no prompt tokens: {case_id}")
        features = rendered.get("features")
        if not isinstance(features, dict):
            raise RuntimeError(f"render returned no media features: {case_id}")
        sampling = rendered.get("sampling_params")
        if not isinstance(sampling, dict) or sampling.get("max_tokens") != 8:
            raise RuntimeError(f"render sampling contract changed: {case_id}")
        processor = oracle_case["processor"]
        private_token_ids = processor["prompt_token_ids"]
        prompt_hash = canonical_json_sha256(token_ids)
        private_hash = processor["prompt_token_ids_sha256"]
        cases.append(
            {
                "case_id": case_id,
                "oracle_request_sha256": canonical_json_sha256(
                    oracle_case["request"]
                ),
                "render_transport_request_sha256": sha256_bytes(transport_body),
                "prompt_tokens": len(token_ids),
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": prompt_hash,
                "mm_placeholders": normalized_placeholders(
                    features.get("mm_placeholders"), token_ids
                ),
                "private_prompt_tokens": len(private_token_ids),
                "private_prompt_token_ids_sha256": private_hash,
                "private_prompt_matches_real_http": (
                    len(private_token_ids) == len(token_ids)
                    and private_hash == prompt_hash
                ),
                "max_tokens": sampling["max_tokens"],
            }
        )
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "event": "serving_render_complete",
                    "prompt_tokens": len(token_ids),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    decisions = {
        "five_serving_render_cases_5_of_5": tuple(
            case["case_id"] for case in cases
        )
        == SERVING_RENDER_CASES,
        "private_preprocessor_boundary_distinguished_5_of_5": all(
            case["private_prompt_matches_real_http"] is False for case in cases
        ),
    }
    qualified = all(decisions.values())
    payload = seal_manifest(
        {
            "schema": SERVING_RENDER_SCHEMA,
            "captured_at": utc_now(),
            "complete": len(cases) == len(SERVING_RENDER_CASES),
            "qualified": qualified,
            "scope": SERVING_RENDER_SCOPE,
            "host": {"label": "amd395", "hostname": socket.gethostname()},
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in (
                        ROOT / "aima_engine/vl_reference.py",
                        ROOT / "aima_engine/vl_serving_render.py",
                        Path(__file__).resolve(),
                        QUALIFIER_SCRIPT,
                    )
                ],
            },
            "runtime": {
                "vllm": version,
                "endpoint": {
                    "scheme": "http",
                    "host": "127.0.0.1",
                    "port": endpoint_port,
                },
            },
            "bindings": {
                "fixture_manifest": file_component(
                    fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
                ),
                "oracle_manifest": file_component(
                    oracle_path,
                    "benchmarks/results/vl-oracle-manifest.json",
                ),
            },
            "contract": {
                "content_format": "auto-resolved-string",
                "request_identity": "oracle-request-with-model-id-only-translation",
                "request_serialization": "native-serving-client-json-separators",
                "render_runtime_uses_gpu": False,
            },
            "cases": cases,
            "decision": {
                **decisions,
                "g1_passed": False,
                "g2_passed": False,
            },
        }
    )
    validation_errors = validate_serving_render_manifest(payload)
    if validation_errors:
        raise RuntimeError(
            "serving render manifest failed validation:\n- "
            + "\n- ".join(validation_errors)
        )
    digest = atomic_json(output, payload)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "output": str(output),
                "qualified": qualified,
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
