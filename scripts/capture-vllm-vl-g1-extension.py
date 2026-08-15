#!/usr/bin/env python3
"""Capture the frozen vLLM reference for the five VL G1 extension cases."""

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

from aima_engine.vl_g1_extension import (  # noqa: E402
    CASE_ORDER,
    build_cases,
    finish_reason,
    normalize_contract_request,
    response_content,
    usage_signature,
)
from aima_engine.vl_reference import (  # noqa: E402
    PINNED_PACKAGES,
    REFERENCE_SCHEMA,
    atomic_json,
    canonical_json_sha256,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    validate_launch_config,
    verify_manifest_integrity,
)


SCHEMA = "aima-amd395-qwen36/vl-g1-mixed-conversation-reference/v1"
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-capability-v0.1.0"
REFERENCE_LAUNCH = ROOT / "benchmarks/results/vl-reference-launch.json"
REFERENCE_MANIFEST = ROOT / "benchmarks/results/vl-reference-manifest.json"
PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
RENDER_SCRIPT = ROOT / "scripts/capture-vllm-vl-api-render.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load frozen module: {path.name}")
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
        raise ValueError("reference endpoint must be an explicit loopback port")
    return value.rstrip("/")


def request_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    payload_bytes: bytes | None = None,
    timeout: float = 300.0,
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
            f"reference endpoint returned HTTP {error.code}: "
            f"{error.read().decode('utf-8', errors='replace')}"
        ) from error
    if status != 200:
        raise RuntimeError(f"reference endpoint returned HTTP {status}")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError("reference endpoint response is not an object")
    return value


def frozen_reference_inputs(
    launch_path: Path, reference_path: Path
) -> tuple[dict[str, Any], dict[str, Any], str]:
    launch = load_json_object(launch_path)
    errors = validate_launch_config(launch)
    reference = load_json_object(reference_path)
    errors.extend(verify_manifest_integrity(reference))
    if reference.get("schema") != REFERENCE_SCHEMA:
        errors.append("reference manifest schema changed")
    if reference.get("complete") is not True:
        errors.append("reference manifest is incomplete")
    if reference.get("qualified_for_oracle_capture") is not True:
        errors.append("reference manifest is not qualified")
    if reference.get("launch") != launch:
        errors.append("reference manifest launch payload changed")
    runtime = reference.get("reference_runtime")
    packages = runtime.get("packages") if isinstance(runtime, dict) else None
    vllm = (
        packages.get("vllm", {}).get("version")
        if isinstance(packages, dict)
        and isinstance(packages.get("vllm"), dict)
        else None
    )
    expected = PINNED_PACKAGES["vllm"]
    if not isinstance(vllm, str) or not (
        vllm == expected or vllm.startswith(expected + ".")
    ):
        errors.append("reference runtime vLLM pin changed")
    if errors:
        raise RuntimeError("invalid frozen reference:\n- " + "\n- ".join(errors))
    return launch, reference, vllm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--fixture-root", type=Path, default=FIXTURE_ROOT)
    parser.add_argument("--reference-launch", type=Path, default=REFERENCE_LAUNCH)
    parser.add_argument(
        "--reference-manifest", type=Path, default=REFERENCE_MANIFEST
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    endpoint = loopback_endpoint(args.endpoint)
    fixture_root = args.fixture_root.resolve()
    launch_path = args.reference_launch.resolve()
    reference_path = args.reference_manifest.resolve()
    output = args.output.resolve()
    required = (
        fixture_root / "fixtures-manifest.json",
        launch_path,
        reference_path,
        PROBE_SCRIPT,
        RENDER_SCRIPT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"G1 extension capture inputs are missing: {missing}")
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise SystemExit("G1 extension output and sidecar must not exist")
    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("G1 extension reference capture requires clean source")

    launch, reference, frozen_vllm = frozen_reference_inputs(
        launch_path, reference_path
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    version = request_json(opener, endpoint + "/version")
    actual_vllm = version.get("version")
    expected_vllm = PINNED_PACKAGES["vllm"]
    if not isinstance(actual_vllm, str) or not (
        actual_vllm == expected_vllm
        or actual_vllm.startswith(expected_vllm + ".")
    ):
        raise SystemExit(
            f"reference vLLM version differs: {actual_vllm!r}"
        )

    probe = load_module(PROBE_SCRIPT, "vl_g1_reference_probe")
    render = load_module(RENDER_SCRIPT, "vl_g1_reference_render")
    fixtures = probe.Fixtures(fixture_root, "http://127.0.0.1:1")
    specs = build_cases(fixtures, "qwen36-vl-reference")
    cases: list[dict[str, Any]] = []
    for spec in specs:
        print(f"CASE {spec['case_id']}", flush=True)
        transport_bytes = probe.canonical_bytes(spec["payload"])
        result = probe.execute_case(
            endpoint,
            timeout=args.timeout_seconds,
            response_redactions={
                str(fixture_root): "${AIMA_VL_FIXTURE_ROOT}"
            },
            **spec,
        )
        rendered = request_json(
            opener,
            endpoint + "/v1/chat/completions/render",
            payload_bytes=transport_bytes,
            timeout=args.timeout_seconds,
        )
        token_ids = rendered.get("token_ids")
        features = rendered.get("features")
        sampling = rendered.get("sampling_params")
        if not isinstance(token_ids, list) or not token_ids:
            raise RuntimeError(f"render returned no tokens: {spec['case_id']}")
        if not isinstance(features, dict) or not isinstance(sampling, dict):
            raise RuntimeError(
                f"render returned no features or sampling params: {spec['case_id']}"
            )
        placeholders = render.normalized_placeholders(
            features.get("mm_placeholders"), token_ids
        )
        normalized_request = normalize_contract_request(result["request"])
        transport_request_sha256 = result.pop("request_sha256")
        result["request"] = normalized_request
        result["request_sha256"] = canonical_json_sha256(normalized_request)
        result["transport_request_sha256"] = transport_request_sha256
        result["render"] = {
            "prompt_tokens": len(token_ids),
            "prompt_token_ids": token_ids,
            "prompt_token_ids_sha256": canonical_json_sha256(token_ids),
            "mm_placeholders": placeholders,
            "max_tokens": sampling.get("max_tokens"),
        }
        stream = bool(spec["payload"].get("stream"))
        reference_usage = usage_signature(result["response"])
        checks = {
            "http_200": result["status_code"] == 200,
            "surface_accepted": result["passed"] is True,
            "render_prompt_nonempty": len(token_ids) > 0,
            "render_max_tokens_exact": sampling.get("max_tokens")
            == spec["payload"]["max_tokens"],
            "finish_reason_length": finish_reason(result["response"])
            == "length",
            "generated_content_nonempty": bool(
                response_content(result["response"])
            ),
            "usage_or_stream_contract": (
                reference_usage is None
                if stream
                else reference_usage
                == (
                    len(token_ids),
                    spec["payload"]["max_tokens"],
                    len(token_ids) + spec["payload"]["max_tokens"],
                )
            ),
            "stream_complete": (
                result["response"].get("done") is True
                and result["response"].get("event_count", 0) > 0
                if stream
                else True
            ),
        }
        result["qualification_checks"] = checks
        result["qualified"] = all(checks.values())
        cases.append(result)
        print(
            json.dumps(
                {
                    "case_id": result["case_id"],
                    "event": "reference_case_complete",
                    "prompt_tokens": len(token_ids),
                    "qualified": result["qualified"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    complete = (
        tuple(case["case_id"] for case in cases) == CASE_ORDER
        and all(case["qualified"] for case in cases)
    )
    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "captured_at": utc_now(),
            "complete": complete,
            "qualified": complete,
            "scope": "fixed-vllm-five-case-mixed-conversation-stream-extension",
            "host": {"label": "amd395", "hostname": socket.gethostname()},
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in (
                        ROOT / "aima_engine/vl_capability.py",
                        ROOT / "aima_engine/vl_g1_extension.py",
                        ROOT / "aima_engine/vl_reference.py",
                        PROBE_SCRIPT,
                        RENDER_SCRIPT,
                        Path(__file__).resolve(),
                    )
                ],
            },
            "runtime": {
                "vllm": actual_vllm,
                "endpoint": {
                    "scheme": "http",
                    "host": "127.0.0.1",
                    "port": urllib.parse.urlparse(endpoint).port,
                },
            },
            "bindings": {
                "fixture_manifest": file_component(
                    fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
                ),
                "reference_launch": file_component(
                    launch_path,
                    "benchmarks/results/vl-reference-launch.json",
                ),
                "reference_manifest": file_component(
                    reference_path,
                    "benchmarks/results/vl-reference-manifest.json",
                ),
            },
            "launch": launch,
            "reference_identity": {
                "model": reference["model"]["repository"],
                "revision": reference["model"]["revision"],
                "runtime_version": frozen_vllm,
            },
            "cases": cases,
            "decision": {
                "five_reference_cases_accepted": complete,
                "five_render_prompt_vectors_frozen": complete,
                "mixed_multi_item_orders_frozen": all(
                    case["qualified"] for case in cases[:2]
                ),
                "video_and_mixed_history_frozen": all(
                    case["qualified"] for case in cases[2:4]
                ),
                "mixed_sse_frozen": cases[4]["qualified"],
                "g1_passed": False,
                "g2_passed": False,
                "g3_passed": False,
                "g4_passed": False,
                "g5_passed": False,
            },
        }
    )
    digest = atomic_json(output, payload)
    print(
        json.dumps(
            {
                "complete": complete,
                "output": str(output),
                "qualified": complete,
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
