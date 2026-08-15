#!/usr/bin/env python3
"""Capture frozen vLLM image-IO and media error/limit behavior."""

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
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_error_limits import (  # noqa: E402
    REFERENCE_CASE_ORDER,
    REFERENCE_ERROR_CONTRACT,
    build_reference_cases,
)
from aima_engine.vl_error_media_server import (  # noqa: E402
    LARGE_IMAGE_BYTES,
    ErrorLimitMediaServer,
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
from aima_engine.vl_transport_cache import (  # noqa: E402
    error_signature,
    finish_reason,
    normalize_contract_request,
    response_content,
    usage_signature,
)


SCHEMA = "aima-amd395-qwen36/vl-error-limits-reference/v1"
MEDIA_IO_ORACLE_SCHEMA = "aima-amd395-qwen36/vl-media-io-reference/v1"
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-capability-v0.1.0"
ERROR_FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-error-v0.1.0"
REFERENCE_LAUNCH = ROOT / "benchmarks/results/vl-reference-launch.json"
REFERENCE_MANIFEST = ROOT / "benchmarks/results/vl-reference-manifest.json"
PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
RENDER_SCRIPT = ROOT / "scripts/capture-vllm-vl-api-render.py"
REFERENCE_RUNNER = ROOT / "scripts/run-vllm-vl-error-limits-reference.sh"


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
    with opener.open(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"reference endpoint returned HTTP {response.status}")
        value = json.loads(response.read())
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
    packages = reference.get("reference_runtime", {}).get("packages", {})
    vllm = packages.get("vllm", {}).get("version")
    expected = PINNED_PACKAGES["vllm"]
    if not isinstance(vllm, str) or not (
        vllm == expected or vllm.startswith(expected + ".")
    ):
        errors.append("reference runtime vLLM pin changed")
    if errors:
        raise RuntimeError("invalid frozen reference:\n- " + "\n- ".join(errors))
    return launch, reference, vllm


def error_message(response: Any) -> str | None:
    if not isinstance(response, dict) or not isinstance(response.get("error"), dict):
        return None
    message = response["error"].get("message")
    return message if isinstance(message, str) and message else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--fixture-root", type=Path, default=FIXTURE_ROOT)
    parser.add_argument(
        "--error-fixture-root", type=Path, default=ERROR_FIXTURE_ROOT
    )
    parser.add_argument("--reference-launch", type=Path, default=REFERENCE_LAUNCH)
    parser.add_argument(
        "--reference-manifest", type=Path, default=REFERENCE_MANIFEST
    )
    parser.add_argument("--media-io-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    endpoint = loopback_endpoint(args.endpoint)
    fixture_root = args.fixture_root.resolve()
    error_fixture_root = args.error_fixture_root.resolve()
    launch_path = args.reference_launch.resolve()
    reference_path = args.reference_manifest.resolve()
    media_io_oracle_path = args.media_io_oracle.resolve()
    output = args.output.resolve()
    required = (
        fixture_root / "fixtures-manifest.json",
        error_fixture_root / "fixtures-manifest.json",
        launch_path,
        reference_path,
        media_io_oracle_path,
        PROBE_SCRIPT,
        RENDER_SCRIPT,
        REFERENCE_RUNNER,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"error/limit reference inputs are missing: {missing}")
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise SystemExit("error/limit output and sidecar must not exist")
    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("error/limit reference capture requires clean source")

    launch, reference, frozen_vllm = frozen_reference_inputs(
        launch_path, reference_path
    )
    media_io_oracle = load_json_object(media_io_oracle_path)
    oracle_errors = verify_manifest_integrity(media_io_oracle)
    if media_io_oracle.get("schema") != MEDIA_IO_ORACLE_SCHEMA:
        oracle_errors.append("media-IO oracle schema changed")
    if (
        media_io_oracle.get("complete") is not True
        or media_io_oracle.get("qualified") is not True
    ):
        oracle_errors.append("media-IO oracle is incomplete")
    if media_io_oracle.get("source", {}).get("commit") != source["commit"]:
        oracle_errors.append("media-IO oracle source commit changed")
    if oracle_errors:
        raise SystemExit("invalid media-IO oracle:\n- " + "\n- ".join(oracle_errors))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    version = request_json(opener, endpoint + "/version")
    actual_vllm = version.get("version")
    expected_vllm = PINNED_PACKAGES["vllm"]
    if not isinstance(actual_vllm, str) or not (
        actual_vllm == expected_vllm
        or actual_vllm.startswith(expected_vllm + ".")
    ):
        raise SystemExit(f"reference vLLM version differs: {actual_vllm!r}")

    probe = load_module(PROBE_SCRIPT, "vl_error_limits_probe")
    render = load_module(RENDER_SCRIPT, "vl_error_limits_render")
    with ErrorLimitMediaServer() as media_server:
        fixtures = probe.Fixtures(fixture_root, media_server.http_base)
        specs = build_reference_cases(
            fixtures,
            error_fixture_root,
            "qwen36-vl-reference",
            media_server,
        )
        response_redactions = {
            str(fixture_root): "${AIMA_VL_FIXTURE_ROOT}",
            str(error_fixture_root): "${AIMA_VL_ERROR_FIXTURE_ROOT}",
            media_server.http_base: "${AIMA_VL_ERROR_HTTP}",
            media_server.unreachable_base: "${AIMA_VL_UNREACHABLE_HTTP}",
        }
        cases: list[dict[str, Any]] = []
        for spec in specs:
            print(f"CASE {spec['case_id']}", flush=True)
            transport_bytes = probe.canonical_bytes(spec["payload"])
            result = probe.execute_case(
                endpoint,
                timeout=args.timeout_seconds,
                response_redactions=response_redactions,
                **spec,
            )
            transport_request_sha256 = result.pop("request_sha256")
            result["request"] = normalize_contract_request(result["request"])
            result["request_sha256"] = canonical_json_sha256(result["request"])
            result["transport_request_sha256"] = transport_request_sha256
            if spec["expected_accept"]:
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
                    raise RuntimeError(f"render omitted features: {spec['case_id']}")
                result["render"] = {
                    "prompt_tokens": len(token_ids),
                    "prompt_token_ids": token_ids,
                    "prompt_token_ids_sha256": canonical_json_sha256(token_ids),
                    "mm_placeholders": render.normalized_placeholders(
                        features.get("mm_placeholders"), token_ids
                    ),
                    "max_tokens": sampling.get("max_tokens"),
                }
                usage = usage_signature(result["response"])
                checks = {
                    "http_200": result["status_code"] == 200,
                    "surface_accepted": result["passed"] is True,
                    "render_prompt_nonempty": len(token_ids) > 0,
                    "render_max_tokens_exact": sampling.get("max_tokens") == 1,
                    "finish_reason_length": finish_reason(result["response"])
                    == "length",
                    "generated_content_nonempty": bool(
                        response_content(result["response"])
                    ),
                    "usage_exact": usage == (len(token_ids), 1, len(token_ids) + 1),
                }
            else:
                result["render"] = None
                signature = error_signature(result["response"])
                expected_error = REFERENCE_ERROR_CONTRACT[spec["case_id"]]
                checks = {
                    "reference_status_exact": result["status_code"]
                    == expected_error[0],
                    "surface_rejected": result["passed"] is True,
                    "reference_error_signature_exact": signature
                    == expected_error[1:3],
                    "error_message_compatible": expected_error[0] == 500
                    or error_message(result["response"]) is not None,
                }
            result["qualification_checks"] = checks
            result["qualified"] = all(checks.values())
            cases.append(result)
            print(
                json.dumps(
                    {
                        "case_id": result["case_id"],
                        "event": "reference_case_complete",
                        "qualified": result["qualified"],
                        "status_code": result["status_code"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        media_statistics = media_server.statistics

    by_id = {case["case_id"]: case for case in cases}
    success_hash = lambda case_id: by_id[case_id]["render"][  # noqa: E731
        "prompt_token_ids_sha256"
    ]
    error_ids = REFERENCE_CASE_ORDER[5:]
    reference_checks = {
        "case_order_exact": tuple(by_id) == REFERENCE_CASE_ORDER,
        "all_cases_qualified": all(case["qualified"] for case in cases),
        "rgba_requests_distinct": by_id["rgba_default_white"]
        ["transport_request_sha256"]
        != by_id["rgba_background_red"]["transport_request_sha256"],
        "rgba_prompt_shape_stable": success_hash("rgba_default_white")
        == success_hash("rgba_background_red"),
        "empty_video_mapping_preserves_prompt": success_hash(
            "video_sampling_default"
        )
        == success_hash("video_sampling_empty_mapping"),
        "long_duration_video_accepted": by_id["video_long_duration"][
            "status_code"
        ]
        == 200,
        "five_error_contracts_exact": all(
            (
                by_id[case_id]["status_code"],
                *(error_signature(by_id[case_id]["response"]) or (None, None)),
            )
            == REFERENCE_ERROR_CONTRACT[case_id][:3]
            for case_id in error_ids
        ),
        "oversize_body_fully_fetched": media_statistics["bytes_sent"][
            "large_image"
        ]
        == LARGE_IMAGE_BYTES,
        "timeout_retry_path_exercised": media_statistics["requests"][
            "slow_image"
        ]
        >= 3
        and by_id["timeout_image_remote"]["timings"]["total_seconds"] >= 200.0,
    }
    complete = all(reference_checks.values())
    source_files = tuple(
        ROOT / path
        for path in (
            "aima_engine/vl_error_limits.py",
            "aima_engine/vl_error_media_server.py",
            "aima_engine/vl_reference.py",
            "aima_engine/vl_transport_cache.py",
            "scripts/probe-vllm-vl-api-capabilities.py",
            "scripts/capture-vllm-vl-api-render.py",
            "scripts/run-vllm-vl-error-limits-reference.sh",
            "scripts/capture-vllm-vl-error-limits.py",
        )
    )
    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "captured_at": utc_now(),
            "complete": complete,
            "qualified": complete,
            "scope": "fixed-vllm-image-io-and-media-error-limit-reference",
            "host": {"label": "amd395", "hostname": socket.gethostname()},
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in source_files
                ],
            },
            "runtime": {
                "vllm": actual_vllm,
                "endpoint": {
                    "scheme": "http",
                    "host": "127.0.0.1",
                    "port": urllib.parse.urlparse(endpoint).port,
                },
                "media_server": media_statistics,
            },
            "bindings": {
                "fixture_manifest": file_component(
                    fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
                ),
                "error_fixture_manifest": file_component(
                    error_fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-error-v0.1.0/fixtures-manifest.json",
                ),
                "reference_launch": file_component(
                    launch_path, "benchmarks/results/vl-reference-launch.json"
                ),
                "reference_manifest": file_component(
                    reference_path, "benchmarks/results/vl-reference-manifest.json"
                ),
                "media_io_oracle": file_component(
                    media_io_oracle_path,
                    "benchmarks/results/vl-media-io-reference-v0.1.0.json",
                ),
            },
            "launch": launch,
            "reference_identity": {
                "model": reference["model"]["repository"],
                "revision": reference["model"]["revision"],
                "runtime_version": frozen_vllm,
            },
            "cases": cases,
            "qualification_checks": reference_checks,
            "decision": {
                "ten_reference_cases_frozen": complete,
                "rgba_request_surface_frozen": reference_checks[
                    "rgba_requests_distinct"
                ]
                and reference_checks["rgba_prompt_shape_stable"]
                and media_io_oracle["decision"]["default_white_exact"]
                and media_io_oracle["decision"]["request_red_exact"],
                "empty_video_mapping_semantics_frozen": reference_checks[
                    "empty_video_mapping_preserves_prompt"
                ],
                "long_duration_semantics_frozen": reference_checks[
                    "long_duration_video_accepted"
                ],
                "error_limit_categories_frozen": reference_checks[
                    "five_error_contracts_exact"
                ],
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
