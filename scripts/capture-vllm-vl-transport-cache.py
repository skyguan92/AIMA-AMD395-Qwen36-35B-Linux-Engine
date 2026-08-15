#!/usr/bin/env python3
"""Capture the frozen vLLM HTTPS, sampling and cache-identity reference."""

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

from aima_engine.vl_local_media_server import LocalMediaServers  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    PINNED_PACKAGES,
    REFERENCE_SCHEMA,
    atomic_json,
    canonical_json_sha256,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
    validate_launch_config,
    verify_manifest_integrity,
)
from aima_engine.vl_transport_cache import (  # noqa: E402
    REFERENCE_CASE_ORDER,
    build_reference_cases,
    error_signature,
    finish_reason,
    normalize_contract_request,
    response_content,
    usage_signature,
)


SCHEMA = "aima-amd395-qwen36/vl-transport-cache-reference/v1"
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-capability-v0.1.0"
REFERENCE_LAUNCH = ROOT / "benchmarks/results/vl-reference-launch.json"
REFERENCE_MANIFEST = ROOT / "benchmarks/results/vl-reference-manifest.json"
PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
RENDER_SCRIPT = ROOT / "scripts/capture-vllm-vl-api-render.py"
REFERENCE_RUNNER = ROOT / "scripts/run-vllm-vl-transport-cache-reference.sh"
TLS_GENERATOR = ROOT / "scripts/generate-vl-test-tls-material.sh"


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
            raise RuntimeError(
                f"reference endpoint returned HTTP {response.status}"
            )
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--fixture-root", type=Path, default=FIXTURE_ROOT)
    parser.add_argument("--tls-certificate", type=Path, required=True)
    parser.add_argument("--tls-private-key", type=Path, required=True)
    parser.add_argument("--reference-launch", type=Path, default=REFERENCE_LAUNCH)
    parser.add_argument(
        "--reference-manifest", type=Path, default=REFERENCE_MANIFEST
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    endpoint = loopback_endpoint(args.endpoint)
    fixture_root = args.fixture_root.resolve()
    certificate = args.tls_certificate.resolve()
    private_key = args.tls_private_key.resolve()
    launch_path = args.reference_launch.resolve()
    reference_path = args.reference_manifest.resolve()
    output = args.output.resolve()
    required = (
        fixture_root / "fixtures-manifest.json",
        certificate,
        private_key,
        launch_path,
        reference_path,
        PROBE_SCRIPT,
        RENDER_SCRIPT,
        REFERENCE_RUNNER,
        TLS_GENERATOR,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"transport/cache reference inputs are missing: {missing}")
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise SystemExit("transport/cache output and sidecar must not exist")
    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("transport/cache reference capture requires clean source")

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
        raise SystemExit(f"reference vLLM version differs: {actual_vllm!r}")

    probe = load_module(PROBE_SCRIPT, "vl_transport_cache_probe")
    render = load_module(RENDER_SCRIPT, "vl_transport_cache_render")
    with LocalMediaServers(
        fixture_root, certificate, private_key
    ) as media_servers:
        fixtures = probe.Fixtures(fixture_root, media_servers.http_base)
        specs = build_reference_cases(
            fixtures,
            "qwen36-vl-reference",
            media_servers.http_base,
            media_servers.https_base,
        )
        cases: list[dict[str, Any]] = []
        for spec in specs:
            media_servers.set_mode(spec["mutable_mode"])
            print(f"CASE {spec['case_id']}", flush=True)
            transport_bytes = probe.canonical_bytes(spec["payload"])
            result = probe.execute_case(
                endpoint,
                timeout=args.timeout_seconds,
                response_redactions={str(fixture_root): "${AIMA_VL_FIXTURE_ROOT}"},
                **{
                    key: value
                    for key, value in spec.items()
                    if key != "mutable_mode"
                },
            )
            normalized_request = normalize_contract_request(result["request"])
            transport_request_sha256 = result.pop("request_sha256")
            result["request"] = normalized_request
            result["request_sha256"] = canonical_json_sha256(normalized_request)
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
                    raise RuntimeError(
                        f"render returned no tokens: {spec['case_id']}"
                    )
                if not isinstance(features, dict) or not isinstance(
                    sampling, dict
                ):
                    raise RuntimeError(
                        f"render omitted features: {spec['case_id']}"
                    )
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
                    "render_max_tokens_exact": sampling.get("max_tokens")
                    == spec["payload"]["max_tokens"],
                    "finish_reason_length": finish_reason(result["response"])
                    == "length",
                    "generated_content_nonempty": bool(
                        response_content(result["response"])
                    ),
                    "usage_exact": usage
                    == (
                        len(token_ids),
                        spec["payload"]["max_tokens"],
                        len(token_ids) + spec["payload"]["max_tokens"],
                    ),
                }
            else:
                result["render"] = None
                checks = {
                    "http_400": result["status_code"] == 400,
                    "surface_rejected": result["passed"] is True,
                    "structured_error": error_signature(result["response"])
                    is not None,
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
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        by_id = {case["case_id"]: case for case in cases}
        sampling_hashes = {
            by_id[case_id]["render"]["prompt_token_ids_sha256"]
            for case_id in (
                "video_sampling_default",
                "video_sampling_fps_1",
                "video_sampling_num_frames_6",
            )
        }
        reference_checks = {
            "case_order_exact": tuple(by_id) == REFERENCE_CASE_ORDER,
            "all_cases_qualified": all(case["qualified"] for case in cases),
            "https_origin_used": media_servers.request_counts["https"] >= 2,
            "sampling_changes_prompt": len(sampling_hashes) == 3,
            "same_url_a_b_wire_request_exact": by_id["video_content_a"]
            ["transport_request_sha256"]
            == by_id["video_content_b"]["transport_request_sha256"],
            "same_url_a_b_prompt_distinct": by_id["video_content_a"]["render"]
            ["prompt_token_ids_sha256"]
            != by_id["video_content_b"]["render"]["prompt_token_ids_sha256"],
            "mixed_order_prompt_distinct": by_id["mixed_image_video"]["render"]
            ["prompt_token_ids_sha256"]
            != by_id["mixed_video_image"]["render"]["prompt_token_ids_sha256"],
            "mixed_mutation_prompt_distinct": by_id["mixed_image_video"]
            ["render"]["prompt_token_ids_sha256"]
            != by_id["mixed_mutated_image_video"]["render"]
            ["prompt_token_ids_sha256"],
        }
        media_server_record = {
            "http": {
                "scheme": "http",
                "host": "127.0.0.1",
                "port": urllib.parse.urlparse(media_servers.http_base).port,
            },
            "https": {
                "scheme": "https",
                "host": "127.0.0.1",
                "port": urllib.parse.urlparse(media_servers.https_base).port,
            },
            "request_counts": media_servers.request_counts,
        }

    complete = all(reference_checks.values())
    source_files = tuple(
        ROOT / path
        for path in (
            "aima_engine/vl_local_media_server.py",
            "aima_engine/vl_transport_cache.py",
            "aima_engine/vl_reference.py",
            "scripts/probe-vllm-vl-api-capabilities.py",
            "scripts/capture-vllm-vl-api-render.py",
            "scripts/generate-vl-test-tls-material.sh",
            "scripts/run-vllm-vl-transport-cache-reference.sh",
            "scripts/capture-vllm-vl-transport-cache.py",
        )
    )
    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "captured_at": utc_now(),
            "complete": complete,
            "qualified": complete,
            "scope": "fixed-vllm-https-video-sampling-content-and-order-reference",
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
                "media_servers": media_server_record,
                "additive_environment": {
                    "REQUESTS_CA_BUNDLE": "${AIMA_VL_TLS_CA_BUNDLE}",
                    "SSL_CERT_FILE": "${AIMA_VL_TLS_CA_BUNDLE}",
                },
                "test_ca": {
                    "bytes": certificate.stat().st_size,
                    "sha256": sha256_file(certificate),
                    "private_key_recorded": False,
                },
            },
            "bindings": {
                "fixture_manifest": file_component(
                    fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
                ),
                "reference_launch": file_component(
                    launch_path, "benchmarks/results/vl-reference-launch.json"
                ),
                "reference_manifest": file_component(
                    reference_path, "benchmarks/results/vl-reference-manifest.json"
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
                "verified_https_frozen": reference_checks["https_origin_used"],
                "video_sampling_overrides_frozen": reference_checks[
                    "sampling_changes_prompt"
                ],
                "same_url_content_mutation_frozen": reference_checks[
                    "same_url_a_b_prompt_distinct"
                ],
                "mixed_order_and_mutation_frozen": reference_checks[
                    "mixed_order_prompt_distinct"
                ]
                and reference_checks["mixed_mutation_prompt_distinct"],
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
