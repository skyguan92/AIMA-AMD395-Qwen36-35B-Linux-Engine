#!/usr/bin/env python3
"""Capture exact prompts from the fixed vLLM GPU-less OpenAI render path."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import http.server
import importlib.util
import json
from pathlib import Path
import socket
import sys
import threading
from types import ModuleType
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_capability import (  # noqa: E402
    API_RENDER_MEDIA_COUNTS,
    API_RENDER_SCHEMA,
    API_RENDER_TOOL_CASES,
    EXPECTED_TOOL_JSON_SCHEMA,
    REQUIRED_API_RENDER_CASES,
    validate_api_render_manifest,
    validate_capability_manifest,
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
    sha256_file,
    validate_launch_config,
    verify_manifest_integrity,
)


CAPABILITY_MANIFEST = ROOT / "benchmarks/results/vl-capability-manifest.json"
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-capability-v0.1.0"
REFERENCE_LAUNCH = ROOT / "benchmarks/results/vl-reference-launch.json"
REFERENCE_MANIFEST = ROOT / "benchmarks/results/vl-reference-manifest.json"
PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_probe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aima_vl_api_render_probe", PROBE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen VL API probe")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QuietFixtureHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *args: object) -> None:
        del args


def request_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = None
    method = "GET"
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method=method
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"render endpoint returned HTTP {error.code}: "
            f"{error.read().decode('utf-8', errors='replace')}"
        ) from error
    if status != 200:
        raise RuntimeError(f"render endpoint returned HTTP {status}")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError("render endpoint response is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("render endpoint response must be an object")
    return value


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


def normalized_placeholders(value: Any) -> dict[str, list[dict[str, int]]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError("render features.mm_placeholders must be an object")
    result: dict[str, list[dict[str, int]]] = {}
    for modality, ranges in value.items():
        if not isinstance(modality, str) or not isinstance(ranges, list):
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
            ):
                raise RuntimeError("render placeholder offset or length is malformed")
            result[modality].append(
                {"offset": offset, "length": length}
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--fixture-port", type=int, default=18127)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capability-manifest", type=Path, default=CAPABILITY_MANIFEST
    )
    parser.add_argument("--fixture-root", type=Path, default=FIXTURE_ROOT)
    parser.add_argument("--reference-launch", type=Path, default=REFERENCE_LAUNCH)
    parser.add_argument(
        "--reference-manifest", type=Path, default=REFERENCE_MANIFEST
    )
    args = parser.parse_args()

    endpoint = loopback_endpoint(args.endpoint)
    endpoint_port = urllib.parse.urlparse(endpoint).port
    if not 1 <= args.fixture_port <= 65_535:
        raise SystemExit("fixture port must be between 1 and 65535")
    if args.fixture_port == endpoint_port:
        raise SystemExit("fixture port must differ from the render endpoint port")
    output = args.output.resolve()
    capability_path = args.capability_manifest.resolve()
    fixture_root = args.fixture_root.resolve()
    reference_launch = args.reference_launch.resolve()
    reference_manifest = args.reference_manifest.resolve()
    for path in (
        capability_path,
        fixture_root / "fixtures-manifest.json",
        reference_launch,
        reference_manifest,
        PROBE_SCRIPT,
    ):
        if not path.exists():
            raise SystemExit(f"API render capture input is missing: {path}")
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise SystemExit("API render output and sidecar must not exist")

    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("API render capture requires clean source")
    capability = load_json_object(capability_path)
    errors = verify_manifest_integrity(capability)
    errors.extend(validate_capability_manifest(capability))
    if errors:
        raise SystemExit("invalid capability manifest:\n- " + "\n- ".join(errors))

    launch = load_json_object(reference_launch)
    launch_errors = validate_launch_config(launch)
    if launch_errors:
        raise SystemExit(
            "invalid reference launch:\n- " + "\n- ".join(launch_errors)
        )
    reference = load_json_object(reference_manifest)
    reference_errors = verify_manifest_integrity(reference)
    if reference.get("schema") != REFERENCE_SCHEMA:
        reference_errors.append("reference manifest schema changed")
    if reference.get("complete") is not True:
        reference_errors.append("reference manifest is incomplete")
    if reference.get("qualified_for_oracle_capture") is not True:
        reference_errors.append("reference manifest is not qualified")
    if reference.get("launch") != launch:
        reference_errors.append("reference launch payload drifted")
    capability_digest = sha256_file(capability_path)
    launch_digest = sha256_file(reference_launch)
    launch_capability = launch.get("capability_manifest")
    if (
        not isinstance(launch_capability, dict)
        or launch_capability.get("sha256") != capability_digest
    ):
        reference_errors.append("launch capability binding drifted")
    reference_capability = reference.get("capability_manifest")
    if (
        not isinstance(reference_capability, dict)
        or reference_capability.get("sha256") != capability_digest
    ):
        reference_errors.append("reference capability binding drifted")
    capture_inputs = reference.get("capture_inputs")
    launch_binding = (
        capture_inputs.get("launch_config")
        if isinstance(capture_inputs, dict)
        else None
    )
    if (
        not isinstance(launch_binding, dict)
        or launch_binding.get("sha256") != launch_digest
    ):
        reference_errors.append("reference launch file binding drifted")
    reference_runtime = reference.get("reference_runtime")
    packages = (
        reference_runtime.get("packages")
        if isinstance(reference_runtime, dict)
        else None
    )
    reference_vllm = (
        packages.get("vllm", {}).get("version")
        if isinstance(packages, dict)
        and isinstance(packages.get("vllm"), dict)
        else None
    )
    expected_vllm = PINNED_PACKAGES["vllm"]
    if not isinstance(reference_vllm, str) or not (
        reference_vllm == expected_vllm
        or reference_vllm.startswith(expected_vllm + ".")
    ):
        reference_errors.append("reference runtime vLLM pin drifted")
    if reference_errors:
        raise SystemExit(
            "invalid reference manifest:\n- " + "\n- ".join(reference_errors)
        )

    references = {
        item["case_id"]: item
        for item in capability["cases"]
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    version = request_json(opener, endpoint + "/version")
    actual_version = version.get("version")
    if not isinstance(actual_version, str) or not (
        actual_version == expected_vllm
        or actual_version.startswith(expected_vllm + ".")
    ):
        raise SystemExit(f"render vLLM pin mismatch: {actual_version!r}")

    try:
        fixture_server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", args.fixture_port),
            partial(QuietFixtureHandler, directory=str(fixture_root)),
        )
    except OSError as error:
        raise SystemExit(
            f"could not bind fixture server to 127.0.0.1:{args.fixture_port}: {error}"
        ) from error
    fixture_thread = threading.Thread(
        target=fixture_server.serve_forever,
        name="vllm-api-render-fixtures",
        daemon=True,
    )
    fixture_thread.start()
    cases: list[dict[str, Any]] = []
    try:
        probe = load_probe_module()
        fixtures = probe.Fixtures(
            fixture_root, f"http://127.0.0.1:{args.fixture_port}"
        )
        specs = probe.build_cases(fixtures, "qwen36-vl-reference")
        accepted = [item for item in specs if item["expected_accept"]]
        if tuple(item["case_id"] for item in accepted) != REQUIRED_API_RENDER_CASES:
            raise RuntimeError("frozen success-case order changed")
        for spec in accepted:
            case_id = spec["case_id"]
            rendered = request_json(
                opener,
                endpoint + "/v1/chat/completions/render",
                payload=spec["payload"],
            )
            token_ids = rendered.get("token_ids")
            if not isinstance(token_ids, list) or not token_ids:
                raise RuntimeError(f"render returned no prompt tokens: {case_id}")
            features = rendered.get("features")
            if not isinstance(features, dict) and API_RENDER_MEDIA_COUNTS[case_id]:
                raise RuntimeError(f"render returned no feature object: {case_id}")
            placeholders = normalized_placeholders(
                features.get("mm_placeholders")
                if isinstance(features, dict)
                else None
            )
            sampling = rendered.get("sampling_params")
            if not isinstance(sampling, dict):
                raise RuntimeError(f"render returned no sampling params: {case_id}")
            reference_response = references[case_id].get("response")
            usage = (
                reference_response.get("usage")
                if isinstance(reference_response, dict)
                else None
            )
            reference_prompt_tokens = (
                usage.get("prompt_tokens") if isinstance(usage, dict) else None
            )
            if (
                not isinstance(reference_prompt_tokens, int)
                or isinstance(reference_prompt_tokens, bool)
                or reference_prompt_tokens <= 0
            ):
                raise RuntimeError(
                    f"reference prompt usage is invalid: {case_id}"
                )
            normalized_request = probe.recursive_replace(
                spec["payload"], spec["replacements"]
            )
            if normalized_request != references[case_id].get("request"):
                raise RuntimeError(
                    f"fixture-normalized request drifted: {case_id}"
                )
            cases.append(
                {
                    "case_id": case_id,
                    "surfaces": list(spec["surfaces"]),
                    "request": normalized_request,
                    "request_sha256": canonical_json_sha256(normalized_request),
                    "reference_transport_request_sha256": references[case_id][
                        "request_sha256"
                    ],
                    "render_transport_request_sha256": canonical_json_sha256(
                        spec["payload"]
                    ),
                    "prompt_tokens": len(token_ids),
                    "prompt_token_ids": token_ids,
                    "prompt_token_ids_sha256": canonical_json_sha256(token_ids),
                    "mm_placeholders": placeholders,
                    "reference_usage_prompt_tokens": reference_prompt_tokens,
                    "reference_usage_delta": reference_prompt_tokens - len(token_ids),
                    "max_tokens": sampling.get("max_tokens"),
                    "structured_outputs": sampling.get("structured_outputs"),
                }
            )
            print(
                json.dumps(
                    {
                        "case_id": case_id,
                        "event": "render_complete",
                        "prompt_tokens": len(token_ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        fixture_server.shutdown()
        fixture_server.server_close()
        fixture_thread.join(timeout=5)

    non_tool_comparable = [
        case
        for case in cases
        if case["case_id"] not in API_RENDER_TOOL_CASES
    ]
    tool_comparable = [
        case
        for case in cases
        if case["case_id"] in API_RENDER_TOOL_CASES
    ]
    forced_structured = next(
        case["structured_outputs"]
        for case in cases
        if case["case_id"] == "tool_forced_image"
    )
    decisions = {
        "success_render_cases_20_of_20": tuple(
            case["case_id"] for case in cases
        )
        == REQUIRED_API_RENDER_CASES,
        "non_tool_render_matches_full_usage": all(
            case["reference_usage_delta"] == 0 for case in non_tool_comparable
        ),
        "tool_full_server_usage_offset_one": all(
            case["reference_usage_delta"] == 1 for case in tool_comparable
        ),
        "named_tool_json_schema_bound": forced_structured
        == {"json": EXPECTED_TOOL_JSON_SCHEMA},
    }
    qualified = all(decisions.values())
    payload = seal_manifest(
        {
            "schema": API_RENDER_SCHEMA,
            "captured_at": utc_now(),
            "complete": tuple(case["case_id"] for case in cases)
            == REQUIRED_API_RENDER_CASES,
            "qualified": qualified,
            "scope": "fixed-vllm-openai-gpu-less-render-token-boundary",
            "host": {"label": "amd395", "hostname": socket.gethostname()},
            "source": {
                **source,
                "files": [
                    file_component(
                        path, path.relative_to(ROOT).as_posix()
                    )
                    for path in (
                        ROOT / "aima_engine/vl_capability.py",
                        ROOT / "aima_engine/vl_reference.py",
                        PROBE_SCRIPT,
                        Path(__file__).resolve(),
                    )
                ],
            },
            "runtime": {
                "vllm": actual_version,
                "endpoint": {
                    "scheme": "http",
                    "host": "127.0.0.1",
                    "port": endpoint_port,
                },
            },
            "bindings": {
                "capability_manifest": file_component(
                    capability_path,
                    "benchmarks/results/vl-capability-manifest.json",
                ),
                "fixture_manifest": file_component(
                    fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
                ),
                "reference_launch": file_component(
                    reference_launch,
                    "benchmarks/results/vl-reference-launch.json",
                ),
                "reference_manifest": file_component(
                    reference_manifest,
                    "benchmarks/results/vl-reference-manifest.json",
                ),
            },
            "contract": {
                "content_format": "auto-resolved-string",
                "request_identity": "fixture-normalized-reference-request",
                "tool_normalization": "ChatCompletionRequest-Pydantic-model_dump",
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
    validation_errors = validate_api_render_manifest(payload)
    if validation_errors:
        raise RuntimeError(
            "API render manifest failed validation:\n- "
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
