#!/usr/bin/env python3
"""Capture the frozen vLLM reference for long greedy VL task quality."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
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

from aima_engine.vl_reference import (  # noqa: E402
    PINNED_PACKAGES,
    REFERENCE_SCHEMA as BASE_REFERENCE_SCHEMA,
    atomic_json,
    canonical_json_sha256,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    validate_launch_config,
    verify_manifest_integrity,
)
from aima_engine.vl_task_quality import (  # noqa: E402
    CASE_ORDER,
    MAX_TOKENS,
    MIN_REFERENCE_CASE_SCORE_MILLIONTHS,
    MIN_REFERENCE_MODALITY_SCORE_MILLIONTHS,
    REFERENCE_SCHEMA,
    TASK_CASES,
    aggregate_scores,
    build_cases,
    complete_output_token_ids,
    finish_reason,
    normalize_contract_request,
    output_token_ids_sha256,
    response_content,
    score_text,
    usage_signature,
    validate_fixture_manifest,
    validate_reference_manifest,
)


FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-task-quality-v0.1.0"
REFERENCE_LAUNCH = ROOT / "benchmarks/results/vl-reference-launch.json"
REFERENCE_MANIFEST = ROOT / "benchmarks/results/vl-reference-manifest.json"
PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
RENDER_SCRIPT = ROOT / "scripts/capture-vllm-vl-api-render.py"
MIN_COMPLETION_TOKENS = 4


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


def validate_timeout(value: float) -> None:
    if not math.isfinite(value) or value <= 0 or value > 600:
        raise ValueError(
            "reference request timeout must be positive and at most 600 seconds"
        )


def request_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    body = (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if payload is not None
        else None
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            response_body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"reference endpoint returned HTTP {error.code}: {detail}"
        ) from error
    if status != 200:
        raise RuntimeError(f"reference endpoint returned HTTP {status}")
    value = json.loads(response_body)
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
    if reference.get("schema") != BASE_REFERENCE_SCHEMA:
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


def tokenize_output(
    opener: urllib.request.OpenerDirector,
    endpoint: str,
    *,
    model: str,
    content: str,
    completion_tokens: int,
    finish: str | None,
    timeout: float,
) -> tuple[list[int], dict[str, Any]]:
    response = request_json(
        opener,
        endpoint + "/tokenize",
        payload={
            "model": model,
            "prompt": content,
            "add_special_tokens": False,
        },
        timeout=timeout,
    )
    tokenized = response.get("tokens")
    if not isinstance(tokenized, list) or any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id < 0
        for token_id in tokenized
    ):
        raise RuntimeError("reference /tokenize returned invalid token IDs")
    if isinstance(response.get("count"), int) and response["count"] != len(
        tokenized
    ):
        raise RuntimeError("reference /tokenize count differs from token IDs")
    token_ids, eos_appended = complete_output_token_ids(
        tokenized,
        completion_tokens=completion_tokens,
        finish=finish,
    )
    return token_ids, {
        "method": "vllm-tokenize-decoded-content-with-terminal-eos-recovery",
        "retokenized_tokens": len(tokenized),
        "eos_appended": eos_appended,
    }


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

    try:
        endpoint = loopback_endpoint(args.endpoint)
        validate_timeout(args.timeout_seconds)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    fixture_root = args.fixture_root.resolve()
    launch_path = args.reference_launch.resolve()
    reference_path = args.reference_manifest.resolve()
    output = args.output.resolve()
    fixture_manifest_path = fixture_root / "fixtures-manifest.json"
    required = (
        fixture_manifest_path,
        launch_path,
        reference_path,
        PROBE_SCRIPT,
        RENDER_SCRIPT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"task-quality capture inputs are missing: {missing}")
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise SystemExit("task-quality output and sidecar must not exist")
    fixture_manifest = load_json_object(fixture_manifest_path)
    fixture_errors = validate_fixture_manifest(fixture_manifest, fixture_root)
    if fixture_errors:
        raise SystemExit(
            "invalid task-quality fixtures:\n- " + "\n- ".join(fixture_errors)
        )
    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("task-quality reference capture requires clean source")

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

    probe = load_module(PROBE_SCRIPT, "vl_task_quality_reference_probe")
    render = load_module(RENDER_SCRIPT, "vl_task_quality_reference_render")
    fixtures = probe.Fixtures(fixture_root, "http://127.0.0.1:1")
    specs = build_cases(fixtures, "qwen36-vl-reference")
    contracts = {case["case_id"]: case for case in TASK_CASES}
    cases: list[dict[str, Any]] = []
    for spec in specs:
        case_id = spec["case_id"]
        print(f"CASE {case_id}", flush=True)
        transport_payload = spec["payload"]
        result = probe.execute_case(
            endpoint,
            case_id=case_id,
            surfaces=spec["surfaces"],
            expected_accept=spec["expected_accept"],
            payload=transport_payload,
            replacements=spec["replacements"],
            require_tool_call=spec["require_tool_call"],
            timeout=args.timeout_seconds,
            response_redactions={
                str(fixture_root): "${AIMA_VL_TASK_FIXTURE_ROOT}"
            },
        )
        rendered = request_json(
            opener,
            endpoint + "/v1/chat/completions/render",
            payload=transport_payload,
            timeout=args.timeout_seconds,
        )
        prompt_ids = rendered.get("token_ids")
        features = rendered.get("features")
        sampling = rendered.get("sampling_params")
        if not isinstance(prompt_ids, list) or not prompt_ids or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in prompt_ids
        ):
            raise RuntimeError(f"render returned invalid tokens: {case_id}")
        if not isinstance(features, dict) or not isinstance(sampling, dict):
            raise RuntimeError(
                f"render returned no features or sampling params: {case_id}"
            )
        placeholders = render.normalized_placeholders(
            features.get("mm_placeholders"), prompt_ids
        )
        normalized_request = normalize_contract_request(result["request"])
        transport_request_sha256 = result.pop("request_sha256")
        result["modality"] = spec["modality"]
        result["request"] = normalized_request
        result["request_sha256"] = canonical_json_sha256(normalized_request)
        result["transport_request_sha256"] = transport_request_sha256
        result["render"] = {
            "prompt_tokens": len(prompt_ids),
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
            "mm_placeholders": placeholders,
            "max_tokens": sampling.get("max_tokens"),
        }

        content = response_content(result["response"])
        finish = finish_reason(result["response"])
        usage = usage_signature(result["response"])
        if usage is None:
            raise RuntimeError(f"reference returned invalid usage: {case_id}")
        output_ids, reconstruction = tokenize_output(
            opener,
            endpoint,
            model=transport_payload["model"],
            content=content,
            completion_tokens=usage[1],
            finish=finish,
            timeout=args.timeout_seconds,
        )
        score = score_text(content, contracts[case_id]["rubric"])
        result["rubric"] = spec["rubric"]
        result["output_text"] = content
        result["output_token_ids"] = output_ids
        result["output_token_ids_sha256"] = output_token_ids_sha256(output_ids)
        result["output_token_reconstruction"] = reconstruction
        result["score"] = score
        checks = {
            "http_200": result["status_code"] == 200,
            "surface_accepted": result["passed"] is True,
            "render_prompt_nonempty": len(prompt_ids) > 0,
            "render_prompt_usage_exact": usage[0] == len(prompt_ids),
            "render_max_tokens_exact": sampling.get("max_tokens") == MAX_TOKENS,
            "finish_reason_supported": finish in {"stop", "length"},
            "generated_content_nonempty": bool(content),
            "completion_is_longer_greedy": usage[1] >= MIN_COMPLETION_TOKENS,
            "completion_within_budget": usage[1] <= MAX_TOKENS,
            "usage_accounting_exact": usage[2] == usage[0] + usage[1],
            "output_token_vector_exact": len(output_ids) == usage[1],
            "reference_case_score_floor": score["score_millionths"]
            >= MIN_REFERENCE_CASE_SCORE_MILLIONTHS,
        }
        result["qualification_checks"] = checks
        result["qualified"] = all(checks.values())
        cases.append(result)
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "completion_tokens": usage[1],
                    "event": "task_quality_reference_case_complete",
                    "qualified": result["qualified"],
                    "score_millionths": score["score_millionths"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    aggregate = aggregate_scores(cases)
    modality_floors = all(
        score["score_millionths"] >= MIN_REFERENCE_MODALITY_SCORE_MILLIONTHS
        for score in aggregate.values()
    )
    complete = (
        tuple(case["case_id"] for case in cases) == CASE_ORDER
        and all(case["qualified"] for case in cases)
        and modality_floors
    )
    payload = seal_manifest(
        {
            "schema": REFERENCE_SCHEMA,
            "captured_at": utc_now(),
            "complete": complete,
            "qualified_for_native_replay": complete,
            "scope": "fixed-vllm-twelve-case-long-greedy-image-video-quality",
            "host": {"label": "amd395", "hostname": socket.gethostname()},
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in (
                        ROOT / "aima_engine/vl_capability.py",
                        ROOT / "aima_engine/vl_reference.py",
                        ROOT / "aima_engine/vl_task_quality.py",
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
                "output_token_capture": (
                    "vllm-tokenize-decoded-content-with-terminal-eos-recovery"
                ),
            },
            "bindings": {
                "fixture_manifest": file_component(
                    fixture_manifest_path,
                    "benchmarks/fixtures/vl-task-quality-v0.1.0/"
                    "fixtures-manifest.json",
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
            "aggregate": aggregate,
            "decision": {
                "twelve_reference_cases_accepted": all(
                    case["qualified"] for case in cases
                ),
                "twelve_prompt_vectors_frozen": len(cases) == len(CASE_ORDER)
                and all(case["render"]["prompt_token_ids"] for case in cases),
                "twelve_output_token_vectors_frozen": len(cases)
                == len(CASE_ORDER)
                and all(case["output_token_ids"] for case in cases),
                "image_reference_quality_floor": aggregate["image"][
                    "score_millionths"
                ]
                >= MIN_REFERENCE_MODALITY_SCORE_MILLIONTHS,
                "video_reference_quality_floor": aggregate["video"][
                    "score_millionths"
                ]
                >= MIN_REFERENCE_MODALITY_SCORE_MILLIONTHS,
                "g1_passed": False,
                "g2_passed": False,
                "g3_passed": False,
                "g4_passed": False,
                "g5_passed": False,
            },
        }
    )
    if complete:
        validation_errors = validate_reference_manifest(payload)
        if validation_errors:
            raise RuntimeError(
                "captured task-quality reference failed validation:\n- "
                + "\n- ".join(validation_errors)
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
