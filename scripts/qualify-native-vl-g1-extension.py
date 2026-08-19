#!/usr/bin/env python3
"""Qualify native serving against the frozen five-case VL G1 extension."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType
from typing import Any
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.aotriton_closure import (  # noqa: E402
    require_aotriton_closure,
)
from aima_engine.vl_g1_extension import (  # noqa: E402
    CASE_ORDER,
    build_cases,
    finish_reason,
    normalize_contract_request,
    request_media_counts,
    response_content,
    usage_signature,
)
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    canonical_json_sha256,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
    verify_manifest_integrity,
)


MODEL_ID = "aima-amd395-qwen36-35b"
REFERENCE_SCHEMA = (
    "aima-amd395-qwen36/vl-g1-mixed-conversation-reference/v1"
)
VISION_ATTENTION_SHA256 = (
    "e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c"
)
PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
CAPABILITY_QUALIFIER = ROOT / "scripts/qualify-native-vl-capabilities.py"
CAPTURE_SCRIPT = ROOT / "scripts/capture-vllm-vl-g1-extension.py"


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


def validate_reference(reference: dict[str, Any]) -> list[str]:
    errors = verify_manifest_integrity(reference)
    if reference.get("schema") != REFERENCE_SCHEMA:
        errors.append("G1 extension reference schema changed")
    if reference.get("complete") is not True or reference.get("qualified") is not True:
        errors.append("G1 extension reference is incomplete")
    cases = reference.get("cases")
    if not isinstance(cases, list) or tuple(
        item.get("case_id") for item in cases if isinstance(item, dict)
    ) != CASE_ORDER:
        errors.append("G1 extension reference case order changed")
    if isinstance(cases, list) and not all(
        isinstance(item, dict) and item.get("qualified") is True
        for item in cases
    ):
        errors.append("G1 extension reference contains a failed case")
    source = reference.get("source")
    components = source.get("files") if isinstance(source, dict) else None
    if not isinstance(components, list):
        errors.append("G1 extension reference source files are missing")
    else:
        for component in components:
            if not isinstance(component, dict):
                errors.append("G1 extension reference source binding is malformed")
                continue
            path = ROOT / str(component.get("path", ""))
            if (
                not path.is_file()
                or path.stat().st_size != component.get("bytes")
                or sha256_file(path) != component.get("sha256")
            ):
                errors.append(
                    f"G1 extension reference source changed: {component.get('path')}"
                )
    return errors


def case_checks(
    case: dict[str, Any], reference: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, bool]:
    expected_counts = request_media_counts(case["request"])
    vl = metrics.get("vl") if isinstance(metrics.get("vl"), dict) else {}
    mrope = (
        metrics.get("mrope")
        if isinstance(metrics.get("mrope"), dict)
        else {}
    )
    stream = "stream" in case["surfaces"]
    response = case["response"]
    reference_response = reference["response"]
    checks = {
        "surface_accepted": case.get("passed") is True,
        "reference_status_exact": case.get("status_code")
        == reference.get("status_code")
        == 200,
        "request_contract_exact": case.get("request")
        == reference.get("request"),
        "resident_native_request": metrics.get("model_loads") == 1
        and metrics.get("oracle_tensor_reads") == 0
        and str(metrics.get("runtime", "")).startswith("native-resident-q"),
        "media_counts_exact": vl.get("image_count") == expected_counts["image"]
        and vl.get("video_count") == expected_counts["video"]
        and vl.get("media_count")
        == expected_counts["image"] + expected_counts["video"],
        "media_executed": vl.get("enabled") is True
        and isinstance(vl.get("vision_patches"), int)
        and vl["vision_patches"] > 0
        and isinstance(vl.get("visual_tokens"), int)
        and vl["visual_tokens"] > 0,
        "mrope_enabled": mrope.get("enabled") is True,
        "render_prompt_tokens_exact": metrics.get("prompt_tokens")
        == reference["render"]["prompt_tokens"],
        "render_prompt_token_ids_exact": metrics.get(
            "prompt_token_ids_sha256"
        )
        == reference["render"]["prompt_token_ids_sha256"],
        "finish_reason_exact": finish_reason(response)
        == finish_reason(reference_response),
        "generated_content_exact": response_content(response)
        == response_content(reference_response),
        "usage_exact": usage_signature(response)
        == usage_signature(reference_response),
        "stream_complete": (
            isinstance(response, dict)
            and response.get("done") is True
            and response.get("event_count", 0) > 0
            if stream
            else True
        ),
    }
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--fmha-provider", type=Path, required=True)
    parser.add_argument("--vision-attention-image", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18156)
    parser.add_argument("--ready-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    binary = args.binary.resolve()
    model_dir = args.model_dir.resolve()
    fixture_root = args.fixture_root.resolve()
    fmha_provider = args.fmha_provider.resolve()
    aotriton = require_aotriton_closure(fmha_provider)
    vision_attention_image = args.vision_attention_image.resolve()
    reference_path = args.reference.resolve()
    output = args.output.resolve()
    raw_root = output.parent / f"{output.stem}-raw"
    required = (
        binary,
        fmha_provider,
        vision_attention_image,
        fixture_root / "fixtures-manifest.json",
        reference_path,
        PROBE_SCRIPT,
        CAPABILITY_QUALIFIER,
        CAPTURE_SCRIPT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"native G1 extension inputs are missing: {missing}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")
    if output.exists() or raw_root.exists():
        raise SystemExit("qualification output and raw directory must not exist")
    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("native G1 extension qualification requires clean source")
    build_info = json.loads(
        subprocess.run(
            [str(binary), "--build-info"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if build_info.get("source_commit") != source["commit"]:
        raise SystemExit("native binary source commit differs from checkout")
    if sha256_file(vision_attention_image) != VISION_ATTENTION_SHA256:
        raise SystemExit("vision-attention image differs from frozen artifact")

    reference = load_json_object(reference_path)
    reference_errors = validate_reference(reference)
    if reference_errors:
        raise SystemExit(
            "invalid G1 extension reference:\n- "
            + "\n- ".join(reference_errors)
        )
    references = {case["case_id"]: case for case in reference["cases"]}
    probe = load_module(PROBE_SCRIPT, "native_vl_g1_extension_probe")
    capability = load_module(
        CAPABILITY_QUALIFIER, "native_vl_g1_extension_helpers"
    )
    fixtures = probe.Fixtures(fixture_root, "http://127.0.0.1:1")
    specs = build_cases(fixtures, MODEL_ID)

    raw_root.mkdir(parents=True)
    isolated_home = raw_root / "home"
    isolated_home.mkdir()
    stdout_path = raw_root / "server.stdout.log"
    stderr_path = raw_root / "server.stderr.log"
    load_report = raw_root / "native-weight-load.json"
    raw_cases_path = raw_root / "cases.json"
    command = [
        str(binary),
        "serve",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        "1024",
        "--cache-capacity",
        "2048",
        "--fmha-provider",
        str(fmha_provider),
        "--vision-attention-image",
        str(vision_attention_image),
        "--allowed-local-media-path",
        str(fixture_root),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-requests",
        str(len(CASE_ORDER)),
        "--request-timeout-ms",
        str(int(args.request_timeout_seconds * 1000)),
        "--report",
        str(load_report),
    ]
    environment = {
        "HOME": str(isolated_home),
        "LANG": "C",
        "PATH": "/usr/bin:/bin",
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint = f"http://{args.host}:{args.port}"
    process: subprocess.Popen[bytes] | None = None
    health: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
            health = capability.wait_ready(
                opener,
                endpoint + "/health",
                process,
                args.ready_timeout_seconds,
            )
            for spec in specs:
                result = probe.execute_case(
                    endpoint,
                    timeout=args.request_timeout_seconds,
                    response_redactions={
                        str(fixture_root): "${AIMA_VL_FIXTURE_ROOT}"
                    },
                    **spec,
                )
                transport_request_sha256 = result.pop("request_sha256")
                result["request"] = normalize_contract_request(
                    result["request"]
                )
                result["request_sha256"] = canonical_json_sha256(
                    result["request"]
                )
                result["transport_request_sha256"] = (
                    transport_request_sha256
                )
                metrics = capability.native_metrics(result)
                checks = case_checks(
                    result, references[result["case_id"]], metrics
                )
                result["qualification_checks"] = checks
                result["qualified"] = all(checks.values())
                cases.append(result)
                print(
                    json.dumps(
                        {
                            "case_id": result["case_id"],
                            "event": "native_case_complete",
                            "qualified": result["qualified"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            atomic_json(raw_cases_path, {"cases": cases})
            process.wait(timeout=60)
            if process.returncode != 0:
                raise RuntimeError(
                    f"native server exited with code {process.returncode}"
                )
    finally:
        if process is not None and process.poll() is None:
            try:
                capability.post_shutdown(opener, endpoint)
                process.wait(timeout=10)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    events = capability.parse_server_events(stdout_path)
    ready_events = [item for item in events if item.get("event") == "ready"]
    stopped_events = [item for item in events if item.get("event") == "stopped"]
    ready = ready_events[0] if len(ready_events) == 1 else {}
    stopped = stopped_events[0] if len(stopped_events) == 1 else {}
    server_checks = {
        "one_ready_event": len(ready_events) == 1,
        "one_stopped_event": len(stopped_events) == 1,
        "all_requests_served": stopped.get("served") == len(CASE_ORDER),
        "one_model_load": stopped.get("model_loads") == 1,
        "native_only": all(
            ready.get(f"runtime_{name}") is False
            for name in ("python", "torch", "vllm", "triton")
        ),
        "visual_weights_resident": ready.get("visual_model_tensor_count") == 333
        and ready.get("visual_model_payload_bytes") == 893_142_496,
        "vl_ready": ready.get("native_vl") is True,
        "stderr_empty": stderr_path.stat().st_size == 0,
    }
    complete = (
        tuple(case["case_id"] for case in cases) == CASE_ORDER
        and all(case["qualified"] for case in cases)
        and all(server_checks.values())
    )
    replacements = [
        (str(ROOT), "${AIMA_REPO_ROOT}"),
        (str(model_dir), "${AIMA_MODEL_DIR}"),
        (str(fixture_root), "${AIMA_VL_FIXTURE_ROOT}"),
        (str(fmha_provider), "${AIMA_FMHA_PROVIDER}"),
        (str(vision_attention_image), "${AIMA_VISION_ATTENTION_IMAGE}"),
        (str(binary.parent), "${AIMA_NATIVE_BUILD_DIR}"),
        (str(raw_root), "${AIMA_QUALIFICATION_RAW_DIR}"),
    ]
    source_files = tuple(
        ROOT / path
        for path in (
            "native/include/aima/native_chat_protocol.h",
            "native/include/aima/native_vl_request.h",
            "native/include/aima/native_resident_engine.h",
            "native/src/native_chat_protocol.cpp",
            "native/src/native_vl_request.cpp",
            "native/src/native_resident_engine.hip.cpp",
            "native/src/native_http_server.cpp",
            "aima_engine/vl_g1_extension.py",
            "aima_engine/aotriton_closure.py",
            "scripts/probe-vllm-vl-api-capabilities.py",
            "scripts/capture-vllm-vl-g1-extension.py",
            "scripts/qualify-native-vl-g1-extension.py",
        )
    )
    payload = seal_manifest(
        {
            "schema": "aima-amd395-qwen36/native-vl-g1-extension-qualification/v1",
            "captured_at": utc_now(),
            "complete": complete,
            "qualified": complete,
            "scope": "resident-native-five-case-mixed-conversation-stream-extension",
            "host": {"label": "amd395", "hostname": socket.gethostname()},
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in source_files
                ],
            },
            "binary": file_component(binary, "build/native/aima-engine-native"),
            "build_info": build_info,
            "dependencies": {
                "reference": file_component(
                    reference_path,
                    "benchmarks/results/vl-g1-mixed-conversation-reference-v0.1.0.json",
                ),
                "fixture_manifest": file_component(
                    fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
                ),
                "fmha_provider": file_component(
                    fmha_provider, "build/native/libaima-fmha-aotriton.so"
                ),
                "aotriton_runtime": file_component(
                    aotriton.runtime, "build/native/libaotriton_v2.so.0.11.1"
                ),
                "aotriton_image": file_component(
                    aotriton.image,
                    "build/native/aotriton.images/amd-gfx11xx/flash/attn_fwd/"
                    "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2",
                ),
                "vision_attention_image": file_component(
                    vision_attention_image,
                    "build/native/aima-vision-attention.hsaco",
                ),
            },
            "launch": {
                "command": capability.publicize(command, replacements),
                "environment_keys": sorted(environment),
                "health": health,
                "ready": capability.publicize(ready, replacements),
                "stopped": stopped,
                "checks": server_checks,
            },
            "cases": cases,
            "raw": {
                "stdout": file_component(
                    stdout_path, f"{raw_root.name}/server.stdout.log"
                ),
                "stderr": file_component(
                    stderr_path, f"{raw_root.name}/server.stderr.log"
                ),
                "weight_load": file_component(
                    load_report, f"{raw_root.name}/native-weight-load.json"
                ),
                "cases": file_component(
                    raw_cases_path, f"{raw_root.name}/cases.json"
                ),
            },
            "decision": {
                "five_cases_reference_exact": complete,
                "mixed_multi_item_orders_qualified": all(
                    case["qualified"] for case in cases[:2]
                ),
                "video_and_mixed_history_qualified": all(
                    case["qualified"] for case in cases[2:4]
                ),
                "mixed_sse_qualified": cases[4]["qualified"],
                "single_resident_model_load": server_checks["one_model_load"],
                "runtime_python": False,
                "runtime_torch": False,
                "runtime_vllm": False,
                "runtime_triton": False,
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
