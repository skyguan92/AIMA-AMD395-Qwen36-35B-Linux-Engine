#!/usr/bin/env python3
"""Qualify the native resident server against the frozen 30-case VL API surface."""

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
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_capability import (  # noqa: E402
    REQUIRED_API_CASES,
    validate_capability_manifest,
)
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
)


MODEL_ID = "aima-amd395-qwen36-35b"
VISION_ATTENTION_SHA256 = (
    "b709a058a77d61e14db73c1ff7d7f4c20859d997bec811cad7339d3e59223d00"
)
REFERENCE_MANIFEST = ROOT / "benchmarks/results/vl-capability-manifest.json"
PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_probe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aima_vl_api_capability_probe", PROBE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen VL API probe")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def publicize(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for actual, symbolic in replacements:
            value = value.replace(actual, symbolic)
        return value
    if isinstance(value, list):
        return [publicize(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: publicize(item, replacements) for key, item in value.items()
        }
    return value


class QuietFixtureHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *args: object) -> None:
        del args


def wait_ready(
    opener: urllib.request.OpenerDirector,
    health_url: str,
    process: subprocess.Popen[bytes],
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"native server exited before READY with code {process.returncode}"
            )
        try:
            with opener.open(health_url, timeout=1.0) as response:
                value = json.loads(response.read())
            if (
                response.status == 200
                and isinstance(value, dict)
                and value.get("status") == "ok"
                and value.get("model_loaded") is True
                and value.get("native_vl") is True
            ):
                return value
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError("native VL server did not become ready before timeout")


def post_shutdown(
    opener: urllib.request.OpenerDirector, endpoint: str
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/shutdown",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=5.0) as response:
        value = json.loads(response.read())
    if response.status != 200 or value.get("status") != "shutting_down":
        raise RuntimeError("native HTTP shutdown did not acknowledge")
    return value


def parse_server_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("event"), str):
            events.append(value)
    return events


def final_stream_event(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    events = response.get("events")
    if not isinstance(events, list):
        return {}
    for event in reversed(events):
        if isinstance(event, dict) and isinstance(event.get("aima_amd395"), dict):
            return event
    return {}


def native_metrics(case: dict[str, Any]) -> dict[str, Any]:
    response = case.get("response")
    if not isinstance(response, dict):
        return {}
    if isinstance(response.get("aima_amd395"), dict):
        return response["aima_amd395"]
    return final_stream_event(response).get("aima_amd395", {})


def finish_reason(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and len(choices) == 1:
        choice = choices[0]
        if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
            return choice["finish_reason"]
    events = response.get("events")
    if isinstance(events, list):
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            choices = event.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                continue
            choice = choices[0]
            if isinstance(choice, dict) and isinstance(
                choice.get("finish_reason"), str
            ):
                return choice["finish_reason"]
    return None


def usage(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    value = response.get("usage")
    return value if isinstance(value, dict) else {}


def usage_signature(response: Any) -> tuple[Any, Any, Any] | None:
    value = usage(response)
    if not value:
        return None
    return tuple(
        value.get(name)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    )


def valid_inspect_visual_call(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    choices = response.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or function.get("name") != "inspect_visual":
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict) and isinstance(decoded.get("label"), str):
                return bool(decoded["label"])
    return False


def reference_case_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["case_id"]: item
        for item in manifest.get("cases", [])
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }


def case_checks(
    case: dict[str, Any], reference: dict[str, Any]
) -> dict[str, bool]:
    accepted = bool(case.get("accepted"))
    metrics = native_metrics(case)
    response = case.get("response")
    has_media = bool(set(case.get("surfaces", [])) & {"image", "video", "mixed"})
    checks = {
        "surface_expectation": case.get("passed") is True,
        "reference_status_exact": case.get("status_code")
        == reference.get("status_code"),
    }
    if accepted:
        vl = metrics.get("vl") if isinstance(metrics.get("vl"), dict) else {}
        mrope = (
            metrics.get("mrope") if isinstance(metrics.get("mrope"), dict) else {}
        )
        checks.update(
            {
                "resident_request_metrics": metrics.get("model_loads") == 1
                and metrics.get("oracle_tensor_reads") == 0,
                "native_runtime": str(metrics.get("runtime", "")).startswith(
                    "native-resident-q"
                ),
                "vl_boundary_exact": vl.get("enabled") is has_media,
                "mrope_boundary_exact": mrope.get("enabled") is has_media,
            }
        )
        if has_media:
            checks["media_executed"] = (
                isinstance(vl.get("media_count"), int)
                and vl["media_count"] > 0
                and isinstance(vl.get("vision_patches"), int)
                and vl["vision_patches"] > 0
                and isinstance(vl.get("visual_tokens"), int)
                and vl["visual_tokens"] > 0
            )
        if case.get("case_id") in {"tool_forced_image", "tool_auto_image"}:
            checks["structured_tool_call"] = valid_inspect_visual_call(response)
        if "stream" in case.get("surfaces", []):
            checks["complete_sse"] = (
                isinstance(response, dict)
                and response.get("done") is True
                and isinstance(response.get("event_count"), int)
                and response["event_count"] > 0
                and bool(final_stream_event(response))
            )
    else:
        error = response.get("error") if isinstance(response, dict) else None
        checks.update(
            {
                "http_400": case.get("status_code") == 400,
                "compatible_error_category": isinstance(error, dict)
                and error.get("type") == "invalid_request_error"
                and error.get("code") == "bad_request"
                and isinstance(error.get("message"), str)
                and bool(error["message"]),
            }
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--fmha-provider", type=Path, required=True)
    parser.add_argument("--vision-attention-image", type=Path, required=True)
    parser.add_argument(
        "--reference-manifest", type=Path, default=REFERENCE_MANIFEST
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18096)
    parser.add_argument("--fixture-port", type=int, default=18097)
    parser.add_argument("--ready-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    binary = args.binary.resolve()
    model_dir = args.model_dir.resolve()
    fixture_root = args.fixture_root.resolve()
    fmha_provider = args.fmha_provider.resolve()
    vision_attention_image = args.vision_attention_image.resolve()
    reference_path = args.reference_manifest.resolve()
    output = args.output.resolve()
    raw_root = output.parent / f"{output.stem}-raw"
    required_paths = (
        binary,
        fmha_provider,
        vision_attention_image,
        fixture_root / "fixtures-manifest.json",
        reference_path,
        PROBE_SCRIPT,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"native VL capability inputs are missing: {missing}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")
    if output.exists() or raw_root.exists():
        raise SystemExit("qualification output and raw directory must not exist")

    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("native VL capability qualification requires clean source")
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
    reference_errors = validate_capability_manifest(reference)
    if reference_errors:
        raise SystemExit(
            "invalid frozen capability reference:\n- "
            + "\n- ".join(reference_errors)
        )
    references = reference_case_map(reference)
    expected_order = tuple(REQUIRED_API_CASES)
    if tuple(item["case_id"] for item in reference["cases"]) != expected_order:
        raise SystemExit("frozen capability reference case order changed")

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
        "--allowed-media-domain",
        "127.0.0.1",
        "--host",
        args.host,
        "--port",
        str(args.port),
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
    fixture_base = f"http://127.0.0.1:{args.fixture_port}"
    probe = load_probe_module()
    fixtures = probe.Fixtures(fixture_root, fixture_base)
    specs = probe.build_cases(fixtures, MODEL_ID)
    if tuple(item["case_id"] for item in specs) != expected_order:
        raise SystemExit("native probe case order differs from frozen reference")

    fixture_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", args.fixture_port),
        partial(QuietFixtureHandler, directory=str(fixture_root)),
    )
    fixture_thread = threading.Thread(
        target=fixture_server.serve_forever,
        name="native-vl-fixture-http",
        daemon=True,
    )
    fixture_thread.start()
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
            health = wait_ready(
                opener,
                endpoint + "/health",
                process,
                args.ready_timeout_seconds,
            )
            for spec in specs:
                print(f"CASE {spec['case_id']}", flush=True)
                result = probe.execute_case(
                    endpoint,
                    timeout=args.request_timeout_seconds,
                    response_redactions={
                        str(fixture_root): "${AIMA_VL_FIXTURE_ROOT}"
                    },
                    **spec,
                )
                checks = case_checks(result, references[result["case_id"]])
                result["qualification_checks"] = checks
                result["qualified"] = all(checks.values())
                cases.append(result)
                print(
                    f"RESULT {result['case_id']} status={result['status_code']} "
                    f"qualified={result['qualified']}",
                    flush=True,
                )
            atomic_json(raw_cases_path, {"cases": cases})
            post_shutdown(opener, endpoint)
            process.wait(timeout=60)
            if process.returncode != 0:
                raise RuntimeError(
                    f"native server exited with code {process.returncode}"
                )
    finally:
        fixture_server.shutdown()
        fixture_server.server_close()
        fixture_thread.join(timeout=5)
        if process is not None and process.poll() is None:
            try:
                post_shutdown(opener, endpoint)
                process.wait(timeout=10)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    successful_metrics = [native_metrics(case) for case in cases if case["accepted"]]
    request_indexes = [item.get("request_index") for item in successful_metrics]
    expected_successes = sum(1 for value in REQUIRED_API_CASES.values() if value)
    expected_errors = len(REQUIRED_API_CASES) - expected_successes
    events = parse_server_events(stdout_path)
    ready_events = [item for item in events if item.get("event") == "ready"]
    stopped_events = [item for item in events if item.get("event") == "stopped"]
    ready = ready_events[0] if len(ready_events) == 1 else {}
    stopped = stopped_events[0] if len(stopped_events) == 1 else {}
    server_checks = {
        "one_ready_event": len(ready_events) == 1,
        "one_stopped_event": len(stopped_events) == 1,
        "twenty_successful_requests_served": stopped.get("served")
        == expected_successes,
        "request_indexes_contiguous": request_indexes
        == list(range(1, expected_successes + 1)),
        "one_model_load": stopped.get("model_loads") == 1,
        "native_only": all(
            ready.get(f"runtime_{name}") is False
            for name in ("python", "torch", "vllm", "triton")
        ),
        "visual_weights_resident": ready.get("visual_model_tensor_count") == 333
        and ready.get("visual_model_payload_bytes") == 893_142_496,
        "vl_ready": ready.get("native_vl") is True,
    }
    success_cases = [case for case in cases if case["accepted"]]
    error_cases = [case for case in cases if not case["accepted"]]
    status_exact = sum(
        case["qualification_checks"]["reference_status_exact"] for case in cases
    )
    finish_exact = sum(
        finish_reason(case["response"])
        == finish_reason(references[case["case_id"]].get("response"))
        for case in success_cases
    )
    usage_comparable = [
        case
        for case in success_cases
        if usage_signature(case["response"]) is not None
        and usage_signature(references[case["case_id"]].get("response"))
        is not None
    ]
    usage_exact = sum(
        usage_signature(case["response"])
        == usage_signature(references[case["case_id"]].get("response"))
        for case in usage_comparable
    )
    matrix_complete = (
        len(cases) == len(REQUIRED_API_CASES)
        and len(success_cases) == expected_successes
        and len(error_cases) == expected_errors
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
        ROOT / relative
        for relative in (
            "native/include/aima/native_chat_protocol.h",
            "native/src/native_chat_protocol.cpp",
            "native/src/native_vl_request.cpp",
            "native/src/native_http_server.cpp",
            "scripts/probe-vllm-vl-api-capabilities.py",
            "scripts/qualify-native-vl-capabilities.py",
        )
    )
    payload = {
        "schema": "aima-amd395-qwen36/native-vl-capability-qualification/v1",
        "captured_at": utc_now(),
        "complete": matrix_complete,
        "qualified": matrix_complete,
        "scope": "resident-native-frozen-30-case-vl-api-surface",
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
            "reference_capability_manifest": file_component(
                reference_path,
                "benchmarks/results/vl-capability-manifest.json",
            ),
            "fixture_manifest": file_component(
                fixture_root / "fixtures-manifest.json",
                "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
            ),
            "fmha_provider": file_component(
                fmha_provider, "build/native/libaima-fmha-aotriton.so"
            ),
            "vision_attention_image": file_component(
                vision_attention_image,
                "build/native/aima-vision-attention.hsaco",
            ),
        },
        "launch": {
            "command": publicize(command, replacements),
            "environment_keys": sorted(environment),
            "fixture_http": {
                "purpose": "static-input-transport-only-not-inference",
                "host": "127.0.0.1",
                "port": args.fixture_port,
            },
            "health": health,
            "ready": publicize(ready, replacements),
            "stopped": stopped,
            "checks": server_checks,
        },
        "matrix": {
            "required_cases": len(REQUIRED_API_CASES),
            "success_cases": len(success_cases),
            "error_cases": len(error_cases),
            "reference_status_exact": f"{status_exact}/{len(cases)}",
            "reference_finish_reason_exact": (
                f"{finish_exact}/{len(success_cases)}"
            ),
            "reference_usage_exact": f"{usage_exact}/{len(usage_comparable)}",
            "cases": cases,
        },
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
            "frozen_surface_matrix_30_of_30": matrix_complete,
            "success_surfaces_20_of_20": len(success_cases) == expected_successes
            and all(case["qualified"] for case in success_cases),
            "error_surfaces_10_of_10": len(error_cases) == expected_errors
            and all(case["qualified"] for case in error_cases),
            "http_status_parity_30_of_30": status_exact == len(cases),
            "structured_vl_tool_calls_2_of_2": all(
                valid_inspect_visual_call(
                    next(
                        case["response"]
                        for case in cases
                        if case["case_id"] == case_id
                    )
                )
                for case_id in ("tool_forced_image", "tool_auto_image")
            ),
            "complete_vl_sse_2_of_2": all(
                next(
                    case["qualification_checks"].get("complete_sse", False)
                    for case in cases
                    if case["case_id"] == case_id
                )
                for case_id in ("stream_image", "stream_video")
            ),
            "single_resident_model_load": server_checks["one_model_load"],
            "runtime_python": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
            "deterministic_reference_usage_exact": usage_exact
            == len(usage_comparable),
            "g1_passed": False,
            "g2_passed": False,
            "g3_passed": False,
            "g4_passed": False,
            "g5_passed": False,
        },
    }
    digest = atomic_json(output, seal_manifest(payload))
    print(
        json.dumps(
            {
                "complete": matrix_complete,
                "qualified": matrix_complete,
                "output": str(output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if matrix_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
