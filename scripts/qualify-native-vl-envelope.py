#!/usr/bin/env python3
"""Qualify every frozen native VL execution-envelope cell."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import ModuleType
from typing import Any
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.aotriton_closure import (  # noqa: E402
    require_aotriton_closure,
)
from aima_engine.vl_envelope import validate_envelope  # noqa: E402
from aima_engine.vl_execution import (  # noqa: E402
    MODEL_ID,
    NON_HTTP_CELL_MODES,
    QUALIFICATION_SCHEMA,
    build_http_probe_specs,
    execution_cell_coverage,
    validate_fixture_manifest,
    validate_http_observation,
    validate_processor_probe_observation,
    validate_vision_probe_observation,
)
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
)


ENVELOPE = ROOT / "benchmarks/results/vl-capability-envelope-v0.1.0.json"
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-envelope-v0.1.0"
API_PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
VISION_ATTENTION_SHA256 = (
    "e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_api_probe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aima_vl_envelope_api_probe", API_PROBE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the native VL API request helper")
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


def require_clean_binary(binary: Path, source: dict[str, Any]) -> dict[str, Any]:
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
    return build_info


def run_processor_probe(
    binary: Path, stdout_path: Path, stderr_path: Path,
) -> tuple[dict[str, bool], list[str]]:
    completed = subprocess.run(
        [str(binary)], capture_output=True, text=True, check=False, timeout=120
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return (
        validate_processor_probe_observation(
            completed.returncode, completed.stdout, completed.stderr
        ),
        [str(binary)],
    )


def run_vision_probe(
    binary: Path,
    model_dir: Path,
    attention_image: Path,
    load_report: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
    envelope_cell: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], list[str]]:
    command = [
        str(binary),
        str(model_dir),
        str(attention_image),
        str(load_report),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("native vision envelope probe emitted invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("native vision envelope probe emitted a non-object")
    checks = validate_vision_probe_observation(result, envelope_cell)
    checks["exit_zero"] = completed.returncode == 0
    checks["empty_stderr"] = completed.stderr == ""
    checks["attention_image_exact"] = (
        result.get("attention_image_sha256") == VISION_ATTENTION_SHA256
    )
    return result, checks, command


def source_components() -> list[dict[str, Any]]:
    relative_paths = (
        "aima_engine/vl_envelope.py",
        "aima_engine/aotriton_closure.py",
        "aima_engine/vl_execution.py",
        "native/include/aima/native_media.h",
        "native/include/aima/native_resident_engine.h",
        "native/include/aima/native_vl_processor.h",
        "native/include/aima/native_vl_request.h",
        "native/include/aima/native_vision_pipeline.h",
        "native/src/native_chat_protocol.cpp",
        "native/src/native_http_server.cpp",
        "native/src/native_image_decoder.cpp",
        "native/src/native_media.cpp",
        "native/src/native_resident_engine.hip.cpp",
        "native/src/native_video_decoder.cpp",
        "native/src/native_vl_processor.cpp",
        "native/src/native_vl_request.cpp",
        "native/src/native_vision_pipeline.hip.cpp",
        "native/tools/vl_envelope_vision_probe.hip.cpp",
        "scripts/build-native-vl-envelope-vision-probe.sh",
        "scripts/generate-vl-envelope-fixtures.py",
        "scripts/probe-vllm-vl-api-capabilities.py",
        "scripts/qualify-native-vl-envelope.py",
        "tests/native_vl_processor_test.cpp",
    )
    return [file_component(ROOT / path, path) for path in relative_paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--processor-probe-binary", type=Path, required=True)
    parser.add_argument("--vision-probe-binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fmha-provider", type=Path, required=True)
    parser.add_argument("--vision-attention-image", type=Path, required=True)
    parser.add_argument("--envelope", type=Path, default=ENVELOPE)
    parser.add_argument("--fixture-root", type=Path, default=FIXTURE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18150)
    parser.add_argument("--ready-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--client-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--vision-timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args()

    binary = args.binary.resolve()
    processor_probe_binary = args.processor_probe_binary.resolve()
    vision_probe_binary = args.vision_probe_binary.resolve()
    model_dir = args.model_dir.resolve()
    fmha_provider = args.fmha_provider.resolve()
    aotriton = require_aotriton_closure(fmha_provider)
    long_context_fmha_provider = fmha_provider.with_name(
        "libaima-fmha-ck.so"
    )
    attention_image = args.vision_attention_image.resolve()
    envelope_path = args.envelope.resolve()
    fixture_root = args.fixture_root.resolve()
    fixture_manifest_path = fixture_root / "fixtures-manifest.json"
    output = args.output.resolve()
    raw_root = output.parent / f"{output.stem}-raw"
    required_paths = (
        binary,
        processor_probe_binary,
        vision_probe_binary,
        fmha_provider,
        long_context_fmha_provider,
        attention_image,
        envelope_path,
        fixture_manifest_path,
        API_PROBE_SCRIPT,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"native VL envelope inputs are missing: {missing}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")
    if output.exists() or raw_root.exists():
        raise SystemExit("qualification output and raw directory must not exist")
    if args.request_timeout_seconds > 600.0:
        raise SystemExit("native HTTP request timeout cannot exceed 600 seconds")
    if (
        args.client_timeout_seconds < args.request_timeout_seconds
        or args.client_timeout_seconds > 7200.0
    ):
        raise SystemExit(
            "qualification client timeout must cover the server timeout and "
            "cannot exceed 7200 seconds"
        )

    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("native VL envelope qualification requires clean source")
    build_info = require_clean_binary(binary, source)
    if sha256_file(attention_image) != VISION_ATTENTION_SHA256:
        raise SystemExit("vision-attention image differs from frozen artifact")
    envelope = load_json_object(envelope_path)
    envelope_errors = validate_envelope(envelope)
    if envelope_errors:
        raise SystemExit(
            "invalid frozen VL envelope:\n- " + "\n- ".join(envelope_errors)
        )
    fixture_manifest = load_json_object(fixture_manifest_path)
    fixture_errors = validate_fixture_manifest(fixture_manifest, fixture_root)
    if fixture_errors:
        raise SystemExit(
            "invalid VL envelope fixtures:\n- " + "\n- ".join(fixture_errors)
        )
    specs = build_http_probe_specs(envelope, fixture_manifest, fixture_root)
    coverage = execution_cell_coverage(envelope, specs)
    if "missing" in coverage.values() or len(coverage) != 23:
        raise SystemExit("native VL envelope execution plan is incomplete")

    raw_root.mkdir(parents=True)
    isolated_home = raw_root / "home"
    isolated_home.mkdir()
    processor_stdout = raw_root / "processor-probe.stdout.log"
    processor_stderr = raw_root / "processor-probe.stderr.log"
    vision_stdout = raw_root / "vision-probe.stdout.json"
    vision_stderr = raw_root / "vision-probe.stderr.log"
    vision_load_report = raw_root / "vision-weight-load.json"
    server_stdout = raw_root / "server.stdout.log"
    server_stderr = raw_root / "server.stderr.log"
    server_load_report = raw_root / "server-weight-load.json"
    raw_cases_path = raw_root / "http-observations.json"

    cells = {cell["cell_id"]: cell for cell in envelope["execution_cells"]}
    processor_checks, processor_command = run_processor_probe(
        processor_probe_binary, processor_stdout, processor_stderr
    )
    vision_result, vision_checks, vision_command = run_vision_probe(
        vision_probe_binary,
        model_dir,
        attention_image,
        vision_load_report,
        vision_stdout,
        vision_stderr,
        args.vision_timeout_seconds,
        cells["image_full_encoder_budget"],
    )

    command = [
        str(binary),
        "serve",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        "262143",
        "--cache-capacity",
        "262144",
        "--vision-attention-image",
        str(attention_image),
        "--allowed-local-media-path",
        str(fixture_root),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--request-timeout-ms",
        str(int(args.request_timeout_seconds * 1000)),
        "--report",
        str(server_load_report),
    ]
    environment = {
        "HOME": str(isolated_home),
        "LANG": "C",
        "PATH": "/usr/bin:/bin",
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint = f"http://{args.host}:{args.port}"
    api_probe = load_api_probe_module()
    api_probe.urllib.request.install_opener(opener)
    process: subprocess.Popen[bytes] | None = None
    health: dict[str, Any] = {}
    observations: list[dict[str, Any]] = []
    try:
        with server_stdout.open("wb") as stdout, server_stderr.open("wb") as stderr:
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, env=environment
            )
            health = wait_ready(
                opener,
                endpoint + "/health",
                process,
                args.ready_timeout_seconds,
            )
            for spec in specs:
                print(f"PROBE {spec['probe_id']}", flush=True)
                result = api_probe.execute_case(
                    endpoint,
                    case_id=spec["probe_id"],
                    surfaces=spec["surfaces"],
                    expected_accept=spec["expected_accept"],
                    payload=spec["payload"],
                    replacements=spec["replacements"],
                    timeout=args.client_timeout_seconds,
                    response_redactions={
                        str(fixture_root): "${AIMA_VL_ENVELOPE_FIXTURE_ROOT}"
                    },
                )
                result["cell_id"] = spec["cell_id"]
                result["expected"] = spec["expected"]
                result["qualification_checks"] = validate_http_observation(
                    result, spec["expected"]
                )
                result["qualified"] = all(
                    result["qualification_checks"].values()
                )
                observations.append(result)
                print(
                    f"RESULT {spec['probe_id']} status={result['status_code']} "
                    f"qualified={result['qualified']}",
                    flush=True,
                )
            atomic_json(raw_cases_path, {"observations": observations})
            post_shutdown(opener, endpoint)
            process.wait(timeout=60)
            if process.returncode != 0:
                raise RuntimeError(
                    f"native server exited with code {process.returncode}"
                )
    finally:
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

    events = parse_server_events(server_stdout)
    ready_events = [item for item in events if item.get("event") == "ready"]
    stopped_events = [item for item in events if item.get("event") == "stopped"]
    ready = ready_events[0] if len(ready_events) == 1 else {}
    stopped = stopped_events[0] if len(stopped_events) == 1 else {}
    successes = [item for item in observations if item["accepted"]]
    errors = [item for item in observations if not item["accepted"]]
    request_indexes = [
        item.get("response", {}).get("aima_amd395", {}).get("request_index")
        for item in successes
    ]
    server_checks = {
        "one_ready_event": len(ready_events) == 1,
        "one_stopped_event": len(stopped_events) == 1,
        "seventeen_successful_requests_served": stopped.get("served") == 17,
        "request_indexes_contiguous": request_indexes == list(range(1, 18)),
        "one_model_load": stopped.get("model_loads") == 1,
        "native_only": all(
            ready.get(f"runtime_{name}") is False
            for name in ("python", "torch", "vllm", "triton")
        ),
        "visual_weights_resident": ready.get("visual_model_tensor_count") == 333
        and ready.get("visual_model_payload_bytes") == 893_142_496,
        "vl_ready": ready.get("native_vl") is True,
        "full_window_admitted": ready.get("context_capacity") == 262_144
        and ready.get("static_prefill_tokens") == 262_143,
        "automatic_long_context_fmha_policy": (
            ready.get("fmha_provider_path")
            == str(long_context_fmha_provider)
            and ready.get("secondary_fmha_provider_path")
            == str(fmha_provider)
        ),
    }
    actual_cell_observations = Counter(
        item["cell_id"] for item in observations
    )
    expected_cell_observations = Counter(spec["cell_id"] for spec in specs)
    matrix_complete = (
        len(observations) == 23
        and len(successes) == 17
        and len(errors) == 6
        and all(item["qualified"] for item in observations)
        and all(processor_checks.values())
        and all(vision_checks.values())
        and all(server_checks.values())
        and actual_cell_observations == expected_cell_observations
    )

    replacements = [
        (str(ROOT), "${AIMA_REPO_ROOT}"),
        (str(model_dir), "${AIMA_MODEL_DIR}"),
        (str(fixture_root), "${AIMA_VL_ENVELOPE_FIXTURE_ROOT}"),
        (
            str(long_context_fmha_provider),
            "${AIMA_LONG_CONTEXT_FMHA_PROVIDER}",
        ),
        (str(fmha_provider), "${AIMA_FMHA_PROVIDER}"),
        (str(attention_image), "${AIMA_VISION_ATTENTION_IMAGE}"),
        (str(binary.parent), "${AIMA_NATIVE_BUILD_DIR}"),
        (str(raw_root), "${AIMA_QUALIFICATION_RAW_DIR}"),
    ]
    payload = {
        "schema": QUALIFICATION_SCHEMA,
        "captured_at": utc_now(),
        "complete": matrix_complete,
        "qualified": matrix_complete,
        "scope": "resident-native-frozen-23-cell-vl-execution-envelope",
        "host": {"label": "amd395", "hostname": socket.gethostname()},
        "source": {**source, "files": source_components()},
        "binary": file_component(binary, "build/native/aima-engine-native"),
        "processor_probe_binary": file_component(
            processor_probe_binary, "build/native_vl_processor_test"
        ),
        "vision_probe_binary": file_component(
            vision_probe_binary,
            "build/native/native-vl-envelope-vision-probe",
        ),
        "build_info": build_info,
        "dependencies": {
            "capability_envelope": file_component(
                envelope_path,
                "benchmarks/results/vl-capability-envelope-v0.1.0.json",
            ),
            "fixture_manifest": file_component(
                fixture_manifest_path,
                "benchmarks/fixtures/vl-envelope-v0.1.0/fixtures-manifest.json",
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
            "long_context_fmha_provider": file_component(
                long_context_fmha_provider,
                "build/native/libaima-fmha-ck.so",
            ),
            "vision_attention_image": file_component(
                attention_image, "build/native/aima-vision-attention.hsaco"
            ),
        },
        "execution_plan": {
            "cells": len(coverage),
            "http_observations": len(specs),
            "client_timeout_seconds": args.client_timeout_seconds,
            "modes": coverage,
            "non_http_modes": NON_HTTP_CELL_MODES,
        },
        "processor_probe": {
            "command": publicize(processor_command, replacements),
            "checks": processor_checks,
        },
        "vision_probe": {
            "command": publicize(vision_command, replacements),
            "result": vision_result,
            "checks": vision_checks,
        },
        "server": {
            "command": publicize(command, replacements),
            "environment_keys": sorted(environment),
            "health": health,
            "ready": publicize(ready, replacements),
            "stopped": stopped,
            "checks": server_checks,
        },
        "matrix": {
            "required_cells": len(coverage),
            "http_observations": len(observations),
            "successful_observations": len(successes),
            "error_observations": len(errors),
            "qualified_observations": sum(
                item["qualified"] for item in observations
            ),
            "observations": observations,
        },
        "raw": {
            "processor_stdout": file_component(
                processor_stdout,
                f"{raw_root.name}/processor-probe.stdout.log",
            ),
            "processor_stderr": file_component(
                processor_stderr,
                f"{raw_root.name}/processor-probe.stderr.log",
            ),
            "vision_stdout": file_component(
                vision_stdout, f"{raw_root.name}/vision-probe.stdout.json"
            ),
            "vision_stderr": file_component(
                vision_stderr, f"{raw_root.name}/vision-probe.stderr.log"
            ),
            "vision_weight_load": file_component(
                vision_load_report, f"{raw_root.name}/vision-weight-load.json"
            ),
            "server_stdout": file_component(
                server_stdout, f"{raw_root.name}/server.stdout.log"
            ),
            "server_stderr": file_component(
                server_stderr, f"{raw_root.name}/server.stderr.log"
            ),
            "server_weight_load": file_component(
                server_load_report, f"{raw_root.name}/server-weight-load.json"
            ),
            "http_observations": file_component(
                raw_cases_path, f"{raw_root.name}/http-observations.json"
            ),
        },
        "decision": {
            "native_execution_qualification_complete": matrix_complete,
            "execution_cells_23_of_23": matrix_complete,
            "http_observations_23_of_23": len(observations) == 23
            and all(item["qualified"] for item in observations),
            "processor_option_boundary_qualified": all(
                processor_checks.values()
            ),
            "full_encoder_budget_vision_qualified": all(
                vision_checks.values()
            ),
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
