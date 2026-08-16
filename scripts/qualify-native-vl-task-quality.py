#!/usr/bin/env python3
"""Qualify native long greedy VL task quality against frozen vLLM."""

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
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    canonical_json_sha256,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
)
from aima_engine.vl_task_quality import (  # noqa: E402
    CASE_ORDER,
    NATIVE_SCHEMA,
    aggregate_scores,
    build_cases,
    finish_reason,
    normalize_contract_request,
    response_content,
    score_not_below,
    score_text,
    usage_signature,
    validate_fixture_manifest,
    validate_reference_manifest,
)


MODEL_ID = "aima-amd395-qwen36-35b"
VISION_ATTENTION_SHA256 = (
    "b709a058a77d61e14db73c1ff7d7f4c20859d997bec811cad7339d3e59223d00"
)
PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
CAPABILITY_QUALIFIER = ROOT / "scripts/qualify-native-vl-capabilities.py"
CAPTURE_SCRIPT = ROOT / "scripts/capture-vllm-vl-task-quality.py"


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


def validate_cli_contract(
    *,
    host: str,
    port: int,
    ready_timeout_seconds: float,
    request_timeout_seconds: float,
) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("native task-quality host must be loopback")
    if not 1024 <= port <= 65535:
        raise SystemExit("native task-quality port must be non-privileged")
    if not math.isfinite(ready_timeout_seconds) or ready_timeout_seconds <= 0:
        raise SystemExit("ready timeout must be a positive finite number")
    if (
        not math.isfinite(request_timeout_seconds)
        or request_timeout_seconds <= 0
        or request_timeout_seconds > 600
    ):
        raise SystemExit(
            "request timeout must be positive and cannot exceed 600 seconds"
        )


def validate_reference_bindings(
    reference: dict[str, Any], fixture_manifest_path: Path
) -> list[str]:
    errors = validate_reference_manifest(reference)
    source = reference.get("source")
    if not isinstance(source, dict) or source.get("dirty") is not False:
        errors.append("task-quality reference source is not clean")
        return errors
    components = source.get("files")
    if not isinstance(components, list) or not components:
        errors.append("task-quality reference source files are missing")
        return errors
    for component in components:
        if not isinstance(component, dict):
            errors.append("task-quality reference source binding is malformed")
            continue
        relative = component.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            errors.append("task-quality reference source path is invalid")
            continue
        path = ROOT / relative
        if (
            not path.is_file()
            or path.stat().st_size != component.get("bytes")
            or sha256_file(path) != component.get("sha256")
        ):
            errors.append(f"task-quality reference source changed: {relative}")
    bindings = reference.get("bindings")
    fixture_binding = (
        bindings.get("fixture_manifest")
        if isinstance(bindings, dict)
        else None
    )
    if (
        not isinstance(fixture_binding, dict)
        or fixture_binding.get("sha256") != sha256_file(fixture_manifest_path)
        or fixture_binding.get("bytes") != fixture_manifest_path.stat().st_size
    ):
        errors.append("task-quality reference fixture binding changed")
    return errors


def case_checks(
    case: dict[str, Any], reference: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, bool]:
    response = case.get("response")
    reference_response = reference.get("response")
    modality = reference.get("modality")
    vl = metrics.get("vl") if isinstance(metrics.get("vl"), dict) else {}
    mrope = (
        metrics.get("mrope")
        if isinstance(metrics.get("mrope"), dict)
        else {}
    )
    structured = (
        metrics.get("structured_decoding")
        if isinstance(metrics.get("structured_decoding"), dict)
        else {}
    )
    expected_image = int(modality == "image")
    expected_video = int(modality == "video")
    reference_score = reference.get("score")
    candidate_score = case.get("score")
    reference_usage = usage_signature(reference_response)
    return {
        "surface_accepted": case.get("passed") is True,
        "reference_status_exact": case.get("status_code")
        == reference.get("status_code")
        == 200,
        "request_contract_exact": case.get("request")
        == reference.get("request"),
        "resident_native_request": metrics.get("model_loads") == 1
        and metrics.get("oracle_tensor_reads") == 0
        and str(metrics.get("runtime", "")).startswith("native-resident-q"),
        "media_counts_exact": vl.get("image_count") == expected_image
        and vl.get("video_count") == expected_video
        and vl.get("media_count") == 1,
        "media_executed": vl.get("enabled") is True
        and isinstance(vl.get("vision_patches"), int)
        and vl["vision_patches"] > 0
        and isinstance(vl.get("visual_tokens"), int)
        and vl["visual_tokens"] > 0,
        "mrope_enabled": mrope.get("enabled") is True,
        "structured_decoding_disabled": structured.get("enabled") is False,
        "render_prompt_tokens_exact": metrics.get("prompt_tokens")
        == reference["render"]["prompt_tokens"],
        "render_prompt_token_ids_exact": metrics.get(
            "prompt_token_ids_sha256"
        )
        == reference["render"]["prompt_token_ids_sha256"],
        "finish_reason_exact": finish_reason(response)
        == finish_reason(reference_response),
        "generated_content_exact": response_content(response)
        == reference.get("output_text"),
        "usage_exact": usage_signature(response) == reference_usage,
        "completion_metric_exact": reference_usage is not None
        and metrics.get("completion_tokens") == reference_usage[1],
        "output_token_ids_exact": metrics.get(
            "output_token_ids_canonical_sha256"
        )
        == reference.get("output_token_ids_sha256"),
        "score_recomputed": candidate_score
        == score_text(
            response_content(response),
            reference.get("rubric", []),
        ),
        "task_quality_not_below_reference": isinstance(candidate_score, dict)
        and isinstance(reference_score, dict)
        and score_not_below(candidate_score, reference_score),
    }


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
    parser.add_argument("--port", type=int, default=18166)
    parser.add_argument("--ready-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    validate_cli_contract(
        host=args.host,
        port=args.port,
        ready_timeout_seconds=args.ready_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
    )

    binary = args.binary.resolve()
    model_dir = args.model_dir.resolve()
    fixture_root = args.fixture_root.resolve()
    fixture_manifest_path = fixture_root / "fixtures-manifest.json"
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
        fixture_manifest_path,
        reference_path,
        PROBE_SCRIPT,
        CAPABILITY_QUALIFIER,
        CAPTURE_SCRIPT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"native task-quality inputs are missing: {missing}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")
    if output.exists() or raw_root.exists():
        raise SystemExit("qualification output and raw directory must not exist")

    fixture_manifest = load_json_object(fixture_manifest_path)
    fixture_errors = validate_fixture_manifest(fixture_manifest, fixture_root)
    if fixture_errors:
        raise SystemExit(
            "invalid task-quality fixtures:\n- " + "\n- ".join(fixture_errors)
        )
    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("native task-quality qualification requires clean source")
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
    reference_errors = validate_reference_bindings(
        reference, fixture_manifest_path
    )
    if reference_errors:
        raise SystemExit(
            "invalid task-quality reference:\n- "
            + "\n- ".join(reference_errors)
        )
    references = {case["case_id"]: case for case in reference["cases"]}

    probe = load_module(PROBE_SCRIPT, "native_vl_task_quality_probe")
    capability = load_module(
        CAPABILITY_QUALIFIER, "native_vl_task_quality_helpers"
    )
    fixtures = probe.Fixtures(fixture_root, "http://127.0.0.1:1")
    specs = build_cases(fixtures, MODEL_ID)
    if tuple(spec["case_id"] for spec in specs) != CASE_ORDER:
        raise SystemExit("native task-quality probe case order changed")

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
                case_id = spec["case_id"]
                print(f"CASE {case_id}", flush=True)
                result = probe.execute_case(
                    endpoint,
                    case_id=case_id,
                    surfaces=spec["surfaces"],
                    expected_accept=spec["expected_accept"],
                    payload=spec["payload"],
                    replacements=spec["replacements"],
                    require_tool_call=spec["require_tool_call"],
                    timeout=args.request_timeout_seconds,
                    response_redactions={
                        str(fixture_root): "${AIMA_VL_TASK_FIXTURE_ROOT}"
                    },
                )
                transport_request_sha256 = result.pop("request_sha256")
                result["modality"] = spec["modality"]
                result["request"] = normalize_contract_request(
                    result["request"]
                )
                result["request_sha256"] = canonical_json_sha256(
                    result["request"]
                )
                result["transport_request_sha256"] = (
                    transport_request_sha256
                )
                reference_case = references[case_id]
                metrics = capability.native_metrics(result)
                content = response_content(result["response"])
                result["rubric"] = reference_case["rubric"]
                result["output_text"] = content
                result["output_token_ids_sha256"] = metrics.get(
                    "output_token_ids_canonical_sha256"
                )
                result["score"] = score_text(
                    content, reference_case["rubric"]
                )
                checks = case_checks(result, reference_case, metrics)
                result["qualification_checks"] = checks
                result["qualified"] = all(checks.values())
                cases.append(result)
                print(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "event": "native_task_quality_case_complete",
                            "qualified": result["qualified"],
                            "score_millionths": result["score"][
                                "score_millionths"
                            ],
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

    metrics_by_case = [capability.native_metrics(case) for case in cases]
    request_indexes = [metrics.get("request_index") for metrics in metrics_by_case]
    events = capability.parse_server_events(stdout_path)
    ready_events = [item for item in events if item.get("event") == "ready"]
    stopped_events = [item for item in events if item.get("event") == "stopped"]
    ready = ready_events[0] if len(ready_events) == 1 else {}
    stopped = stopped_events[0] if len(stopped_events) == 1 else {}
    server_checks = {
        "one_ready_event": len(ready_events) == 1,
        "one_stopped_event": len(stopped_events) == 1,
        "all_requests_served": stopped.get("served") == len(CASE_ORDER),
        "request_indexes_contiguous": request_indexes
        == list(range(1, len(CASE_ORDER) + 1)),
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
    aggregate = aggregate_scores(cases)
    reference_aggregate = reference["aggregate"]
    modality_non_regression = {
        modality: score_not_below(aggregate[modality], reference_aggregate[modality])
        for modality in ("image", "video")
    }
    complete = (
        tuple(case["case_id"] for case in cases) == CASE_ORDER
        and all(case["qualified"] for case in cases)
        and all(server_checks.values())
        and all(modality_non_regression.values())
    )

    replacements = [
        (str(ROOT), "${AIMA_REPO_ROOT}"),
        (str(model_dir), "${AIMA_MODEL_DIR}"),
        (str(fixture_root), "${AIMA_VL_TASK_FIXTURE_ROOT}"),
        (str(fmha_provider), "${AIMA_FMHA_PROVIDER}"),
        (str(vision_attention_image), "${AIMA_VISION_ATTENTION_IMAGE}"),
        (str(binary.parent), "${AIMA_NATIVE_BUILD_DIR}"),
        (str(raw_root), "${AIMA_QUALIFICATION_RAW_DIR}"),
    ]
    source_files = tuple(
        ROOT / relative
        for relative in (
            "native/include/aima/native_chat_protocol.h",
            "native/include/aima/native_decode_runner.h",
            "native/include/aima/native_resident_engine.h",
            "native/include/aima/native_tokenizer.h",
            "native/include/aima/native_vl_request.h",
            "native/src/native_chat_protocol.cpp",
            "native/src/native_decode_runner.hip.cpp",
            "native/src/native_resident_engine.hip.cpp",
            "native/src/native_tokenizer.cpp",
            "native/src/native_vl_request.cpp",
            "native/src/native_http_server.cpp",
            "aima_engine/aotriton_closure.py",
            "aima_engine/vl_task_quality.py",
            "scripts/probe-vllm-vl-api-capabilities.py",
            "scripts/qualify-native-vl-capabilities.py",
            "scripts/capture-vllm-vl-task-quality.py",
            "scripts/qualify-native-vl-task-quality.py",
        )
    )
    exact_content = sum(
        case["qualification_checks"]["generated_content_exact"] for case in cases
    )
    exact_output_tokens = sum(
        case["qualification_checks"]["output_token_ids_exact"] for case in cases
    )
    exact_prompts = sum(
        case["qualification_checks"]["render_prompt_token_ids_exact"]
        for case in cases
    )
    payload = seal_manifest(
        {
            "schema": NATIVE_SCHEMA,
            "captured_at": utc_now(),
            "complete": complete,
            "qualified": complete,
            "scope": "resident-native-twelve-case-long-greedy-vl-quality",
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
                    "benchmarks/results/vl-task-quality-reference-v0.1.0.json",
                ),
                "fixture_manifest": file_component(
                    fixture_manifest_path,
                    "benchmarks/fixtures/vl-task-quality-v0.1.0/"
                    "fixtures-manifest.json",
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
                "health": capability.publicize(health, replacements),
                "ready": capability.publicize(ready, replacements),
                "stopped": stopped,
                "checks": server_checks,
            },
            "matrix": {
                "required_cases": len(CASE_ORDER),
                "exact_generated_content": f"{exact_content}/{len(CASE_ORDER)}",
                "exact_output_token_vectors": (
                    f"{exact_output_tokens}/{len(CASE_ORDER)}"
                ),
                "exact_render_prompt_vectors": (
                    f"{exact_prompts}/{len(CASE_ORDER)}"
                ),
                "reference_aggregate": reference_aggregate,
                "native_aggregate": aggregate,
                "modality_non_regression": modality_non_regression,
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
                "twelve_long_greedy_cases_reference_exact": complete,
                "twelve_render_prompt_vectors_exact": exact_prompts
                == len(CASE_ORDER),
                "twelve_output_token_vectors_exact": exact_output_tokens
                == len(CASE_ORDER),
                "image_task_quality_not_below_reference": (
                    modality_non_regression["image"]
                ),
                "video_task_quality_not_below_reference": (
                    modality_non_regression["video"]
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
