#!/usr/bin/env python3
"""Qualify native image-IO and media error/limit parity against vLLM."""

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
from aima_engine.vl_error_limits import (  # noqa: E402
    NATIVE_COMPATIBLE_ERROR,
    NATIVE_REPLAY,
    REFERENCE_CASE_ORDER,
    REFERENCE_ERROR_CONTRACT,
    build_reference_cases,
)
from aima_engine.vl_error_media_server import ErrorLimitMediaServer  # noqa: E402
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
from aima_engine.vl_transport_cache import (  # noqa: E402
    error_signature,
    normalize_contract_request,
)


MODEL_ID = "aima-amd395-qwen36-35b"
REFERENCE_SCHEMA = "aima-amd395-qwen36/vl-error-limits-reference/v1"
VISION_ATTENTION_SHA256 = (
    "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e"
)
PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
CAPABILITY_QUALIFIER = ROOT / "scripts/qualify-native-vl-capabilities.py"
TRANSPORT_QUALIFIER = ROOT / "scripts/qualify-native-vl-transport-cache.py"
CAPTURE_SCRIPT = ROOT / "scripts/capture-vllm-vl-error-limits.py"


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
        errors.append("error/limit reference schema changed")
    if reference.get("complete") is not True or reference.get("qualified") is not True:
        errors.append("error/limit reference is incomplete")
    cases = reference.get("cases")
    if not isinstance(cases, list) or tuple(
        item.get("case_id") for item in cases if isinstance(item, dict)
    ) != REFERENCE_CASE_ORDER:
        errors.append("error/limit reference case order changed")
    if isinstance(cases, list) and not all(
        isinstance(item, dict) and item.get("qualified") is True for item in cases
    ):
        errors.append("error/limit reference contains a failed case")
    source = reference.get("source")
    if not isinstance(source, dict) or source.get("dirty") is not False:
        errors.append("error/limit reference source identity is not clean")
    components = source.get("files") if isinstance(source, dict) else None
    if not isinstance(components, list):
        errors.append("error/limit reference source files are missing")
    else:
        for component in components:
            if not isinstance(component, dict):
                errors.append("reference source binding is malformed")
                continue
            path = ROOT / str(component.get("path", ""))
            if (
                not path.is_file()
                or path.stat().st_size != component.get("bytes")
                or sha256_file(path) != component.get("sha256")
            ):
                errors.append(
                    "error/limit reference source changed: "
                    + str(component.get("path"))
                )
    return errors


def error_message(response: Any) -> str | None:
    if not isinstance(response, dict) or not isinstance(response.get("error"), dict):
        return None
    message = response["error"].get("message")
    return message if isinstance(message, str) and message else None


def rejected_case_checks(
    case: dict[str, Any], reference: dict[str, Any]
) -> dict[str, bool]:
    case_id = reference.get("case_id")
    expected_reference = REFERENCE_ERROR_CONTRACT.get(case_id)
    native_signature = error_signature(case.get("response"))
    reference_signature = error_signature(reference.get("response"))
    return {
        "surface_rejected": case.get("passed") is True,
        "reference_contract_exact": expected_reference is not None
        and (
            reference.get("status_code"),
            *(reference_signature or (None, None)),
        )
        == expected_reference[:3],
        "native_compatible_status": case.get("status_code")
        == NATIVE_COMPATIBLE_ERROR[0],
        "request_contract_exact": case.get("request") == reference.get("request"),
        "native_compatible_error_shape": native_signature
        == NATIVE_COMPATIBLE_ERROR[1:],
        "compatible_error_category": expected_reference is not None
        and isinstance(expected_reference[3], str)
        and bool(expected_reference[3]),
        "native_error_message_nonempty": error_message(case.get("response"))
        is not None,
        "reference_error_message_compatible": reference.get("status_code") == 500
        or error_message(reference.get("response")) is not None,
    }


def run_server(
    *,
    port: int,
    args: argparse.Namespace,
    source: dict[str, Any],
    references: dict[str, dict[str, Any]],
    probe: ModuleType,
    capability: ModuleType,
    transport: ModuleType,
    raw_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    isolated_home = raw_root / "home"
    isolated_home.mkdir()
    stdout_path = raw_root / "server.stdout.log"
    stderr_path = raw_root / "server.stderr.log"
    load_report = raw_root / "native-weight-load.json"
    raw_cases_path = raw_root / "cases.json"
    success_count = sum(
        references[reference_id]["status_code"] == 200
        for _, reference_id in NATIVE_REPLAY
    )
    command = [
        str(args.binary),
        "serve",
        "--model-dir",
        str(args.model_dir),
        "--context-tokens",
        "1024",
        "--cache-capacity",
        "2048",
        "--fmha-provider",
        str(args.fmha_provider),
        "--vision-attention-image",
        str(args.vision_attention_image),
        "--allowed-local-media-path",
        str(args.fixture_root),
        "--allowed-local-media-path",
        str(args.error_fixture_root),
        "--allowed-media-domain",
        "127.0.0.1",
        "--allowed-private-media-domain",
        "127.0.0.1",
        "--host",
        args.host,
        "--port",
        str(port),
        "--max-requests",
        str(success_count),
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
    endpoint = f"http://{args.host}:{port}"
    process: subprocess.Popen[bytes] | None = None
    health: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    media_statistics: dict[str, dict[str, int]] = {}
    try:
        with ErrorLimitMediaServer() as media_server:
            fixtures = probe.Fixtures(args.fixture_root, media_server.http_base)
            specs = {
                item["case_id"]: item
                for item in build_reference_cases(
                    fixtures,
                    args.error_fixture_root,
                    MODEL_ID,
                    media_server,
                )
            }
            response_redactions = {
                str(args.fixture_root): "${AIMA_VL_FIXTURE_ROOT}",
                str(args.error_fixture_root): "${AIMA_VL_ERROR_FIXTURE_ROOT}",
                media_server.http_base: "${AIMA_VL_ERROR_HTTP}",
                media_server.unreachable_base: "${AIMA_VL_UNREACHABLE_HTTP}",
            }
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command, stdout=stdout, stderr=stderr, env=environment
                )
                health = capability.wait_ready(
                    opener,
                    endpoint + "/health",
                    process,
                    args.ready_timeout_seconds,
                )
                for observation_id, reference_id in NATIVE_REPLAY:
                    spec = specs[reference_id]
                    result = probe.execute_case(
                        endpoint,
                        timeout=args.request_timeout_seconds,
                        response_redactions=response_redactions,
                        **spec,
                    )
                    transport_request_sha256 = result.pop("request_sha256")
                    result["request"] = normalize_contract_request(
                        result["request"]
                    )
                    result["request_sha256"] = canonical_json_sha256(
                        result["request"]
                    )
                    result["transport_request_sha256"] = transport_request_sha256
                    result["observation_id"] = observation_id
                    result["reference_case_id"] = reference_id
                    reference = references[reference_id]
                    if reference["status_code"] == 200:
                        metrics = capability.native_metrics(result)
                        checks = transport.successful_case_checks(
                            result, reference, metrics
                        )
                        result["cache"] = transport.cache_summary(metrics)
                    else:
                        checks = rejected_case_checks(result, reference)
                        result["cache"] = None
                    result["qualification_checks"] = checks
                    result["qualified"] = all(checks.values())
                    cases.append(result)
                    print(
                        json.dumps(
                            {
                                "event": "native_case_complete",
                                "observation_id": observation_id,
                                "qualified": result["qualified"],
                                "status_code": result["status_code"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                atomic_json(raw_cases_path, {"cases": cases})
                process.wait(timeout=60)
                if process.returncode != 0:
                    raise RuntimeError(
                        f"native error/limit server exited with code {process.returncode}"
                    )
            media_statistics = media_server.statistics
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
        "all_successes_served": stopped.get("served") == success_count,
        "one_model_load": stopped.get("model_loads") == 1,
        "native_only": all(
            ready.get(f"runtime_{runtime}") is False
            for runtime in ("python", "torch", "vllm", "triton")
        ),
        "visual_weights_resident": ready.get("visual_model_tensor_count") == 333
        and ready.get("visual_model_payload_bytes") == 893_142_496,
        "vl_ready": ready.get("native_vl") is True,
        "media_cache_enabled": ready.get("media_cache_capacity_bytes")
        == 4 * 1024 * 1024 * 1024,
        "stderr_empty": stderr_path.stat().st_size == 0,
    }
    replacements = [
        (str(ROOT), "${AIMA_REPO_ROOT}"),
        (str(args.model_dir), "${AIMA_MODEL_DIR}"),
        (str(args.fixture_root), "${AIMA_VL_FIXTURE_ROOT}"),
        (str(args.error_fixture_root), "${AIMA_VL_ERROR_FIXTURE_ROOT}"),
        (str(args.fmha_provider), "${AIMA_FMHA_PROVIDER}"),
        (str(args.vision_attention_image), "${AIMA_VISION_ATTENTION_IMAGE}"),
        (str(args.binary.parent), "${AIMA_NATIVE_BUILD_DIR}"),
        (str(raw_root), "${AIMA_QUALIFICATION_RAW_DIR}"),
    ]
    run = {
        "source_commit": source["commit"],
        "command": capability.publicize(command, replacements),
        "environment_keys": sorted(environment),
        "health": capability.publicize(health, replacements),
        "ready": capability.publicize(ready, replacements),
        "stopped": stopped,
        "checks": server_checks,
        "cases": cases,
        "raw": {
            "stdout": file_component(stdout_path, f"{raw_root.name}/server.stdout.log"),
            "stderr": file_component(stderr_path, f"{raw_root.name}/server.stderr.log"),
            "weight_load": file_component(
                load_report, f"{raw_root.name}/native-weight-load.json"
            ),
            "cases": file_component(raw_cases_path, f"{raw_root.name}/cases.json"),
        },
    }
    return run, media_statistics


def cache_correctness_checks(run: dict[str, Any]) -> dict[str, bool]:
    observations = {item["observation_id"]: item for item in run["cases"]}

    def cache(observation_id: str) -> dict[str, Any]:
        value = observations[observation_id].get("cache")
        if not isinstance(value, dict):
            raise RuntimeError("successful cache observation has no metrics")
        return value

    white = cache("rgba_default_cold")
    red = cache("rgba_red_miss")
    white_restored = cache("rgba_default_restored")
    video_default = cache("video_default_cold")
    video_empty = cache("video_empty_mapping_exact")
    video_restored = cache("video_default_restored")
    long_video = cache("video_long_duration")
    red_after = cache("rgba_red_after_errors")
    return {
        "rgba_default_and_red_both_miss": white["media_cache_misses"] == 1
        and red["media_cache_misses"] == 1
        and white["media_cache_hits"] == red["media_cache_hits"] == 0,
        "rgba_background_changes_prefix_identity": white["prefix_lookup"] == "miss"
        and red["prefix_lookup"] == "miss",
        "rgba_default_a_b_a_recovers": white_restored["media_cache_hits"] == 1
        and white_restored["prefix_lookup"] == "exact"
        and white_restored["output_token_ids_sha256"]
        == white["output_token_ids_sha256"],
        "empty_video_mapping_reuses_default_media": video_default[
            "media_cache_misses"
        ]
        == 1
        and video_empty["media_cache_hits"] == 1
        and video_empty["media_cache_misses"] == 0,
        "empty_video_mapping_reuses_default_prefix": video_empty[
            "prefix_lookup"
        ]
        == "exact"
        and video_empty["output_token_ids_sha256"]
        == video_default["output_token_ids_sha256"],
        "video_default_restored_exact": video_restored["media_cache_hits"] == 1
        and video_restored["prefix_lookup"] == "exact"
        and video_restored["output_token_ids_sha256"]
        == video_default["output_token_ids_sha256"],
        "long_duration_video_executes": long_video["media_cache_misses"] == 1
        and long_video["media_count"] == 1,
        "errors_do_not_pollute_media_cache": red_after["media_cache_entries"]
        == long_video["media_cache_entries"]
        and red_after["media_cache_hits"] == 1,
        "rgba_red_recovers_after_errors": red_after["prefix_lookup"] == "exact"
        and red_after["output_token_ids_sha256"]
        == red["output_token_ids_sha256"],
        "all_observations_reference_exact": all(
            item["qualified"] for item in run["cases"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--error-fixture-root", type=Path, required=True)
    parser.add_argument("--fmha-provider", type=Path, required=True)
    parser.add_argument("--vision-attention-image", type=Path, required=True)
    parser.add_argument("--media-io-oracle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18158)
    parser.add_argument("--ready-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    args = parser.parse_args()

    for name in (
        "binary",
        "model_dir",
        "fixture_root",
        "error_fixture_root",
        "fmha_provider",
        "vision_attention_image",
        "media_io_oracle",
        "reference",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    aotriton = require_aotriton_closure(args.fmha_provider)
    raw_root = args.output.parent / f"{args.output.stem}-raw"
    required = (
        args.binary,
        args.fmha_provider,
        args.vision_attention_image,
        args.media_io_oracle,
        args.fixture_root / "fixtures-manifest.json",
        args.error_fixture_root / "fixtures-manifest.json",
        args.reference,
        PROBE_SCRIPT,
        CAPABILITY_QUALIFIER,
        TRANSPORT_QUALIFIER,
        CAPTURE_SCRIPT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"native error/limit inputs are missing: {missing}")
    if not args.model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {args.model_dir}")
    if args.output.exists() or raw_root.exists():
        raise SystemExit("qualification output and raw directory must not exist")
    if args.port < 1024 or args.port > 65535:
        raise SystemExit("qualification needs a non-privileged port")
    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("native error/limit qualification requires clean source")
    build_info = json.loads(
        subprocess.run(
            [str(args.binary), "--build-info"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if build_info.get("source_commit") != source["commit"]:
        raise SystemExit("native binary source commit differs from checkout")
    if sha256_file(args.vision_attention_image) != VISION_ATTENTION_SHA256:
        raise SystemExit("vision-attention image differs from frozen artifact")

    reference = load_json_object(args.reference)
    errors = validate_reference(reference)
    oracle_binding = reference.get("bindings", {}).get("media_io_oracle")
    if (
        not isinstance(oracle_binding, dict)
        or oracle_binding.get("bytes") != args.media_io_oracle.stat().st_size
        or oracle_binding.get("sha256") != sha256_file(args.media_io_oracle)
    ):
        errors.append("media-IO oracle differs from the reference binding")
    if errors:
        raise SystemExit("invalid error/limit reference:\n- " + "\n- ".join(errors))
    references = {case["case_id"]: case for case in reference["cases"]}
    probe = load_module(PROBE_SCRIPT, "native_vl_error_limits_probe")
    capability = load_module(
        CAPABILITY_QUALIFIER, "native_vl_error_limits_capability"
    )
    transport = load_module(
        TRANSPORT_QUALIFIER, "native_vl_error_limits_transport"
    )
    raw_root.mkdir(parents=True)
    run, media_statistics = run_server(
        port=args.port,
        args=args,
        source=source,
        references=references,
        probe=probe,
        capability=capability,
        transport=transport,
        raw_root=raw_root,
    )

    cache_checks = cache_correctness_checks(run)
    server_checks = run["checks"]
    complete = all(cache_checks.values()) and all(server_checks.values())
    source_files = tuple(
        ROOT / path
        for path in (
            "native/include/aima/native_chat_protocol.h",
            "native/include/aima/native_image_decoder.h",
            "native/include/aima/native_media.h",
            "native/include/aima/native_video_decoder.h",
            "native/include/aima/native_vl_processor.h",
            "native/src/native_chat_protocol.cpp",
            "native/src/native_http_server.cpp",
            "native/src/native_image_decoder.cpp",
            "native/src/native_remote_media.cpp",
            "native/src/native_video_decoder.cpp",
            "native/src/native_vl_processor.cpp",
            "native/src/native_vl_request.cpp",
            "native/src/native_resident_engine.hip.cpp",
            "aima_engine/vl_error_limits.py",
            "aima_engine/aotriton_closure.py",
            "aima_engine/vl_error_media_server.py",
            "aima_engine/vl_transport_cache.py",
            "scripts/probe-vllm-vl-api-capabilities.py",
            "scripts/capture-vllm-vl-error-limits.py",
            "scripts/qualify-native-vl-capabilities.py",
            "scripts/qualify-native-vl-transport-cache.py",
            "scripts/qualify-native-vl-error-limits.py",
        )
    )
    payload = seal_manifest(
        {
            "schema": "aima-amd395-qwen36/native-vl-error-limits/v1",
            "captured_at": utc_now(),
            "complete": complete,
            "qualified": complete,
            "scope": "native-image-io-and-media-error-limit-parity",
            "host": {"label": "amd395", "hostname": socket.gethostname()},
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in source_files
                ],
            },
            "binary": file_component(args.binary, "build/native/aima-engine-native"),
            "build_info": build_info,
            "dependencies": {
                "reference": file_component(
                    args.reference,
                    "benchmarks/results/vl-error-limits-reference-v0.1.0.json",
                ),
                "media_io_oracle": file_component(
                    args.media_io_oracle,
                    "benchmarks/results/vl-media-io-reference-v0.1.0.json",
                ),
                "fixture_manifest": file_component(
                    args.fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
                ),
                "error_fixture_manifest": file_component(
                    args.error_fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-error-v0.1.0/fixtures-manifest.json",
                ),
                "fmha_provider": file_component(
                    args.fmha_provider, "build/native/libaima-fmha-aotriton.so"
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
                    args.vision_attention_image,
                    "build/native/aima-vision-attention.hsaco",
                ),
            },
            "media_server": media_statistics,
            "run": run,
            "qualification_checks": {
                "cache": cache_checks,
                "server": server_checks,
            },
            "decision": {
                "thirteen_observations_reference_exact": cache_checks[
                    "all_observations_reference_exact"
                ],
                "rgba_background_cache_identity_qualified": all(
                    cache_checks[name]
                    for name in (
                        "rgba_default_and_red_both_miss",
                        "rgba_background_changes_prefix_identity",
                        "rgba_default_a_b_a_recovers",
                        "rgba_red_recovers_after_errors",
                    )
                ),
                "empty_video_mapping_qualified": all(
                    cache_checks[name]
                    for name in (
                        "empty_video_mapping_reuses_default_media",
                        "empty_video_mapping_reuses_default_prefix",
                        "video_default_restored_exact",
                    )
                ),
                "long_duration_video_qualified": cache_checks[
                    "long_duration_video_executes"
                ],
                "error_limit_categories_qualified": all(
                    rejected_case_checks(item, references[item["reference_case_id"]])[
                        "compatible_error_category"
                    ]
                    for item in run["cases"]
                    if item["reference_case_id"] in REFERENCE_CASE_ORDER[5:]
                ),
                "error_cache_non_pollution_qualified": cache_checks[
                    "errors_do_not_pollute_media_cache"
                ],
                "one_resident_model_load": server_checks["one_model_load"],
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
    digest = atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "complete": complete,
                "output": str(args.output),
                "qualified": complete,
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
