#!/usr/bin/env python3
"""Qualify native HTTPS, video sampling and cache invariance against vLLM."""

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
from aima_engine.vl_local_media_server import LocalMediaServers  # noqa: E402
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
    DISABLED_REPLAY,
    ENABLED_REPLAY,
    REFERENCE_CASE_ORDER,
    build_reference_cases,
    error_signature,
    finish_reason,
    normalize_contract_request,
    request_media_counts,
    response_content,
    usage_signature,
)


MODEL_ID = "aima-amd395-qwen36-35b"
REFERENCE_SCHEMA = "aima-amd395-qwen36/vl-transport-cache-reference/v1"
VISION_ATTENTION_SHA256 = (
    "e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c"
)
PROBE_SCRIPT = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
CAPABILITY_QUALIFIER = ROOT / "scripts/qualify-native-vl-capabilities.py"
CAPTURE_SCRIPT = ROOT / "scripts/capture-vllm-vl-transport-cache.py"


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
        errors.append("transport/cache reference schema changed")
    if reference.get("complete") is not True or reference.get("qualified") is not True:
        errors.append("transport/cache reference is incomplete")
    cases = reference.get("cases")
    if not isinstance(cases, list) or tuple(
        item.get("case_id") for item in cases if isinstance(item, dict)
    ) != REFERENCE_CASE_ORDER:
        errors.append("transport/cache reference case order changed")
    if isinstance(cases, list) and not all(
        isinstance(item, dict) and item.get("qualified") is True
        for item in cases
    ):
        errors.append("transport/cache reference contains a failed case")
    source = reference.get("source")
    components = source.get("files") if isinstance(source, dict) else None
    if not isinstance(components, list):
        errors.append("transport/cache reference source files are missing")
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
                    "transport/cache reference source changed: "
                    + str(component.get("path"))
                )
    return errors


def successful_case_checks(
    case: dict[str, Any], reference: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, bool]:
    counts = request_media_counts(case["request"])
    vl = metrics.get("vl") if isinstance(metrics.get("vl"), dict) else {}
    mrope = metrics.get("mrope") if isinstance(metrics.get("mrope"), dict) else {}
    return {
        "surface_accepted": case.get("passed") is True,
        "reference_status_exact": case.get("status_code")
        == reference.get("status_code")
        == 200,
        "request_contract_exact": case.get("request") == reference.get("request"),
        "resident_native_request": metrics.get("model_loads") == 1
        and metrics.get("oracle_tensor_reads") == 0
        and str(metrics.get("runtime", "")).startswith("native-resident-q"),
        "media_counts_exact": vl.get("image_count") == counts["image"]
        and vl.get("video_count") == counts["video"]
        and vl.get("media_count") == counts["image"] + counts["video"],
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
        "finish_reason_exact": finish_reason(case["response"])
        == finish_reason(reference["response"]),
        "generated_content_exact": response_content(case["response"])
        == response_content(reference["response"]),
        "usage_exact": usage_signature(case["response"])
        == usage_signature(reference["response"]),
    }


def rejected_case_checks(
    case: dict[str, Any], reference: dict[str, Any]
) -> dict[str, bool]:
    return {
        "surface_rejected": case.get("passed") is True,
        "reference_status_exact": case.get("status_code")
        == reference.get("status_code")
        == 400,
        "request_contract_exact": case.get("request") == reference.get("request"),
        "structured_native_error": error_signature(case.get("response"))
        is not None,
        "structured_reference_error": error_signature(reference.get("response"))
        is not None,
    }


def cache_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    vl = metrics["vl"]
    prefix = metrics["prefix_cache"]
    return {
        "prompt_tokens": metrics["prompt_tokens"],
        "completion_tokens": metrics.get("completion_tokens"),
        "output_token_ids_sha256": metrics[
            "output_token_ids_canonical_sha256"
        ],
        "prefix_lookup": prefix["lookup"],
        "prefix_matched_tokens": prefix["matched_tokens"],
        "media_cache_hits": vl["media_cache_hits"],
        "media_cache_misses": vl["media_cache_misses"],
        "media_cache_entries": vl["media_cache_entries"],
        "media_cache_resident_bytes": vl["media_cache_resident_bytes"],
        "media_count": vl["media_count"],
        "image_count": vl["image_count"],
        "video_count": vl["video_count"],
        "media_decode_wall_ms": vl["media_decode_wall_ms"],
        "processor_wall_ms": vl["processor_wall_ms"],
        "vision_encode_wall_ms": vl["vision_encode_wall_ms"],
    }


def run_server(
    *,
    name: str,
    cache_enabled: bool,
    sequence: tuple[tuple[str, str, str], ...],
    port: int,
    args: argparse.Namespace,
    source: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    media_servers: LocalMediaServers,
    probe: ModuleType,
    capability: ModuleType,
    raw_root: Path,
) -> dict[str, Any]:
    run_root = raw_root / name
    run_root.mkdir()
    isolated_home = run_root / "home"
    isolated_home.mkdir()
    stdout_path = run_root / "server.stdout.log"
    stderr_path = run_root / "server.stderr.log"
    load_report = run_root / "native-weight-load.json"
    raw_cases_path = run_root / "cases.json"
    success_count = sum(
        references[reference_id]["status_code"] == 200
        for _, reference_id, _ in sequence
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
        "--allowed-media-domain",
        "127.0.0.1",
        "--allowed-private-media-domain",
        "127.0.0.1",
        "--remote-tls-ca-bundle",
        str(args.tls_certificate),
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
    if not cache_enabled:
        command.append("--disable-media-cache")
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
    try:
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
            for observation_id, reference_id, mode in sequence:
                media_servers.set_mode(mode)
                spec = specs[reference_id]
                result = probe.execute_case(
                    endpoint,
                    timeout=args.request_timeout_seconds,
                    response_redactions={
                        str(args.fixture_root): "${AIMA_VL_FIXTURE_ROOT}"
                    },
                    **{
                        key: value
                        for key, value in spec.items()
                        if key != "mutable_mode"
                    },
                )
                transport_request_sha256 = result.pop("request_sha256")
                result["request"] = normalize_contract_request(result["request"])
                result["request_sha256"] = canonical_json_sha256(result["request"])
                result["transport_request_sha256"] = transport_request_sha256
                result["observation_id"] = observation_id
                result["reference_case_id"] = reference_id
                result["cache_enabled"] = cache_enabled
                reference = references[reference_id]
                if reference["status_code"] == 200:
                    metrics = capability.native_metrics(result)
                    checks = successful_case_checks(result, reference, metrics)
                    result["cache"] = cache_summary(metrics)
                else:
                    checks = rejected_case_checks(result, reference)
                    result["cache"] = None
                result["qualification_checks"] = checks
                result["qualified"] = all(checks.values())
                cases.append(result)
                print(
                    json.dumps(
                        {
                            "cache_enabled": cache_enabled,
                            "event": "native_case_complete",
                            "observation_id": observation_id,
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
                    f"native {name} server exited with code {process.returncode}"
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
    expected_capacity = 4 * 1024 * 1024 * 1024 if cache_enabled else 0
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
        "media_cache_mode_exact": ready.get("media_cache_capacity_bytes")
        == expected_capacity,
        "stderr_empty": stderr_path.stat().st_size == 0,
    }
    replacements = [
        (str(ROOT), "${AIMA_REPO_ROOT}"),
        (str(args.model_dir), "${AIMA_MODEL_DIR}"),
        (str(args.fixture_root), "${AIMA_VL_FIXTURE_ROOT}"),
        (str(args.fmha_provider), "${AIMA_FMHA_PROVIDER}"),
        (str(args.vision_attention_image), "${AIMA_VISION_ATTENTION_IMAGE}"),
        (str(args.tls_certificate), "${AIMA_VL_TLS_CA_BUNDLE}"),
        (str(args.binary.parent), "${AIMA_NATIVE_BUILD_DIR}"),
        (str(raw_root), "${AIMA_QUALIFICATION_RAW_DIR}"),
    ]
    return {
        "name": name,
        "cache_enabled": cache_enabled,
        "source_commit": source["commit"],
        "command": capability.publicize(command, replacements),
        "environment_keys": sorted(environment),
        "health": capability.publicize(health, replacements),
        "ready": capability.publicize(ready, replacements),
        "stopped": stopped,
        "checks": server_checks,
        "cases": cases,
        "raw": {
            "stdout": file_component(
                stdout_path, f"{raw_root.name}/{name}/server.stdout.log"
            ),
            "stderr": file_component(
                stderr_path, f"{raw_root.name}/{name}/server.stderr.log"
            ),
            "weight_load": file_component(
                load_report, f"{raw_root.name}/{name}/native-weight-load.json"
            ),
            "cases": file_component(
                raw_cases_path, f"{raw_root.name}/{name}/cases.json"
            ),
        },
    }


def cache_correctness_checks(
    enabled: dict[str, Any], disabled: dict[str, Any]
) -> dict[str, bool]:
    on = {item["observation_id"]: item for item in enabled["cases"]}
    off = {item["observation_id"]: item for item in disabled["cases"]}

    def cache(observation: dict[str, Any]) -> dict[str, Any]:
        value = observation.get("cache")
        if not isinstance(value, dict):
            raise RuntimeError("successful cache observation has no metrics")
        return value

    on_https_cold = cache(on["https_image_cold"])
    on_https_exact = cache(on["https_image_exact"])
    on_a = cache(on["video_content_a_cold"])
    on_b = cache(on["video_content_b_miss"])
    on_a_restored = cache(on["video_content_a_restored"])
    on_a_after_error = cache(on["video_content_a_after_error"])
    on_default = cache(on["video_sampling_default"])
    on_default_exact = cache(on["video_sampling_default_exact"])
    on_fps = cache(on["video_sampling_fps_1"])
    on_default_restored = cache(on["video_sampling_default_restored"])
    on_frames = cache(on["video_sampling_num_frames_6"])
    on_frames_exact = cache(on["video_sampling_num_frames_6_exact"])
    on_mixed = cache(on["mixed_image_video"])
    on_reordered = cache(on["mixed_video_image_reordered"])
    on_mutated = cache(on["mixed_mutated_image_video"])
    on_mixed_restored = cache(on["mixed_image_video_restored"])
    disabled_successes = [
        item for item in disabled["cases"] if item.get("cache") is not None
    ]
    paired_outputs = all(
        cache(item)["output_token_ids_sha256"]
        == cache(
            next(
                candidate
                for candidate in enabled["cases"]
                if candidate["reference_case_id"] == item["reference_case_id"]
                and candidate.get("cache") is not None
            )
        )["output_token_ids_sha256"]
        for item in disabled_successes
    )
    return {
        "https_cold_then_exact_hit": on_https_cold["media_cache_misses"] == 1
        and on_https_exact["media_cache_hits"] == 1
        and on_https_exact["prefix_lookup"] == "exact",
        "https_output_exact": on_https_exact["output_token_ids_sha256"]
        == on_https_cold["output_token_ids_sha256"],
        "video_same_url_a_b_both_miss": on_a["media_cache_misses"] == 1
        and on_b["media_cache_misses"] == 1
        and on_a["media_cache_hits"] == on_b["media_cache_hits"] == 0,
        "video_same_url_a_b_prefix_miss": on_a["prefix_lookup"] == "miss"
        and on_b["prefix_lookup"] == "miss",
        "video_a_b_a_recovers_exact": on_a_restored["media_cache_hits"] == 1
        and on_a_restored["prefix_lookup"] == "exact"
        and on_a_restored["output_token_ids_sha256"]
        == on_a["output_token_ids_sha256"],
        "cache_error_does_not_pollute": on_a_after_error[
            "media_cache_entries"
        ]
        == on_a_restored["media_cache_entries"]
        and on_a_after_error["media_cache_hits"] == 1
        and on_a_after_error["prefix_lookup"] == "exact",
        "sampling_default_media_reuse": on_default["media_cache_hits"] == 1
        and on_default_exact["media_cache_hits"] == 1
        and on_default_exact["prefix_lookup"] == "exact",
        "sampling_fps_changes_identity": on_fps["media_cache_misses"] == 1
        and on_fps["media_cache_hits"] == 0
        and on_fps["prefix_lookup"] == "miss",
        "sampling_default_a_b_a_recovers": on_default_restored[
            "media_cache_hits"
        ]
        == 1
        and on_default_restored["prefix_lookup"] == "exact"
        and on_default_restored["output_token_ids_sha256"]
        == on_default["output_token_ids_sha256"],
        "sampling_num_frames_changes_identity": on_frames[
            "media_cache_misses"
        ]
        == 1
        and on_frames["media_cache_hits"] == 0
        and on_frames["prefix_lookup"] == "miss",
        "sampling_num_frames_exact_hit": on_frames_exact[
            "media_cache_hits"
        ]
        == 1
        and on_frames_exact["prefix_lookup"] == "exact"
        and on_frames_exact["output_token_ids_sha256"]
        == on_frames["output_token_ids_sha256"],
        "mixed_first_one_hit_one_miss": on_mixed["media_cache_hits"] == 1
        and on_mixed["media_cache_misses"] == 1
        and on_mixed["prefix_lookup"] == "miss",
        "mixed_reorder_conservative_prefix_miss": on_reordered[
            "media_cache_hits"
        ]
        == 2
        and on_reordered["prefix_lookup"] == "miss",
        "mixed_mutation_conservative_prefix_miss": on_mutated[
            "media_cache_hits"
        ]
        == 1
        and on_mutated["media_cache_misses"] == 1
        and on_mutated["prefix_lookup"] == "miss",
        "mixed_a_b_a_recovers_exact": on_mixed_restored[
            "media_cache_hits"
        ]
        == 2
        and on_mixed_restored["prefix_lookup"] == "exact"
        and on_mixed_restored["output_token_ids_sha256"]
        == on_mixed["output_token_ids_sha256"],
        "disabled_cache_always_misses": all(
            cache(item)["media_cache_hits"] == 0
            and cache(item)["media_cache_misses"]
            == cache(item)["media_count"]
            and cache(item)["media_cache_entries"] == 0
            and cache(item)["media_cache_resident_bytes"] == 0
            for item in disabled_successes
        ),
        "disabled_cache_outputs_exact": paired_outputs,
        "enabled_disabled_error_status_exact": on["video_content_error"]
        ["status_code"]
        == off["video_content_error_disabled"]["status_code"]
        == 400,
        "enabled_disabled_error_payload_exact": on["video_content_error"]
        ["response"]
        == off["video_content_error_disabled"]["response"],
        "all_observations_reference_exact": all(
            item["qualified"]
            for item in (*enabled["cases"], *disabled["cases"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--fmha-provider", type=Path, required=True)
    parser.add_argument("--vision-attention-image", type=Path, required=True)
    parser.add_argument("--tls-certificate", type=Path, required=True)
    parser.add_argument("--tls-private-key", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18156)
    parser.add_argument("--ready-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    for name in (
        "binary",
        "model_dir",
        "fixture_root",
        "fmha_provider",
        "vision_attention_image",
        "tls_certificate",
        "tls_private_key",
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
        args.fixture_root / "fixtures-manifest.json",
        args.tls_certificate,
        args.tls_private_key,
        args.reference,
        PROBE_SCRIPT,
        CAPABILITY_QUALIFIER,
        CAPTURE_SCRIPT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"native transport/cache inputs are missing: {missing}")
    if not args.model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {args.model_dir}")
    if args.output.exists() or raw_root.exists():
        raise SystemExit("qualification output and raw directory must not exist")
    if args.port < 1024 or args.port >= 65535:
        raise SystemExit("qualification needs two adjacent non-privileged ports")
    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("native transport/cache qualification requires clean source")
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
    if reference.get("runtime", {}).get("test_ca", {}).get(
        "sha256"
    ) != sha256_file(args.tls_certificate):
        errors.append("test CA differs from transport/cache reference")
    if errors:
        raise SystemExit(
            "invalid transport/cache reference:\n- " + "\n- ".join(errors)
        )
    references = {case["case_id"]: case for case in reference["cases"]}
    probe = load_module(PROBE_SCRIPT, "native_vl_transport_cache_probe")
    capability = load_module(
        CAPABILITY_QUALIFIER, "native_vl_transport_cache_helpers"
    )
    raw_root.mkdir(parents=True)
    with LocalMediaServers(
        args.fixture_root, args.tls_certificate, args.tls_private_key
    ) as media_servers:
        fixtures = probe.Fixtures(args.fixture_root, media_servers.http_base)
        spec_list = build_reference_cases(
            fixtures,
            MODEL_ID,
            media_servers.http_base,
            media_servers.https_base,
        )
        specs = {item["case_id"]: item for item in spec_list}
        enabled = run_server(
            name="cache-enabled",
            cache_enabled=True,
            sequence=ENABLED_REPLAY,
            port=args.port,
            args=args,
            source=source,
            specs=specs,
            references=references,
            media_servers=media_servers,
            probe=probe,
            capability=capability,
            raw_root=raw_root,
        )
        disabled = run_server(
            name="cache-disabled",
            cache_enabled=False,
            sequence=DISABLED_REPLAY,
            port=args.port + 1,
            args=args,
            source=source,
            specs=specs,
            references=references,
            media_servers=media_servers,
            probe=probe,
            capability=capability,
            raw_root=raw_root,
        )
        media_request_counts = media_servers.request_counts

    cache_checks = cache_correctness_checks(enabled, disabled)
    server_checks = {
        "enabled_server_qualified": all(enabled["checks"].values()),
        "disabled_server_qualified": all(disabled["checks"].values()),
        "two_total_model_loads": enabled["stopped"].get("model_loads") == 1
        and disabled["stopped"].get("model_loads") == 1,
        "verified_https_origin_used": media_request_counts["https"] >= 2,
    }
    complete = all(cache_checks.values()) and all(server_checks.values())
    source_files = tuple(
        ROOT / path
        for path in (
            "native/include/aima/native_chat_protocol.h",
            "native/include/aima/native_media.h",
            "native/include/aima/native_video_decoder.h",
            "native/include/aima/native_vl_processor.h",
            "native/include/aima/native_vl_request.h",
            "native/src/native_chat_protocol.cpp",
            "native/src/native_http_server.cpp",
            "native/src/native_video_decoder.cpp",
            "native/src/native_vl_processor.cpp",
            "native/src/native_vl_request.cpp",
            "native/src/native_resident_engine.hip.cpp",
            "aima_engine/vl_local_media_server.py",
            "aima_engine/aotriton_closure.py",
            "aima_engine/vl_transport_cache.py",
            "scripts/probe-vllm-vl-api-capabilities.py",
            "scripts/capture-vllm-vl-transport-cache.py",
            "scripts/qualify-native-vl-capabilities.py",
            "scripts/qualify-native-vl-transport-cache.py",
        )
    )
    payload = seal_manifest(
        {
            "schema": "aima-amd395-qwen36/native-vl-transport-cache/v1",
            "captured_at": utc_now(),
            "complete": complete,
            "qualified": complete,
            "scope": "paired-native-https-sampling-content-order-cache-invariance",
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
                    "benchmarks/results/vl-transport-cache-reference-v0.1.0.json",
                ),
                "fixture_manifest": file_component(
                    args.fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
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
                "test_ca": {
                    "bytes": args.tls_certificate.stat().st_size,
                    "sha256": sha256_file(args.tls_certificate),
                    "private_key_recorded": False,
                },
            },
            "media_server_request_counts": media_request_counts,
            "runs": {"cache_enabled": enabled, "cache_disabled": disabled},
            "qualification_checks": {
                "cache": cache_checks,
                "servers": server_checks,
            },
            "decision": {
                "all_observations_reference_exact": cache_checks[
                    "all_observations_reference_exact"
                ],
                "verified_https_qualified": server_checks[
                    "verified_https_origin_used"
                ],
                "video_sampling_cache_identity_qualified": all(
                    cache_checks[name]
                    for name in (
                        "sampling_fps_changes_identity",
                        "sampling_default_a_b_a_recovers",
                        "sampling_num_frames_changes_identity",
                        "sampling_num_frames_exact_hit",
                    )
                ),
                "video_content_a_b_a_qualified": cache_checks[
                    "video_a_b_a_recovers_exact"
                ],
                "mixed_order_mutation_qualified": all(
                    cache_checks[name]
                    for name in (
                        "mixed_reorder_conservative_prefix_miss",
                        "mixed_mutation_conservative_prefix_miss",
                        "mixed_a_b_a_recovers_exact",
                    )
                ),
                "long_generation_usage_qualified": cache_checks[
                    "all_observations_reference_exact"
                ],
                "cache_disabled_and_error_invariance_qualified": all(
                    cache_checks[name]
                    for name in (
                        "disabled_cache_always_misses",
                        "disabled_cache_outputs_exact",
                        "enabled_disabled_error_status_exact",
                        "enabled_disabled_error_payload_exact",
                    )
                ),
                "two_resident_model_loads": server_checks[
                    "two_total_model_loads"
                ],
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
