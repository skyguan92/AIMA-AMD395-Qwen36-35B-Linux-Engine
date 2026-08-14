#!/usr/bin/env python3
"""Qualify the resident native VL HTTP path against frozen greedy oracles."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import zlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_oracle import validate_oracle_manifest  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_bytes,
    sha256_file,
)


MODEL_ID = "aima-amd395-qwen36-35b"
CASE_ORDER = (
    "image_local_png",
    "video_local_mp4",
    "multi_image",
    "multi_video",
    "mixed_image_video",
)
VISION_ATTENTION_SHA256 = (
    "b709a058a77d61e14db73c1ff7d7f4c20859d997bec811cad7339d3e59223d00"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def build_request(case: dict[str, Any], fixture_root: Path) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for item in case["request"]["content"]:
        kind = item["type"]
        if kind == "text":
            parts.append({"type": "text", "text": item["text"]})
            continue
        if kind not in {"image", "video"}:
            raise RuntimeError(f"unsupported frozen media kind: {kind}")
        fixture = fixture_root / item["fixture"]
        if (
            not fixture.is_file()
            or fixture.stat().st_size != int(item["bytes"])
            or sha256_file(fixture) != item["sha256"]
        ):
            raise RuntimeError(f"frozen media fixture changed: {fixture.name}")
        field = f"{kind}_url"
        parts.append(
            {
                "type": field,
                field: {"url": fixture.resolve().as_uri()},
            }
        )
    return {
        "model": MODEL_ID,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "max_tokens": 8,
        "messages": [{"role": case["request"]["role"], "content": parts}],
    }


def image_request(source: str, text: str) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "max_tokens": 1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": source}},
                ],
            }
        ],
    }


def write_cache_variant_png(path: Path) -> None:
    """Write a deterministic RGB PNG with A's 160x320 dimensions."""

    width = 160
    height = 320
    row = bytearray([0])
    for x in range(width):
        row.extend(((x * 3) & 0xFF, (255 - x) & 0xFF, (x * 7) & 0xFF))
    pixels = bytes(row) * height

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + chunk(b"IDAT", zlib.compress(pixels, level=9))
        + chunk(b"IEND", b"")
    )


def post_json(
    opener: urllib.request.OpenerDirector,
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> tuple[int, dict[str, Any], float]:
    wire = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=wire,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read()
    elapsed_ms = (time.monotonic() - started) * 1000.0
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError("native HTTP response is not a JSON object")
    return status, value, elapsed_ms


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


def oracle_case_result(
    case: dict[str, Any], status: int, response: dict[str, Any], wall_ms: float
) -> dict[str, Any]:
    generation = case["generation"]
    processor = case["processor"]
    metrics = response.get("aima_amd395", {})
    usage = response.get("usage", {})
    choices = response.get("choices", [])
    choice = choices[0] if len(choices) == 1 else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    output_text_sha256 = (
        sha256_bytes(content.encode("utf-8")) if isinstance(content, str) else None
    )
    vl = metrics.get("vl", {}) if isinstance(metrics, dict) else {}
    mrope = metrics.get("mrope", {}) if isinstance(metrics, dict) else {}
    prompt_ids = processor["prompt_token_ids"]
    expected_visual_tokens = sum(
        int(span["num_embeds"])
        for spans in processor["placeholders"].values()
        for span in spans
    )
    expected_patches = sum(
        int(tensor["shape"][0])
        for name, tensor in processor["tensors"].items()
        if name in {"pixel_values", "pixel_values_videos"}
    )
    expected_mrope_delta = case["boundaries"]["mrope_positions"][
        "position_delta"
    ]
    checks = {
        "http_200": status == 200,
        "prompt_tokens_exact": usage.get("prompt_tokens") == len(prompt_ids),
        "prompt_token_ids_sha256_exact": metrics.get(
            "prompt_token_ids_sha256"
        )
        == processor["prompt_token_ids_sha256"],
        "completion_tokens_exact": usage.get("completion_tokens")
        == generation["completion_tokens"],
        "output_token_ids_sha256_exact": metrics.get(
            "output_token_ids_canonical_sha256"
        )
        == generation["output_token_ids_sha256"],
        "output_text_sha256_exact": output_text_sha256
        == generation["output_text_sha256"],
        "finish_reason_exact": choice.get("finish_reason")
        == generation["finish_reason"],
        "vision_shape_exact": vl.get("vision_patches") == expected_patches
        and vl.get("visual_tokens") == expected_visual_tokens,
        "mrope_exact": mrope.get("enabled") is True
        and mrope.get("position_delta") == expected_mrope_delta,
        "resident_native_execution": metrics.get("model_loads") == 1
        and metrics.get("oracle_tensor_reads") == 0
        and vl.get("enabled") is True,
    }
    return {
        "case_id": case["case_id"],
        "passed": all(checks.values()),
        "checks": checks,
        "prompt_tokens": usage.get("prompt_tokens"),
        "prompt_token_ids_sha256": metrics.get("prompt_token_ids_sha256"),
        "completion_tokens": usage.get("completion_tokens"),
        "output_token_ids_sha256": metrics.get(
            "output_token_ids_canonical_sha256"
        ),
        "output_text_sha256": output_text_sha256,
        "finish_reason": choice.get("finish_reason"),
        "request_wall_ms": wall_ms,
        "native_metrics": metrics,
    }


def cache_observation(response: dict[str, Any]) -> dict[str, Any]:
    metrics = response["aima_amd395"]
    return {
        "content": response["choices"][0]["message"]["content"],
        "output_token_ids_sha256": metrics[
            "output_token_ids_canonical_sha256"
        ],
        "prefix_lookup": metrics["prefix_cache"]["lookup"],
        "vl": metrics["vl"],
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--fmha-provider", type=Path, required=True)
    parser.add_argument(
        "--vision-attention-image", type=Path, required=True
    )
    parser.add_argument(
        "--oracle-manifest",
        type=Path,
        default=ROOT / "benchmarks/results/vl-oracle-manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--ready-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    binary = args.binary.resolve()
    model_dir = args.model_dir.resolve()
    fixture_root = args.fixture_root.resolve()
    fmha_provider = args.fmha_provider.resolve()
    vision_attention_image = args.vision_attention_image.resolve()
    oracle_path = args.oracle_manifest.resolve()
    output = args.output.resolve()
    raw_root = output.parent / f"{output.stem}-raw"
    required_paths = (
        binary,
        fmha_provider,
        vision_attention_image,
        oracle_path,
        fixture_root / "fixtures-manifest.json",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"native VL qualification inputs are missing: {missing}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")
    if output.exists() or raw_root.exists():
        raise SystemExit("qualification output and raw directory must not exist")

    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("native VL serving qualification requires clean source")
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

    oracle = load_json_object(oracle_path)
    oracle_errors = validate_oracle_manifest(oracle)
    if oracle_errors:
        raise SystemExit("invalid VL oracle:\n- " + "\n- ".join(oracle_errors))
    cases_by_id = {case["case_id"]: case for case in oracle["cases"]}
    if tuple(case_id for case_id in CASE_ORDER if case_id in cases_by_id) != CASE_ORDER:
        raise SystemExit("VL oracle case order is incomplete")

    raw_root.mkdir(parents=True)
    cache_media_root = raw_root / "cache-media"
    cache_media_root.mkdir()
    isolated_home = raw_root / "home"
    isolated_home.mkdir()
    mutable_image = cache_media_root / "mutable.png"
    cache_a = fixture_root / "image-transparent-160x320.png"
    cache_b = cache_media_root / "alternate-160x320.png"
    write_cache_variant_png(cache_b)
    shutil.copyfile(cache_a, mutable_image)

    stdout_path = raw_root / "server.stdout.log"
    stderr_path = raw_root / "server.stderr.log"
    load_report = raw_root / "native-weight-load.json"
    request_count = len(CASE_ORDER) + 5
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
        "--allowed-local-media-path",
        str(cache_media_root),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-requests",
        str(request_count),
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
    base_url = f"http://{args.host}:{args.port}"
    process: subprocess.Popen[bytes] | None = None
    oracle_results: list[dict[str, Any]] = []
    cache_result: dict[str, Any] = {}
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
                base_url + "/health",
                process,
                args.ready_timeout_seconds,
            )
            for case_id in CASE_ORDER:
                case = cases_by_id[case_id]
                status, response, wall_ms = post_json(
                    opener,
                    base_url + "/v1/chat/completions",
                    build_request(case, fixture_root),
                    timeout=args.request_timeout_seconds,
                )
                result = oracle_case_result(case, status, response, wall_ms)
                oracle_results.append(result)
                print(
                    json.dumps(
                        {
                            "event": "oracle_case_complete",
                            "case_id": case_id,
                            "passed": result["passed"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            cache_prompt = "Identify the visual content."
            mutable_uri = mutable_image.resolve().as_uri()
            cache_observations: list[dict[str, Any]] = []
            for source_value, text in (
                (mutable_uri, cache_prompt),
                (mutable_uri, cache_prompt),
                (mutable_uri, cache_prompt),
                (
                    "data:image/png;base64,"
                    + base64.b64encode(cache_a.read_bytes()).decode("ascii"),
                    cache_prompt,
                ),
                (
                    "data:image/png;base64,"
                    + base64.b64encode(cache_a.read_bytes()).decode("ascii"),
                    "Identify the visual content again.",
                ),
            ):
                index = len(cache_observations)
                if index == 1:
                    shutil.copyfile(cache_b, mutable_image)
                elif index == 2:
                    shutil.copyfile(cache_a, mutable_image)
                status, response, _wall_ms = post_json(
                    opener,
                    base_url + "/v1/chat/completions",
                    image_request(source_value, text),
                    timeout=args.request_timeout_seconds,
                )
                if status != 200:
                    raise RuntimeError(
                        f"cache qualification request {index} failed: {response}"
                    )
                cache_observations.append(cache_observation(response))

            first, changed, restored, equivalent, variant = cache_observations
            cache_checks = {
                "first_a_processor_miss": first["vl"]["media_cache_misses"] == 1,
                "same_path_b_processor_miss": changed["vl"]["media_cache_misses"]
                == 1
                and changed["vl"]["media_cache_hits"] == 0,
                "same_path_b_prefix_miss": changed["prefix_lookup"] == "miss"
                and changed["vl"]["vision_plan_cache_hit"] is True
                and changed["vl"]["vision_encode_wall_ms"] > 0.0,
                "restored_a_media_hit": restored["vl"]["media_cache_hits"] == 1,
                "restored_a_exact_prefix_hit": restored["prefix_lookup"] == "exact",
                "restored_a_output_exact": restored["output_token_ids_sha256"]
                == first["output_token_ids_sha256"],
                "data_local_equivalent_hit": equivalent["vl"]["media_cache_hits"]
                == 1
                and equivalent["prefix_lookup"] == "exact",
                "data_local_output_exact": equivalent["output_token_ids_sha256"]
                == first["output_token_ids_sha256"],
                "variant_reuses_processed_media": variant["vl"]["media_cache_hits"]
                == 1
                and variant["vl"]["media_decode_wall_ms"] == 0.0
                and variant["vl"]["processor_wall_ms"] == 0.0,
                "variant_reuses_shape_plan_only": variant["prefix_lookup"] == "miss"
                and variant["vl"]["vision_plan_cache_hit"] is True
                and variant["vl"]["vision_encode_wall_ms"] > 0.0,
            }
            cache_result = {
                "passed": all(cache_checks.values()),
                "checks": cache_checks,
                "inputs": {
                    "a": file_component(
                        cache_a,
                        "benchmarks/fixtures/vl-capability-v0.1.0/"
                        "image-transparent-160x320.png",
                    ),
                    "b": file_component(
                        cache_b,
                        f"{raw_root.name}/cache-media/{cache_b.name}",
                    ),
                    "mutable_final": file_component(
                        mutable_image,
                        f"{raw_root.name}/cache-media/{mutable_image.name}",
                    ),
                },
                "observations": cache_observations,
            }
            process.wait(timeout=60)
            if process.returncode != 0:
                raise RuntimeError(
                    f"native server exited with code {process.returncode}"
                )
    finally:
        if process is not None and process.poll() is None:
            try:
                post_json(
                    opener,
                    base_url + "/shutdown",
                    {},
                    timeout=2.0,
                )
                process.wait(timeout=10)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    events = parse_server_events(stdout_path)
    ready_events = [item for item in events if item.get("event") == "ready"]
    stopped_events = [item for item in events if item.get("event") == "stopped"]
    ready = ready_events[0] if len(ready_events) == 1 else {}
    stopped = stopped_events[0] if len(stopped_events) == 1 else {}
    server_checks = {
        "one_ready_event": len(ready_events) == 1,
        "one_stopped_event": len(stopped_events) == 1,
        "all_requests_served": stopped.get("served") == request_count,
        "one_model_load": stopped.get("model_loads") == 1,
        "native_only": all(
            ready.get(f"runtime_{name}") is False
            for name in ("python", "torch", "vllm", "triton")
        ),
        "visual_weights_resident": ready.get("visual_model_tensor_count") == 333
        and ready.get("visual_model_payload_bytes") == 893_142_496,
        "vl_ready": ready.get("native_vl") is True,
    }
    replacements = [
        (str(ROOT), "${AIMA_REPO_ROOT}"),
        (str(model_dir), "${AIMA_MODEL_DIR}"),
        (str(fixture_root), "${AIMA_VL_FIXTURE_ROOT}"),
        (str(binary.parent), "${AIMA_NATIVE_BUILD_DIR}"),
        (str(raw_root), "${AIMA_QUALIFICATION_RAW_DIR}"),
    ]
    source_files = tuple(
        ROOT / relative
        for relative in (
            "native/include/aima/native_vl_request.h",
            "native/src/native_vl_request.cpp",
            "native/include/aima/native_resident_engine.h",
            "native/src/native_resident_engine.hip.cpp",
            "native/src/native_http_server.cpp",
            "scripts/build-native-runtime.sh",
            "scripts/qualify-native-vl-serving.py",
        )
    )
    complete = (
        all(result["passed"] for result in oracle_results)
        and cache_result.get("passed") is True
        and all(server_checks.values())
    )
    payload = {
        "schema": "aima-amd395-qwen36/native-vl-serving-qualification/v1",
        "captured_at": utc_now(),
        "complete": complete,
        "qualified": complete,
        "scope": (
            "resident-native-http-five-frozen-greedy-oracles-and-"
            "content-addressed-cache-correctness"
        ),
        "host": {
            "label": "amd395",
            "hostname": socket.gethostname(),
        },
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
            "oracle_manifest": file_component(
                oracle_path, "benchmarks/results/vl-oracle-manifest.json"
            ),
            "fixture_manifest": file_component(
                fixture_root / "fixtures-manifest.json",
                "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
            ),
            "fmha_provider": file_component(
                fmha_provider, "build/native/libaima-fmha-aotriton.so"
            ),
            "vision_attention_image": file_component(
                vision_attention_image, "build/native/aima-vision-attention.hsaco"
            ),
        },
        "launch": {
            "command": publicize(command, replacements),
            "environment_keys": sorted(environment),
            "health": health,
            "ready": publicize(ready, replacements),
            "stopped": stopped,
            "checks": server_checks,
        },
        "oracle_cases": oracle_results,
        "cache_correctness": cache_result,
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
        },
        "decision": {
            "five_frozen_prompt_hashes_exact": all(
                item["checks"]["prompt_token_ids_sha256_exact"]
                for item in oracle_results
            ),
            "five_frozen_generations_exact": all(
                item["checks"]["output_token_ids_sha256_exact"]
                and item["checks"]["output_text_sha256_exact"]
                for item in oracle_results
            ),
            "content_addressed_media_cache_qualified": cache_result.get(
                "passed"
            )
            is True,
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
                "complete": complete,
                "qualified": complete,
                "output": str(output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
