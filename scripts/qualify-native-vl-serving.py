#!/usr/bin/env python3
"""Qualify the resident native VL HTTP path against frozen greedy oracles."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from functools import partial
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request
import zlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.aotriton_closure import (  # noqa: E402
    require_aotriton_closure,
)
from aima_engine.vl_oracle import validate_oracle_manifest  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    canonical_json_sha256,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_bytes,
    sha256_file,
)
from aima_engine.vl_serving_render import (  # noqa: E402
    SERVING_RENDER_CASES,
    validate_serving_render_manifest,
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
    "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e"
)
SERVING_RENDER_MANIFEST = (
    ROOT / "benchmarks/results/vl-serving-render-manifest-v0.1.0.json"
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


def media_request(
    media: list[tuple[str, str]], text: str
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for kind, source in media:
        if kind not in {"image", "video"}:
            raise ValueError(f"unsupported cache media kind: {kind}")
        field = f"{kind}_url"
        content.append({"type": field, field: {"url": source}})
    return {
        "model": MODEL_ID,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "max_tokens": 1,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }


def image_request(source: str, text: str) -> dict[str, Any]:
    return media_request([("image", source)], text)


def video_request(source: str, text: str) -> dict[str, Any]:
    return media_request([("video", source)], text)


def mixed_request(image: str, video: str, text: str) -> dict[str, Any]:
    return media_request([("image", image), ("video", video)], text)


def write_cache_variant_png(path: Path, phase: int = 0) -> None:
    """Write a deterministic RGB PNG with A's 160x320 dimensions."""

    if phase < 0 or phase > 255:
        raise ValueError("cache image phase must fit one byte")
    width = 160
    height = 320
    row = bytearray([0])
    for x in range(width):
        row.extend(
            (
                (x * 3 + phase * 17) & 0xFF,
                (255 - x + phase * 29) & 0xFF,
                (x * 7 + phase * 43) & 0xFF,
            )
        )
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


def request_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize the serving request exactly as the qualification client does."""

    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def post_json(
    opener: urllib.request.OpenerDirector,
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> tuple[int, dict[str, Any], float]:
    wire = request_bytes(payload)
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
    case: dict[str, Any],
    render: dict[str, Any],
    status: int,
    response: dict[str, Any],
    wall_ms: float,
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
        "real_http_prompt_tokens_exact": usage.get("prompt_tokens")
        == render["prompt_tokens"],
        "real_http_prompt_token_ids_sha256_exact": metrics.get(
            "prompt_token_ids_sha256"
        )
        == render["prompt_token_ids_sha256"],
        "private_prompt_boundary_distinguished": render[
            "private_prompt_matches_real_http"
        ]
        is False
        and render["private_prompt_tokens"] == len(prompt_ids)
        and render["private_prompt_token_ids_sha256"]
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
        "render_prompt_tokens": render["prompt_tokens"],
        "render_prompt_token_ids_sha256": render[
            "prompt_token_ids_sha256"
        ],
        "completion_tokens": usage.get("completion_tokens"),
        "output_token_ids_sha256": metrics.get(
            "output_token_ids_canonical_sha256"
        ),
        "output_text_sha256": output_text_sha256,
        "finish_reason": choice.get("finish_reason"),
        "request_wall_ms": wall_ms,
        "native_metrics": metrics,
    }


def cache_observation(
    case_id: str, response: dict[str, Any]
) -> dict[str, Any]:
    metrics = response["aima_amd395"]
    return {
        "case_id": case_id,
        "content": response["choices"][0]["message"]["content"],
        "output_token_ids_sha256": metrics[
            "output_token_ids_canonical_sha256"
        ],
        "prefix_lookup": metrics["prefix_cache"]["lookup"],
        "vl": metrics["vl"],
    }


def cache_correctness_checks(
    observations: list[dict[str, Any]],
) -> dict[str, bool]:
    by_id = {item["case_id"]: item for item in observations}
    if len(by_id) != len(observations):
        raise RuntimeError("cache qualification case ids are not unique")

    first = by_id["image_local_a"]
    changed = by_id["image_local_b"]
    restored = by_id["image_local_a_restored"]
    equivalent = by_id["image_data_a_equivalent"]
    variant = by_id["image_data_a_prompt_variant"]
    http_first = by_id["image_http_a"]
    http_changed = by_id["image_http_b"]
    http_restored = by_id["image_http_a_restored"]
    video_first = by_id["video_local_cold"]
    video_equivalent = by_id["video_data_equivalent"]
    mixed_first = by_id["mixed_local_cold"]
    mixed_exact = by_id["mixed_local_exact"]
    return {
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
        "same_http_url_a_processor_miss": http_first["vl"][
            "media_cache_misses"
        ]
        == 1
        and http_first["vl"]["media_cache_hits"] == 0
        and http_first["prefix_lookup"] == "miss",
        "same_http_url_b_processor_miss": http_changed["vl"][
            "media_cache_misses"
        ]
        == 1
        and http_changed["vl"]["media_cache_hits"] == 0,
        "same_http_url_b_prefix_miss": http_changed["prefix_lookup"] == "miss"
        and http_changed["vl"]["vision_plan_cache_hit"] is True
        and http_changed["vl"]["vision_encode_wall_ms"] > 0.0,
        "same_http_url_restored_a_hit": http_restored["vl"][
            "media_cache_hits"
        ]
        == 1
        and http_restored["prefix_lookup"] == "exact",
        "same_http_url_restored_a_output_exact": http_restored[
            "output_token_ids_sha256"
        ]
        == http_first["output_token_ids_sha256"],
        "video_local_cold_miss": video_first["vl"]["media_cache_misses"] == 1
        and video_first["vl"]["media_cache_hits"] == 0
        and video_first["prefix_lookup"] == "miss",
        "video_data_local_equivalent_hit": video_equivalent["vl"][
            "media_cache_hits"
        ]
        == 1
        and video_equivalent["prefix_lookup"] == "exact",
        "video_data_local_output_exact": video_equivalent[
            "output_token_ids_sha256"
        ]
        == video_first["output_token_ids_sha256"],
        "mixed_cold_two_media_misses": mixed_first["vl"][
            "media_cache_misses"
        ]
        == 2
        and mixed_first["vl"]["media_cache_hits"] == 0
        and mixed_first["prefix_lookup"] == "miss",
        "mixed_exact_two_media_hits": mixed_exact["vl"]["media_cache_hits"]
        == 2
        and mixed_exact["vl"]["media_cache_misses"] == 0
        and mixed_exact["prefix_lookup"] == "exact",
        "mixed_hit_miss_output_exact": mixed_exact["output_token_ids_sha256"]
        == mixed_first["output_token_ids_sha256"],
    }


class QuietFixtureHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *args: object) -> None:
        del args


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
    parser.add_argument(
        "--render-manifest", type=Path, default=SERVING_RENDER_MANIFEST
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--fixture-port", type=int, default=18087)
    parser.add_argument("--ready-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    binary = args.binary.resolve()
    model_dir = args.model_dir.resolve()
    fixture_root = args.fixture_root.resolve()
    fmha_provider = args.fmha_provider.resolve()
    aotriton = require_aotriton_closure(fmha_provider)
    vision_attention_image = args.vision_attention_image.resolve()
    oracle_path = args.oracle_manifest.resolve()
    render_path = args.render_manifest.resolve()
    output = args.output.resolve()
    raw_root = output.parent / f"{output.stem}-raw"
    required_paths = (
        binary,
        fmha_provider,
        vision_attention_image,
        oracle_path,
        render_path,
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

    render_manifest = load_json_object(render_path)
    render_errors = validate_serving_render_manifest(render_manifest)
    if render_errors:
        raise SystemExit(
            "invalid serving HTTP render manifest:\n- "
            + "\n- ".join(render_errors)
        )
    render_cases = {
        case["case_id"]: case for case in render_manifest["cases"]
    }
    if tuple(render_cases) != SERVING_RENDER_CASES or (
        SERVING_RENDER_CASES != CASE_ORDER
    ):
        raise SystemExit("serving render case order changed")
    for case_id in CASE_ORDER:
        case = cases_by_id[case_id]
        render = render_cases[case_id]
        processor = case["processor"]
        if render.get("oracle_request_sha256") != canonical_json_sha256(
            case["request"]
        ):
            raise SystemExit(f"serving render request drifted: {case_id}")
        if (
            render.get("private_prompt_tokens")
            != len(processor["prompt_token_ids"])
            or render.get("private_prompt_token_ids_sha256")
            != processor["prompt_token_ids_sha256"]
        ):
            raise SystemExit(f"serving private prompt binding drifted: {case_id}")

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
    http_a = cache_media_root / "http-a-160x320.png"
    http_b = cache_media_root / "http-b-160x320.png"
    http_mutable = cache_media_root / "http-mutable.png"
    write_cache_variant_png(http_a, phase=1)
    write_cache_variant_png(http_b, phase=2)
    shutil.copyfile(http_a, http_mutable)
    video_cache = fixture_root / "video-8f-4fps-128.mp4"
    mixed_image = fixture_root / "image-portrait-192x512.webp"
    mixed_video = fixture_root / "video-12f-6fps-192x128.avi"

    stdout_path = raw_root / "server.stdout.log"
    stderr_path = raw_root / "server.stderr.log"
    load_report = raw_root / "native-weight-load.json"
    request_count = len(CASE_ORDER) + 12
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
        "--allowed-media-domain",
        "127.0.0.1",
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
    fixture_base = f"http://127.0.0.1:{args.fixture_port}"
    fixture_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", args.fixture_port),
        partial(QuietFixtureHandler, directory=str(cache_media_root)),
    )
    fixture_thread = threading.Thread(
        target=fixture_server.serve_forever,
        name="native-vl-serving-cache-http",
        daemon=True,
    )
    fixture_thread.start()
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
            cache_prompt = "Identify the visual content."
            mutable_uri = mutable_image.resolve().as_uri()
            cache_observations: list[dict[str, Any]] = []

            def run_cache_case(
                case_id: str, payload: dict[str, Any]
            ) -> None:
                status, response, _wall_ms = post_json(
                    opener,
                    base_url + "/v1/chat/completions",
                    payload,
                    timeout=args.request_timeout_seconds,
                )
                if status != 200:
                    raise RuntimeError(
                        f"cache qualification request {case_id} failed: {response}"
                    )
                cache_observations.append(
                    cache_observation(case_id, response)
                )

            run_cache_case(
                "image_local_a", image_request(mutable_uri, cache_prompt)
            )
            shutil.copyfile(cache_b, mutable_image)
            run_cache_case(
                "image_local_b", image_request(mutable_uri, cache_prompt)
            )
            shutil.copyfile(cache_a, mutable_image)
            run_cache_case(
                "image_local_a_restored",
                image_request(mutable_uri, cache_prompt),
            )
            image_data_a = "data:image/png;base64," + base64.b64encode(
                cache_a.read_bytes()
            ).decode("ascii")
            run_cache_case(
                "image_data_a_equivalent",
                image_request(image_data_a, cache_prompt),
            )
            run_cache_case(
                "image_data_a_prompt_variant",
                image_request(
                    image_data_a, "Identify the visual content again."
                ),
            )

            http_mutable_url = f"{fixture_base}/{http_mutable.name}"
            run_cache_case(
                "image_http_a",
                image_request(http_mutable_url, "Identify the HTTP image."),
            )
            shutil.copyfile(http_b, http_mutable)
            run_cache_case(
                "image_http_b",
                image_request(http_mutable_url, "Identify the HTTP image."),
            )
            shutil.copyfile(http_a, http_mutable)
            run_cache_case(
                "image_http_a_restored",
                image_request(http_mutable_url, "Identify the HTTP image."),
            )

            video_prompt = "Identify the video content."
            video_uri = video_cache.resolve().as_uri()
            run_cache_case(
                "video_local_cold", video_request(video_uri, video_prompt)
            )
            video_data = "data:video/mp4;base64," + base64.b64encode(
                video_cache.read_bytes()
            ).decode("ascii")
            run_cache_case(
                "video_data_equivalent",
                video_request(video_data, video_prompt),
            )

            mixed_prompt = "Identify the image and video."
            mixed_payload = mixed_request(
                mixed_image.resolve().as_uri(),
                mixed_video.resolve().as_uri(),
                mixed_prompt,
            )
            run_cache_case("mixed_local_cold", mixed_payload)
            run_cache_case("mixed_local_exact", mixed_payload)

            cache_checks = cache_correctness_checks(cache_observations)
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
                    "http_a": file_component(
                        http_a,
                        f"{raw_root.name}/cache-media/{http_a.name}",
                    ),
                    "http_b": file_component(
                        http_b,
                        f"{raw_root.name}/cache-media/{http_b.name}",
                    ),
                    "http_mutable_final": file_component(
                        http_mutable,
                        f"{raw_root.name}/cache-media/{http_mutable.name}",
                    ),
                    "video": file_component(
                        video_cache,
                        "benchmarks/fixtures/vl-capability-v0.1.0/"
                        f"{video_cache.name}",
                    ),
                    "mixed_image": file_component(
                        mixed_image,
                        "benchmarks/fixtures/vl-capability-v0.1.0/"
                        f"{mixed_image.name}",
                    ),
                    "mixed_video": file_component(
                        mixed_video,
                        "benchmarks/fixtures/vl-capability-v0.1.0/"
                        f"{mixed_video.name}",
                    ),
                },
                "observations": cache_observations,
            }

            for case_id in CASE_ORDER:
                case = cases_by_id[case_id]
                status, response, wall_ms = post_json(
                    opener,
                    base_url + "/v1/chat/completions",
                    build_request(case, fixture_root),
                    timeout=args.request_timeout_seconds,
                )
                result = oracle_case_result(
                    case, render_cases[case_id], status, response, wall_ms
                )
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
        "remote_fixture_domain_bound": ready.get("allowed_media_domains") == 1,
        "structured_token_mask_resident": health.get(
            "structured_token_mask_bytes"
        )
        == 248_320
        and ready.get("structured_token_mask_bytes") == 248_320,
    }
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
            "native/include/aima/native_vl_request.h",
            "native/src/native_chat_protocol.cpp",
            "native/src/native_vl_request.cpp",
            "native/include/aima/native_resident_engine.h",
            "native/src/native_resident_engine.hip.cpp",
            "native/src/native_http_server.cpp",
            "aima_engine/vl_serving_render.py",
            "aima_engine/aotriton_closure.py",
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
            "resident-native-five-real-http-render-prompts-private-greedy-"
            "oracles-http-mutation-video-mixed-and-content-addressed-cache-"
            "correctness"
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
            "serving_render_manifest": file_component(
                render_path,
                "benchmarks/results/vl-serving-render-manifest-v0.1.0.json",
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
            "five_real_http_prompt_hashes_exact": all(
                item["checks"]["real_http_prompt_token_ids_sha256_exact"]
                and item["checks"]["real_http_prompt_tokens_exact"]
                for item in oracle_results
            ),
            "five_private_oracle_generations_preserved": all(
                item["checks"]["output_token_ids_sha256_exact"]
                and item["checks"]["output_text_sha256_exact"]
                for item in oracle_results
            ),
            "five_private_prompt_boundaries_distinguished": all(
                item["checks"]["private_prompt_boundary_distinguished"]
                for item in oracle_results
            ),
            "content_addressed_media_cache_qualified": cache_result.get(
                "passed"
            )
            is True,
            "same_http_url_content_mutation_qualified": all(
                cache_result.get("checks", {}).get(name) is True
                for name in (
                    "same_http_url_a_processor_miss",
                    "same_http_url_b_processor_miss",
                    "same_http_url_b_prefix_miss",
                    "same_http_url_restored_a_hit",
                    "same_http_url_restored_a_output_exact",
                )
            ),
            "video_transport_cache_equivalence_qualified": all(
                cache_result.get("checks", {}).get(name) is True
                for name in (
                    "video_local_cold_miss",
                    "video_data_local_equivalent_hit",
                    "video_data_local_output_exact",
                )
            ),
            "mixed_cache_invariance_qualified": all(
                cache_result.get("checks", {}).get(name) is True
                for name in (
                    "mixed_cold_two_media_misses",
                    "mixed_exact_two_media_hits",
                    "mixed_hit_miss_output_exact",
                )
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
