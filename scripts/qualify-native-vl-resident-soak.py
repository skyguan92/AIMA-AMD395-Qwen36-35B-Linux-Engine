#!/usr/bin/env python3
"""Run the formal one-hour mixed-workload soak from an exact portable archive."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.public_hygiene import scan_bytes  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
    verify_manifest_integrity,
)


SCHEMA = "aima-amd395-qwen36/native-vl-resident-soak/v1"
PRODUCT_SCHEMA = "aima-amd395-qwen36/native-vl-product-qualification/v1"
MINIMUM_DURATION_SECONDS = 3600.0
MINIMUM_REQUESTS = 240
MAXIMUM_RSS_GROWTH_BYTES = 1024 * 1024 * 1024
MAXIMUM_RSS_BYTES = 96 * 1024 * 1024 * 1024
MAXIMUM_GTT_BYTES = 96 * 1024 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float = 600.0,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = int(error.code)
        raw = error.read()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"HTTP endpoint returned non-object JSON: {url}")
    return status, value


def materialize_request(path: Path, media_root: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8").replace(
        "${AIMA_VL_MEDIA_ROOT}", str(media_root)
    )
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"soak request is not an object: {path}")
    value["model"] = "aima-amd395-qwen36-35b"
    value["temperature"] = 0
    value["max_tokens"] = 1
    value["stream"] = False
    return value


def process_rss_bytes(pid: int) -> int:
    status = Path(f"/proc/{pid}/status").read_text(
        encoding="utf-8", errors="replace"
    )
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    raise RuntimeError("resident process RSS is unavailable")


def require_gpu_idle() -> None:
    occupied = subprocess.run(
        ["fuser", "/dev/kfd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if occupied.returncode != 0:
        return
    subprocess.run(["fuser", "-v", "/dev/kfd"], check=False)
    raise SystemExit(75)


def target_gtt_bytes() -> int:
    matches: list[Path] = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        device = card / "device"
        try:
            vendor = (device / "vendor").read_text(encoding="ascii").strip()
            product = (device / "device").read_text(encoding="ascii").strip()
        except OSError:
            continue
        if vendor == "0x1002" and product == "0x1586":
            matches.append(device / "mem_info_gtt_used")
    if len(matches) != 1 or not matches[0].is_file():
        raise RuntimeError("exact AMD395 GTT telemetry path is unavailable")
    return int(matches[0].read_text(encoding="ascii").strip())


def publicize(
    value: Any, replacements: tuple[tuple[str, str], ...]
) -> Any:
    if isinstance(value, str):
        result = value
        for private, logical in replacements:
            result = result.replace(private, logical)
        return result
    if isinstance(value, list):
        return [publicize(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: publicize(item, replacements) for key, item in value.items()
        }
    return value


def sanitize_text(
    path: Path, replacements: tuple[tuple[str, str], ...]
) -> None:
    value = path.read_text(encoding="utf-8", errors="replace")
    for private, logical in replacements:
        value = value.replace(private, logical)
    path.write_text(value, encoding="utf-8")


def public_hygiene_passes(root: Path) -> bool:
    return not any(
        scan_bytes(path.relative_to(root).as_posix(), path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    )


def verify_recursive_manifest(bundle: Path) -> dict[str, Any]:
    manifest = load_object(bundle / "manifest.json")
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            raise RuntimeError("portable manifest contains a non-object entry")
        path = bundle / str(item.get("path", ""))
        if item.get("type") == "symlink":
            if not path.is_symlink() or path.readlink().as_posix() != item.get(
                "target"
            ):
                raise RuntimeError(f"portable manifest symlink differs: {path}")
        elif (
            item.get("type") != "file"
            or not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or sha256(path) != item.get("sha256")
        ):
            raise RuntimeError(f"portable manifest file differs: {path}")
    if manifest.get("complete") is not True:
        raise RuntimeError("portable manifest is incomplete")
    return manifest


def request_specs() -> tuple[tuple[str, Path | None, int, int], ...]:
    request_root = ROOT / "benchmarks/fixtures/vl-performance-v0.1.0/requests"
    return (
        ("text", None, 0, 0),
        ("image_a", request_root / "image_min_short_output1.json", 1, 0),
        ("video_b", request_root / "video_min_short_output1.json", 0, 1),
        (
            "mixed",
            request_root / "mixed_multi_turn_q8k_output512.json",
            1,
            1,
        ),
        (
            "image_a_restored",
            request_root / "image_min_short_output1.json",
            1,
            0,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--product-result", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host-role", default="primary_amd395")
    parser.add_argument("--duration-seconds", type=float, default=3600.0)
    parser.add_argument("--minimum-requests", type=int, default=240)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    args = parser.parse_args()

    if args.duration_seconds < MINIMUM_DURATION_SECONDS:
        parser.error(
            f"duration-seconds must be at least {MINIMUM_DURATION_SECONDS:g}"
        )
    if args.minimum_requests < MINIMUM_REQUESTS:
        parser.error(f"minimum-requests must be at least {MINIMUM_REQUESTS}")
    if args.interval_seconds <= 0:
        parser.error("interval-seconds must be positive")

    archive = args.archive.expanduser().resolve()
    product_result_path = args.product_result.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    media_root = args.media_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise SystemExit(f"output directory must be empty: {output_dir}")
    for label, path in (
        ("archive", archive),
        ("product result", product_result_path),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} is missing: {path}")
    for label, path in (
        ("model directory", model_dir),
        ("media root", media_root),
    ):
        if not path.is_dir():
            raise SystemExit(f"{label} is missing: {path}")

    archive_digest = sha256(archive)
    checksum = archive.with_name(archive.name + ".sha256")
    if (
        not checksum.is_file()
        or checksum.read_text(encoding="utf-8").split()[0] != archive_digest
    ):
        raise SystemExit("portable archive checksum is missing or mismatched")
    product_result = load_object(product_result_path)
    if (
        product_result.get("schema") != PRODUCT_SCHEMA
        or product_result.get("complete") is not True
        or product_result.get("qualified") is not True
        or verify_manifest_integrity(product_result)
    ):
        raise SystemExit("native VL product result is incomplete or unsealed")
    source = product_result.get("components", {}).get("source", {})
    release = product_result.get("release")
    if not isinstance(source, dict) or not isinstance(release, str):
        raise SystemExit("native VL product source identity is missing")

    specs = request_specs()
    for _, request_path, _, _ in specs:
        if request_path is not None and not request_path.is_file():
            raise SystemExit(f"soak request fixture is missing: {request_path}")
    media_files = (
        media_root / "vl-envelope-v0.1.0/image-minimum-1x1.png",
        media_root / "vl-envelope-v0.1.0/video-minimum-2f-2fps-32x32.mp4",
        media_root / "vl-envelope-v0.1.0/image-portrait-256x1024.png",
        media_root / "vl-envelope-v0.1.0/video-typical-4f-2fps-256x256.mp4",
    )
    if not all(path.is_file() for path in media_files):
        raise SystemExit("soak media fixtures are incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = raw_dir / "server.stdout.jsonl"
    stderr_path = raw_dir / "server.stderr"
    load_report = raw_dir / "server.load.json"
    requests_path = raw_dir / "requests.json"
    health_before_path = raw_dir / "health-before.json"
    health_after_path = raw_dir / "health-after.json"

    process: subprocess.Popen[str] | None = None
    health_before: dict[str, Any] | None = None
    health_after: dict[str, Any] | None = None
    shutdown: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    rss_samples: list[int] = []
    gtt_samples: list[int] = []
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    soak_elapsed = 0.0
    with tempfile.TemporaryDirectory(prefix="aima-native-vl-soak-") as temporary:
        temporary_root = Path(temporary)
        extraction = temporary_root / "extract"
        extraction.mkdir()
        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(extraction)],
            check=True,
        )
        roots = [path for path in extraction.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("portable archive must contain one root directory")
        bundle = roots[0]
        manifest = verify_recursive_manifest(bundle)
        launcher = bundle / "bin/aima-engine"
        engine = bundle / "libexec/aima-engine.real"
        components = product_result["components"]
        if (
            sha256(launcher) != components["static_launcher"]["sha256"]
            or sha256(engine) != components["native_engine"]["sha256"]
            or manifest.get("release") != release
            or manifest.get("source", {}).get("commit")
            != source.get("release_commit")
            or manifest.get("native_vl", {}).get("enabled") is not True
        ):
            raise RuntimeError("portable archive identity differs from qualification")

        isolated_home = temporary_root / "home"
        isolated_home.mkdir()
        environment = {
            "HOME": str(isolated_home),
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
        }
        port = free_loopback_port()
        endpoint = f"http://127.0.0.1:{port}"
        command = [
            str(launcher),
            "serve",
            "--model-dir",
            str(model_dir),
            "--context-tokens",
            "16384",
            "--cache-capacity",
            "16384",
            "--allowed-local-media-path",
            str(media_root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--request-timeout-ms",
            "600000",
            "--report",
            str(load_report),
        ]
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            try:
                require_gpu_idle()
                process = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    env=environment,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 300.0
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("soak server exited before readiness")
                    try:
                        status, health = http_json(
                            endpoint + "/health", timeout=2.0
                        )
                        if status == 200 and health.get("status") == "ok":
                            health_before = health
                            break
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
                    time.sleep(0.5)
                if health_before is None:
                    raise RuntimeError("soak server was not ready")
                rss_samples.append(process_rss_bytes(process.pid))
                gtt_samples.append(target_gtt_bytes())
                soak_start = time.monotonic()
                next_request_at = soak_start
                output_hashes: dict[str, str] = {}
                while (
                    time.monotonic() - soak_start < args.duration_seconds
                    or len(records) < args.minimum_requests
                ):
                    case_id, request_path, image_count, video_count = specs[
                        len(records) % len(specs)
                    ]
                    if request_path is None:
                        request_payload = {
                            "model": "aima-amd395-qwen36-35b",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Reply with exactly one token.",
                                }
                            ],
                            "temperature": 0,
                            "max_tokens": 1,
                            "stream": False,
                        }
                    else:
                        request_payload = materialize_request(
                            request_path, media_root
                        )
                    status, response = http_json(
                        endpoint + "/v1/chat/completions",
                        method="POST",
                        payload=request_payload,
                    )
                    metrics = response.get("aima_amd395", {})
                    raw_vl = metrics.get("vl")
                    vl = raw_vl if isinstance(raw_vl, dict) else {}
                    output_hash = metrics.get("output_token_ids_sha256")
                    stable_key = (
                        "image_a" if case_id == "image_a_restored" else case_id
                    )
                    first_hash = output_hashes.setdefault(stable_key, output_hash)
                    checks = {
                        "status_200": status == 200,
                        "one_completion_token": (
                            response.get("usage", {}).get("completion_tokens") == 1
                        ),
                        "one_model_load": metrics.get("model_loads") == 1,
                        "request_index_exact": metrics.get("request_index")
                        == len(records) + 1,
                        "native_runtime": str(metrics.get("runtime", "")).startswith(
                            "native-resident-q"
                        ),
                        "no_oracle_reads": metrics.get("oracle_tensor_reads") == 0,
                        "image_count_exact": vl.get("image_count", 0)
                        == image_count,
                        "video_count_exact": vl.get("video_count", 0)
                        == video_count,
                        "deterministic_case_output": (
                            isinstance(output_hash, str)
                            and output_hash == first_hash
                        ),
                    }
                    if not all(checks.values()):
                        raise RuntimeError(f"soak request failed: {case_id}")
                    prefix = metrics.get("prefix_cache", {})
                    records.append(
                        {
                            "request_index": len(records) + 1,
                            "elapsed_seconds": time.monotonic() - soak_start,
                            "case_id": case_id,
                            "checks": checks,
                            "output_token_ids_sha256": output_hash,
                            "prompt_tokens": metrics.get("prompt_tokens"),
                            "request_wall_ms": metrics.get("request_wall_ms"),
                            "ttft_ms": metrics.get("ttft_ms"),
                            "prefix_lookup": (
                                prefix.get("lookup")
                                if isinstance(prefix, dict)
                                else None
                            ),
                            "media_cache_hits": vl.get("media_cache_hits", 0),
                            "vision_embedding_cache_hit": vl.get(
                                "vision_embedding_cache_hit", False
                            ),
                        }
                    )
                    rss_samples.append(process_rss_bytes(process.pid))
                    gtt_samples.append(target_gtt_bytes())
                    if len(records) % 20 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "native_vl_soak_progress",
                                    "elapsed_seconds": records[-1][
                                        "elapsed_seconds"
                                    ],
                                    "requests": len(records),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    next_request_at += args.interval_seconds
                    delay = next_request_at - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                soak_elapsed = time.monotonic() - soak_start
                status, health_after = http_json(
                    endpoint + "/health", timeout=5.0
                )
                if status != 200:
                    raise RuntimeError("soak health endpoint failed")
                shutdown_status, shutdown = http_json(
                    endpoint + "/shutdown",
                    method="POST",
                    payload={},
                    timeout=10.0,
                )
                if shutdown_status != 200:
                    raise RuntimeError("soak shutdown endpoint failed")
                process.wait(timeout=60.0)
                if process.returncode != 0:
                    raise RuntimeError("soak server stopped nonzero")
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10.0)

        replacements = (
            (str(bundle), "${AIMA_BUNDLE_ROOT}"),
            (str(model_dir), "${AIMA_MODEL_DIR}"),
            (str(media_root), "${AIMA_VL_MEDIA_ROOT}"),
            (str(output_dir), "${AIMA_OUTPUT_DIR}"),
            (str(temporary_root), "${AIMA_ISOLATED_ROOT}"),
        )
        command = publicize(command, replacements)
        health_before = publicize(health_before, replacements)
        health_after = publicize(health_after, replacements)
        shutdown = publicize(shutdown, replacements)
        sanitize_text(stdout_path, replacements)
        sanitize_text(stderr_path, replacements)
        load_payload = publicize(load_object(load_report), replacements)
        atomic_json(load_report, load_payload)

    atomic_json(health_before_path, health_before)
    atomic_json(health_after_path, health_after)
    atomic_json(requests_path, {"requests": records})
    events = [
        json.loads(line)
        for line in stdout_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ready_events = [item for item in events if item.get("event") == "ready"]
    stopped_events = [item for item in events if item.get("event") == "stopped"]
    case_counts = {
        case_id: sum(record["case_id"] == case_id for record in records)
        for case_id, _, _, _ in specs
    }
    warm_index = min(len(rss_samples) - 1, len(specs) * 2)
    warm_rss = rss_samples[warm_index]
    post_warm_peak_rss = max(rss_samples[warm_index:])
    rss_growth = post_warm_peak_rss - warm_rss
    reuse_observed = any(
        record["case_id"] != "text"
        and (
            record["prefix_lookup"] == "exact"
            or record["media_cache_hits"] >= 1
            or record["vision_embedding_cache_hit"] is True
        )
        for record in records[len(specs) :]
    )
    checks = {
        "minimum_duration": soak_elapsed >= MINIMUM_DURATION_SECONDS,
        "minimum_requests": len(records) >= MINIMUM_REQUESTS,
        "every_case_repeated": all(count >= 2 for count in case_counts.values()),
        "all_requests_qualified": all(
            all(record["checks"].values()) for record in records
        ),
        "one_ready_event": len(ready_events) == 1,
        "one_stopped_event": len(stopped_events) == 1,
        "single_model_load": (
            len(stopped_events) == 1
            and stopped_events[0].get("model_loads") == 1
        ),
        "served_count_exact": (
            health_after.get("served") == len(records)
            and len(stopped_events) == 1
            and stopped_events[0].get("served") == len(records)
        ),
        "native_vl_ready": (
            health_before.get("native_vl") is True
            and health_before.get("vision_warmup_completed") is True
        ),
        "cache_reuse_observed": reuse_observed,
        "rss_within_96_gib": max(rss_samples) <= MAXIMUM_RSS_BYTES,
        "gtt_within_96_gib": max(gtt_samples) <= MAXIMUM_GTT_BYTES,
        "rss_growth_bounded": rss_growth <= MAXIMUM_RSS_GROWTH_BYTES,
        "clean_shutdown": shutdown.get("status") == "shutting_down",
        "stderr_empty": stderr_path.stat().st_size == 0,
        "public_hygiene": public_hygiene_passes(output_dir),
    }
    if not all(checks.values()):
        raise RuntimeError(f"native VL resident soak failed: {checks}")

    request_inputs = {
        path.name: file_component(path, str(path.relative_to(ROOT)))
        for _, path, _, _ in specs
        if path is not None
    }
    media_inputs = {
        path.name: file_component(
            path,
            "${AIMA_VL_MEDIA_ROOT}/" + str(path.relative_to(media_root)),
        )
        for path in media_files
    }
    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "release": release,
            "recorded_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "started_at": started_at,
            "complete": True,
            "qualified": True,
            "host_role": args.host_role,
            "source": source,
            "archive": {
                "name": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": archive_digest,
                "checksum_sidecar": checksum.name,
                "checksum_verified": True,
            },
            "product_result": file_component(
                product_result_path,
                "share/aima/qualification.json",
            ),
            "protocol": {
                "minimum_duration_seconds": MINIMUM_DURATION_SECONDS,
                "minimum_requests": MINIMUM_REQUESTS,
                "requested_duration_seconds": args.duration_seconds,
                "requested_minimum_requests": args.minimum_requests,
                "interval_seconds": args.interval_seconds,
                "workload_order": [item[0] for item in specs],
                "single_process": True,
                "temperature": 0,
                "output_tokens": 1,
                "isolated_environment_keys": ["HOME", "LANG", "PATH"],
            },
            "command": command,
            "inputs": {
                "requests": request_inputs,
                "media": media_inputs,
            },
            "measurement": {
                "elapsed_seconds": soak_elapsed,
                "request_count": len(records),
                "case_counts": case_counts,
                "rss": {
                    "samples": len(rss_samples),
                    "ready_bytes": rss_samples[0],
                    "warm_bytes": warm_rss,
                    "post_warm_peak_bytes": post_warm_peak_rss,
                    "final_bytes": rss_samples[-1],
                    "peak_bytes": max(rss_samples),
                    "growth_after_warm_bytes": rss_growth,
                    "growth_limit_bytes": MAXIMUM_RSS_GROWTH_BYTES,
                },
                "gtt": {
                    "samples": len(gtt_samples),
                    "ready_bytes": gtt_samples[0],
                    "peak_bytes": max(gtt_samples),
                    "final_bytes": gtt_samples[-1],
                    "limit_bytes": MAXIMUM_GTT_BYTES,
                    "source": (
                        "/sys/class/drm/card*/device/mem_info_gtt_used for "
                        "PCI 1002:1586"
                    ),
                },
            },
            "checks": checks,
            "raw_artifacts": {
                name: file_component(path, f"raw/{path.name}")
                for name, path in (
                    ("requests", requests_path),
                    ("health_before", health_before_path),
                    ("health_after", health_after_path),
                    ("load_report", load_report),
                    ("server_stdout", stdout_path),
                    ("server_stderr", stderr_path),
                )
            },
            "decision": {
                "one_hour_resident_mixed_workload_passed": True,
                "single_model_load_preserved": True,
                "request_level_oracle_reads": 0,
                "memory_growth_bounded": True,
            },
        }
    )
    output = output_dir / "soak.json"
    digest = atomic_json(output, payload)
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": True,
                "elapsed_seconds": soak_elapsed,
                "requests": len(records),
                "output": str(output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
