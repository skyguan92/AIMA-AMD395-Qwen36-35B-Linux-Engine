#!/usr/bin/env python3
"""Qualify an extracted release archive with no host userspace environment."""

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
import stat
import subprocess
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aima_engine.public_hygiene import scan_bytes
from aima_engine.vl_reference import seal_manifest, verify_manifest_integrity
from native_bundle_closure import audit_bundle


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def public_hygiene_summary(root: Path) -> dict[str, int | bool]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    findings = [
        finding
        for path in files
        for finding in scan_bytes(
            path.relative_to(root).as_posix(), path.read_bytes()
        )
    ]
    return {
        "checked_files": len(files),
        "finding_count": len(findings),
        "passed": not findings,
    }


def publicize(
    value: Any,
    model_dir: Path,
    replacements: tuple[tuple[str, str], ...] = (),
) -> Any:
    if isinstance(value, str):
        result = value.replace(str(ROOT), "${AIMA_REPO_ROOT}").replace(
            str(model_dir), "${AIMA_MODEL_DIR}"
        )
        for private, logical in replacements:
            result = result.replace(private, logical)
        return result
    if isinstance(value, list):
        return [publicize(item, model_dir, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: publicize(item, model_dir, replacements)
            for key, item in value.items()
        }
    return value


def verify_manifest(bundle: Path) -> dict[str, Any]:
    manifest = load_json(bundle / "manifest.json")
    if manifest.get("complete") is not True:
        raise RuntimeError("bundle manifest is not complete")
    checked_files = 0
    checked_symlinks = 0
    for entry in manifest["files"]:
        path = bundle / entry["path"]
        if entry["type"] == "symlink":
            if not path.is_symlink() or path.readlink().as_posix() != entry[
                "target"
            ]:
                raise RuntimeError(f"bundle symlink mismatch: {entry['path']}")
            checked_symlinks += 1
            continue
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["bytes"])
            or sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"bundle file mismatch: {entry['path']}")
        checked_files += 1
    return {
        "schema": manifest["schema"],
        "complete": True,
        "release": manifest["release"],
        "source": manifest["source"],
        "checked_files": checked_files,
        "checked_symlinks": checked_symlinks,
        "payload_bytes_excluding_manifest": manifest[
            "payload_bytes_excluding_manifest"
        ],
        "attention_providers": manifest["attention_providers"],
        "native_vl": manifest.get("native_vl", {"enabled": False}),
    }


def run_smoke(
    *,
    launcher: Path,
    model_dir: Path,
    output_dir: Path,
    context: int,
    expected_primary: str,
    expected_secondary: str,
    expected_secondary_layers: list[int],
    require_native_vl: bool,
    environment: dict[str, str],
) -> dict[str, Any]:
    report = output_dir / "raw" / f"q{context}-output1.json"
    load_report = report.with_name(report.stem + ".load.json")
    command = [
        str(launcher),
        "resident-session-probe",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        str(context),
        "--uniform-input-token-id",
        "1",
        "--max-new-tokens",
        "1",
        "--requests",
        "1",
        "--disable-prefix-cache",
        "--report",
        str(load_report),
    ]
    print(
        json.dumps(
            {"event": "bundle_smoke_start", "context_tokens": context},
            sort_keys=True,
        ),
        flush=True,
    )
    require_gpu_idle()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        env=environment,
        check=False,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("bundle smoke emitted non-object JSON")
    load = payload["load"]
    request = payload["requests"][0]
    native_vl_qualified = bool(
        not require_native_vl
        or (
            load.get("model_tensor_count") == 1026
            and load.get("visual_model_tensor_count") == 333
            and load.get("visual_model_payload_bytes") == 893142496
            and load.get("vision_warmup_completed") is True
            and load.get("vision_attention_image_sha256")
            == "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e"
            and load.get("vision_dense_image_attention_image_sha256")
            == "e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c"
        )
    )
    qualified = bool(
        completed.returncode == 0
        and payload["complete"] is True
        and payload["runtime_python"] is False
        and payload["runtime_torch"] is False
        and payload["runtime_vllm"] is False
        and payload["runtime_triton"] is False
        and payload["model_loads"] == 1
        and native_vl_qualified
        and request["prompt_tokens"] == context
        and request["completion_tokens"] == 1
        and request["first_token_certified"] is True
        and request["all_decode_tokens_certified"] is True
        and load["fmha_provider_backend"] == expected_primary
        and load["secondary_fmha_provider_backend"] == expected_secondary
        and load["secondary_fmha_layers"] == expected_secondary_layers
    )
    replacements = (
        (str(launcher.parents[1]), "${AIMA_BUNDLE_ROOT}"),
        (str(output_dir), "${AIMA_OUTPUT_DIR}"),
    )
    load_payload = publicize(load_json(load_report), model_dir, replacements)
    atomic_json(load_report, load_payload)
    payload["bundle_qualification"] = {
        "command": publicize(command, model_dir, replacements),
        "environment_keys": sorted(environment),
        "load_report": load_report.name,
        "load_report_sha256": sha256(load_report),
        "qualified": qualified,
    }
    payload = publicize(payload, model_dir, replacements)
    atomic_json(report, payload)
    if not qualified:
        raise RuntimeError(f"bundle smoke failed: q{context}")
    print(
        json.dumps(
            {
                "event": "bundle_smoke_complete",
                "context_tokens": context,
                "command_to_ready_wall_ms": load[
                    "command_to_ready_wall_ms"
                ],
                "prefill_tokens_per_second": request[
                    "prefill_tokens_per_second"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "context_tokens": context,
        "output_tokens": 1,
        "primary_provider": load["fmha_provider_backend"],
        "secondary_provider": load["secondary_fmha_provider_backend"],
        "secondary_layers": load["secondary_fmha_layers"],
        "command_to_ready_wall_ms": load["command_to_ready_wall_ms"],
        "prefill_tokens_per_second": request["prefill_tokens_per_second"],
        "output_token_ids_sha256": request["output_token_ids_sha256"],
        "report": f"raw/{report.name}",
        "report_sha256": sha256(report),
        "qualified": True,
        "native_vl_ready": native_vl_qualified if require_native_vl else None,
    }


def read_os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, raw_value = line.partition("=")
        if separator and key in {"ID", "VERSION_ID", "PRETTY_NAME"}:
            result[key.lower()] = raw_value.strip().strip('"')
    return result


def host_fingerprint_sha256() -> str:
    """Return a non-reversible identity for proving two physical hosts differ."""
    components: list[bytes] = []
    for label, path in (
        (b"machine-id", Path("/etc/machine-id")),
        (b"product-uuid", Path("/sys/class/dmi/id/product_uuid")),
    ):
        if not path.is_file():
            continue
        try:
            value = path.read_bytes().strip().lower()
        except OSError:
            continue
        if value:
            components.append(label + b"\0" + value)
    if not components:
        raise RuntimeError("stable host identity inputs are unavailable")
    payload = b"aima/native-vl/host-fingerprint/v1\0" + b"\0".join(
        components
    )
    return hashlib.sha256(payload).hexdigest()


def run_doctor(
    *,
    launcher: Path,
    model_dir: Path,
    output_dir: Path,
    expected_engine_version: str,
    expected_native_commit: str,
    host_role: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    report = output_dir / "raw" / "doctor.json"
    command = [
        str(launcher),
        "doctor",
        "--model-dir",
        str(model_dir),
        "--json",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("portable doctor emitted invalid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("portable doctor emitted non-object JSON")
    check_records = payload.get("checks", [])
    checks_by_id = {
        item.get("id"): item
        for item in check_records
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    bundle_check = checks_by_id.get("runtime.bundle", {})
    required_checks = [
        item
        for item in check_records
        if isinstance(item, dict) and item.get("required") is True
    ]
    qualified = bool(
        completed.returncode == 0
        and payload.get("complete") is True
        and payload.get("qualified") is True
        and payload.get("version") == expected_engine_version
        and payload.get("source_commit") == expected_native_commit
        and payload.get("model_checked") is True
        and len(required_checks) >= 13
        and all(item.get("passed") is True for item in required_checks)
        and bundle_check.get("required") is True
        and bundle_check.get("passed") is True
        and bundle_check.get("actual")
        == {"detected": True, "complete": True}
    )
    if not qualified:
        detail = completed.stderr.strip()
        raise RuntimeError(f"portable doctor failed: {detail}")
    replacements = (
        (str(launcher.parents[1]), "${AIMA_BUNDLE_ROOT}"),
        (str(output_dir), "${AIMA_OUTPUT_DIR}"),
    )
    public_payload = publicize(payload, model_dir, replacements)
    atomic_json(report, public_payload)

    uname = os.uname()

    def actual(check_id: str) -> Any:
        record = checks_by_id.get(check_id, {})
        return record.get("actual") if isinstance(record, dict) else None

    return {
        "complete": True,
        "qualified": True,
        "command": publicize(command, model_dir, replacements),
        "environment_keys": sorted(environment),
        "host": {
            "role": host_role,
            "fingerprint_sha256": host_fingerprint_sha256(),
            "os": read_os_release(),
            "kernel": uname.release,
            "architecture": uname.machine,
            "gpu": actual("gpu.architecture"),
            "vram_bytes": actual("memory.vram"),
            "gtt_bytes": actual("memory.gtt"),
        },
        "required_check_count": len(required_checks),
        "required_checks_passed": all(
            item.get("passed") is True for item in required_checks
        ),
        "bundle_detected_complete": True,
        "report": "raw/doctor.json",
        "report_sha256": sha256(report),
    }


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> tuple[int, dict[str, Any]]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            body = response.read()
    except urllib.error.HTTPError as error:
        status = int(error.code)
        body = error.read()
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError(f"HTTP endpoint returned non-object JSON: {url}")
    return status, value


def materialize_request(
    path: Path, media_root: Path, *, max_tokens: int = 1
) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8").replace(
        "${AIMA_VL_MEDIA_ROOT}", str(media_root)
    )
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"VL smoke request is not an object: {path}")
    value["model"] = "aima-amd395-qwen36-35b"
    value["temperature"] = 0
    value["max_tokens"] = max_tokens
    value["stream"] = False
    return value


def sanitize_text_file(
    path: Path, replacements: tuple[tuple[str, str], ...]
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for private, logical in replacements:
        text = text.replace(private, logical)
    path.write_text(text, encoding="utf-8")


def run_vl_smoke(
    *,
    launcher: Path,
    model_dir: Path,
    media_root: Path,
    output_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    request_root = ROOT / "benchmarks/fixtures/vl-performance-v0.1.0/requests"
    request_specs = (
        (
            "text",
            None,
            {"enabled": False, "image_count": 0, "video_count": 0},
        ),
        (
            "image_a",
            request_root / "image_min_short_output1.json",
            {"enabled": True, "image_count": 1, "video_count": 0},
        ),
        (
            "video_b",
            request_root / "video_min_short_output1.json",
            {"enabled": True, "image_count": 0, "video_count": 1},
        ),
        (
            "mixed",
            request_root / "mixed_multi_turn_q8k_output512.json",
            {"enabled": True, "image_count": 1, "video_count": 1},
        ),
        (
            "image_a_restored",
            request_root / "image_min_short_output1.json",
            {"enabled": True, "image_count": 1, "video_count": 0},
        ),
    )
    for _, request_path, _ in request_specs:
        if request_path is not None and not request_path.is_file():
            raise RuntimeError(f"isolated VL smoke fixture is missing: {request_path}")
    if not (media_root / "vl-envelope-v0.1.0").is_dir():
        raise RuntimeError("isolated VL smoke media root is incomplete")

    raw_dir = output_dir / "raw" / "vl-smoke"
    raw_dir.mkdir(parents=True, exist_ok=True)
    port = free_loopback_port()
    endpoint = f"http://127.0.0.1:{port}"
    load_report = raw_dir / "server.load.json"
    stdout_path = raw_dir / "server.stdout.jsonl"
    stderr_path = raw_dir / "server.stderr"
    command = [
        str(launcher),
        "serve",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        "16384",
        "--cache-capacity",
        "17408",
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
    process: subprocess.Popen[str] | None = None
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
            health_before: dict[str, Any] | None = None
            deadline = time.monotonic() + 300.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        "isolated native VL server exited before readiness"
                    )
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
                raise RuntimeError("isolated native VL server was not ready")

            responses: list[dict[str, Any]] = []
            first_image_hash: str | None = None
            for request_index, (case_id, request_path, expected_vl) in enumerate(
                request_specs, start=1
            ):
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
                vl = metrics.get("vl", {})
                vl_enabled = vl.get("enabled")
                checks = {
                    "status_200": status == 200,
                    "one_completion_token": (
                        response.get("usage", {}).get("completion_tokens") == 1
                    ),
                    "one_model_load": metrics.get("model_loads") == 1,
                    "request_index_exact": metrics.get("request_index")
                    == request_index,
                    "vl_enabled_exact": (
                        vl_enabled is True
                        if expected_vl["enabled"]
                        else vl_enabled in (None, False)
                    ),
                    "image_count_exact": vl.get("image_count", 0)
                    == expected_vl["image_count"],
                    "video_count_exact": vl.get("video_count", 0)
                    == expected_vl["video_count"],
                }
                output_hash = metrics.get("output_token_ids_sha256")
                if case_id == "image_a":
                    first_image_hash = output_hash
                if case_id == "image_a_restored":
                    prefix_cache = metrics.get("prefix_cache", {})
                    reuse_signals = {
                        "prefix_cache_exact": prefix_cache.get("lookup")
                        == "exact",
                        "media_cache_hit": vl.get("media_cache_hits", 0) >= 1,
                        "vision_embedding_cache_hit": (
                            vl.get("vision_embedding_cache_hit") is True
                        ),
                    }
                    checks["a_b_a_output_exact"] = (
                        isinstance(output_hash, str)
                        and output_hash == first_image_hash
                    )
                    checks["a_b_a_cache_reuse_observed"] = any(
                        reuse_signals.values()
                    )
                else:
                    reuse_signals = None
                record = {
                    "case_id": case_id,
                    "status": status,
                    "checks": checks,
                    "response": publicize(response, model_dir),
                    "cache_reuse_signals": reuse_signals,
                    "qualified": all(checks.values()),
                }
                response_path = raw_dir / f"{request_index:02d}-{case_id}.json"
                atomic_json(response_path, record)
                responses.append(
                    {
                        "case_id": case_id,
                        "response": f"raw/vl-smoke/{response_path.name}",
                        "response_sha256": sha256(response_path),
                        "checks": checks,
                        "cache_reuse_signals": reuse_signals,
                        "qualified": all(checks.values()),
                    }
                )
                if not all(checks.values()):
                    raise RuntimeError(f"isolated VL smoke failed: {case_id}")

            status, health_after = http_json(endpoint + "/health", timeout=5.0)
            if status != 200 or health_after.get("served") != len(request_specs):
                raise RuntimeError("isolated VL health did not retain all requests")
            shutdown_status, shutdown = http_json(
                endpoint + "/shutdown", method="POST", payload={}, timeout=10.0
            )
            if shutdown_status != 200:
                raise RuntimeError("isolated native VL shutdown failed")
            process.wait(timeout=60.0)
            if process.returncode != 0:
                raise RuntimeError("isolated native VL server stopped nonzero")
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10.0)

    replacements = (
        (str(launcher.parents[1]), "${AIMA_BUNDLE_ROOT}"),
        (str(model_dir), "${AIMA_MODEL_DIR}"),
        (str(media_root), "${AIMA_VL_MEDIA_ROOT}"),
        (str(output_dir), "${AIMA_OUTPUT_DIR}"),
    )
    sanitize_text_file(stdout_path, replacements)
    sanitize_text_file(stderr_path, replacements)
    load_payload = publicize(load_json(load_report), model_dir, replacements)
    atomic_json(load_report, load_payload)
    health_before = publicize(health_before, model_dir, replacements)
    health_after = publicize(health_after, model_dir, replacements)
    shutdown = publicize(shutdown, model_dir, replacements)
    checks = {
        "five_requests_qualified": len(responses) == 5
        and all(item["qualified"] for item in responses),
        "single_model_load": all(
            item["checks"]["one_model_load"] for item in responses
        ),
        "ready_includes_native_vl": health_before.get("native_vl") is True,
        "ready_includes_vision_warmup": (
            health_before.get("vision_warmup_completed") is True
        ),
        "served_count_exact": health_after.get("served") == 5,
        "shutdown_clean": shutdown.get("status") == "shutting_down",
    }
    if not all(checks.values()):
        raise RuntimeError("isolated native VL aggregate smoke failed")
    return {
        "complete": True,
        "qualified": True,
        "command": publicize(command, model_dir, replacements),
        "environment_keys": sorted(environment),
        "checks": checks,
        "health_before": health_before,
        "health_after": health_after,
        "responses": responses,
        "raw_artifacts": {
            "load_report": {
                "path": "raw/vl-smoke/server.load.json",
                "sha256": sha256(load_report),
            },
            "server_stdout": {
                "path": "raw/vl-smoke/server.stdout.jsonl",
                "sha256": sha256(stdout_path),
            },
            "server_stderr": {
                "path": "raw/vl-smoke/server.stderr",
                "sha256": sha256(stderr_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--host-role", default="qualification_amd395")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--product-result",
        type=Path,
        required=True,
    )
    cli = parser.parse_args()

    archive = cli.archive.expanduser().resolve()
    model_dir = cli.model_dir.expanduser().resolve()
    media_root = (
        cli.media_root.expanduser().resolve()
        if cli.media_root is not None
        else None
    )
    output_dir = cli.output_dir.expanduser().resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise SystemExit(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    product_result = load_json(cli.product_result.resolve())
    release = product_result.get("release")
    source = product_result.get("components", {}).get("source", {})
    engine_version = product_result.get("engine_version", f"{release}-native")
    require_native_vl = (
        product_result.get("schema")
        == "aima-amd395-qwen36/native-vl-product-qualification/v1"
    )
    if not isinstance(release, str) or not release:
        raise SystemExit("product result release is missing")
    if not isinstance(source, dict) or not source.get("release_commit"):
        raise SystemExit("product result source provenance is missing")
    if require_native_vl and verify_manifest_integrity(product_result):
        raise SystemExit("native VL product result integrity failed")
    if require_native_vl and not source.get("native_source_commit"):
        raise SystemExit("native VL product native source identity is missing")
    if not archive.is_file():
        raise SystemExit(f"archive is missing: {archive}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")
    if require_native_vl and (media_root is None or not media_root.is_dir()):
        raise SystemExit(
            "--media-root must name the native VL fixture root for this release"
        )
    archive_sha256 = sha256(archive)
    checksum_path = archive.with_name(archive.name + ".sha256")
    checksum_verified = False
    if checksum_path.is_file():
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        checksum_verified = expected == archive_sha256
    if not checksum_verified:
        raise SystemExit("archive checksum sidecar is missing or mismatched")

    with tempfile.TemporaryDirectory(prefix="aima-bundle-qualification-") as tmp:
        extraction = Path(tmp) / "extract"
        extraction.mkdir()
        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(extraction)],
            check=True,
        )
        roots = [path for path in extraction.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("archive must contain exactly one root directory")
        bundle = roots[0]
        if stat.S_IMODE(bundle.stat().st_mode) != 0o755:
            raise RuntimeError("archive root directory must have mode 0755")
        launcher = bundle / "bin/aima-engine"
        engine = bundle / "libexec/aima-engine.real"
        if sha256(engine) != product_result["components"]["native_engine"][
            "sha256"
        ]:
            raise RuntimeError("archive engine does not match product result")
        if sha256(launcher) != product_result["components"]["static_launcher"][
            "sha256"
        ]:
            raise RuntimeError("archive launcher does not match product result")
        manifest = verify_manifest(bundle)
        if manifest.get("release") != release:
            raise RuntimeError("bundle manifest release does not match product result")
        if require_native_vl and manifest.get("native_vl", {}).get("enabled") is not True:
            raise RuntimeError("bundle manifest does not contain native VL closure")
        manifest_source = manifest.get("source", {})
        if (
            not isinstance(manifest_source, dict)
            or manifest_source.get("release_tag") != source.get("release_tag")
            or manifest_source.get("commit") != source.get("release_commit")
            or manifest_source.get("native_commit")
            != source.get("native_source_commit")
            or manifest_source.get("dirty") is not False
        ):
            raise RuntimeError("bundle manifest source does not match product result")
        closure = audit_bundle(bundle)
        isolated_home = Path(tmp) / "home"
        isolated_home.mkdir()
        environment = {
            "HOME": str(isolated_home),
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
        }
        version = subprocess.run(
            [str(launcher), "--version"],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        ).stdout.strip()
        help_text = subprocess.run(
            [str(launcher), "--help"],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        ).stdout
        if (
            version != f"aima-engine-native {engine_version}"
            or "131072" not in help_text
            or "input261120/output1024" not in help_text
        ):
            raise RuntimeError("portable launcher public CLI is incomplete")
        doctor = run_doctor(
            launcher=launcher,
            model_dir=model_dir,
            output_dir=output_dir,
            expected_engine_version=engine_version,
            expected_native_commit=source["native_source_commit"],
            host_role=cli.host_role,
            environment=environment,
        )
        smokes = [
            run_smoke(
                launcher=launcher,
                model_dir=model_dir,
                output_dir=output_dir,
                context=context,
                expected_primary=primary,
                expected_secondary=secondary,
                expected_secondary_layers=layers,
                require_native_vl=require_native_vl,
                environment=environment,
            )
            for context, primary, secondary, layers in (
                (1024, "AOTriton 0.11.1", "", []),
                (
                    16384,
                    "packed-GQA/CK-Tile hybrid",
                    "CK-Tile",
                    [39],
                ),
                (65536, "CK-Tile", "AOTriton 0.11.1", [39]),
            )
        ]
        vl_smoke = (
            run_vl_smoke(
                launcher=launcher,
                model_dir=model_dir,
                media_root=media_root,
                output_dir=output_dir,
                environment=environment,
            )
            if require_native_vl and media_root is not None
            else None
        )

    hygiene = public_hygiene_summary(output_dir)
    if hygiene["passed"] is not True:
        raise RuntimeError("portable qualification output failed public hygiene")

    result = {
        "schema": "aima-amd395-qwen36/native-portable-bundle-qualification/v1",
        "release": release,
        "recorded_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "host_role": cli.host_role,
        "source": source,
        "complete": True,
        "qualified": True,
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
            "checksum_sidecar": checksum_path.name,
            "checksum_verified": checksum_verified,
            "root_mode": "0755",
        },
        "manifest": manifest,
        "elf_closure": {
            "complete": closure["complete"],
            "launcher_static": closure["launcher_static"],
            "x86_64_dynamic_object_count": closure[
                "x86_64_dynamic_object_count"
            ],
            "provided_soname_count": closure["provided_soname_count"],
            "unresolved_userspace_dependencies": closure[
                "unresolved_userspace_dependencies"
            ],
            "non_relocatable_runpaths": closure[
                "non_relocatable_runpaths"
            ],
            "host_userspace_dependencies": closure[
                "host_userspace_dependencies"
            ],
            "maximum_bundled_glibc_abi": closure[
                "maximum_bundled_glibc_abi"
            ],
        },
        "isolated_environment": {
            "keys": ["HOME", "LANG", "PATH"],
            "host_ld_library_path": False,
            "host_rocm_path": False,
            "host_python_path": False,
            "version": version,
            "expected_engine_version": engine_version,
        },
        "doctor": doctor,
        "provider_smokes": smokes,
        "vl_smoke": vl_smoke,
        "public_hygiene": hygiene,
        "product_result": (
            {
                "path": "share/aima/qualification.json",
                "bytes": cli.product_result.resolve().stat().st_size,
                "sha256": sha256(cli.product_result.resolve()),
            }
            if require_native_vl
            else None
        ),
        "host_requirements": [
            "Linux x86_64 kernel ABI",
            "amdgpu kernel driver with KFD and render nodes",
            "AMD gfx1151 GPU",
        ],
    }
    if require_native_vl:
        result = seal_manifest(result)
    output = output_dir / "bundle.json"
    atomic_json(output, result)
    if require_native_vl:
        digest = sha256(output)
        output.with_name(output.name + ".sha256").write_text(
            f"{digest}  {output.name}\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": True,
                "archive_sha256": archive_sha256,
                "output": str(output_dir / "bundle.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
