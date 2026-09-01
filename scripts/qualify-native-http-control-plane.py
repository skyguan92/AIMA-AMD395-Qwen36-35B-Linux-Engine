#!/usr/bin/env python3
"""Qualify responsive native HTTP control-plane behavior during inference."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.public_hygiene import scan_bytes  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
)


SCHEMA = "aima-amd395-qwen36/native-http-control-plane/v1"
MODEL_ID = "aima-amd395-qwen36-35b"
MAXIMUM_HEALTH_LATENCY_MS = 1000.0
MAXIMUM_SHUTDOWN_ACK_LATENCY_MS = 1000.0
MAXIMUM_SIGNAL_SHUTDOWN_LATENCY_MS = 5000.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def require_gpu_idle() -> None:
    occupied = subprocess.run(
        ["fuser", "/dev/kfd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if occupied.returncode == 0:
        subprocess.run(["fuser", "-v", "/dev/kfd"], check=False)
        raise RuntimeError("/dev/kfd is owned by another process")


def host_fingerprint_sha256() -> str:
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
    require(bool(components), "stable host identity inputs are unavailable")
    payload = b"aima/native-http/host-fingerprint/v1\0" + b"\0".join(components)
    return hashlib.sha256(payload).hexdigest()


def publicize(value: Any, replacements: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, str):
        result = value
        for private, logical in replacements:
            result = result.replace(private, logical)
        if result.startswith(("/home/", "/Users/", "/data/", "/tmp/")):
            return "${AIMA_HOST_PATH}/" + Path(result).name
        return result
    if isinstance(value, list):
        return [publicize(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: publicize(item, replacements) for key, item in value.items()}
    return value


def http_json(
    port: int,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        status = response.status
    finally:
        connection.close()
    value = json.loads(raw)
    require(isinstance(value, dict), f"{method} {path} returned non-object JSON")
    return status, value


def chat_request(port: int, *, tokens: int, stream: bool = False) -> dict[str, Any]:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": "Count upward slowly and explain each number briefly.",
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": tokens,
        "stream": stream,
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read()
        status = response.status
        content_type = response.getheader("Content-Type") or ""
    finally:
        connection.close()
    if stream:
        return {
            "status": status,
            "content_type": content_type,
            "done": b"data: [DONE]" in raw,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    value = json.loads(raw)
    require(isinstance(value, dict), "chat response is not an object")
    message = value.get("choices", [{}])[0].get("message", {})
    metrics = value.get("aima_amd395", {})
    return {
        "status": status,
        "finish_reason": value.get("choices", [{}])[0].get("finish_reason"),
        "content_sha256": hashlib.sha256(
            str(message.get("content", "")).encode("utf-8")
        ).hexdigest(),
        "request_index": metrics.get("request_index"),
        "usage": value.get("usage"),
    }


def wait_ready(process: subprocess.Popen[str], timeout: float = 240.0) -> dict[str, Any]:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        events = selector.select(timeout)
    finally:
        selector.close()
    require(bool(events), "native server readiness timed out")
    line = process.stdout.readline()
    require(bool(line), f"native server exited before readiness: {process.poll()}")
    value = json.loads(line)
    require(isinstance(value, dict) and value.get("event") == "ready", "bad ready event")
    return value


def health_sample(port: int) -> tuple[float, dict[str, Any]]:
    started = time.monotonic()
    status, value = http_json(port, "GET", "/health", timeout=5)
    elapsed_ms = (time.monotonic() - started) * 1000
    require(status == 200 and value.get("status") == "ok", "health request failed")
    return elapsed_ms, value


def wait_busy(port: int, timeout: float = 10.0) -> tuple[list[float], bool]:
    deadline = time.monotonic() + timeout
    latencies: list[float] = []
    while time.monotonic() < deadline:
        latency, health = health_sample(port)
        latencies.append(latency)
        if health.get("busy") is True:
            return latencies, True
        time.sleep(0.01)
    return latencies, False


def poll_futures(port: int, futures: list[Future[Any]]) -> tuple[list[float], bool]:
    latencies: list[float] = []
    busy = False
    while not all(future.done() for future in futures):
        latency, health = health_sample(port)
        latencies.append(latency)
        busy = busy or health.get("busy") is True
        time.sleep(0.01)
    return latencies, busy


def sanitize_report(path: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(publicize(value, replacements), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def start_server(
    *, engine: Path, model_dir: Path, port: int, timeout_ms: int, report: Path,
    stderr: Path,
) -> tuple[subprocess.Popen[str], list[str], Any]:
    command = [
        str(engine),
        "serve",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        "8192",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--request-timeout-ms",
        str(timeout_ms),
        "--report",
        str(report),
    ]
    stderr_stream = stderr.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=stderr_stream, text=True
    )
    return process, command, stderr_stream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-engine-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--host-role", default="patch_qualification_amd395")
    args = parser.parse_args()

    engine = args.engine.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require(engine.is_file(), "native engine is missing")
    require(model_dir.is_dir(), "model directory is missing")
    require(
        not output_dir.exists() or not any(output_dir.iterdir()),
        "output directory must be empty",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    engine_digest = sha256(engine)
    require(engine_digest == args.expected_engine_sha256, "engine SHA-256 differs")
    build_info = json.loads(
        subprocess.run(
            [str(engine), "--build-info"], capture_output=True, text=True, check=True
        ).stdout
    )
    require(
        build_info.get("source_commit") == args.expected_source_commit,
        "engine source commit differs",
    )
    replacements = (
        (str(engine), "${AIMA_ENGINE}"),
        (str(engine.parent), "${AIMA_ENGINE_DIR}"),
        (str(model_dir), "${AIMA_MODEL_DIR}"),
        (str(output_dir), "${AIMA_OUTPUT_DIR}"),
        (str(ROOT), "${AIMA_REPO_ROOT}"),
    )
    observations: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    all_health_latencies: list[float] = []
    require_gpu_idle()

    port = free_loopback_port()
    first_report = raw_dir / "server.json"
    first_stderr = raw_dir / "server.stderr"
    process, command, stderr_stream = start_server(
        engine=engine,
        model_dir=model_dir,
        port=port,
        timeout_ms=600001,
        report=first_report,
        stderr=first_stderr,
    )
    try:
        ready = wait_ready(process)
        initial_latency, initial_health = health_sample(port)
        all_health_latencies.append(initial_latency)
        checks["large_timeout_preserved_exactly"] = (
            ready.get("request_timeout_ms") == 600001
            and initial_health.get("request_timeout_ms") == 600001
        )
        checks["idle_health_reports_not_busy"] = initial_health.get("busy") is False

        with ThreadPoolExecutor(max_workers=3) as pool:
            nonstream_future = pool.submit(chat_request, port, tokens=128)
            latencies, busy = poll_futures(port, [nonstream_future])
            all_health_latencies.extend(latencies)
            nonstream = nonstream_future.result()
            checks["health_responsive_during_nonstream"] = (
                busy and nonstream.get("status") == 200
            )

            stream_future = pool.submit(chat_request, port, tokens=128, stream=True)
            latencies, busy = poll_futures(port, [stream_future])
            all_health_latencies.extend(latencies)
            stream = stream_future.result()
            checks["health_responsive_during_stream"] = (
                busy and stream.get("status") == 200 and stream.get("done") is True
            )

            first = pool.submit(chat_request, port, tokens=128)
            latencies, first_busy = wait_busy(port)
            all_health_latencies.extend(latencies)
            second = pool.submit(chat_request, port, tokens=64)
            latencies, queued_busy = poll_futures(port, [first, second])
            all_health_latencies.extend(latencies)
            queued = [first.result(), second.result()]
            request_indices = [item.get("request_index") for item in queued]
            checks["two_chats_execute_serially_without_rejection"] = (
                first_busy
                and queued_busy
                and all(item.get("status") == 200 for item in queued)
                and all(isinstance(index, int) for index in request_indices)
                and request_indices[1] == request_indices[0] + 1
            )

            active = pool.submit(chat_request, port, tokens=256)
            latencies, shutdown_busy = wait_busy(port)
            all_health_latencies.extend(latencies)
            shutdown_started = time.monotonic()
            shutdown_status, shutdown = http_json(port, "POST", "/shutdown")
            shutdown_latency_ms = (time.monotonic() - shutdown_started) * 1000
            active_result = active.result()
            checks["shutdown_ack_responsive_during_chat"] = (
                shutdown_busy
                and shutdown_status == 200
                and shutdown.get("status") == "shutting_down"
                and shutdown_latency_ms <= MAXIMUM_SHUTDOWN_ACK_LATENCY_MS
            )
            checks["active_chat_completes_during_shutdown"] = (
                active_result.get("status") == 200
            )
        process.wait(timeout=300)
        checks["graceful_http_shutdown_exit_zero"] = process.returncode == 0
        observations["live_server"] = {
            "ready": ready,
            "initial_health": initial_health,
            "nonstream": nonstream,
            "stream": stream,
            "queued": queued,
            "active_during_shutdown": active_result,
            "shutdown_latency_ms": shutdown_latency_ms,
            "health_sample_count": len(all_health_latencies),
            "health_max_latency_ms": max(all_health_latencies),
        }
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=30)
        stderr_stream.close()
    checks["health_latency_bounded"] = (
        bool(all_health_latencies)
        and max(all_health_latencies) <= MAXIMUM_HEALTH_LATENCY_MS
    )

    require_gpu_idle()
    zero_port = free_loopback_port()
    zero_report = raw_dir / "zero-timeout-server.json"
    zero_stderr = raw_dir / "zero-timeout-server.stderr"
    zero_process, zero_command, zero_stderr_stream = start_server(
        engine=engine,
        model_dir=model_dir,
        port=zero_port,
        timeout_ms=0,
        report=zero_report,
        stderr=zero_stderr,
    )
    incomplete = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        zero_ready = wait_ready(zero_process)
        incomplete.settimeout(5)
        incomplete.connect(("127.0.0.1", zero_port))
        incomplete.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 100000\r\n\r\n{"
        )
        time.sleep(0.1)
        signal_started = time.monotonic()
        zero_process.send_signal(signal.SIGTERM)
        zero_process.wait(timeout=MAXIMUM_SIGNAL_SHUTDOWN_LATENCY_MS / 1000)
        signal_latency_ms = (time.monotonic() - signal_started) * 1000
        checks["zero_timeout_incomplete_read_is_interruptible"] = (
            zero_ready.get("request_timeout_ms") == 0
            and zero_process.returncode == 0
            and signal_latency_ms <= MAXIMUM_SIGNAL_SHUTDOWN_LATENCY_MS
        )
        observations["zero_timeout_signal"] = {
            "ready": zero_ready,
            "signal_shutdown_latency_ms": signal_latency_ms,
            "exit_code": zero_process.returncode,
        }
    finally:
        incomplete.close()
        if zero_process.poll() is None:
            zero_process.kill()
            zero_process.wait(timeout=30)
        zero_stderr_stream.close()

    raw_files: list[Path] = []
    for report in (first_report, zero_report):
        for path in (
            report,
            report.with_suffix(".language.json"),
            report.with_suffix(".visual.json"),
        ):
            require(path.is_file(), f"server report is missing: {path.name}")
            sanitize_report(path, replacements)
            raw_files.append(path)
    for path in (first_stderr, zero_stderr):
        text = publicize(path.read_text(encoding="utf-8", errors="replace"), replacements)
        path.write_text(str(text), encoding="utf-8")
        raw_files.append(path)

    observations = publicize(observations, replacements)
    commands = publicize({"live_server": command, "zero_timeout": zero_command}, replacements)
    public_payload = json.dumps(
        {"commands": commands, "observations": observations}, sort_keys=True
    ).encode("utf-8")
    checks["artifact_paths_sanitized"] = not any(
        prefix in public_payload for prefix in (b"/home/", b"/Users/", b"/data/", b"/tmp/")
    )
    checks["raw_artifacts_public"] = not any(
        scan_bytes(path.relative_to(output_dir).as_posix(), path.read_bytes())
        for path in raw_files
    )
    qualified = bool(checks) and all(checks.values())
    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "release": "1.5.1-native-vl.5",
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "complete": True,
            "qualified": qualified,
            "candidate": {
                "native_engine_sha256": engine_digest,
                "native_source_commit": build_info["source_commit"],
                "engine_version": build_info["version"],
            },
            "host": {
                "role": args.host_role,
                "fingerprint_sha256": host_fingerprint_sha256(),
                "kernel": os.uname().release,
                "architecture": os.uname().machine,
            },
            "limits": {
                "maximum_health_latency_ms": MAXIMUM_HEALTH_LATENCY_MS,
                "maximum_shutdown_ack_latency_ms": MAXIMUM_SHUTDOWN_ACK_LATENCY_MS,
                "maximum_signal_shutdown_latency_ms": MAXIMUM_SIGNAL_SHUTDOWN_LATENCY_MS,
            },
            "commands": commands,
            "observations": observations,
            "checks": checks,
            "raw_artifacts": {
                path.name: file_component(path, f"raw/{path.name}") for path in raw_files
            },
            "decision": {
                "health_remains_responsive_during_inference": True,
                "chat_execution_remains_serial": True,
                "shutdown_interrupts_unbounded_reads": True,
                "http_control_plane_qualified": qualified,
            },
        }
    )
    output = output_dir / "http-control-plane.json"
    digest = atomic_json(output, payload)
    output.with_name(output.name + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"qualified": qualified, "output": str(output), "sha256": digest}))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
