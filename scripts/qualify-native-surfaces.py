#!/usr/bin/env python3
"""Qualify startup, prefix cache and resident native HTTP lifecycle."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def publicize(value: Any, model_dir: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(ROOT), "${AIMA_REPO_ROOT}").replace(
            str(model_dir), "${AIMA_MODEL_DIR}"
        )
    if isinstance(value, list):
        return [publicize(item, model_dir) for item in value]
    if isinstance(value, dict):
        return {
            key: publicize(item, model_dir) for key, item in value.items()
        }
    return value


def http_json(
    port: int,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    encoded = (
        json.dumps(body, separators=(",", ":")).encode("utf-8")
        if body is not None
        else None
    )
    headers = {"Content-Type": "application/json"} if encoded else {}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.request(method, path, body=encoded, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{method} {path} returned invalid JSON: {raw[:200]!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{method} {path} returned non-object JSON")
    return response.status, payload


def read_json_line(
    process: subprocess.Popen[str], timeout_seconds: float
) -> dict[str, Any]:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    events = selector.select(timeout_seconds)
    selector.close()
    if not events:
        process.terminate()
        raise RuntimeError("native server did not emit lifecycle JSON in time")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(
            f"native server exited before lifecycle JSON: {process.poll()}"
        )
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise RuntimeError("native server emitted non-object lifecycle JSON")
    return payload


def make_exact_chat_user(
    engine: Path, model_dir: Path, context_tokens: int
) -> tuple[str, dict[str, Any]]:
    if context_tokens < 13:
        raise RuntimeError("chat fixture context must be at least 13")
    # "test" is one token. The production disable-thinking template
    # contributes twelve fixed tokens around the user content.
    user = "test" + " test" * (context_tokens - 13)
    completed = subprocess.run(
        [
            str(engine),
            "chat-template-probe",
            "--model-dir",
            str(model_dir),
            "--user",
            user,
            "--disable-thinking",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    if (
        not isinstance(payload, dict)
        or payload.get("complete") is not True
        or len(payload.get("token_ids", [])) != context_tokens
    ):
        raise RuntimeError("failed to construct exact-length native chat fixture")
    return user, payload


def prefix_cache_report_qualified(
    payload: dict[str, Any],
    *,
    engine_sha256: str,
    minimum_ttft_speedup: float,
    minimum_decode_retention: float,
) -> bool:
    try:
        cold, hit = payload["requests"]
        speedup = float(cold["prefill_wall_ms"]) / float(
            hit["prefill_wall_ms"]
        )
        decode_retention = float(hit["decode_tokens_per_second"]) / float(
            cold["decode_tokens_per_second"]
        )
        return bool(
            payload["schema"]
            == "aima-amd395-qwen36/native-resident-session-probe/v1"
            and payload["complete"] is True
            and payload["model_loads"] == 1
            and payload["request_count"] == 2
            and payload["repeat_tokens_identical"] is True
            and cold["prefix_cache_lookup"] == "miss"
            and hit["prefix_cache_lookup"] == "exact"
            and int(hit["prefix_cache_matched_tokens"]) == 32768
            and int(hit["prefix_cache_suffix_tokens"]) == 0
            and int(hit["prefix_cache_suffix_aot_launches"]) == 0
            and int(hit["prefix_cache_suffix_native_launches"]) == 0
            and cold["output_token_ids_sha256"]
            == hit["output_token_ids_sha256"]
            and cold["first_token_certified"] is True
            and hit["first_token_certified"] is True
            and cold["all_decode_tokens_certified"] is True
            and hit["all_decode_tokens_certified"] is True
            and speedup >= minimum_ttft_speedup
            and decode_retention >= minimum_decode_retention
            and payload["qualification"]["engine_sha256"] == engine_sha256
        )
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return False


def run_prefix_cache(
    *,
    engine: Path,
    model_dir: Path,
    output_dir: Path,
    engine_sha256: str,
    minimum_ttft_speedup: float,
    minimum_decode_retention: float,
) -> Path:
    report = output_dir / "raw" / "prefix-cache-q32768-o512.json"
    if report.is_file() and prefix_cache_report_qualified(
        load_json(report),
        engine_sha256=engine_sha256,
        minimum_ttft_speedup=minimum_ttft_speedup,
        minimum_decode_retention=minimum_decode_retention,
    ):
        return report
    load_report = report.with_name(report.stem + ".load.json")
    command = [
        str(engine),
        "resident-session-probe",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        "32768",
        "--uniform-input-token-id",
        "1",
        "--max-new-tokens",
        "512",
        "--requests",
        "2",
        "--report",
        str(load_report),
    ]
    print(
        json.dumps({"event": "prefix_cache_run_start"}, sort_keys=True),
        flush=True,
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("native prefix-cache run emitted non-object JSON")
    payload["qualification"] = {
        "command": command,
        "engine_sha256": engine_sha256,
        "load_report": str(load_report),
        "load_report_sha256": sha256(load_report),
    }
    payload = publicize(payload, model_dir)
    atomic_json(report, payload)
    if completed.returncode != 0 or not prefix_cache_report_qualified(
        payload,
        engine_sha256=engine_sha256,
        minimum_ttft_speedup=minimum_ttft_speedup,
        minimum_decode_retention=minimum_decode_retention,
    ):
        raise RuntimeError(
            f"native prefix-cache qualification failed: {report}"
        )
    print(
        json.dumps({"event": "prefix_cache_run_complete"}, sort_keys=True),
        flush=True,
    )
    return report


def server_run_qualified(
    payload: dict[str, Any],
    *,
    engine_sha256: str,
    with_chat: bool,
) -> bool:
    try:
        ready = payload["ready"]
        stopped = payload["stopped"]
        common = bool(
            payload["complete"] is True
            and payload["engine_sha256"] == engine_sha256
            and ready["event"] == "ready"
            and ready["runtime_python"] is False
            and ready["runtime_torch"] is False
            and ready["runtime_vllm"] is False
            and ready["runtime_triton"] is False
            and payload["health"]["status"] == 200
            and payload["health"]["body"]["model_loaded"] is True
            and payload["health"]["body"]["resident"] is True
            and payload["models"]["status"] == 200
            and payload["models"]["body"]["data"][0]["id"]
            == "aima-amd395-qwen36-35b"
            and payload["shutdown"]["status"] == 200
            and payload["shutdown"]["body"]["status"] == "shutting_down"
            and stopped["event"] == "stopped"
            and stopped["model_loads"] == 1
        )
        if not with_chat:
            return common and stopped["served"] == 0
        first, second = payload["chat"]
        first_metrics = first["body"]["aima_amd395"]
        second_metrics = second["body"]["aima_amd395"]
        return bool(
            common
            and stopped["served"] == 2
            and first["status"] == 200
            and second["status"] == 200
            and first_metrics["model_loads"] == 1
            and second_metrics["model_loads"] == 1
            and first_metrics["prefix_cache"]["lookup"] == "miss"
            and second_metrics["prefix_cache"]["lookup"] == "exact"
            and first_metrics["output_token_ids_sha256"]
            == second_metrics["output_token_ids_sha256"]
            and float(second_metrics["ttft_ms"])
            < float(first_metrics["ttft_ms"])
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def run_server(
    *,
    engine: Path,
    model_dir: Path,
    output_dir: Path,
    engine_sha256: str,
    run_index: int,
    port: int,
    user: str,
    with_chat: bool,
) -> Path:
    report = output_dir / "raw" / f"server-q8192-r{run_index}.json"
    if report.is_file() and server_run_qualified(
        load_json(report),
        engine_sha256=engine_sha256,
        with_chat=with_chat,
    ):
        return report
    load_report = report.with_name(report.stem + ".load.json")
    stderr_path = report.with_name(report.stem + ".stderr.txt")
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
        "--report",
        str(load_report),
    ]
    print(
        json.dumps(
            {
                "event": "server_run_start",
                "run_index": run_index,
                "with_chat": with_chat,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
        try:
            ready = read_json_line(process, 180)
            health_status, health = http_json(port, "GET", "/health")
            models_status, models = http_json(port, "GET", "/v1/models")
            chats: list[dict[str, Any]] = []
            if with_chat:
                request = {
                    "model": "aima-amd395-qwen36-35b",
                    "messages": [{"role": "user", "content": user}],
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": 4,
                }
                for _ in range(2):
                    status, body = http_json(
                        port, "POST", "/v1/chat/completions", request
                    )
                    chats.append({"status": status, "body": body})
            shutdown_status, shutdown = http_json(
                port, "POST", "/shutdown"
            )
            stopped = read_json_line(process, 30)
            returncode = process.wait(timeout=30)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    payload = {
        "schema": "aima-amd395-qwen36/native-server-qualification-run/v1",
        "complete": returncode == 0,
        "engine_sha256": engine_sha256,
        "command": command,
        "load_report": str(load_report),
        "load_report_sha256": sha256(load_report),
        "ready": ready,
        "health": {"status": health_status, "body": health},
        "models": {"status": models_status, "body": models},
        "chat": chats,
        "shutdown": {"status": shutdown_status, "body": shutdown},
        "stopped": stopped,
        "stderr": str(stderr_path),
    }
    payload = publicize(payload, model_dir)
    atomic_json(report, payload)
    if not server_run_qualified(
        payload, engine_sha256=engine_sha256, with_chat=with_chat
    ):
        raise RuntimeError(f"native server qualification failed: {report}")
    print(
        json.dumps(
            {
                "event": "server_run_complete",
                "run_index": run_index,
                "command_to_ready_wall_ms": ready[
                    "command_to_ready_wall_ms"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine", type=Path, default=Path("build/native/aima-engine-native")
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=18080)
    parser.add_argument(
        "--startup-ceiling-ms", type=float, default=51408.20149378851
    )
    parser.add_argument(
        "--minimum-prefix-ttft-speedup",
        type=float,
        default=110.11994260509346,
    )
    parser.add_argument(
        "--minimum-prefix-decode-retention",
        type=float,
        default=0.999653457424567,
    )
    cli = parser.parse_args()

    engine = cli.engine.expanduser().resolve()
    model_dir = cli.model_dir.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    if not engine.is_file() or not os.access(engine, os.X_OK):
        raise SystemExit(f"native engine is not executable: {engine}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")
    engine_sha256 = sha256(engine)

    user, chat_fixture = make_exact_chat_user(engine, model_dir, 8192)
    prefix_report = run_prefix_cache(
        engine=engine,
        model_dir=model_dir,
        output_dir=output_dir,
        engine_sha256=engine_sha256,
        minimum_ttft_speedup=cli.minimum_prefix_ttft_speedup,
        minimum_decode_retention=cli.minimum_prefix_decode_retention,
    )
    server_reports = [
        run_server(
            engine=engine,
            model_dir=model_dir,
            output_dir=output_dir,
            engine_sha256=engine_sha256,
            run_index=run_index,
            port=cli.port_base + run_index,
            user=user,
            with_chat=run_index == 1,
        )
        for run_index in (1, 2, 3)
    ]

    prefix_payload = load_json(prefix_report)
    cold, hit = prefix_payload["requests"]
    prefix_speedup = float(cold["prefill_wall_ms"]) / float(
        hit["prefill_wall_ms"]
    )
    prefix_decode_retention = float(
        hit["decode_tokens_per_second"]
    ) / float(cold["decode_tokens_per_second"])
    server_payloads = [load_json(path) for path in server_reports]
    startup_runs = [
        float(payload["ready"]["command_to_ready_wall_ms"])
        for payload in server_payloads
    ]
    startup_median = statistics.median(startup_runs)
    first_chat, second_chat = server_payloads[0]["chat"]
    first_metrics = first_chat["body"]["aima_amd395"]
    second_metrics = second_chat["body"]["aima_amd395"]
    prefix_pass = prefix_cache_report_qualified(
        prefix_payload,
        engine_sha256=engine_sha256,
        minimum_ttft_speedup=cli.minimum_prefix_ttft_speedup,
        minimum_decode_retention=cli.minimum_prefix_decode_retention,
    )
    startup_pass = startup_median <= cli.startup_ceiling_ms
    http_pass = server_run_qualified(
        server_payloads[0],
        engine_sha256=engine_sha256,
        with_chat=True,
    )
    result = {
        "schema": "aima-amd395-qwen36/native-product-surfaces/v1",
        "complete": True,
        "qualified": prefix_pass and startup_pass and http_pass,
        "engine": {
            "path": "${AIMA_REPO_ROOT}/build/native/aima-engine-native",
            "sha256": engine_sha256,
        },
        "model_dir": "${AIMA_MODEL_DIR}",
        "host": {
            "hostname": os.uname().nodename,
            "sysname": os.uname().sysname,
            "release": os.uname().release,
            "machine": os.uname().machine,
        },
        "prefix_cache": {
            "report": str(prefix_report.relative_to(output_dir)),
            "report_sha256": sha256(prefix_report),
            "context_tokens": 32768,
            "output_tokens": 512,
            "cold_ttft_ms": cold["prefill_wall_ms"],
            "hit_ttft_ms": hit["prefill_wall_ms"],
            "ttft_speedup": prefix_speedup,
            "minimum_ttft_speedup": cli.minimum_prefix_ttft_speedup,
            "cold_decode_tps": cold["decode_tokens_per_second"],
            "hit_decode_tps": hit["decode_tokens_per_second"],
            "decode_retention": prefix_decode_retention,
            "minimum_decode_retention": (
                cli.minimum_prefix_decode_retention
            ),
            "output_token_sha256_equal": (
                cold["output_token_ids_sha256"]
                == hit["output_token_ids_sha256"]
            ),
            "pass": prefix_pass,
        },
        "startup": {
            "context_tokens": 8192,
            "protocol": "three fresh resident HTTP processes",
            "command_to_ready_runs_ms": startup_runs,
            "command_to_ready_median_ms": startup_median,
            "ceiling_ms": cli.startup_ceiling_ms,
            "pass": startup_pass,
        },
        "http": {
            "server_reports": [
                str(path.relative_to(output_dir)) for path in server_reports
            ],
            "server_report_sha256": [
                sha256(path) for path in server_reports
            ],
            "chat_fixture_token_count": len(chat_fixture["token_ids"]),
            "resident": True,
            "model_loads": first_metrics["model_loads"],
            "served_requests": server_payloads[0]["stopped"]["served"],
            "first_request": {
                "prompt_tokens": first_chat["body"]["usage"][
                    "prompt_tokens"
                ],
                "completion_tokens": first_chat["body"]["usage"][
                    "completion_tokens"
                ],
                "prefix_lookup": first_metrics["prefix_cache"]["lookup"],
                "ttft_ms": first_metrics["ttft_ms"],
                "decode_tps": first_metrics["decode_tokens_per_second"],
            },
            "second_identical_request": {
                "prompt_tokens": second_chat["body"]["usage"][
                    "prompt_tokens"
                ],
                "completion_tokens": second_chat["body"]["usage"][
                    "completion_tokens"
                ],
                "prefix_lookup": second_metrics["prefix_cache"]["lookup"],
                "ttft_ms": second_metrics["ttft_ms"],
                "decode_tps": second_metrics["decode_tokens_per_second"],
            },
            "output_token_sha256_equal": (
                first_metrics["output_token_ids_sha256"]
                == second_metrics["output_token_ids_sha256"]
            ),
            "health_before_request": True,
            "models_endpoint": True,
            "shutdown_endpoint": True,
            "runtime_python": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
            "pass": http_pass,
        },
    }
    atomic_json(output_dir / "surfaces.json", result)
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": result["qualified"],
                "output": str(output_dir / "surfaces.json"),
            },
            sort_keys=True,
        )
    )
    if not result["qualified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
