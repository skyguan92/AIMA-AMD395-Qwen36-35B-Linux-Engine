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
import re
import selectors
import shutil
import statistics
import subprocess
import time
from typing import Any

from native_text_metrics import text_path_is_idle


ROOT = Path(__file__).resolve().parents[1]
STARTUP_CEILING_MS = 44_900.0
MINIMUM_PREFIX_PAIRS = 5
DEFAULT_PREFIX_PAIRS = 5
V151_SOURCE_COMMIT = "65c198415709dad6d046c247acab3dc9df2a95a0"
V151_VERSION = "1.5.1-native"

# These are the exact values from the immutable v1.5.1 public evidence, not
# qualification floors.  README/PERFORMANCE round them to 2637x and 1.0003x;
# treating those display strings as lower bounds would reject the frozen
# release that produced them.
FROZEN_V151_PREFIX_OBSERVATION = {
    "ttft_speedup": 2636.9250000546567,
    "decode_retention": 1.0002958825348782,
}

# The old product contract remains an independent safety floor.  Native-VL
# no-regression is stricter: paired candidate/release medians must be >= 1.0.
MINIMUM_PREFIX_TTFT_SPEEDUP = 110.11994260509346
MINIMUM_PREFIX_DECODE_RETENTION = 0.999653457424567


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


def publicize(
    value: Any,
    *,
    engine: Path,
    model_dir: Path,
    output_dir: Path,
    baseline_engine: Path | None = None,
) -> Any:
    if isinstance(value, str):
        result = value
        if baseline_engine is not None:
            result = result.replace(
                str(baseline_engine), "${AIMA_BASELINE_ENGINE}"
            )
        result = (
            result.replace(str(engine), "${AIMA_ENGINE}")
            .replace(str(model_dir), "${AIMA_MODEL_DIR}")
            .replace(str(output_dir), "${AIMA_OUTPUT_DIR}")
            .replace(str(ROOT), "${AIMA_REPO_ROOT}")
        )
        return result
    if isinstance(value, list):
        return [
            publicize(
                item,
                engine=engine,
                model_dir=model_dir,
                output_dir=output_dir,
                baseline_engine=baseline_engine,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: publicize(
                item,
                engine=engine,
                model_dir=model_dir,
                output_dir=output_dir,
                baseline_engine=baseline_engine,
            )
            for key, item in value.items()
        }
    return value


def engine_build_info(engine: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(engine), "--build-info"],
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("native --build-info returned a non-object")
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


def resident_probe_text_path_is_idle(request: dict[str, Any]) -> bool:
    try:
        if request.get("mrope_enabled") is not False:
            return False
        for name in (
            "mrope_position_upload_bytes",
            "mrope_full_attention_launches",
            "mrope_decode_steps",
            "prefill_vl_unified_attention_launches",
            "vl_logical_projection_tokens",
            "vl_logical_projection_plan_count",
            "vl_logical_projection_workspace_bytes",
        ):
            if int(request.get(name, -1)) != 0:
                return False
        return bool(
            request.get("vl_logical_projections_enabled") is False
            and float(
                request.get("vl_logical_projection_plan_build_wall_ms", -1)
            )
            == 0.0
        )
    except (TypeError, ValueError):
        return False


def prefix_cache_report_valid(
    payload: dict[str, Any],
    *,
    engine_sha256: str,
    require_text_path_idle: bool = True,
    require_active_kv_reuse: bool = True,
) -> bool:
    try:
        cold, hit = payload["requests"]
        return bool(
            payload["schema"]
            == "aima-amd395-qwen36/native-resident-session-probe/v1"
            and payload["complete"] is True
            and payload["model_loads"] == 1
            and payload["request_count"] == 2
            and payload["repeat_tokens_identical"] is True
            and payload.get("runtime_python") is False
            and payload.get("runtime_torch") is False
            and payload.get("runtime_vllm") is False
            and payload.get("runtime_triton") is False
            and cold["prefix_cache_lookup"] == "miss"
            and hit["prefix_cache_lookup"] == "exact"
            and int(hit["prefix_cache_matched_tokens"]) == 32768
            and int(hit["prefix_cache_suffix_tokens"]) == 0
            and int(hit["prefix_cache_suffix_aot_launches"]) == 0
            and int(hit["prefix_cache_suffix_native_launches"]) == 0
            and (
                not require_active_kv_reuse
                or (
                    float(cold.get("prefix_cache_restore_wall_ms", -1.0))
                    == 0.0
                    and float(hit["prefix_cache_restore_wall_ms"]) > 0.0
                    and cold.get("prefix_cache_active_kv_reused", False)
                    is False
                    and hit["prefix_cache_active_kv_reused"] is True
                )
            )
            and cold["output_token_ids_sha256"]
            == hit["output_token_ids_sha256"]
            and cold["first_token_certified"] is True
            and hit["first_token_certified"] is True
            and cold["all_decode_tokens_certified"] is True
            and hit["all_decode_tokens_certified"] is True
            and (
                not require_text_path_idle
                or (
                    resident_probe_text_path_is_idle(cold)
                    and resident_probe_text_path_is_idle(hit)
                )
            )
            and payload["qualification"]["engine_sha256"] == engine_sha256
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def prefix_cache_report_qualified(
    payload: dict[str, Any],
    *,
    engine_sha256: str,
    minimum_ttft_speedup: float,
    minimum_decode_retention: float,
) -> bool:
    if not prefix_cache_report_valid(
        payload, engine_sha256=engine_sha256
    ):
        return False
    try:
        cold, hit = payload["requests"]
        speedup = float(cold["prefill_wall_ms"]) / float(
            hit["prefill_wall_ms"]
        )
        decode_retention = float(hit["decode_tokens_per_second"]) / float(
            cold["decode_tokens_per_second"]
        )
        return bool(
            speedup >= minimum_ttft_speedup
            and decode_retention >= minimum_decode_retention
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
    if report.is_file() and prefix_cache_report_valid(
        load_json(report), engine_sha256=engine_sha256
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
    payload = publicize(
        payload,
        engine=engine,
        model_dir=model_dir,
        output_dir=output_dir,
    )
    atomic_json(report, payload)
    if completed.returncode != 0 or not prefix_cache_report_valid(
        payload, engine_sha256=engine_sha256
    ):
        raise RuntimeError(
            f"native prefix-cache qualification failed: {report}"
        )
    print(
        json.dumps(
            {
                "event": "prefix_cache_run_complete",
                "qualified": prefix_cache_report_qualified(
                    payload,
                    engine_sha256=engine_sha256,
                    minimum_ttft_speedup=minimum_ttft_speedup,
                    minimum_decode_retention=minimum_decode_retention,
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def prefix_pair_order(pair_index: int) -> tuple[str, str]:
    if pair_index <= 0:
        raise ValueError("prefix pair index must be positive")
    return (
        ("baseline", "candidate")
        if pair_index % 2 == 1
        else ("candidate", "baseline")
    )


def parse_cpu_list(text: str) -> str:
    if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", text):
        raise argparse.ArgumentTypeError(
            "CPU list must use taskset syntax such as 8-15 or 2,4-7"
        )
    for item in text.split(","):
        bounds = [int(value) for value in item.split("-")]
        if len(bounds) == 2 and bounds[0] > bounds[1]:
            raise argparse.ArgumentTypeError(
                "CPU list ranges must be ascending"
            )
    return text


def paired_prefix_report_path(
    output_dir: Path, pair_index: int, role: str
) -> Path:
    return (
        output_dir
        / "raw"
        / "prefix-cache-q32768-o512-paired"
        / f"pair-{pair_index:02d}"
        / f"{role}.json"
    )


def paired_prefix_report_valid(
    payload: dict[str, Any],
    *,
    role: str,
    pair_index: int,
    order: tuple[str, str],
    engine_sha256: str,
    cpu_list: str | None,
    taskset_sha256: str | None,
) -> bool:
    if not prefix_cache_report_valid(
        payload,
        engine_sha256=engine_sha256,
        require_text_path_idle=role == "candidate",
        require_active_kv_reuse=role == "candidate",
    ):
        return False
    try:
        qualification = payload["qualification"]
        affinity = qualification.get("cpu_affinity")
        expected_affinity = (
            None
            if cpu_list is None
            else {
                "cpu_list": cpu_list,
                "taskset_sha256": taskset_sha256,
            }
        )
        return bool(
            qualification.get("schema")
            == "aima-amd395-qwen36/native-paired-prefix-binding/v2"
            and qualification.get("engine_role") == role
            and int(qualification.get("pair_index")) == pair_index
            and tuple(qualification.get("pair_order", [])) == order
            and affinity == expected_affinity
        )
    except (KeyError, TypeError, ValueError):
        return False


def paired_prefix_artifacts_valid(
    report: Path, payload: dict[str, Any]
) -> bool:
    load_report = report.with_suffix(".load.json")
    stderr_path = report.with_suffix(".stderr.txt")
    try:
        qualification = payload["qualification"]
        return bool(
            load_report.is_file()
            and stderr_path.is_file()
            and stderr_path.stat().st_size == 0
            and qualification.get("load_report_sha256")
            == sha256(load_report)
            and qualification.get("stderr_sha256") == sha256(stderr_path)
        )
    except (KeyError, OSError, TypeError):
        return False


def prefix_cache_measurement(payload: dict[str, Any]) -> dict[str, float]:
    cold, hit = payload["requests"]
    cold_ttft_ms = float(cold["prefill_wall_ms"])
    hit_ttft_ms = float(hit["prefill_wall_ms"])
    cold_decode_tps = float(cold["decode_tokens_per_second"])
    hit_decode_tps = float(hit["decode_tokens_per_second"])
    if min(
        cold_ttft_ms,
        hit_ttft_ms,
        cold_decode_tps,
        hit_decode_tps,
    ) <= 0.0:
        raise RuntimeError("prefix-cache measurement must be positive")
    return {
        "cold_ttft_ms": cold_ttft_ms,
        "hit_ttft_ms": hit_ttft_ms,
        "ttft_speedup": cold_ttft_ms / hit_ttft_ms,
        "cold_decode_tps": cold_decode_tps,
        "hit_decode_tps": hit_decode_tps,
        "decode_retention": hit_decode_tps / cold_decode_tps,
    }


def summarize_paired_prefix_cache(
    pairs: list[dict[str, Any]], *, required_pair_count: int
) -> dict[str, Any]:
    if pairs:
        paired_medians = {
            name: float(
                statistics.median(
                    pair["candidate_over_baseline"][name]
                    for pair in pairs
                )
            )
            for name in ("ttft_speedup", "decode_retention")
        }
        baseline_medians = {
            name: float(
                statistics.median(
                    pair["measurements"]["baseline"][name]
                    for pair in pairs
                )
            )
            for name in prefix_cache_measurement_keys()
        }
        candidate_medians = {
            name: float(
                statistics.median(
                    pair["measurements"]["candidate"][name]
                    for pair in pairs
                )
            )
            for name in prefix_cache_measurement_keys()
        }
    else:
        paired_medians = {}
        baseline_medians = {}
        candidate_medians = {}
    complete = (
        len(pairs) == required_pair_count
        and len(pairs) >= MINIMUM_PREFIX_PAIRS
    )
    checks = {
        "minimum_five_pairs": len(pairs) >= MINIMUM_PREFIX_PAIRS,
        "all_requested_pairs_complete": len(pairs) == required_pair_count,
        "candidate_ttft_speedup_not_below_release": bool(
            pairs and paired_medians["ttft_speedup"] >= 1.0
        ),
        "candidate_decode_retention_not_below_release": bool(
            pairs and paired_medians["decode_retention"] >= 1.0
        ),
        "candidate_ttft_speedup_above_legacy_floor": bool(
            pairs
            and candidate_medians["ttft_speedup"]
            >= MINIMUM_PREFIX_TTFT_SPEEDUP
        ),
        "candidate_decode_retention_above_legacy_floor": bool(
            pairs
            and candidate_medians["decode_retention"]
            >= MINIMUM_PREFIX_DECODE_RETENTION
        ),
    }
    return {
        "complete": complete,
        "qualified": complete and all(checks.values()),
        "pair_count": len(pairs),
        "required_pair_count": required_pair_count,
        "minimum_pair_count": MINIMUM_PREFIX_PAIRS,
        "pairs": pairs,
        "paired_candidate_over_baseline_medians": paired_medians,
        "baseline_medians": baseline_medians,
        "candidate_medians": candidate_medians,
        "legacy_absolute_floors": {
            "ttft_speedup": MINIMUM_PREFIX_TTFT_SPEEDUP,
            "decode_retention": MINIMUM_PREFIX_DECODE_RETENTION,
        },
        "frozen_v151_single_run_observation_not_a_floor": (
            FROZEN_V151_PREFIX_OBSERVATION
        ),
        "checks": checks,
    }


def prefix_cache_measurement_keys() -> tuple[str, ...]:
    return (
        "cold_ttft_ms",
        "hit_ttft_ms",
        "ttft_speedup",
        "cold_decode_tps",
        "hit_decode_tps",
        "decode_retention",
    )


def run_paired_prefix_report(
    *,
    role: str,
    engine: Path,
    candidate_engine: Path,
    baseline_engine: Path,
    model_dir: Path,
    output_dir: Path,
    engine_sha256: str,
    pair_index: int,
    order: tuple[str, str],
    sequence_index: int,
    cpu_list: str | None,
    taskset: Path | None,
    taskset_sha256: str | None,
) -> Path:
    report = paired_prefix_report_path(output_dir, pair_index, role)
    if report.is_file():
        existing = load_json(report)
        if paired_prefix_report_valid(
            existing,
            role=role,
            pair_index=pair_index,
            order=order,
            engine_sha256=engine_sha256,
            cpu_list=cpu_list,
            taskset_sha256=taskset_sha256,
        ) and paired_prefix_artifacts_valid(report, existing):
            return report
    report.parent.mkdir(parents=True, exist_ok=True)
    load_report = report.with_suffix(".load.json")
    stderr_path = report.with_suffix(".stderr.txt")
    engine_command = [
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
    command = (
        [str(taskset), "--cpu-list", cpu_list, *engine_command]
        if taskset is not None and cpu_list is not None
        else engine_command
    )
    print(
        json.dumps(
            {
                "event": "paired_prefix_run_start",
                "pair_index": pair_index,
                "pair_order": list(order),
                "engine_role": role,
                "sequence_index": sequence_index,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"paired prefix run emitted invalid JSON for {role}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("paired prefix run emitted non-object JSON")
    payload["qualification"] = {
        "schema": "aima-amd395-qwen36/native-paired-prefix-binding/v2",
        "engine_role": role,
        "engine_sha256": engine_sha256,
        "pair_index": pair_index,
        "pair_order": list(order),
        "sequence_index": sequence_index,
        "cpu_affinity": (
            None
            if cpu_list is None
            else {
                "cpu_list": cpu_list,
                "taskset_sha256": taskset_sha256,
            }
        ),
        "command": command,
        "load_report": str(load_report),
        "load_report_sha256": (
            sha256(load_report) if load_report.is_file() else None
        ),
        "stderr": str(stderr_path),
        "stderr_sha256": sha256(stderr_path),
    }
    payload = publicize(
        payload,
        engine=candidate_engine,
        baseline_engine=baseline_engine,
        model_dir=model_dir,
        output_dir=output_dir,
    )
    atomic_json(report, payload)
    if (
        completed.returncode != 0
        or not paired_prefix_report_valid(
            payload,
            role=role,
            pair_index=pair_index,
            order=order,
            engine_sha256=engine_sha256,
            cpu_list=cpu_list,
            taskset_sha256=taskset_sha256,
        )
        or not paired_prefix_artifacts_valid(report, payload)
    ):
        raise RuntimeError(
            f"paired prefix run failed with exit {completed.returncode}: "
            f"{role} pair {pair_index}"
        )
    print(
        json.dumps(
            {
                "event": "paired_prefix_run_complete",
                "pair_index": pair_index,
                "engine_role": role,
                "sequence_index": sequence_index,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def qualify_paired_prefix_cache(
    *,
    candidate_engine: Path,
    baseline_engine: Path,
    model_dir: Path,
    output_dir: Path,
    candidate_sha256: str,
    baseline_sha256: str,
    pair_count: int,
    cpu_list: str | None,
    taskset: Path | None,
    taskset_sha256: str | None,
) -> dict[str, Any]:
    engines = {
        "candidate": candidate_engine,
        "baseline": baseline_engine,
    }
    engine_sha256 = {
        "candidate": candidate_sha256,
        "baseline": baseline_sha256,
    }
    sequence_index = 0
    for pair_index in range(1, pair_count + 1):
        order = prefix_pair_order(pair_index)
        for role in order:
            sequence_index += 1
            run_paired_prefix_report(
                role=role,
                engine=engines[role],
                candidate_engine=candidate_engine,
                baseline_engine=baseline_engine,
                model_dir=model_dir,
                output_dir=output_dir,
                engine_sha256=engine_sha256[role],
                pair_index=pair_index,
                order=order,
                sequence_index=sequence_index,
                cpu_list=cpu_list,
                taskset=taskset,
                taskset_sha256=taskset_sha256,
            )
    pairs: list[dict[str, Any]] = []
    for pair_index in range(1, pair_count + 1):
        order = prefix_pair_order(pair_index)
        paths = {
            role: paired_prefix_report_path(
                output_dir, pair_index, role
            )
            for role in ("baseline", "candidate")
        }
        payloads = {
            role: load_json(path) for role, path in paths.items()
            if path.is_file()
        }
        if not all(
            role in payloads
            and paired_prefix_report_valid(
                payloads[role],
                role=role,
                pair_index=pair_index,
                order=order,
                engine_sha256=engine_sha256[role],
                cpu_list=cpu_list,
                taskset_sha256=taskset_sha256,
            )
            and paired_prefix_artifacts_valid(path, payloads[role])
            for role, path in paths.items()
        ):
            continue
        measurements = {
            role: prefix_cache_measurement(payloads[role])
            for role in paths
        }
        pairs.append(
            {
                "pair_index": pair_index,
                "execution_order": list(order),
                "reports": {
                    role: {
                        "path": str(path.relative_to(output_dir)),
                        "sha256": sha256(path),
                    }
                    for role, path in paths.items()
                },
                "measurements": measurements,
                "candidate_over_baseline": {
                    name: (
                        measurements["candidate"][name]
                        / measurements["baseline"][name]
                    )
                    for name in ("ttft_speedup", "decode_retention")
                },
            }
        )
    summary = summarize_paired_prefix_cache(
        pairs, required_pair_count=pair_count
    )
    return {
        "mode": "paired-v151-no-regression",
        "context_tokens": 32768,
        "output_tokens": 512,
        "protocol": {
            "execution_order": (
                "baseline,candidate for odd pairs; "
                "candidate,baseline for even pairs"
            ),
            "pair_locality": "adjacent fresh processes",
            "per_process_requests": "one cold request then one exact hit",
            "cpu_affinity": (
                None
                if cpu_list is None
                else {
                    "cpu_list": cpu_list,
                    "taskset": str(taskset),
                    "taskset_sha256": taskset_sha256,
                }
            ),
            "decision": (
                "paired candidate/release median >= 1.0 for TTFT "
                "speedup and decode retention"
            ),
        },
        "baseline_engine": {
            "path": "${AIMA_BASELINE_ENGINE}",
            "sha256": baseline_sha256,
            "build_info": engine_build_info(baseline_engine),
        },
        **summary,
        "pass": summary["qualified"],
    }


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
            and text_path_is_idle(first_metrics)
            and text_path_is_idle(second_metrics)
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
    payload = publicize(
        payload,
        engine=engine,
        model_dir=model_dir,
        output_dir=output_dir,
    )
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
    parser.add_argument("--engine-sha256")
    parser.add_argument("--baseline-engine", type=Path)
    parser.add_argument("--baseline-engine-sha256")
    parser.add_argument(
        "--baseline-source-commit", default=V151_SOURCE_COMMIT
    )
    parser.add_argument("--baseline-version", default=V151_VERSION)
    parser.add_argument(
        "--prefix-pairs", type=int, default=DEFAULT_PREFIX_PAIRS
    )
    parser.add_argument(
        "--prefix-cpu-list",
        type=parse_cpu_list,
        help=(
            "optional taskset CPU list applied symmetrically to paired "
            "prefix-cache processes"
        ),
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=18080)
    parser.add_argument(
        "--startup-ceiling-ms", type=float, default=STARTUP_CEILING_MS
    )
    parser.add_argument(
        "--minimum-prefix-ttft-speedup",
        type=float,
        default=MINIMUM_PREFIX_TTFT_SPEEDUP,
    )
    parser.add_argument(
        "--minimum-prefix-decode-retention",
        type=float,
        default=MINIMUM_PREFIX_DECODE_RETENTION,
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
    if cli.engine_sha256 is not None and engine_sha256 != cli.engine_sha256:
        raise SystemExit("candidate engine SHA-256 changed")
    engine_info = engine_build_info(engine)

    paired_prefix_requested = (
        cli.baseline_engine is not None
        or cli.baseline_engine_sha256 is not None
    )
    if paired_prefix_requested and (
        cli.baseline_engine is None
        or cli.baseline_engine_sha256 is None
    ):
        raise SystemExit(
            "paired prefix qualification requires baseline engine and SHA-256"
        )
    baseline_engine: Path | None = None
    baseline_sha256: str | None = None
    taskset: Path | None = None
    taskset_sha256: str | None = None
    if cli.prefix_cpu_list is not None:
        if not paired_prefix_requested:
            raise SystemExit(
                "--prefix-cpu-list requires paired prefix qualification"
            )
        taskset_command = shutil.which("taskset")
        if taskset_command is None:
            raise SystemExit("--prefix-cpu-list requires taskset")
        taskset = Path(taskset_command).resolve()
        taskset_sha256 = sha256(taskset)
    if paired_prefix_requested:
        if cli.prefix_pairs < MINIMUM_PREFIX_PAIRS:
            raise SystemExit(
                "paired prefix qualification requires at least "
                f"{MINIMUM_PREFIX_PAIRS} pairs"
            )
        assert cli.baseline_engine is not None
        assert cli.baseline_engine_sha256 is not None
        baseline_engine = cli.baseline_engine.expanduser().resolve()
        if not baseline_engine.is_file() or not os.access(
            baseline_engine, os.X_OK
        ):
            raise SystemExit(
                f"baseline engine is not executable: {baseline_engine}"
            )
        baseline_sha256 = sha256(baseline_engine)
        if baseline_sha256 != cli.baseline_engine_sha256:
            raise SystemExit("baseline engine SHA-256 changed")
        baseline_info = engine_build_info(baseline_engine)
        if baseline_info.get("source_commit") != cli.baseline_source_commit:
            raise SystemExit("baseline engine source commit changed")
        if baseline_info.get("version") != cli.baseline_version:
            raise SystemExit("baseline engine version changed")

    user, chat_fixture = make_exact_chat_user(engine, model_dir, 8192)
    if paired_prefix_requested:
        assert baseline_engine is not None
        assert baseline_sha256 is not None
        prefix_result = qualify_paired_prefix_cache(
            candidate_engine=engine,
            baseline_engine=baseline_engine,
            model_dir=model_dir,
            output_dir=output_dir,
            candidate_sha256=engine_sha256,
            baseline_sha256=baseline_sha256,
            pair_count=cli.prefix_pairs,
            cpu_list=cli.prefix_cpu_list,
            taskset=taskset,
            taskset_sha256=taskset_sha256,
        )
        prefix_pass = bool(prefix_result["qualified"])
    else:
        prefix_report = run_prefix_cache(
            engine=engine,
            model_dir=model_dir,
            output_dir=output_dir,
            engine_sha256=engine_sha256,
            minimum_ttft_speedup=cli.minimum_prefix_ttft_speedup,
            minimum_decode_retention=cli.minimum_prefix_decode_retention,
        )
        prefix_payload = load_json(prefix_report)
        prefix_measurement = prefix_cache_measurement(prefix_payload)
        cold, hit = prefix_payload["requests"]
        prefix_pass = prefix_cache_report_qualified(
            prefix_payload,
            engine_sha256=engine_sha256,
            minimum_ttft_speedup=cli.minimum_prefix_ttft_speedup,
            minimum_decode_retention=cli.minimum_prefix_decode_retention,
        )
        prefix_result = {
            "mode": "legacy-single-run-absolute-floor",
            "complete": True,
            "qualified": prefix_pass,
            "report": str(prefix_report.relative_to(output_dir)),
            "report_sha256": sha256(prefix_report),
            "context_tokens": 32768,
            "output_tokens": 512,
            **prefix_measurement,
            "minimum_ttft_speedup": cli.minimum_prefix_ttft_speedup,
            "minimum_decode_retention": (
                cli.minimum_prefix_decode_retention
            ),
            "output_token_sha256_equal": (
                cold["output_token_ids_sha256"]
                == hit["output_token_ids_sha256"]
            ),
            "pass": prefix_pass,
        }
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

    server_payloads = [load_json(path) for path in server_reports]
    startup_runs = [
        float(payload["ready"]["command_to_ready_wall_ms"])
        for payload in server_payloads
    ]
    startup_median = statistics.median(startup_runs)
    first_chat, second_chat = server_payloads[0]["chat"]
    first_metrics = first_chat["body"]["aima_amd395"]
    second_metrics = second_chat["body"]["aima_amd395"]
    startup_pass = startup_median <= cli.startup_ceiling_ms
    http_pass = server_run_qualified(
        server_payloads[0],
        engine_sha256=engine_sha256,
        with_chat=True,
    )
    result = {
        "schema": "aima-amd395-qwen36/native-product-surfaces/v1",
        "complete": bool(prefix_result["complete"]),
        "qualified": prefix_pass and startup_pass and http_pass,
        "engine": {
            "path": "${AIMA_ENGINE}",
            "sha256": engine_sha256,
            "build_info": engine_info,
        },
        "model_dir": "${AIMA_MODEL_DIR}",
        "host": {
            "hostname": os.uname().nodename,
            "sysname": os.uname().sysname,
            "release": os.uname().release,
            "machine": os.uname().machine,
        },
        "prefix_cache": prefix_result,
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
            "text_path_idle": (
                text_path_is_idle(first_metrics)
                and text_path_is_idle(second_metrics)
            ),
            "pass": http_pass,
        },
    }
    atomic_json(output_dir / "surfaces.json", result)
    print(
        json.dumps(
            {
                "complete": result["complete"],
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
