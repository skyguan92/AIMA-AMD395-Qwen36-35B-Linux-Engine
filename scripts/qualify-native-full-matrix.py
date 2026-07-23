#!/usr/bin/env python3
"""Run and aggregate the native cold-prefill/decode qualification matrix."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STANDARD_CONTEXTS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
WINDOW_ENDPOINTS = (
    (262143, 1),
    (261632, 512),
    (261120, 1024),
)


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


def parse_contexts(text: str) -> list[int]:
    if text == "standard":
        return list(STANDARD_CONTEXTS)
    values = [int(item) for item in text.split(",") if item]
    unsupported = sorted(set(values) - set(STANDARD_CONTEXTS))
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported standard contexts: {unsupported}"
        )
    return values


def baseline_cells(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    payload = load_json(path)
    result: dict[tuple[int, int], dict[str, float]] = {}
    for row in payload["cold_context_matrix"]:
        result[(int(row["input_tokens"]), int(row["output_tokens"]))] = {
            "prefill_tps": float(row["prefill_tok_s"]),
            "decode_tps": float(row["decode_tok_s"]),
        }
    output_one = payload["output_one"]
    result[(int(output_one["input_tokens"]), 1)] = {
        "prefill_tps": float(output_one["prefill_tok_s"]),
    }
    return result


def complete_report(
    path: Path,
    context: int,
    outputs: tuple[int, ...],
    engine_sha256: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        requests = payload["requests"]
        complete = (
            payload.get("complete") is True
            and payload.get("schema")
            == "aima-amd395-qwen36/native-resident-session-probe/v1"
            and all(int(item["prompt_tokens"]) == context for item in requests)
            and tuple(int(item["completion_tokens"]) for item in requests)
            == outputs
        )
        if engine_sha256 is not None:
            complete = (
                complete
                and payload.get("qualification", {}).get("engine_sha256")
                == engine_sha256
            )
        return complete
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_report(
    engine: Path,
    model_dir: Path,
    report: Path,
    context: int,
    outputs: tuple[int, ...],
    uniform_token_id: int,
    engine_sha256: str,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    load_report = report.with_name(report.stem + ".load.json")
    command = [
        str(engine),
        "resident-session-probe",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        str(context),
        "--uniform-input-token-id",
        str(uniform_token_id),
        "--report",
        str(load_report),
    ]
    if len(outputs) == 1:
        command.extend(
            [
                "--max-new-tokens",
                str(outputs[0]),
                "--requests",
                "1",
                "--disable-prefix-cache",
            ]
        )
    else:
        command.extend(
            ["--max-new-tokens-sequence", ",".join(map(str, outputs))]
        )
    print(
        json.dumps(
            {
                "event": "matrix_run_start",
                "context_tokens": context,
                "output_tokens": list(outputs),
                "report": str(report),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"native matrix run emitted invalid JSON: {report}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"native matrix run emitted non-object JSON: {report}")
    payload["qualification"] = {
        "engine_sha256": engine_sha256,
        "command": command,
        "load_report": str(load_report),
        "load_report_sha256": (
            sha256(load_report) if load_report.is_file() else None
        ),
    }
    payload = publicize(payload, model_dir)
    atomic_json(report, payload)
    if completed.returncode != 0 or not complete_report(
        report, context, outputs, engine_sha256
    ):
        raise RuntimeError(
            f"native matrix run failed with exit {completed.returncode}: "
            f"{report}"
        )
    print(
        json.dumps(
            {
                "event": "matrix_run_complete",
                "context_tokens": context,
                "output_tokens": list(outputs),
                "report": str(report),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def relative_spread(values: list[float]) -> float:
    if not values:
        return 0.0
    low = min(values)
    return (max(values) - low) / low if low > 0 else float("inf")


def report_samples(
    paths: list[Path], context: int, outputs: tuple[int, ...]
) -> dict[int, dict[str, list[float]]]:
    result = {
        output: {"prefill_tps": [], "decode_tps": []} for output in outputs
    }
    for path in paths:
        payload = load_json(path)
        requests = payload["requests"]
        cold_prefill = float(requests[0]["prefill_tokens_per_second"])
        for index, output in enumerate(outputs):
            request = requests[index]
            if int(request["completion_tokens"]) != output:
                raise RuntimeError(f"output order mismatch: {path}")
            result[output]["prefill_tps"].append(cold_prefill)
            if output > 1:
                result[output]["decode_tps"].append(
                    float(request["decode_tokens_per_second"])
                )
    return result


def need_third_sample(samples: dict[int, dict[str, list[float]]]) -> bool:
    for output in samples.values():
        for values in output.values():
            if len(values) >= 2 and relative_spread(values[:2]) > 0.03:
                return True
    return False


def build_result(
    engine: Path,
    model_dir: Path,
    baseline_path: Path,
    raw_dir: Path,
    requested: list[tuple[int, tuple[int, ...], list[Path]]],
    minimum_retention: float,
) -> dict[str, Any]:
    baselines = baseline_cells(baseline_path)
    cells: list[dict[str, Any]] = []
    all_pass = True
    for context, outputs, reports in requested:
        samples = report_samples(reports, context, outputs)
        for output in outputs:
            key = (context, output)
            baseline = baselines[key]
            prefill_values = samples[output]["prefill_tps"]
            decode_values = samples[output]["decode_tps"]
            prefill_median = statistics.median(prefill_values)
            prefill_retention = prefill_median / baseline["prefill_tps"]
            cell: dict[str, Any] = {
                "input_tokens": context,
                "output_tokens": output,
                "reports": [
                    str(path.relative_to(raw_dir.parent)) for path in reports
                ],
                "report_sha256": [sha256(path) for path in reports],
                "sample_count": len(reports),
                "protocol": (
                    "two_within_3_percent"
                    if len(reports) == 2
                    else "three_run_median"
                ),
                "prefill_runs_tps": prefill_values,
                "prefill_spread": relative_spread(prefill_values),
                "prefill_median_tps": prefill_median,
                "baseline_prefill_tps": baseline["prefill_tps"],
                "prefill_retention": prefill_retention,
                "prefill_pass": prefill_retention >= minimum_retention,
            }
            if output > 1:
                decode_median = statistics.median(decode_values)
                decode_retention = decode_median / baseline["decode_tps"]
                cell.update(
                    {
                        "decode_runs_tps": decode_values,
                        "decode_spread": relative_spread(decode_values),
                        "decode_median_tps": decode_median,
                        "baseline_decode_tps": baseline["decode_tps"],
                        "decode_retention": decode_retention,
                        "decode_pass": decode_retention >= minimum_retention,
                    }
                )
            cell["pass"] = bool(
                cell["prefill_pass"] and cell.get("decode_pass", True)
            )
            all_pass = all_pass and cell["pass"]
            cells.append(cell)
    return {
        "schema": "aima-amd395-qwen36/native-full-matrix-qualification/v1",
        "complete": True,
        "qualified": all_pass,
        "engine": {
            "path": "${AIMA_REPO_ROOT}/build/native/aima-engine-native",
            "sha256": sha256(engine),
        },
        "model_dir": "${AIMA_MODEL_DIR}",
        "host": {
            "hostname": os.uname().nodename,
            "sysname": os.uname().sysname,
            "release": os.uname().release,
            "machine": os.uname().machine,
        },
        "baseline": {
            "path": str(baseline_path.relative_to(ROOT)),
            "sha256": sha256(baseline_path),
        },
        "measurement_protocol": {
            "minimum_samples": 2,
            "first_pair_maximum_spread": 0.03,
            "fallback": "third run and median",
            "minimum_retention": minimum_retention,
            "cold_prefill": "first request in each process",
            "standard_decode": (
                "output512 cold request then output1024 exact-prefix restore"
            ),
        },
        "cells": cells,
        "all_cells_pass": all_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine", type=Path, default=Path("build/native/aima-engine-native")
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=Path("benchmarks/results/v1.0.0.json")
    )
    parser.add_argument("--contexts", type=parse_contexts, default="standard")
    parser.add_argument("--include-window-endpoints", action="store_true")
    parser.add_argument("--uniform-input-token-id", type=int, default=1)
    parser.add_argument("--minimum-retention", type=float, default=0.97)
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()

    engine = cli.engine.expanduser().resolve()
    model_dir = cli.model_dir.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    baseline = cli.baseline.expanduser().resolve()
    if not engine.is_file() or not os.access(engine, os.X_OK):
        raise SystemExit(f"native engine is not executable: {engine}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")
    if not baseline.is_file():
        raise SystemExit(f"baseline is missing: {baseline}")
    if not 0 <= cli.uniform_input_token_id < 248320:
        raise SystemExit("--uniform-input-token-id is invalid")
    engine_sha256 = sha256(engine)

    jobs: list[tuple[int, tuple[int, ...]]] = [
        (context, (512, 1024)) for context in cli.contexts
    ]
    if cli.include_window_endpoints:
        jobs.extend((context, (output,)) for context, output in WINDOW_ENDPOINTS)

    requested: list[tuple[int, tuple[int, ...], list[Path]]] = []
    for context, outputs in jobs:
        stem = f"q{context}-o{'-'.join(map(str, outputs))}"
        reports: list[Path] = []
        for run_index in (1, 2):
            path = raw_dir / f"{stem}-r{run_index}.json"
            if not (
                cli.resume
                and complete_report(path, context, outputs, engine_sha256)
            ):
                run_report(
                    engine,
                    model_dir,
                    path,
                    context,
                    outputs,
                    cli.uniform_input_token_id,
                    engine_sha256,
                )
            reports.append(path)
        samples = report_samples(reports, context, outputs)
        if need_third_sample(samples):
            path = raw_dir / f"{stem}-r3.json"
            if not (
                cli.resume
                and complete_report(path, context, outputs, engine_sha256)
            ):
                run_report(
                    engine,
                    model_dir,
                    path,
                    context,
                    outputs,
                    cli.uniform_input_token_id,
                    engine_sha256,
                )
            reports.append(path)
        requested.append((context, outputs, reports))
        atomic_json(
            output_dir / "matrix.json",
            build_result(
                engine,
                model_dir,
                baseline,
                raw_dir,
                requested,
                cli.minimum_retention,
            ),
        )

    result = build_result(
        engine,
        model_dir,
        baseline,
        raw_dir,
        requested,
        cli.minimum_retention,
    )
    atomic_json(output_dir / "matrix.json", result)
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": result["qualified"],
                "cell_count": len(result["cells"]),
                "output": str(output_dir / "matrix.json"),
            },
            sort_keys=True,
        )
    )
    if not result["qualified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
