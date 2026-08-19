#!/usr/bin/env python3
"""Pair two exact native engines on short tail buckets for G3 diagnosis."""

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
from typing import Any, Iterable

from aima_engine.aotriton_closure import require_aotriton_closure


SCHEMA = "aima-amd395-qwen36/native-paired-text-tail-diagnostic/v1"
DEFAULT_CONTEXTS = (7168, 7680, 8191)
MINIMUM_PAIRS = 3
FMHA_AOTRITON_FILENAME = "libaima-fmha-aotriton.so"
FMHA_CK_FILENAME = "libaima-fmha-ck.so"
FMHA_Q16384_HYBRID_FILENAME = "libaima-fmha-q16384-hybrid.so"
VISION_ATTENTION_FILENAME = "aima-vision-attention.hsaco"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
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


def publicize(value: Any, replacements: Iterable[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        result = value
        for private, public in replacements:
            result = result.replace(private, public)
        return result
    if isinstance(value, list):
        return [publicize(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: publicize(item, replacements) for key, item in value.items()
        }
    return value


def parse_contexts(value: str) -> tuple[int, ...]:
    try:
        contexts = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("contexts must be decimal integers") from exc
    if (
        not contexts
        or len(set(contexts)) != len(contexts)
        or any(context <= 0 or context > 262143 for context in contexts)
    ):
        raise argparse.ArgumentTypeError(
            "contexts must be unique integers in [1, 262143]"
        )
    return contexts


def pair_order(pair_index: int) -> tuple[str, str]:
    if pair_index <= 0:
        raise ValueError("pair index must be positive")
    return (
        ("left", "right")
        if pair_index % 2 == 1
        else ("right", "left")
    )


def bind_engine(
    path: Path, expected_sha256: str, expected_source_commit: str
) -> dict[str, Any]:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"native engine is not executable: {path}")
    digest = sha256(path)
    if digest != expected_sha256:
        raise SystemExit(f"native engine SHA-256 changed: {path}")
    completed = subprocess.run(
        [str(path), "--build-info"],
        capture_output=True,
        text=True,
        check=True,
    )
    build_info = json.loads(completed.stdout)
    if (
        not isinstance(build_info, dict)
        or build_info.get("source_commit") != expected_source_commit
    ):
        raise SystemExit(f"native engine source identity changed: {path}")
    runtime_dir = path.parent
    if runtime_dir.name == "libexec":
        runtime_dir = runtime_dir.parent / "lib"
    aotriton = require_aotriton_closure(
        runtime_dir / FMHA_AOTRITON_FILENAME
    )
    required_runtime = {
        "fmha_aotriton": aotriton.provider,
        "aotriton_runtime": aotriton.runtime,
        "aotriton_image": aotriton.image,
        "fmha_ck": runtime_dir / FMHA_CK_FILENAME,
        "fmha_q16384_hybrid": runtime_dir / FMHA_Q16384_HYBRID_FILENAME,
        "vision_attention": runtime_dir / VISION_ATTENTION_FILENAME,
    }
    if not all(component.is_file() for component in required_runtime.values()):
        raise SystemExit(f"native engine runtime closure is incomplete: {path}")
    return {
        "sha256": digest,
        "source_commit": expected_source_commit,
        "build_info": build_info,
        "runtime_dependencies": {
            name: {
                "filename": component.name,
                "sha256": sha256(component),
            }
            for name, component in required_runtime.items()
        },
    }


def report_path(
    output_dir: Path, context: int, pair_index: int, role: str
) -> Path:
    return (
        output_dir
        / "raw"
        / f"q{context}-o1"
        / f"pair-{pair_index:02d}"
        / f"{role}.json"
    )


def text_path_is_idle(request: dict[str, Any]) -> bool:
    try:
        return bool(
            request.get("mrope_enabled") is False
            and request.get("vl_logical_projections_enabled") is False
            and all(
                int(request.get(name, -1)) == 0
                for name in (
                    "mrope_position_upload_bytes",
                    "mrope_full_attention_launches",
                    "mrope_decode_steps",
                    "prefill_vl_unified_attention_launches",
                    "vl_logical_projection_tokens",
                    "vl_logical_projection_plan_count",
                    "vl_logical_projection_workspace_bytes",
                )
            )
            and float(
                request.get("vl_logical_projection_plan_build_wall_ms", -1)
            )
            == 0.0
        )
    except (TypeError, ValueError):
        return False


def report_complete(
    path: Path,
    *,
    context: int,
    pair_index: int,
    role: str,
    engine_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_object(path)
        requests = payload.get("requests")
        qualification = payload.get("qualification")
        if (
            payload.get("schema")
            != "aima-amd395-qwen36/native-resident-session-probe/v1"
            or payload.get("complete") is not True
            or payload.get("model_loads") != 1
            or payload.get("request_count") != 1
            or payload.get("runtime_python") is not False
            or payload.get("runtime_torch") is not False
            or payload.get("runtime_vllm") is not False
            or payload.get("runtime_triton") is not False
            or not isinstance(requests, list)
            or len(requests) != 1
            or not isinstance(requests[0], dict)
            or requests[0].get("prompt_tokens") != context
            or requests[0].get("completion_tokens") != 1
            or requests[0].get("first_token_certified") is not True
            or requests[0].get("all_decode_tokens_certified") is not True
            or requests[0].get("oracle_tensor_reads") != 0
            or not text_path_is_idle(requests[0])
            or not isinstance(qualification, dict)
            or qualification.get("engine_role") != role
            or qualification.get("engine_sha256") != engine_sha256
            or qualification.get("pair_index") != pair_index
            or tuple(qualification.get("pair_order", []))
            != pair_order(pair_index)
        ):
            return False
        return True
    except (
        OSError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def run_report(
    *,
    engine: Path,
    engine_sha256: str,
    model_dir: Path,
    output_dir: Path,
    context: int,
    pair_index: int,
    role: str,
    sequence_index: int,
    replacements: tuple[tuple[str, str], ...],
) -> None:
    path = report_path(output_dir, context, pair_index, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    load_path = path.with_suffix(".load.json")
    stderr_path = path.with_suffix(".stderr.log")
    command = [
        str(engine),
        "resident-session-probe",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        str(context),
        "--uniform-input-token-id",
        "1",
        "--report",
        str(load_path),
        "--max-new-tokens",
        "1",
        "--requests",
        "1",
        "--disable-prefix-cache",
    ]
    print(
        json.dumps(
            {
                "event": "text_tail_run_start",
                "context_tokens": context,
                "pair_index": pair_index,
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
    stderr_path.write_text(
        publicize(completed.stderr, replacements), encoding="utf-8"
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"native tail run emitted invalid JSON: q{context} {role}"
        ) from exc
    if not isinstance(payload, dict) or not load_path.is_file():
        raise RuntimeError(f"native tail run is incomplete: q{context} {role}")
    load_payload = publicize(load_object(load_path), replacements)
    atomic_json(load_path, load_payload)
    payload["qualification"] = {
        "schema": "aima-amd395-qwen36/native-tail-run-binding/v1",
        "engine_role": role,
        "engine_sha256": engine_sha256,
        "pair_index": pair_index,
        "pair_order": list(pair_order(pair_index)),
        "sequence_index": sequence_index,
        "command": command,
        "load_report": str(load_path),
        "load_report_sha256": sha256(load_path),
        "stderr": str(stderr_path),
        "stderr_sha256": sha256(stderr_path),
    }
    atomic_json(path, publicize(payload, replacements))
    if completed.returncode != 0 or not report_complete(
        path,
        context=context,
        pair_index=pair_index,
        role=role,
        engine_sha256=engine_sha256,
    ):
        raise RuntimeError(
            f"native tail run failed: q{context} pair {pair_index} {role}"
        )
    print(
        json.dumps(
            {
                "event": "text_tail_run_complete",
                "context_tokens": context,
                "pair_index": pair_index,
                "engine_role": role,
                "sequence_index": sequence_index,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def measurement(path: Path) -> dict[str, float]:
    request = load_object(path)["requests"][0]
    return {
        "prefill_tps": float(request["prefill_tokens_per_second"]),
        "prefill_wall_ms": float(request["prefill_wall_ms"]),
        "command_to_ready_wall_ms": float(
            load_object(path)["load"]["command_to_ready_wall_ms"]
        ),
    }


def build_result(
    *,
    output_dir: Path,
    contexts: tuple[int, ...],
    pair_count: int,
    identities: dict[str, dict[str, Any]],
    script_path: Path,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for context in contexts:
        pairs: list[dict[str, Any]] = []
        for pair_index in range(1, pair_count + 1):
            paths = {
                role: report_path(output_dir, context, pair_index, role)
                for role in ("left", "right")
            }
            if not all(
                report_complete(
                    paths[role],
                    context=context,
                    pair_index=pair_index,
                    role=role,
                    engine_sha256=identities[role]["sha256"],
                )
                for role in paths
            ):
                continue
            values = {role: measurement(path) for role, path in paths.items()}
            pairs.append(
                {
                    "pair_index": pair_index,
                    "execution_order": list(pair_order(pair_index)),
                    "measurements": values,
                    "ratios": {
                        "prefill_tps_right_over_left": (
                            values["right"]["prefill_tps"]
                            / values["left"]["prefill_tps"]
                        ),
                        "prefill_wall_right_over_left": (
                            values["right"]["prefill_wall_ms"]
                            / values["left"]["prefill_wall_ms"]
                        ),
                    },
                    "reports": {
                        role: {
                            "path": str(path.relative_to(output_dir)),
                            "sha256": sha256(path),
                        }
                        for role, path in paths.items()
                    },
                }
            )
        ratio_names = (
            "prefill_tps_right_over_left",
            "prefill_wall_right_over_left",
        )
        cells.append(
            {
                "context_tokens": context,
                "output_tokens": 1,
                "complete": len(pairs) == pair_count,
                "pair_count": len(pairs),
                "pairs": pairs,
                "paired_medians": {
                    name: float(
                        statistics.median(
                            pair["ratios"][name] for pair in pairs
                        )
                    )
                    for name in ratio_names
                }
                if pairs
                else {},
            }
        )
    complete = len(cells) == len(contexts) and all(
        cell["complete"] for cell in cells
    )
    return {
        "schema": SCHEMA,
        "complete": complete,
        "diagnostic_only": True,
        "host": {
            "hostname": os.uname().nodename,
            "sysname": os.uname().sysname,
            "release": os.uname().release,
            "machine": os.uname().machine,
        },
        "engines": identities,
        "model_dir": "${AIMA_MODEL_DIR}",
        "protocol": {
            "contexts": list(contexts),
            "output_tokens": 1,
            "pair_count": pair_count,
            "minimum_pairs": MINIMUM_PAIRS,
            "execution_order": "left,right on odd pairs; right,left on even pairs",
            "fresh_processes": True,
            "prefix_cache": "disabled",
            "uniform_input_token_id": 1,
        },
        "script": {
            "path": "scripts/diagnose-native-paired-text-tails.py",
            "sha256": sha256(script_path),
        },
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-engine", type=Path, required=True)
    parser.add_argument("--left-engine-sha256", required=True)
    parser.add_argument("--left-source-commit", required=True)
    parser.add_argument("--right-engine", type=Path, required=True)
    parser.add_argument("--right-engine-sha256", required=True)
    parser.add_argument("--right-source-commit", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contexts",
        type=parse_contexts,
        default=DEFAULT_CONTEXTS,
    )
    parser.add_argument("--pairs", type=int, default=MINIMUM_PAIRS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.pairs < MINIMUM_PAIRS:
        raise SystemExit(
            f"tail diagnostic requires at least {MINIMUM_PAIRS} pairs"
        )
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    engines = {
        "left": args.left_engine.expanduser().resolve(),
        "right": args.right_engine.expanduser().resolve(),
    }
    if not model_dir.is_dir():
        raise SystemExit("model directory is missing")
    if output_dir.exists() and not args.resume:
        raise SystemExit(f"refusing to reuse diagnostic output: {output_dir}")
    identities = {
        "left": bind_engine(
            engines["left"],
            args.left_engine_sha256,
            args.left_source_commit,
        ),
        "right": bind_engine(
            engines["right"],
            args.right_engine_sha256,
            args.right_source_commit,
        ),
    }
    if identities["left"]["sha256"] == identities["right"]["sha256"]:
        raise SystemExit("tail diagnostic requires two distinct engine binaries")
    identities["left"]["path"] = "${AIMA_LEFT_ENGINE}"
    identities["right"]["path"] = "${AIMA_RIGHT_ENGINE}"
    replacements = (
        (str(engines["left"]), "${AIMA_LEFT_ENGINE}"),
        (str(engines["right"]), "${AIMA_RIGHT_ENGINE}"),
        (str(model_dir), "${AIMA_MODEL_DIR}"),
        (str(output_dir), "${AIMA_DIAGNOSTIC_DIR}"),
    )
    sequence_index = 0
    for context in args.contexts:
        for pair_index in range(1, args.pairs + 1):
            for role in pair_order(pair_index):
                sequence_index += 1
                path = report_path(output_dir, context, pair_index, role)
                if args.resume and report_complete(
                    path,
                    context=context,
                    pair_index=pair_index,
                    role=role,
                    engine_sha256=identities[role]["sha256"],
                ):
                    continue
                run_report(
                    engine=engines[role],
                    engine_sha256=identities[role]["sha256"],
                    model_dir=model_dir,
                    output_dir=output_dir,
                    context=context,
                    pair_index=pair_index,
                    role=role,
                    sequence_index=sequence_index,
                    replacements=replacements,
                )
                atomic_json(
                    output_dir / "diagnostic.json",
                    build_result(
                        output_dir=output_dir,
                        contexts=args.contexts,
                        pair_count=args.pairs,
                        identities=identities,
                        script_path=Path(__file__).resolve(),
                    ),
                )
    result = build_result(
        output_dir=output_dir,
        contexts=args.contexts,
        pair_count=args.pairs,
        identities=identities,
        script_path=Path(__file__).resolve(),
    )
    atomic_json(output_dir / "diagnostic.json", result)
    digest = sha256(output_dir / "diagnostic.json")
    (output_dir / "diagnostic.json.sha256").write_text(
        f"{digest}  diagnostic.json\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "output": str(output_dir / "diagnostic.json"),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
