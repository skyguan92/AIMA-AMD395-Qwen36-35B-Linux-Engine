#!/usr/bin/env python3
"""Pair the native VL candidate against the immutable v1.5.1 text engine."""

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
from typing import Any, Iterable

from aima_engine.aotriton_closure import require_aotriton_closure


ROOT = Path(__file__).resolve().parents[1]
STANDARD_CONTEXTS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
STANDARD_OUTPUTS = (512, 1024)
WINDOW_ENDPOINTS = ((262143, 1), (261632, 512), (261120, 1024))
MINIMUM_PAIRS = 5
DEFAULT_PAIRS = 6
V151_SOURCE_COMMIT = "65c198415709dad6d046c247acab3dc9df2a95a0"
V151_VERSION = "1.5.1-native"
Q8192_STARTUP_LIMIT_MS = 44_900.0
VISION_ATTENTION_SHA256 = (
    "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e"
)
FMHA_AOTRITON_FILENAME = "libaima-fmha-aotriton.so"
FMHA_CK_FILENAME = "libaima-fmha-ck.so"
FMHA_Q16384_HYBRID_FILENAME = "libaima-fmha-q16384-hybrid.so"
VISION_ATTENTION_FILENAME = "aima-vision-attention.hsaco"
CANDIDATE_RUNTIME_POLICY = "automatic-context-provider/v1"
GPU_PERFORMANCE_LEVEL_PATH = Path(
    "/sys/class/drm/card0/device/power_dpm_force_performance_level"
)

# Frozen public values in docs/NATIVE_VL_GOAL.md.  These remain an independent
# safety floor; the paired v1.5.1 process is the stricter no-regression gate.
FROZEN_V151_FLOORS: dict[tuple[int, int], dict[str, float]] = {
    (1024, 512): {"prefill_tps": 1630.0, "decode_tps": 34.00},
    (1024, 1024): {"prefill_tps": 1630.0, "decode_tps": 34.02},
    (2048, 512): {"prefill_tps": 1693.0, "decode_tps": 33.85},
    (2048, 1024): {"prefill_tps": 1693.0, "decode_tps": 33.85},
    (4096, 512): {"prefill_tps": 1569.0, "decode_tps": 33.32},
    (4096, 1024): {"prefill_tps": 1569.0, "decode_tps": 33.30},
    (8192, 512): {"prefill_tps": 1660.0, "decode_tps": 32.30},
    (8192, 1024): {"prefill_tps": 1660.0, "decode_tps": 32.28},
    (16384, 512): {"prefill_tps": 1440.0, "decode_tps": 30.79},
    (16384, 1024): {"prefill_tps": 1440.0, "decode_tps": 30.78},
    (32768, 512): {"prefill_tps": 1358.0, "decode_tps": 28.22},
    (32768, 1024): {"prefill_tps": 1358.0, "decode_tps": 28.22},
    (65536, 512): {"prefill_tps": 1170.0, "decode_tps": 24.65},
    (65536, 1024): {"prefill_tps": 1170.0, "decode_tps": 24.65},
    (131072, 512): {"prefill_tps": 869.7, "decode_tps": 19.62},
    (131072, 1024): {"prefill_tps": 869.7, "decode_tps": 19.62},
    (262143, 1): {"prefill_tps": 555.2},
    (261632, 512): {"prefill_tps": 555.1, "decode_tps": 14.04},
    (261120, 1024): {"prefill_tps": 559.3, "decode_tps": 14.02},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_gpu_idle() -> None:
    """Refuse to start a fresh GPU process while another owner has /dev/kfd."""
    occupied = subprocess.run(
        ["fuser", "/dev/kfd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if occupied.returncode != 0:
        return
    subprocess.run(["fuser", "-v", "/dev/kfd"], check=False)
    print(
        "paired text matrix paused because /dev/kfd is owned externally",
        file=sys.stderr,
    )
    raise SystemExit(75)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def optional_text(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


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


def sanitize_role_reports(
    output_dir: Path, replacements: tuple[tuple[str, str], ...]
) -> None:
    """Apply current path redactions to role reports retained by --resume."""
    raw = output_dir / "raw"
    if not raw.is_dir():
        return
    for path in sorted(raw.rglob("*.json")):
        if path.name not in {"baseline.json", "candidate.json"}:
            continue
        payload = load_json(path)
        sanitized = publicize(payload, replacements)
        if sanitized != payload:
            atomic_json(path, sanitized)


def engine_build_info(engine: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(engine), "--build-info"],
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError(f"engine emitted non-object build info: {engine}")
    return value


def bind_engine(
    *,
    role: str,
    engine: Path,
    expected_sha256: str,
    expected_source_commit: str,
    expected_version: str | None,
) -> dict[str, Any]:
    actual_sha256 = sha256(engine)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"{role} engine SHA-256 differs from the requested identity"
        )
    build_info = engine_build_info(engine)
    if build_info.get("source_commit") != expected_source_commit:
        raise SystemExit(
            f"{role} engine source commit differs from the requested identity"
        )
    if (
        expected_version is not None
        and build_info.get("version") != expected_version
    ):
        raise SystemExit(f"{role} engine version differs from the requested identity")
    return {
        "role": role,
        "path": "${AIMA_" + role.upper() + "_ENGINE}",
        "sha256": actual_sha256,
        "build_info": build_info,
    }


def automatic_runtime_path(engine: Path, filename: str) -> Path:
    executable_dir = engine.parent
    if executable_dir.name == "libexec":
        return executable_dir.parent / "lib" / filename
    return executable_dir / filename


def bind_candidate_runtime(
    *,
    engine: Path,
    fmha_provider: Path,
    expected_fmha_provider_sha256: str,
    long_context_fmha_provider: Path,
    expected_long_context_fmha_provider_sha256: str,
    q16384_hybrid_provider: Path,
    expected_q16384_hybrid_provider_sha256: str,
    vision_attention_image: Path,
    expected_vision_attention_sha256: str,
) -> dict[str, Any]:
    expected_paths = {
        "AOTriton FMHA provider": automatic_runtime_path(
            engine, FMHA_AOTRITON_FILENAME
        ),
        "long-context CK FMHA provider": automatic_runtime_path(
            engine, FMHA_CK_FILENAME
        ),
        "q16384 hybrid FMHA provider": automatic_runtime_path(
            engine, FMHA_Q16384_HYBRID_FILENAME
        ),
        "vision-attention image": automatic_runtime_path(
            engine, VISION_ATTENTION_FILENAME
        ),
    }
    supplied_paths = {
        "AOTriton FMHA provider": fmha_provider,
        "long-context CK FMHA provider": long_context_fmha_provider,
        "q16384 hybrid FMHA provider": q16384_hybrid_provider,
        "vision-attention image": vision_attention_image,
    }
    for label, expected_path in expected_paths.items():
        if supplied_paths[label] != expected_path:
            raise SystemExit(
                f"candidate {label} must use its automatic runtime path"
            )
    closure = require_aotriton_closure(fmha_provider)
    provider_sha256 = sha256(closure.provider)
    if provider_sha256 != expected_fmha_provider_sha256:
        raise SystemExit("candidate FMHA provider SHA-256 changed")
    if not long_context_fmha_provider.is_file():
        raise SystemExit("candidate long-context CK FMHA provider is missing")
    long_context_sha256 = sha256(long_context_fmha_provider)
    if long_context_sha256 != expected_long_context_fmha_provider_sha256:
        raise SystemExit(
            "candidate long-context CK FMHA provider SHA-256 changed"
        )
    if not q16384_hybrid_provider.is_file():
        raise SystemExit("candidate q16384 hybrid FMHA provider is missing")
    q16384_hybrid_sha256 = sha256(q16384_hybrid_provider)
    if q16384_hybrid_sha256 != expected_q16384_hybrid_provider_sha256:
        raise SystemExit(
            "candidate q16384 hybrid FMHA provider SHA-256 changed"
        )
    if not vision_attention_image.is_file():
        raise SystemExit("candidate vision-attention image is missing")
    vision_sha256 = sha256(vision_attention_image)
    if vision_sha256 != expected_vision_attention_sha256:
        raise SystemExit("candidate vision-attention image SHA-256 changed")
    return {
        "fmha_provider": {
            "path": "${AIMA_CANDIDATE_FMHA_PROVIDER}",
            "sha256": provider_sha256,
        },
        "long_context_fmha_provider": {
            "path": "${AIMA_CANDIDATE_LONG_CONTEXT_FMHA_PROVIDER}",
            "sha256": long_context_sha256,
        },
        "q16384_hybrid_fmha_provider": {
            "path": "${AIMA_CANDIDATE_Q16384_HYBRID_FMHA_PROVIDER}",
            "sha256": q16384_hybrid_sha256,
        },
        "aotriton_runtime": {
            "path": "${AIMA_CANDIDATE_AOTRITON_RUNTIME}",
            "sha256": sha256(closure.runtime),
        },
        "aotriton_image": {
            "path": "${AIMA_CANDIDATE_AOTRITON_IMAGE}",
            "sha256": sha256(closure.image),
        },
        "vision_attention_image": {
            "path": "${AIMA_CANDIDATE_VISION_ATTENTION_IMAGE}",
            "sha256": vision_sha256,
        },
    }


def jobs() -> list[tuple[int, tuple[int, ...]]]:
    result = [(context, STANDARD_OUTPUTS) for context in STANDARD_CONTEXTS]
    result.extend((context, (output,)) for context, output in WINDOW_ENDPOINTS)
    return result


def resolve_pair_counts(
    default_pair_count: int, overrides: Iterable[str]
) -> dict[tuple[int, tuple[int, ...]], int]:
    if default_pair_count < MINIMUM_PAIRS:
        raise ValueError(
            f"paired qualification requires at least {MINIMUM_PAIRS} pairs"
        )
    result = {job: default_pair_count for job in jobs()}
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for raw in overrides:
        try:
            job_text, pair_text = raw.rsplit("=", 1)
            context_text, outputs_text = job_text.split("/", 1)
            job = (
                int(context_text),
                tuple(int(value) for value in outputs_text.split(",")),
            )
            pair_count = int(pair_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "pair-count override must be CONTEXT/OUTPUT[,OUTPUT]=PAIRS"
            ) from exc
        if job not in result:
            raise ValueError(f"pair-count override is not a frozen job: {raw}")
        if job in seen:
            raise ValueError(f"duplicate pair-count override: {raw}")
        if pair_count < default_pair_count:
            raise ValueError(
                "pair-count override cannot reduce the default pair count"
            )
        seen.add(job)
        result[job] = pair_count
    return result


def pair_order(pair_index: int) -> tuple[str, str]:
    if pair_index <= 0:
        raise ValueError("pair index must be positive")
    return (
        ("baseline", "candidate")
        if pair_index % 2 == 1
        else ("candidate", "baseline")
    )


def report_path(
    output_dir: Path,
    context: int,
    outputs: tuple[int, ...],
    pair_index: int,
    role: str,
) -> Path:
    output_label = "-".join(map(str, outputs))
    return (
        output_dir
        / "raw"
        / f"q{context}-o{output_label}"
        / f"pair-{pair_index:02d}"
        / f"{role}.json"
    )


def maximum_recorded_sequence_index(output_dir: Path) -> int:
    maximum = 0
    raw = output_dir / "raw"
    if not raw.is_dir():
        return maximum
    for path in raw.rglob("*.json"):
        try:
            qualification = load_json(path).get("qualification")
            if not isinstance(qualification, dict):
                continue
            value = qualification.get("sequence_index")
            if isinstance(value, int) and not isinstance(value, bool):
                maximum = max(maximum, value)
        except (OSError, RuntimeError, json.JSONDecodeError):
            continue
    return maximum


def candidate_text_path_is_idle(payload: dict[str, Any]) -> bool:
    try:
        for request in payload["requests"]:
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
            if request.get("vl_logical_projections_enabled") is not False:
                return False
            if float(request.get("vl_logical_projection_plan_build_wall_ms", -1)) != 0:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def candidate_disabled_cache_backing_is_absent(
    payload: dict[str, Any], outputs: tuple[int, ...]
) -> bool:
    if len(outputs) != 1:
        return True
    try:
        load = payload["load"]
        return bool(
            int(load.get("prefix_cache_entries", -1)) == 0
            and int(load.get("exact_prefix_cache_bytes", -1)) == 0
            and all(
                request.get("prefix_cache_lookup") == "disabled"
                for request in payload["requests"]
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def report_complete(
    path: Path,
    *,
    context: int,
    outputs: tuple[int, ...],
    role: str,
    pair_index: int,
    order: tuple[str, str],
    engine_sha256: str,
    candidate_runtime_binding_sha256: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        requests = payload["requests"]
        qualification = payload["qualification"]
        complete = bool(
            payload.get("schema")
            == "aima-amd395-qwen36/native-resident-session-probe/v1"
            and payload.get("complete") is True
            and int(payload.get("model_loads")) == 1
            and int(payload.get("request_count")) == len(outputs)
            and payload.get("runtime_python") is False
            and payload.get("runtime_torch") is False
            and payload.get("runtime_vllm") is False
            and payload.get("runtime_triton") is False
            and len(requests) == len(outputs)
            and all(int(item["prompt_tokens"]) == context for item in requests)
            and tuple(int(item["completion_tokens"]) for item in requests)
            == outputs
            and all(item["first_token_certified"] is True for item in requests)
            and all(
                item["all_decode_tokens_certified"] is True for item in requests
            )
            and all(int(item["oracle_tensor_reads"]) == 0 for item in requests)
            and qualification.get("engine_role") == role
            and int(qualification.get("pair_index")) == pair_index
            and tuple(qualification.get("pair_order", [])) == order
            and qualification.get("engine_sha256") == engine_sha256
        )
        if len(outputs) == 2:
            complete = bool(
                complete
                and requests[0]["prefix_cache_lookup"] == "miss"
                and requests[1]["prefix_cache_lookup"] == "exact"
                and int(requests[1]["prefix_cache_matched_tokens"]) == context
            )
        if role == "candidate":
            complete = bool(
                complete
                and candidate_text_path_is_idle(payload)
                and candidate_disabled_cache_backing_is_absent(
                    payload, outputs
                )
                and (
                    candidate_runtime_binding_sha256 is None
                    or (
                        qualification.get("runtime_policy")
                        == CANDIDATE_RUNTIME_POLICY
                        and qualification.get("runtime_binding_sha256")
                        == candidate_runtime_binding_sha256
                    )
                )
            )
        return complete
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_report(
    *,
    engine: Path,
    model_dir: Path,
    output_dir: Path,
    context: int,
    outputs: tuple[int, ...],
    uniform_token_id: int,
    role: str,
    pair_index: int,
    order: tuple[str, str],
    sequence_index: int,
    engine_sha256: str,
    candidate_runtime_binding_sha256: str | None,
    runtime_options: tuple[str, ...],
    replacements: tuple[tuple[str, str], ...],
) -> Path:
    report = report_path(output_dir, context, outputs, pair_index, role)
    report.parent.mkdir(parents=True, exist_ok=True)
    load_report = report.with_suffix(".load.json")
    stderr_path = report.with_suffix(".stderr.log")
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
    command.extend(runtime_options)
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
                "event": "paired_matrix_run_start",
                "context_tokens": context,
                "output_tokens": list(outputs),
                "pair_index": pair_index,
                "pair_order": list(order),
                "engine_role": role,
                "sequence_index": sequence_index,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    require_gpu_idle()
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
            f"paired matrix run emitted invalid JSON for {role} q{context}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"paired matrix run emitted non-object JSON for {role} q{context}"
        )
    payload["qualification"] = {
        "schema": "aima-amd395-qwen36/native-paired-run-binding/v1",
        "engine_role": role,
        "engine_sha256": engine_sha256,
        "runtime_policy": (
            CANDIDATE_RUNTIME_POLICY if role == "candidate" else None
        ),
        "runtime_binding_sha256": candidate_runtime_binding_sha256,
        "pair_index": pair_index,
        "pair_order": list(order),
        "sequence_index": sequence_index,
        "command": command,
        "load_report": str(load_report),
        "load_report_sha256": sha256(load_report) if load_report.is_file() else None,
        "stderr": str(stderr_path),
        "stderr_sha256": sha256(stderr_path),
    }
    payload = publicize(payload, replacements)
    atomic_json(report, payload)
    if completed.returncode != 0 or not report_complete(
        report,
        context=context,
        outputs=outputs,
        role=role,
        pair_index=pair_index,
        order=order,
        engine_sha256=engine_sha256,
        candidate_runtime_binding_sha256=(
            candidate_runtime_binding_sha256
        ),
    ):
        raise RuntimeError(
            f"paired matrix run failed with exit {completed.returncode}: "
            f"{role} q{context} pair {pair_index}"
        )
    print(
        json.dumps(
            {
                "event": "paired_matrix_run_complete",
                "context_tokens": context,
                "output_tokens": list(outputs),
                "pair_index": pair_index,
                "engine_role": role,
                "sequence_index": sequence_index,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def request_measurement(
    payload: dict[str, Any], output_index: int
) -> dict[str, float]:
    cold = payload["requests"][0]
    decode = payload["requests"][output_index]
    prefill_wall_ms = float(cold["prefill_wall_ms"])
    decode_wall_ms = float(decode["decode_wall_ms"])
    result = {
        "prefill_tps": float(cold["prefill_tokens_per_second"]),
        "prefill_wall_ms": prefill_wall_ms,
        "total_wall_ms": prefill_wall_ms + decode_wall_ms,
        "command_to_ready_wall_ms": float(
            payload["load"]["command_to_ready_wall_ms"]
        ),
    }
    if int(decode["completion_tokens"]) > 1:
        result.update(
            {
                "decode_tps": float(decode["decode_tokens_per_second"]),
                "decode_wall_ms": decode_wall_ms,
            }
        )
    return result


def ratio(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise RuntimeError("paired baseline measurement must be positive")
    return candidate / baseline


def median(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot compute a median without measurements")
    return float(statistics.median(values))


def build_cell(
    *,
    output_dir: Path,
    context: int,
    outputs: tuple[int, ...],
    output_index: int,
    pair_count: int,
    engine_sha256: dict[str, str],
    candidate_runtime_binding_sha256: str | None = None,
) -> dict[str, Any] | None:
    output = outputs[output_index]
    pairs: list[dict[str, Any]] = []
    for pair_index in range(1, pair_count + 1):
        order = pair_order(pair_index)
        paths = {
            role: report_path(
                output_dir, context, outputs, pair_index, role
            )
            for role in ("baseline", "candidate")
        }
        if not all(
            report_complete(
                paths[role],
                context=context,
                outputs=outputs,
                role=role,
                pair_index=pair_index,
                order=order,
                engine_sha256=engine_sha256[role],
                candidate_runtime_binding_sha256=(
                    candidate_runtime_binding_sha256
                    if role == "candidate"
                    else None
                ),
            )
            for role in paths
        ):
            continue
        payloads = {role: load_json(path) for role, path in paths.items()}
        measurements = {
            role: request_measurement(payload, output_index)
            for role, payload in payloads.items()
        }
        baseline = measurements["baseline"]
        candidate = measurements["candidate"]
        ratios = {
            "prefill_tps_candidate_over_baseline": ratio(
                candidate["prefill_tps"], baseline["prefill_tps"]
            ),
            "prefill_wall_candidate_over_baseline": ratio(
                candidate["prefill_wall_ms"], baseline["prefill_wall_ms"]
            ),
            "total_wall_candidate_over_baseline": ratio(
                candidate["total_wall_ms"], baseline["total_wall_ms"]
            ),
        }
        if output > 1:
            ratios.update(
                {
                    "decode_tps_candidate_over_baseline": ratio(
                        candidate["decode_tps"], baseline["decode_tps"]
                    ),
                    "decode_wall_candidate_over_baseline": ratio(
                        candidate["decode_wall_ms"], baseline["decode_wall_ms"]
                    ),
                }
            )
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
                "ratios": ratios,
            }
        )
    if not pairs:
        return None
    paired_medians = {
        name: median([pair["ratios"][name] for pair in pairs])
        for name in pairs[0]["ratios"]
    }
    candidate_medians = {
        name: median(
            [pair["measurements"]["candidate"][name] for pair in pairs]
        )
        for name in pairs[0]["measurements"]["candidate"]
    }
    baseline_medians = {
        name: median(
            [pair["measurements"]["baseline"][name] for pair in pairs]
        )
        for name in pairs[0]["measurements"]["baseline"]
    }
    floor = FROZEN_V151_FLOORS[(context, output)]
    floor_checks = {
        "prefill_tps": candidate_medians["prefill_tps"]
        >= 0.97 * floor["prefill_tps"]
    }
    if output > 1:
        floor_checks["decode_tps"] = (
            candidate_medians["decode_tps"] >= 0.97 * floor["decode_tps"]
        )
    paired_checks = {
        "prefill_tps": paired_medians[
            "prefill_tps_candidate_over_baseline"
        ]
        >= 1.0,
        "prefill_wall": paired_medians[
            "prefill_wall_candidate_over_baseline"
        ]
        <= 1.0,
        "total_wall": paired_medians[
            "total_wall_candidate_over_baseline"
        ]
        <= 1.0,
    }
    if output > 1:
        paired_checks.update(
            {
                "decode_tps": paired_medians[
                    "decode_tps_candidate_over_baseline"
                ]
                >= 1.0,
                "decode_wall": paired_medians[
                    "decode_wall_candidate_over_baseline"
                ]
                <= 1.0,
            }
        )
    complete = len(pairs) == pair_count and len(pairs) >= MINIMUM_PAIRS
    qualified = complete and all(paired_checks.values()) and all(
        floor_checks.values()
    )
    return {
        "input_tokens": context,
        "output_tokens": output,
        "complete": complete,
        "qualified": qualified,
        "pair_count": len(pairs),
        "required_pair_count": pair_count,
        "pairs": pairs,
        "paired_medians": paired_medians,
        "baseline_medians": baseline_medians,
        "candidate_medians": candidate_medians,
        "paired_checks": paired_checks,
        "legacy_floor": floor,
        "legacy_floor_retention": 0.97,
        "legacy_floor_checks": floor_checks,
    }


def build_startup_gate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    source = next(
        (
            cell
            for cell in cells
            if cell["input_tokens"] == 8192 and cell["output_tokens"] == 512
        ),
        None,
    )
    if source is None:
        return {
            "complete": False,
            "qualified": False,
            "reason": "q8192 paired measurements are incomplete",
        }
    pairs = source["pairs"]
    baseline = [
        pair["measurements"]["baseline"]["command_to_ready_wall_ms"]
        for pair in pairs
    ]
    candidate = [
        pair["measurements"]["candidate"]["command_to_ready_wall_ms"]
        for pair in pairs
    ]
    paired_ratios = [
        ratio(candidate_value, baseline_value)
        for candidate_value, baseline_value in zip(
            candidate, baseline, strict=True
        )
    ]
    candidate_median = median(candidate)
    paired_median = median(paired_ratios)
    checks = {
        "minimum_five_pairs": len(pairs) >= MINIMUM_PAIRS,
        "candidate_at_most_44_90_seconds": (
            candidate_median <= Q8192_STARTUP_LIMIT_MS
        ),
    }
    complete = source["complete"]
    return {
        "complete": complete,
        "qualified": complete and all(checks.values()),
        "baseline_runs_ms": baseline,
        "candidate_runs_ms": candidate,
        "paired_candidate_over_baseline": paired_ratios,
        "baseline_median_ms": median(baseline),
        "candidate_median_ms": candidate_median,
        "paired_ratio_median": paired_median,
        "paired_ratio_is_diagnostic_only": True,
        "absolute_limit_ms": Q8192_STARTUP_LIMIT_MS,
        "checks": checks,
        "ready_semantics_note": (
            "HTTP READY=1 VL readiness is sealed by the separate native "
            "surface qualification"
        ),
    }


def build_result(
    *,
    output_dir: Path,
    pair_counts: dict[tuple[int, tuple[int, ...]], int],
    identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    engine_sha256 = {
        role: identity["sha256"] for role, identity in identities.items()
    }
    candidate_runtime_binding_sha256 = identities["candidate"].get(
        "runtime_binding_sha256"
    )
    cells: list[dict[str, Any]] = []
    for context, outputs in jobs():
        pair_count = pair_counts[(context, outputs)]
        for output_index in range(len(outputs)):
            cell = build_cell(
                output_dir=output_dir,
                context=context,
                outputs=outputs,
                output_index=output_index,
                pair_count=pair_count,
                engine_sha256=engine_sha256,
                candidate_runtime_binding_sha256=(
                    candidate_runtime_binding_sha256
                ),
            )
            if cell is not None:
                cells.append(cell)
    startup = build_startup_gate(cells)
    complete = (
        len(cells) == len(FROZEN_V151_FLOORS)
        and all(cell["complete"] for cell in cells)
        and startup["complete"]
    )
    qualified = (
        complete
        and all(cell["qualified"] for cell in cells)
        and startup["qualified"]
    )
    return {
        "schema": "aima-amd395-qwen36/native-v151-paired-text-matrix/v1",
        "complete": complete,
        "qualified": qualified,
        "host": {
            "hostname": os.uname().nodename,
            "sysname": os.uname().sysname,
            "release": os.uname().release,
            "machine": os.uname().machine,
            "gpu_performance_level": optional_text(
                GPU_PERFORMANCE_LEVEL_PATH
            ),
        },
        "engines": identities,
        "model_dir": "${AIMA_MODEL_DIR}",
        "protocol": {
            "pair_count": min(pair_counts.values()),
            "pair_count_is_minimum": True,
            "minimum_observed_pair_count": min(pair_counts.values()),
            "maximum_observed_pair_count": max(pair_counts.values()),
            "pair_counts_by_job": {
                f"q{context}-o{'-'.join(map(str, outputs))}": pair_counts[
                    (context, outputs)
                ]
                for context, outputs in jobs()
            },
            "minimum_pairs": MINIMUM_PAIRS,
            "execution_order": (
                "baseline,candidate for odd pairs; candidate,baseline for even pairs"
            ),
            "pair_locality": "adjacent processes for each cell and pair",
            "cold_prefill": "first request in each fresh process",
            "standard_decode": (
                "output512 cold request then output1024 exact-prefix restore"
            ),
            "throughput_gate": "paired candidate/baseline median >= 1.000",
            "latency_gate": "paired candidate/baseline median <= 1.000",
            "startup_gate": "candidate median <= 44.90 seconds",
            "startup_paired_ratio": "diagnostic only",
            "legacy_floor_gate": "candidate median >= 0.97 * frozen v1.5.1 floor",
            "aggregate_cannot_mask_cell_failure": True,
        },
        "expected_cell_count": len(FROZEN_V151_FLOORS),
        "observed_cell_count": len(cells),
        "cells": cells,
        "q8192_startup": startup,
        "all_cells_pass": complete and all(
            cell["qualified"] for cell in cells
        ),
        "text_request_path_idle": complete and all(
            candidate_text_path_is_idle(
                load_json(
                    report_path(
                        output_dir,
                        context,
                        outputs,
                        pair_index,
                        "candidate",
                    )
                )
            )
            for context, outputs in jobs()
            for pair_index in range(
                1, pair_counts[(context, outputs)] + 1
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-engine", type=Path, required=True)
    parser.add_argument("--candidate-engine", type=Path, required=True)
    parser.add_argument("--candidate-fmha-provider", type=Path)
    parser.add_argument("--candidate-fmha-provider-sha256", required=True)
    parser.add_argument("--candidate-long-context-fmha-provider", type=Path)
    parser.add_argument(
        "--candidate-long-context-fmha-provider-sha256", required=True
    )
    parser.add_argument("--candidate-q16384-hybrid-provider", type=Path)
    parser.add_argument(
        "--candidate-q16384-hybrid-provider-sha256", required=True
    )
    parser.add_argument("--candidate-vision-attention-image", type=Path)
    parser.add_argument(
        "--candidate-vision-attention-sha256",
        default=VISION_ATTENTION_SHA256,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-engine-sha256", required=True)
    parser.add_argument("--candidate-engine-sha256", required=True)
    parser.add_argument(
        "--baseline-source-commit", default=V151_SOURCE_COMMIT
    )
    parser.add_argument("--candidate-source-commit", required=True)
    parser.add_argument("--baseline-version", default=V151_VERSION)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument(
        "--pair-count-override",
        action="append",
        default=[],
        metavar="CONTEXT/OUTPUT[,OUTPUT]=PAIRS",
        help=(
            "increase repetitions for one frozen job without rerunning every "
            "matrix cell; may be repeated"
        ),
    )
    parser.add_argument("--uniform-input-token-id", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()

    baseline_engine = cli.baseline_engine.expanduser().resolve()
    candidate_engine = cli.candidate_engine.expanduser().resolve()
    candidate_fmha_provider = (
        cli.candidate_fmha_provider.expanduser().resolve()
        if cli.candidate_fmha_provider is not None
        else automatic_runtime_path(candidate_engine, FMHA_AOTRITON_FILENAME)
    )
    candidate_long_context_fmha_provider = (
        cli.candidate_long_context_fmha_provider.expanduser().resolve()
        if cli.candidate_long_context_fmha_provider is not None
        else automatic_runtime_path(candidate_engine, FMHA_CK_FILENAME)
    )
    candidate_q16384_hybrid_provider = (
        cli.candidate_q16384_hybrid_provider.expanduser().resolve()
        if cli.candidate_q16384_hybrid_provider is not None
        else automatic_runtime_path(
            candidate_engine, FMHA_Q16384_HYBRID_FILENAME
        )
    )
    candidate_vision_attention = (
        cli.candidate_vision_attention_image.expanduser().resolve()
        if cli.candidate_vision_attention_image is not None
        else automatic_runtime_path(candidate_engine, VISION_ATTENTION_FILENAME)
    )
    model_dir = cli.model_dir.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    for role, engine in (
        ("baseline", baseline_engine),
        ("candidate", candidate_engine),
    ):
        if not engine.is_file() or not os.access(engine, os.X_OK):
            raise SystemExit(f"{role} engine is not executable")
    if not model_dir.is_dir():
        raise SystemExit("model directory is missing")
    try:
        pair_counts = resolve_pair_counts(cli.pairs, cli.pair_count_override)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not 0 <= cli.uniform_input_token_id < 248320:
        raise SystemExit("--uniform-input-token-id is invalid")

    identities = {
        "baseline": bind_engine(
            role="baseline",
            engine=baseline_engine,
            expected_sha256=cli.baseline_engine_sha256,
            expected_source_commit=cli.baseline_source_commit,
            expected_version=cli.baseline_version,
        ),
        "candidate": bind_engine(
            role="candidate",
            engine=candidate_engine,
            expected_sha256=cli.candidate_engine_sha256,
            expected_source_commit=cli.candidate_source_commit,
            expected_version=None,
        ),
    }
    identities["candidate"]["runtime_dependencies"] = bind_candidate_runtime(
        engine=candidate_engine,
        fmha_provider=candidate_fmha_provider,
        expected_fmha_provider_sha256=cli.candidate_fmha_provider_sha256,
        long_context_fmha_provider=candidate_long_context_fmha_provider,
        expected_long_context_fmha_provider_sha256=(
            cli.candidate_long_context_fmha_provider_sha256
        ),
        q16384_hybrid_provider=candidate_q16384_hybrid_provider,
        expected_q16384_hybrid_provider_sha256=(
            cli.candidate_q16384_hybrid_provider_sha256
        ),
        vision_attention_image=candidate_vision_attention,
        expected_vision_attention_sha256=(
            cli.candidate_vision_attention_sha256
        ),
    )
    identities["candidate"]["runtime_policy"] = CANDIDATE_RUNTIME_POLICY
    identities["candidate"]["runtime_binding_sha256"] = json_sha256(
        identities["candidate"]["runtime_dependencies"]
    )
    baseline_runtime_root = (
        baseline_engine.parents[1]
        if baseline_engine.parent.name == "libexec"
        else baseline_engine.parent
    )
    replacements = (
        (str(baseline_engine), "${AIMA_BASELINE_ENGINE}"),
        (
            str(baseline_runtime_root),
            "${AIMA_BASELINE_RUNTIME_ROOT}",
        ),
        (str(candidate_engine), "${AIMA_CANDIDATE_ENGINE}"),
        (
            str(candidate_fmha_provider),
            "${AIMA_CANDIDATE_FMHA_PROVIDER}",
        ),
        (
            str(candidate_long_context_fmha_provider),
            "${AIMA_CANDIDATE_LONG_CONTEXT_FMHA_PROVIDER}",
        ),
        (
            str(candidate_q16384_hybrid_provider),
            "${AIMA_CANDIDATE_Q16384_HYBRID_FMHA_PROVIDER}",
        ),
        (
            str(candidate_vision_attention),
            "${AIMA_CANDIDATE_VISION_ATTENTION_IMAGE}",
        ),
        (
            str(candidate_engine.parent),
            "${AIMA_CANDIDATE_RUNTIME_DIR}",
        ),
        (str(model_dir), "${AIMA_MODEL_DIR}"),
        (str(output_dir), "${AIMA_QUALIFICATION_DIR}"),
        (str(ROOT), "${AIMA_REPO_ROOT}"),
    )
    engines = {
        "baseline": baseline_engine,
        "candidate": candidate_engine,
    }
    runtime_options = {
        "baseline": (),
        # Both engines must exercise their packaged automatic provider policy:
        # AOTriton through q4096, CK at q8192/q32768/long context, and the
        # packed-GQA/CK hybrid at q16384. An explicit provider would silently
        # replace that product routing policy and invalidate paired results.
        "candidate": (),
    }
    sequence_index = (
        maximum_recorded_sequence_index(output_dir) if cli.resume else 0
    )
    for context, outputs in jobs():
        pair_count = pair_counts[(context, outputs)]
        for pair_index in range(1, pair_count + 1):
            order = pair_order(pair_index)
            for role in order:
                path = report_path(
                    output_dir, context, outputs, pair_index, role
                )
                if not (
                    cli.resume
                    and report_complete(
                        path,
                        context=context,
                        outputs=outputs,
                        role=role,
                        pair_index=pair_index,
                        order=order,
                        engine_sha256=identities[role]["sha256"],
                        candidate_runtime_binding_sha256=(
                            identities["candidate"][
                                "runtime_binding_sha256"
                            ]
                            if role == "candidate"
                            else None
                        ),
                    )
                ):
                    sequence_index += 1
                    run_report(
                        engine=engines[role],
                        model_dir=model_dir,
                        output_dir=output_dir,
                        context=context,
                        outputs=outputs,
                        uniform_token_id=cli.uniform_input_token_id,
                        role=role,
                        pair_index=pair_index,
                        order=order,
                        sequence_index=sequence_index,
                        engine_sha256=identities[role]["sha256"],
                        candidate_runtime_binding_sha256=(
                            identities["candidate"][
                                "runtime_binding_sha256"
                            ]
                            if role == "candidate"
                            else None
                        ),
                        runtime_options=runtime_options[role],
                        replacements=replacements,
                    )
            atomic_json(
                output_dir / "matrix.json",
                build_result(
                    output_dir=output_dir,
                    pair_counts=pair_counts,
                    identities=identities,
                ),
            )

    sanitize_role_reports(output_dir, replacements)
    result = build_result(
        output_dir=output_dir,
        pair_counts=pair_counts,
        identities=identities,
    )
    atomic_json(output_dir / "matrix.json", result)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "qualified": result["qualified"],
                "cell_count": result["observed_cell_count"],
                "minimum_pair_count": min(pair_counts.values()),
                "maximum_pair_count": max(pair_counts.values()),
                "output": "${AIMA_QUALIFICATION_DIR}/matrix.json",
            },
            sort_keys=True,
        )
    )
    if not result["qualified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
