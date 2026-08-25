#!/usr/bin/env python3
"""Qualify native full-vocabulary logits against frozen source oracles."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from aima_engine.aotriton_closure import require_aotriton_closure


ROOT = Path(__file__).resolve().parents[1]
FMHA_AOTRITON_FILENAME = "libaima-fmha-aotriton.so"
FMHA_CK_FILENAME = "libaima-fmha-ck.so"
FMHA_Q16384_HYBRID_FILENAME = "libaima-fmha-q16384-hybrid.so"
VISION_ATTENTION_FILENAME = "aima-vision-attention.hsaco"
PRODUCTION_TOKEN_PERIOD = (
    55771,
    18,
    24,
    20,
    1167,
    16451,
    18,
    13,
    21,
    48804,
    16,
    21,
    4906,
    6220,
    26388,
    9640,
    13,
    32975,
    8981,
    874,
    64913,
    35464,
    6937,
    18030,
    13092,
    13,
    220,
)
CORRECTNESS_SCHEMA = "aima-amd395-qwen36/native-correctness-qualification/v1"


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
        "native correctness paused because /dev/kfd is owned externally",
        file=sys.stderr,
    )
    raise SystemExit(75)


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
    model_dir: Path,
    replacements: tuple[tuple[str, str], ...] = (),
) -> Any:
    if isinstance(value, str):
        result = value
        for private, public in replacements:
            result = result.replace(private, public)
        return result.replace(str(ROOT), "${AIMA_REPO_ROOT}").replace(
            str(model_dir), "${AIMA_MODEL_DIR}"
        )
    if isinstance(value, list):
        return [publicize(item, model_dir, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: publicize(item, model_dir, replacements)
            for key, item in value.items()
        }
    return value


def bind_reference_correctness(
    *,
    path: Path,
    cases: list[tuple[int, tuple[int, ...], Path]],
    exact_context: int,
    exact_token_id: int,
    exact_completion_tokens: int,
) -> dict[str, Any]:
    reference = load_json(path)
    if (
        reference.get("schema") != CORRECTNESS_SCHEMA
        or reference.get("complete") is not True
        or reference.get("qualified") is not True
    ):
        raise SystemExit("frozen correctness reference is not qualified")
    rows = reference.get("cases")
    if not isinstance(rows, list):
        raise SystemExit("frozen correctness reference has no cases")
    by_context: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("frozen correctness reference has an invalid case")
        context = row.get("context_tokens")
        if not isinstance(context, int) or context in by_context:
            raise SystemExit(
                "frozen correctness reference has duplicate or invalid contexts"
            )
        by_context[context] = row
    requested_contexts = {context for context, _, _ in cases}
    if set(by_context) != requested_contexts:
        raise SystemExit(
            "frozen correctness reference context set differs from --oracle"
        )
    for context, input_cycle, oracle in cases:
        row = by_context[context]
        if row.get("qualified") is not True:
            raise SystemExit(
                f"frozen correctness reference q{context} is not qualified"
            )
        if row.get("input_token_period") != list(input_cycle):
            raise SystemExit(
                f"frozen correctness reference q{context} input period changed"
            )
        if row.get("oracle_sha256") != sha256(oracle):
            raise SystemExit(
                f"frozen correctness reference q{context} oracle SHA-256 changed"
            )
    exact = reference.get("exact_completion")
    if not isinstance(exact, dict) or exact.get("qualified") is not True:
        raise SystemExit(
            "frozen correctness reference exact completion is not qualified"
        )
    if (
        exact.get("context_tokens") != exact_context
        or exact.get("input_token_id") != exact_token_id
        or exact.get("completion_tokens") != exact_completion_tokens
        or exact.get("expected_token_id") != exact_token_id
    ):
        raise SystemExit("frozen correctness exact completion contract changed")
    exact_output_sha256 = exact.get("output_token_ids_sha256")
    if (
        not isinstance(exact_output_sha256, str)
        or len(exact_output_sha256) != 64
    ):
        raise SystemExit(
            "frozen correctness exact completion SHA-256 is invalid"
        )
    reference_engine = reference.get("engine")
    return {
        "path": "${AIMA_FROZEN_CORRECTNESS_REFERENCE}",
        "sha256": sha256(path),
        "schema": reference["schema"],
        "engine_sha256": (
            reference_engine.get("sha256")
            if isinstance(reference_engine, dict)
            else None
        ),
        "case_count": len(rows),
        "exact_output_token_ids_sha256": exact_output_sha256,
    }


def automatic_runtime_path(engine: Path, filename: str) -> Path:
    executable_dir = engine.parent
    if executable_dir.name == "libexec":
        return executable_dir.parent / "lib" / filename
    if executable_dir.name == "bin" and (
        executable_dir.parent / "libexec" / "aima-engine.real"
    ).is_file():
        return executable_dir.parent / "lib" / filename
    return executable_dir / filename


def bind_runtime_dependencies(
    *,
    engine: Path,
    expected_aotriton_provider_sha256: str,
    expected_ck_provider_sha256: str,
    expected_q16384_hybrid_provider_sha256: str,
    expected_vision_attention_sha256: str,
) -> dict[str, Any]:
    aotriton_provider = automatic_runtime_path(
        engine, FMHA_AOTRITON_FILENAME
    )
    closure = require_aotriton_closure(aotriton_provider)
    ck_provider = automatic_runtime_path(engine, FMHA_CK_FILENAME)
    hybrid_provider = automatic_runtime_path(
        engine, FMHA_Q16384_HYBRID_FILENAME
    )
    vision_attention = automatic_runtime_path(
        engine, VISION_ATTENTION_FILENAME
    )
    checks = (
        (
            "AOTriton FMHA provider",
            closure.provider,
            expected_aotriton_provider_sha256,
        ),
        ("CK FMHA provider", ck_provider, expected_ck_provider_sha256),
        (
            "q16384 hybrid FMHA provider",
            hybrid_provider,
            expected_q16384_hybrid_provider_sha256,
        ),
        (
            "vision-attention image",
            vision_attention,
            expected_vision_attention_sha256,
        ),
    )
    for label, path, expected in checks:
        if not path.is_file():
            raise SystemExit(f"candidate {label} is missing")
        if sha256(path) != expected:
            raise SystemExit(f"candidate {label} SHA-256 changed")
    return {
        "automatic_provider_policy": True,
        "aotriton_fmha_provider": {
            "path": "${AIMA_FMHA_AOTRITON_PROVIDER}",
            "sha256": sha256(closure.provider),
        },
        "aotriton_runtime": {
            "path": "${AIMA_AOTRITON_RUNTIME}",
            "sha256": sha256(closure.runtime),
        },
        "aotriton_image": {
            "path": "${AIMA_AOTRITON_IMAGE}",
            "sha256": sha256(closure.image),
        },
        "ck_fmha_provider": {
            "path": "${AIMA_FMHA_CK_PROVIDER}",
            "sha256": sha256(ck_provider),
        },
        "q16384_hybrid_fmha_provider": {
            "path": "${AIMA_FMHA_Q16384_HYBRID_PROVIDER}",
            "sha256": sha256(hybrid_provider),
        },
        "vision_attention_image": {
            "path": "${AIMA_VISION_ATTENTION_IMAGE}",
            "sha256": sha256(vision_attention),
        },
    }


def parse_oracle(text: str) -> tuple[int, tuple[int, ...], Path]:
    case_text, separator, path_text = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            "--oracle must use CONTEXT[@TOKEN[,TOKEN...]]=FINAL_LOGITS_PATH"
        )
    context_text, token_separator, token_text = case_text.partition("@")
    try:
        context = int(context_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid oracle context: {context_text}"
        ) from exc
    if context <= 0:
        raise argparse.ArgumentTypeError("oracle context must be positive")
    if token_separator:
        try:
            input_cycle = tuple(
                int(item) for item in token_text.split(",") if item
            )
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid oracle input token cycle: {token_text}"
            ) from exc
        if not input_cycle or any(
            token < 0 or token >= 248320 for token in input_cycle
        ):
            raise argparse.ArgumentTypeError(
                f"invalid oracle input token cycle: {token_text}"
            )
    else:
        input_cycle = PRODUCTION_TOKEN_PERIOD
    return context, input_cycle, Path(path_text)


def report_qualified(
    payload: dict[str, Any],
    *,
    context: int,
    engine_sha256: str,
    oracle_sha256: str,
    input_cycle: tuple[int, ...],
    runtime_binding_sha256: str | None,
    reference_correctness_sha256: str | None = None,
) -> bool:
    try:
        comparison = payload["reference_logits"]
        request = payload["requests"][0]
        qualification = payload["qualification"]
        return bool(
            payload["schema"]
            == "aima-amd395-qwen36/native-resident-session-probe/v1"
            and payload["complete"] is True
            and payload["qualified"] is True
            and payload["correctness_claim"] is True
            and payload["runtime_python"] is False
            and payload["runtime_torch"] is False
            and payload["runtime_vllm"] is False
            and payload["runtime_triton"] is False
            and int(payload["model_loads"]) == 1
            and int(payload["request_count"]) == 1
            and int(request["prompt_tokens"]) == context
            and int(request["completion_tokens"]) == 1
            and request["first_token_certified"] is True
            and request["all_decode_tokens_certified"] is True
            and int(comparison["elements"]) == 248320
            and int(comparison["finite_elements"]) == 248320
            and comparison["top1_match"] is True
            and comparison["qualified"] is True
            and float(comparison["kl_divergence"]) < 0.005
            and qualification["engine_sha256"] == engine_sha256
            and qualification["oracle_sha256"] == oracle_sha256
            and qualification["input_token_period"] == list(input_cycle)
            and (
                runtime_binding_sha256 is None
                or qualification.get("runtime_binding_sha256")
                == runtime_binding_sha256
            )
            and (
                reference_correctness_sha256 is None
                or qualification.get("reference_correctness_sha256")
                == reference_correctness_sha256
            )
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def run_case(
    *,
    engine: Path,
    model_dir: Path,
    output_dir: Path,
    context: int,
    oracle: Path,
    engine_sha256: str,
    input_cycle: tuple[int, ...],
    runtime_binding_sha256: str | None,
    reference_correctness_sha256: str,
) -> Path:
    report = output_dir / "raw" / f"q{context}-full-vocabulary.json"
    oracle_sha256 = sha256(oracle)
    if report.is_file():
        previous = load_json(report)
        if report_qualified(
            previous,
            context=context,
            engine_sha256=engine_sha256,
            oracle_sha256=oracle_sha256,
            input_cycle=input_cycle,
            runtime_binding_sha256=runtime_binding_sha256,
            reference_correctness_sha256=reference_correctness_sha256,
        ):
            return report

    load_report = report.with_name(report.stem + ".load.json")
    command = [
        str(engine),
        "resident-session-probe",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        str(context),
        "--max-new-tokens",
        "1",
        "--requests",
        "1",
        "--disable-prefix-cache",
        "--reference-logits",
        str(oracle),
        "--report",
        str(load_report),
    ]
    if len(input_cycle) == 1:
        command[6:6] = ["--uniform-input-token-id", str(input_cycle[0])]
    else:
        command[6:6] = [
            "--input-token-id-cycle",
            ",".join(map(str, input_cycle)),
        ]
    print(
        json.dumps(
            {
                "event": "correctness_run_start",
                "context_tokens": context,
                "oracle": f"${{AIMA_ORACLE_Q{context}}}",
            },
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
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"native correctness run emitted invalid JSON at q{context}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"native correctness run emitted non-object JSON at q{context}"
        )
    payload["qualification"] = {
        "command": command,
        "engine_sha256": engine_sha256,
        "oracle_sha256": oracle_sha256,
        "load_report": str(load_report),
        "load_report_sha256": (
            sha256(load_report) if load_report.is_file() else None
        ),
        "input_token_period": list(input_cycle),
        "runtime_binding_sha256": runtime_binding_sha256,
        "reference_correctness_sha256": reference_correctness_sha256,
    }
    replacements = (
        (str(oracle), f"${{AIMA_ORACLE_Q{context}}}"),
        (str(engine), "${AIMA_CANDIDATE_ENGINE}"),
        (str(engine.parent), "${AIMA_CANDIDATE_ENGINE_DIR}"),
        (str(output_dir), "${AIMA_QUALIFICATION_DIR}"),
    )
    payload = publicize(payload, model_dir, replacements)
    atomic_json(report, payload)
    if completed.returncode != 0 or not report_qualified(
        payload,
        context=context,
        engine_sha256=engine_sha256,
        oracle_sha256=oracle_sha256,
        input_cycle=input_cycle,
        runtime_binding_sha256=runtime_binding_sha256,
        reference_correctness_sha256=reference_correctness_sha256,
    ):
        raise RuntimeError(
            f"native correctness run failed with exit {completed.returncode}: "
            f"{report}"
        )
    print(
        json.dumps(
            {
                "event": "correctness_run_complete",
                "context_tokens": context,
                "kl_divergence": payload["reference_logits"]["kl_divergence"],
                "top1_match": payload["reference_logits"]["top1_match"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def exact_completion_qualified(
    payload: dict[str, Any],
    *,
    engine_sha256: str,
    context: int,
    completion_tokens: int,
    runtime_binding_sha256: str | None,
    reference_correctness_sha256: str | None = None,
    expected_output_token_ids_sha256: str | None = None,
) -> bool:
    try:
        request = payload["requests"][0]
        return bool(
            payload["schema"]
            == "aima-amd395-qwen36/native-resident-session-probe/v1"
            and payload["complete"] is True
            and payload["expected_tokens_provided"] is True
            and payload["expected_tokens_match"] is True
            and int(payload["model_loads"]) == 1
            and int(payload["request_count"]) == 1
            and int(request["prompt_tokens"]) == context
            and int(request["completion_tokens"]) == completion_tokens
            and request["first_token_certified"] is True
            and request["all_decode_tokens_certified"] is True
            and payload["qualification"]["engine_sha256"] == engine_sha256
            and (
                runtime_binding_sha256 is None
                or payload["qualification"].get("runtime_binding_sha256")
                == runtime_binding_sha256
            )
            and (
                reference_correctness_sha256 is None
                or payload["qualification"].get(
                    "reference_correctness_sha256"
                )
                == reference_correctness_sha256
            )
            and (
                expected_output_token_ids_sha256 is None
                or request.get("output_token_ids_sha256")
                == expected_output_token_ids_sha256
            )
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def run_exact_completion(
    *,
    engine: Path,
    model_dir: Path,
    output_dir: Path,
    engine_sha256: str,
    context: int,
    token_id: int,
    completion_tokens: int,
    runtime_binding_sha256: str | None,
    reference_correctness_sha256: str,
    expected_output_token_ids_sha256: str,
) -> Path:
    report = (
        output_dir
        / "raw"
        / f"q{context}-exact-{completion_tokens}-token-completion.json"
    )
    if report.is_file() and exact_completion_qualified(
        load_json(report),
        engine_sha256=engine_sha256,
        context=context,
        completion_tokens=completion_tokens,
        runtime_binding_sha256=runtime_binding_sha256,
        reference_correctness_sha256=reference_correctness_sha256,
        expected_output_token_ids_sha256=expected_output_token_ids_sha256,
    ):
        return report
    load_report = report.with_name(report.stem + ".load.json")
    expected = ",".join([str(token_id)] * completion_tokens)
    command = [
        str(engine),
        "resident-session-probe",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        str(context),
        "--uniform-input-token-id",
        str(token_id),
        "--max-new-tokens",
        str(completion_tokens),
        "--requests",
        "1",
        "--disable-prefix-cache",
        "--expected-token-ids",
        expected,
        "--report",
        str(load_report),
    ]
    print(
        json.dumps(
            {
                "event": "exact_completion_run_start",
                "context_tokens": context,
                "completion_tokens": completion_tokens,
            },
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
        check=False,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(
            "native exact-completion run emitted non-object JSON"
        )
    payload["qualification"] = {
        "command": command,
        "engine_sha256": engine_sha256,
        "load_report": str(load_report),
        "load_report_sha256": sha256(load_report),
        "expected_token_id": token_id,
        "expected_completion_tokens": completion_tokens,
        "runtime_binding_sha256": runtime_binding_sha256,
        "reference_correctness_sha256": reference_correctness_sha256,
    }
    replacements = (
        (str(engine), "${AIMA_CANDIDATE_ENGINE}"),
        (str(engine.parent), "${AIMA_CANDIDATE_ENGINE_DIR}"),
        (str(output_dir), "${AIMA_QUALIFICATION_DIR}"),
    )
    payload = publicize(payload, model_dir, replacements)
    atomic_json(report, payload)
    if completed.returncode != 0 or not exact_completion_qualified(
        payload,
        engine_sha256=engine_sha256,
        context=context,
        completion_tokens=completion_tokens,
        runtime_binding_sha256=runtime_binding_sha256,
        reference_correctness_sha256=reference_correctness_sha256,
        expected_output_token_ids_sha256=expected_output_token_ids_sha256,
    ):
        raise RuntimeError(
            f"native exact-completion qualification failed: {report}"
        )
    print(
        json.dumps(
            {
                "event": "exact_completion_run_complete",
                "context_tokens": context,
                "completion_tokens": completion_tokens,
                "output_token_ids_sha256": payload["requests"][0][
                    "output_token_ids_sha256"
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
    parser.add_argument(
        "--oracle",
        type=parse_oracle,
        action="append",
        required=True,
        metavar="CONTEXT[@TOKEN[,TOKEN...]]=FINAL_LOGITS_PATH",
    )
    parser.add_argument("--exact-context", type=int, default=8192)
    parser.add_argument("--exact-token-id", type=int, default=1000)
    parser.add_argument("--exact-completion-tokens", type=int, default=128)
    parser.add_argument(
        "--reference-correctness",
        type=Path,
        required=True,
        help=(
            "qualified frozen correctness aggregate that binds oracle hashes, "
            "input periods, and the exact-completion fixture"
        ),
    )
    parser.add_argument("--aotriton-provider-sha256")
    parser.add_argument("--ck-provider-sha256")
    parser.add_argument("--q16384-hybrid-provider-sha256")
    parser.add_argument("--vision-attention-sha256")
    cli = parser.parse_args()

    engine = cli.engine.expanduser().resolve()
    model_dir = cli.model_dir.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    if not engine.is_file() or not os.access(engine, os.X_OK):
        raise SystemExit(f"native engine is not executable: {engine}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")
    cases: list[tuple[int, tuple[int, ...], Path]] = []
    for context, input_cycle, oracle_argument in cli.oracle:
        oracle = oracle_argument.expanduser().resolve()
        if not oracle.is_file():
            raise SystemExit(f"oracle is missing: {oracle}")
        cases.append((context, input_cycle, oracle))
    if len({context for context, _, _ in cases}) != len(cases):
        raise SystemExit("oracle contexts must be unique")

    reference_path = cli.reference_correctness.expanduser().resolve()
    if not reference_path.is_file():
        raise SystemExit("frozen correctness reference is missing")
    reference_correctness = bind_reference_correctness(
        path=reference_path,
        cases=cases,
        exact_context=cli.exact_context,
        exact_token_id=cli.exact_token_id,
        exact_completion_tokens=cli.exact_completion_tokens,
    )
    reference_correctness_sha256 = reference_correctness["sha256"]

    engine_sha256 = sha256(engine)
    expected_runtime_sha256 = {
        "aotriton_provider": cli.aotriton_provider_sha256,
        "ck_provider": cli.ck_provider_sha256,
        "q16384_hybrid_provider": cli.q16384_hybrid_provider_sha256,
        "vision_attention": cli.vision_attention_sha256,
    }
    supplied_runtime_sha256 = {
        name: value
        for name, value in expected_runtime_sha256.items()
        if value is not None
    }
    if supplied_runtime_sha256 and len(supplied_runtime_sha256) != len(
        expected_runtime_sha256
    ):
        missing = sorted(set(expected_runtime_sha256) - set(supplied_runtime_sha256))
        raise SystemExit(
            "runtime SHA-256 arguments are all-or-none; missing: "
            + ", ".join(missing)
        )
    runtime_dependencies: dict[str, Any] | None = None
    runtime_binding_sha256: str | None = None
    if supplied_runtime_sha256:
        runtime_dependencies = bind_runtime_dependencies(
            engine=engine,
            expected_aotriton_provider_sha256=cli.aotriton_provider_sha256,
            expected_ck_provider_sha256=cli.ck_provider_sha256,
            expected_q16384_hybrid_provider_sha256=(
                cli.q16384_hybrid_provider_sha256
            ),
            expected_vision_attention_sha256=cli.vision_attention_sha256,
        )
        runtime_binding_sha256 = json_sha256(runtime_dependencies)
    reports = [
        run_case(
            engine=engine,
            model_dir=model_dir,
            output_dir=output_dir,
            context=context,
            oracle=oracle,
            engine_sha256=engine_sha256,
            input_cycle=input_cycle,
            runtime_binding_sha256=runtime_binding_sha256,
            reference_correctness_sha256=reference_correctness_sha256,
        )
        for context, input_cycle, oracle in cases
    ]
    exact_report = run_exact_completion(
        engine=engine,
        model_dir=model_dir,
        output_dir=output_dir,
        engine_sha256=engine_sha256,
        context=cli.exact_context,
        token_id=cli.exact_token_id,
        completion_tokens=cli.exact_completion_tokens,
        runtime_binding_sha256=runtime_binding_sha256,
        reference_correctness_sha256=reference_correctness_sha256,
        expected_output_token_ids_sha256=reference_correctness[
            "exact_output_token_ids_sha256"
        ],
    )
    rows: list[dict[str, Any]] = []
    all_pass = True
    for (context, input_cycle, oracle), report in zip(
        cases, reports, strict=True
    ):
        payload = load_json(report)
        comparison = payload["reference_logits"]
        row = {
            "context_tokens": context,
            "oracle_path": f"${{AIMA_ORACLE_Q{context}}}",
            "oracle_sha256": sha256(oracle),
            "input_token_period": list(input_cycle),
            "report": str(report.relative_to(output_dir)),
            "report_sha256": sha256(report),
            "actual_top1_token_id": comparison["actual_top1_token_id"],
            "reference_top1_token_id": comparison[
                "reference_top1_token_id"
            ],
            "top1_match": comparison["top1_match"],
            "kl_divergence": comparison["kl_divergence"],
            "relative_l2_error": comparison["relative_l2_error"],
            "maximum_absolute_error": comparison["maximum_absolute_error"],
            "exact_elements": comparison["exact_elements"],
            "qualified": report_qualified(
                payload,
                context=context,
                engine_sha256=engine_sha256,
                oracle_sha256=sha256(oracle),
                input_cycle=input_cycle,
                runtime_binding_sha256=runtime_binding_sha256,
                reference_correctness_sha256=(
                    reference_correctness_sha256
                ),
            ),
        }
        all_pass = all_pass and row["qualified"]
        rows.append(row)
    exact_payload = load_json(exact_report)
    exact_request = exact_payload["requests"][0]
    exact_pass = exact_completion_qualified(
        exact_payload,
        engine_sha256=engine_sha256,
        context=cli.exact_context,
        completion_tokens=cli.exact_completion_tokens,
        runtime_binding_sha256=runtime_binding_sha256,
        reference_correctness_sha256=reference_correctness_sha256,
        expected_output_token_ids_sha256=reference_correctness[
            "exact_output_token_ids_sha256"
        ],
    )
    all_pass = all_pass and exact_pass
    result = {
        "schema": CORRECTNESS_SCHEMA,
        "complete": True,
        "qualified": all_pass,
        "engine": {
            "path": "${AIMA_CANDIDATE_ENGINE}",
            "sha256": engine_sha256,
        },
        "model_dir": "${AIMA_MODEL_DIR}",
        "runtime_dependencies": runtime_dependencies,
        "runtime_binding_sha256": runtime_binding_sha256,
        "frozen_correctness_reference": reference_correctness,
        "host": {
            "hostname": os.uname().nodename,
            "sysname": os.uname().sysname,
            "release": os.uname().release,
            "machine": os.uname().machine,
        },
        "gate": {
            "full_vocabulary_elements": 248320,
            "kl_divergence_strictly_less_than": 0.005,
            "top1_match": True,
        },
        "input_token_period": list(PRODUCTION_TOKEN_PERIOD),
        "cases": rows,
        "exact_completion": {
            "context_tokens": cli.exact_context,
            "input_token_id": cli.exact_token_id,
            "completion_tokens": cli.exact_completion_tokens,
            "expected_token_id": cli.exact_token_id,
            "expected_tokens_match": exact_payload[
                "expected_tokens_match"
            ],
            "output_token_ids_sha256": exact_request[
                "output_token_ids_sha256"
            ],
            "report": str(exact_report.relative_to(output_dir)),
            "report_sha256": sha256(exact_report),
            "qualified": exact_pass,
        },
    }
    atomic_json(output_dir / "correctness.json", result)
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": all_pass,
                "case_count": len(rows),
                "output": "${AIMA_QUALIFICATION_DIR}/correctness.json",
            },
            sort_keys=True,
        )
    )
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
