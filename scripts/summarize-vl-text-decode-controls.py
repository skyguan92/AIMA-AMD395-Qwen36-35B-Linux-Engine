#!/usr/bin/env python3
"""Qualify G4 VL decode retention with adjacent same-process text controls."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
    verify_manifest_integrity,
)


SCHEMA = "aima-amd395-qwen36/vl-text-decode-retention/v1"
MANIFEST_SCHEMA = "aima-amd395-qwen36/vl-text-decode-control-manifest/v1"
MATRIX_SCHEMA = "aima-amd395-qwen36/vl-performance-comparable-matrix/v1"
TEXT_SAMPLE_SCHEMA = "aima-amd395-qwen36/vl-text-decode-control-sample/v1"
VL_SAMPLE_SCHEMA = "aima-amd395-qwen36/vl-performance-request-sample/v1"
TIMING_BOUNDARY = (
    "perf_counter_ns immediately before HTTP open through SSE stream EOF; "
    "TTFT ends at the first semantic delta; decode throughput is "
    "(completion_tokens - 1) / (total_seconds - ttft_seconds)"
)
TEXT_BENCHMARK_ID = re.compile(
    r"^(text_q[0-9]+_output[0-9]+)\.control-([1-9][0-9]*)$"
)
VL_BENCHMARK_ID = re.compile(r"^(.+)\.vl-control-([1-9][0-9]*)$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MATRIX_PATH = (
    ROOT
    / "benchmarks/fixtures/vl-performance-v0.1.0/comparable-matrix.json"
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def read_index(path: Path, label: str) -> int:
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is missing or invalid") from error
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def paired_identity_module():
    path = Path(__file__).with_name("summarize-vl-performance-pairs.py")
    spec = importlib.util.spec_from_file_location("vl_paired_identity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load paired VL identity helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def health_contract(health: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "status",
        "model",
        "model_loaded",
        "resident",
        "runtime",
        "fmha_provider_backend",
        "secondary_fmha_provider_backend",
        "secondary_fmha_layers",
        "request_timeout_ms",
        "admitted_prompt_tokens",
        "context_capacity",
        "static_prefill_tokens",
        "resident_prefill_buckets",
        "prefix_cache_entries",
        "native_vl",
        "vision_warmup_completed",
        "vision_warmup_patches",
        "vision_warmup_visual_tokens",
        "vision_image_count_warmup_patches",
        "vision_image_count_warmup_visual_tokens",
        "vision_plan_cache_entries_at_ready",
        "vision_attention_image_sha256",
        "vision_dense_image_attention_image_sha256",
        "media_cache_capacity_bytes",
        "media_cache_capacity_entries",
        "prompt_token_ids_extension",
    )
    return {name: health.get(name) for name in fields}


def validate_health(health: Mapping[str, Any], label: str) -> None:
    if not (
        health.get("status") == "ok"
        and health.get("model") == "aima-amd395-qwen36-35b"
        and health.get("model_loaded") is True
        and health.get("resident") is True
        and health.get("runtime") == "native"
        and health.get("native_vl") is True
        and health.get("vision_warmup_completed") is True
        and health.get("vision_warmup_patches") == 1024
        and health.get("vision_warmup_visual_tokens") == 256
        and health.get("vision_image_count_warmup_patches") == 4096
        and health.get("vision_image_count_warmup_visual_tokens") == 1024
        and health.get("vision_plan_cache_entries_at_ready") == 2
        and health.get("context_capacity") == 262_144
        and health.get("static_prefill_tokens") == 262_143
        and health.get("prefix_cache_entries") == 0
        and health.get("media_cache_capacity_bytes") == 0
        and health.get("media_cache_capacity_entries") == 0
        and health.get("prompt_token_ids_extension") is True
    ):
        raise ValueError(f"{label} does not match the disabled-cache G4 service")


def manifest_controls(
    manifest: Mapping[str, Any], manifest_path: Path
) -> dict[str, Mapping[str, Any]]:
    protocol = manifest.get("protocol")
    if not (
        manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("complete") is True
        and not verify_manifest_integrity(manifest)
        and isinstance(protocol, Mapping)
        and protocol.get("timing_boundary") == TIMING_BOUNDARY
        and protocol.get("minimum_fresh_processes") == 5
        and protocol.get("pair_order")
        == (
            "text then VL for odd process indices; VL then text for even "
            "process indices"
        )
        and protocol.get("pairing")
        == (
            "each text/VL control pair is adjacent on one server PID with "
            "consecutive native request indices"
        )
    ):
        raise ValueError("text-control manifest is incomplete or corrupt")
    padding_contract = manifest.get("token_padding")
    if not (
        isinstance(padding_contract, Mapping)
        and padding_contract.get("frozen_single_token_id") == 830
        and padding_contract.get("unit_sha256")
        == "830a815db067f9501633539c2505e8c8a82ecc045b2adea256a55365ef516c4b"
        and padding_contract.get("base_prompt_tokens") == 50
        and padding_contract.get("chat_template_mode")
        == "enable_thinking=false"
    ):
        raise ValueError("text-control token-padding contract is invalid")
    controls = manifest.get("controls")
    if not isinstance(controls, list) or len(controls) != 3:
        raise ValueError("text-control manifest must contain three controls")
    indexed: dict[str, Mapping[str, Any]] = {}
    for control in controls:
        if not isinstance(control, Mapping):
            raise ValueError("text-control manifest contains a malformed control")
        control_id = control.get("control_id")
        request = control.get("request")
        padding_tokens = control.get("text_padding_tokens")
        if (
            not isinstance(control_id, str)
            or control_id in indexed
            or not isinstance(request, Mapping)
            or not isinstance(padding_tokens, int)
            or isinstance(padding_tokens, bool)
            or control.get("expected_prompt_tokens") != 50 + padding_tokens
        ):
            raise ValueError("text-control IDs, requests and padding must be exact")
        request_path = ROOT / str(request.get("path"))
        if (
            not request_path.is_file()
            or request_path.stat().st_size != request.get("bytes")
            or sha256_file(request_path) != request.get("sha256")
        ):
            raise ValueError(f"text-control request binding failed: {control_id}")
        indexed[control_id] = control
    orders = manifest.get("balanced_orders")
    if not (
        isinstance(orders, list)
        and len(orders) == 2
        and all(isinstance(order, list) for order in orders)
        and all(set(order) == set(indexed) for order in orders)
        and orders[1] == list(reversed(orders[0]))
    ):
        raise ValueError("text-control balanced orders are invalid")
    if not manifest_path.is_file():
        raise ValueError("text-control manifest path is missing")
    return indexed


def matrix_cells(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cells = matrix.get("cells")
    if not (
        matrix.get("schema") == MATRIX_SCHEMA
        and matrix.get("complete") is True
        and not verify_manifest_integrity(matrix)
        and isinstance(cells, list)
    ):
        raise ValueError("G4 comparable matrix is incomplete or corrupt")
    indexed: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("cell_id"), str):
            raise ValueError("G4 comparable matrix contains a malformed cell")
        cell_id = str(cell["cell_id"])
        if cell_id in indexed:
            raise ValueError("G4 comparable matrix cell IDs are not unique")
        indexed[cell_id] = cell
    return indexed


def validate_text_sample(
    sample: Mapping[str, Any], control: Mapping[str, Any], run_index: int
) -> dict[str, Any]:
    control_id = str(control["control_id"])
    match = TEXT_BENCHMARK_ID.fullmatch(str(sample.get("benchmark_id", "")))
    request = sample.get("request")
    response = sample.get("response")
    timings = sample.get("timings")
    native = sample.get("native_metrics")
    if not all(isinstance(value, Mapping) for value in (request, response, timings, native)):
        raise ValueError(f"text control {control_id} has incomplete sections")
    assert isinstance(request, Mapping)
    assert isinstance(response, Mapping)
    assert isinstance(timings, Mapping)
    assert isinstance(native, Mapping)
    usage = response.get("usage")
    summary = request.get("summary")
    padding = request.get("text_padding")
    vl = native.get("vl")
    mrope = native.get("mrope")
    prefix = native.get("prefix_cache")
    expected_prompt = control.get("expected_prompt_tokens")
    expected_completion = control.get("expected_completion_tokens")
    if not (
        sample.get("schema") == TEXT_SAMPLE_SCHEMA
        and sample.get("complete") is True
        and sample.get("engine_role") == "candidate"
        and sample.get("timing_boundary") == TIMING_BOUNDARY
        and match is not None
        and match.group(1) == control_id
        and int(match.group(2)) == run_index
        and request.get("path") == control.get("request", {}).get("path")
        and request.get("template_sha256") == control.get("request", {}).get("sha256")
        and isinstance(summary, Mapping)
        and summary.get("image_count") == 0
        and summary.get("video_count") == 0
        and request.get("media") == []
        and isinstance(padding, Mapping)
        and padding.get("tokens") == control.get("text_padding_tokens")
        and padding.get("frozen_single_token_id") == 830
        and isinstance(usage, Mapping)
        and usage.get("prompt_tokens") == expected_prompt
        and usage.get("completion_tokens") == expected_completion
        and usage.get("total_tokens") == expected_prompt + expected_completion
        and native.get("prompt_tokens") == expected_prompt
        and native.get("completion_tokens") == expected_completion
        and str(native.get("runtime", "")).startswith("native-resident")
        and native.get("model_loads") == 1
        and isinstance(prefix, Mapping)
        and prefix.get("lookup") == "disabled"
        and (vl is None or (isinstance(vl, Mapping) and vl.get("enabled") is False))
        and isinstance(mrope, Mapping)
        and mrope.get("enabled") is False
    ):
        raise ValueError(f"text control {control_id} did not stay on the text path")
    return {
        "client_decode_tokens_per_second": positive_float(
            timings.get("decode_tokens_per_second"),
            f"{control_id} client decode throughput",
        ),
        "engine_decode_tokens_per_second": positive_float(
            native.get("decode_tokens_per_second"),
            f"{control_id} engine decode throughput",
        ),
        "content_sha256": response.get("content_sha256"),
        "output_token_ids_sha256": native.get("output_token_ids_sha256"),
        "request_index": native.get("request_index"),
        "server_pid": sample.get("server_pid"),
        "host": sample.get("host", {}).get("hostname"),
    }


def validate_vl_sample(
    sample: Mapping[str, Any],
    control: Mapping[str, Any],
    cell: Mapping[str, Any],
    run_index: int,
) -> dict[str, Any]:
    control_id = str(control["control_id"])
    cell_id = str(control["g4_cell_id"])
    match = VL_BENCHMARK_ID.fullmatch(str(sample.get("benchmark_id", "")))
    request = sample.get("request")
    response = sample.get("response")
    timings = sample.get("timings")
    native = sample.get("native_metrics")
    if not all(isinstance(value, Mapping) for value in (request, response, timings, native)):
        raise ValueError(f"paired run {run_index} VL sample is incomplete")
    assert isinstance(request, Mapping)
    assert isinstance(response, Mapping)
    assert isinstance(timings, Mapping)
    assert isinstance(native, Mapping)
    usage = response.get("usage")
    media = request.get("media")
    vl = native.get("vl")
    mrope = native.get("mrope")
    prefix = native.get("prefix_cache")
    expected_prompt = control.get("expected_prompt_tokens")
    expected_completion = control.get("expected_completion_tokens")
    cell_request = cell.get("request")
    if not (
        sample.get("schema") == VL_SAMPLE_SCHEMA
        and sample.get("complete") is True
        and sample.get("engine_role") == "candidate"
        and sample.get("timing_boundary") == TIMING_BOUNDARY
        and match is not None
        and match.group(1) == cell_id
        and int(match.group(2)) == run_index
        and isinstance(cell_request, Mapping)
        and request.get("template_sha256") == cell_request.get("sha256")
        and isinstance(usage, Mapping)
        and usage.get("prompt_tokens") == expected_prompt
        and usage.get("completion_tokens") == expected_completion
        and usage.get("total_tokens") == expected_prompt + expected_completion
        and isinstance(media, list)
        and bool(media)
        and native.get("prompt_tokens") == expected_prompt
        and native.get("completion_tokens") == expected_completion
        and str(native.get("runtime", "")).startswith("native-resident")
        and native.get("model_loads") == 1
        and isinstance(prefix, Mapping)
        and prefix.get("lookup") == "disabled"
        and isinstance(vl, Mapping)
        and vl.get("enabled") is True
        and isinstance(mrope, Mapping)
        and mrope.get("enabled") is True
    ):
        raise ValueError(f"paired run {run_index} {control_id} VL path drifted")
    return {
        "client_decode_tokens_per_second": positive_float(
            timings.get("decode_tokens_per_second"),
            f"paired run {run_index} VL client decode throughput",
        ),
        "engine_decode_tokens_per_second": positive_float(
            native.get("decode_tokens_per_second"),
            f"paired run {run_index} VL engine decode throughput",
        ),
        "content_sha256": response.get("content_sha256"),
        "output_token_ids_sha256": native.get("output_token_ids_sha256"),
        "request_index": native.get("request_index"),
        "server_pid": sample.get("server_pid"),
        "host": sample.get("host", {}).get("hostname"),
    }


def paired_control_record(
    directory: Path,
    manifest: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]],
    manifest_sha256: str,
    matrix: Mapping[str, Any],
    cells: Mapping[str, Mapping[str, Any]],
    matrix_sha256: str,
) -> dict[str, Any]:
    run_index = read_index(directory / "run-index.txt", "paired-control run index")
    if sha256_file(directory / "manifest.json") != manifest_sha256:
        raise ValueError(f"paired-control run {run_index} manifest drifted")
    if sha256_file(directory / "matrix.json") != matrix_sha256:
        raise ValueError(f"paired-control run {run_index} matrix drifted")
    if matrix.get("schema") != MATRIX_SCHEMA:
        raise ValueError("paired-control matrix identity drifted")
    expected_order = manifest["balanced_orders"][0 if run_index % 2 else 1]
    actual_order = (directory / "request-order.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if actual_order != expected_order:
        raise ValueError(f"paired-control run {run_index} control order drifted")
    expected_pair_order = ["text", "vl"] if run_index % 2 else ["vl", "text"]
    pair_order = (directory / "pair-order.txt").read_text(
        encoding="ascii"
    ).split()
    if pair_order != expected_pair_order:
        raise ValueError(f"paired-control run {run_index} pair order drifted")
    health_path = directory / "health.json"
    health = load_object(health_path)
    validate_health(health, f"paired-control run {run_index} health")
    samples: dict[str, dict[str, Any]] = {}
    pids: set[int] = set()
    hosts: set[str] = set()
    observed_indices: list[int] = []
    expected_index = 2
    for control_id in actual_order:
        control = controls[control_id]
        cell_id = str(control["g4_cell_id"])
        if cell_id not in cells:
            raise ValueError(f"paired-control G4 cell is missing: {cell_id}")
        text_path = directory / "text/requests" / f"{control_id}.json"
        vl_path = directory / "vl/requests" / f"{control_id}.json"
        text_sample = {
            **validate_text_sample(load_object(text_path), control, run_index),
            "raw_sha256": sha256_file(text_path),
        }
        vl_sample = {
            **validate_vl_sample(
                load_object(vl_path), control, cells[cell_id], run_index
            ),
            "raw_sha256": sha256_file(vl_path),
        }
        role_samples = {"text": text_sample, "vl": vl_sample}
        for role in pair_order:
            sample = role_samples[role]
            if sample["request_index"] != expected_index:
                raise ValueError(
                    f"paired-control run {run_index} requests are not adjacent"
                )
            observed_indices.append(expected_index)
            expected_index += 1
        for sample in role_samples.values():
            if isinstance(sample["server_pid"], int):
                pids.add(sample["server_pid"])
            if isinstance(sample["host"], str):
                hosts.add(sample["host"])
        samples[control_id] = role_samples
    if len(pids) != 1 or len(hosts) != 1 or observed_indices != list(range(2, 8)):
        raise ValueError(
            f"paired-control run {run_index} was not one fresh warmed process"
        )
    binary_sha = (directory / "candidate-binary.sha256").read_text(
        encoding="ascii"
    ).strip()
    if SHA256.fullmatch(binary_sha) is None:
        raise ValueError(f"paired-control run {run_index} binary hash is invalid")
    return {
        "run_index": run_index,
        "directory": directory.name,
        "manifest_sha256": manifest_sha256,
        "matrix_sha256": matrix_sha256,
        "health_sha256": sha256_file(health_path),
        "health_contract": health_contract(health),
        "candidate_binary_sha256": binary_sha,
        "server_pid": next(iter(pids)),
        "host": next(iter(hosts)),
        "request_order": actual_order,
        "pair_order": pair_order,
        "samples": samples,
    }


def aggregate(
    paired_runs: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Mapping[str, Any]],
    candidate_identity: Mapping[str, Any],
    minimum_pairs: int = 5,
) -> dict[str, Any]:
    ordered = sorted(paired_runs, key=lambda item: int(item["run_index"]))
    indices = [int(item["run_index"]) for item in ordered]
    candidate_files = candidate_identity.get("files")
    engine_files = (
        [
            item
            for item in candidate_files
            if isinstance(item, Mapping) and item.get("path") == "aima-engine-native"
        ]
        if isinstance(candidate_files, list)
        else []
    )
    engine_sha = engine_files[0].get("sha256") if len(engine_files) == 1 else None
    hosts = {item.get("host") for item in ordered}
    health_contracts = [item.get("health_contract") for item in ordered]
    checks = {
        "minimum_five_fresh_paired_processes": len(ordered) >= minimum_pairs,
        "paired_run_indices_consecutive": indices == list(range(1, len(ordered) + 1)),
        "candidate_binary_exact": (
            isinstance(engine_sha, str)
            and all(item.get("candidate_binary_sha256") == engine_sha for item in ordered)
        ),
        "one_host_exact": len(hosts) == 1 and None not in hosts,
        "same_native_service_contract": (
            bool(health_contracts)
            and all(value == health_contracts[0] for value in health_contracts)
        ),
        "fresh_process_directories_distinct": (
            len({item.get("directory") for item in ordered}) == len(ordered)
        ),
        "control_order_alternates": all(
            item.get("request_order")
            == (list(controls) if index % 2 else list(reversed(controls)))
            for index, item in enumerate(ordered, start=1)
        ),
        "text_vl_pair_order_alternates": all(
            item.get("pair_order") == (["text", "vl"] if index % 2 else ["vl", "text"])
            for index, item in enumerate(ordered, start=1)
        ),
        "pair_orders_balanced": (
            abs(
                sum(item.get("pair_order") == ["text", "vl"] for item in ordered)
                - sum(item.get("pair_order") == ["vl", "text"] for item in ordered)
            )
            <= 1
        ),
    }
    cells: list[dict[str, Any]] = []
    for control_id, control in controls.items():
        pairs: list[dict[str, Any]] = []
        for run in ordered:
            text_sample = run["samples"][control_id]["text"]
            vl_sample = run["samples"][control_id]["vl"]
            client_ratio = (
                vl_sample["client_decode_tokens_per_second"]
                / text_sample["client_decode_tokens_per_second"]
            )
            engine_ratio = (
                vl_sample["engine_decode_tokens_per_second"]
                / text_sample["engine_decode_tokens_per_second"]
            )
            pairs.append(
                {
                    "pair_index": run["run_index"],
                    "pair_order": " ".join(run["pair_order"]),
                    "server_pid": run["server_pid"],
                    "text_request_index": text_sample["request_index"],
                    "vl_request_index": vl_sample["request_index"],
                    "text_client_decode_tokens_per_second": text_sample[
                        "client_decode_tokens_per_second"
                    ],
                    "vl_client_decode_tokens_per_second": vl_sample[
                        "client_decode_tokens_per_second"
                    ],
                    "vl_over_text_client_decode": client_ratio,
                    "text_engine_decode_tokens_per_second": text_sample[
                        "engine_decode_tokens_per_second"
                    ],
                    "vl_engine_decode_tokens_per_second": vl_sample[
                        "engine_decode_tokens_per_second"
                    ],
                    "vl_over_text_engine_decode_diagnostic": engine_ratio,
                    "text_content_sha256": text_sample["content_sha256"],
                    "vl_content_sha256": vl_sample["content_sha256"],
                    "text_output_token_ids_sha256": text_sample[
                        "output_token_ids_sha256"
                    ],
                    "vl_output_token_ids_sha256": vl_sample[
                        "output_token_ids_sha256"
                    ],
                    "raw": {
                        "text_sha256": text_sample["raw_sha256"],
                        "vl_sha256": vl_sample["raw_sha256"],
                    },
                }
            )
        client_median = float(
            statistics.median(pair["vl_over_text_client_decode"] for pair in pairs)
        )
        engine_median = float(
            statistics.median(
                pair["vl_over_text_engine_decode_diagnostic"] for pair in pairs
            )
        )
        cells.append(
            {
                "control_id": control_id,
                "g4_cell_id": control.get("g4_cell_id"),
                "prompt_tokens": control.get("expected_prompt_tokens"),
                "completion_tokens": control.get("expected_completion_tokens"),
                "pair_count": len(pairs),
                "pairs": pairs,
                "paired_median_vl_over_text_client_decode": client_median,
                "paired_median_vl_over_text_engine_decode_diagnostic": engine_median,
                "qualified": len(pairs) >= minimum_pairs and client_median >= 1.0,
            }
        )
    complete = all(checks.values()) and len(cells) == len(controls)
    qualified = complete and all(cell["qualified"] for cell in cells)
    return {
        "schema": SCHEMA,
        "complete": complete,
        "qualified": qualified,
        "scope": (
            "three G4 VL decode cells against exact-shape text-only controls "
            "captured as adjacent requests in the same fresh native process"
        ),
        "protocol": {
            "minimum_pairs": minimum_pairs,
            "pairing": "adjacent text/VL requests on one fresh server PID",
            "control_order": "forward for odd runs; reverse for even runs",
            "pair_order": "text,VL for odd runs; VL,text for even runs",
            "timing_boundary": TIMING_BOUNDARY,
            "blocking_gate": "median paired VL/text client decode ratio >= 1.000",
            "engine_decode_ratio": "diagnostic only",
            "aggregate_cannot_mask_cell_failure": True,
        },
        "pair_count": len(ordered),
        "checks": checks,
        "gates": {"every_vl_decode_cell_gte_same_boundary_text_control": qualified},
        "host": {"hostname": next(iter(hosts)) if len(hosts) == 1 else None},
        "artifact_identity": {"candidate": dict(candidate_identity)},
        "cells": cells,
        "raw_runs": [
            {
                "run_index": run["run_index"],
                "paired_control_dir": run["directory"],
                "manifest_sha256": run["manifest_sha256"],
                "matrix_sha256": run["matrix_sha256"],
                "health_sha256": run["health_sha256"],
                "candidate_binary_sha256": run["candidate_binary_sha256"],
                "pair_order": " ".join(run["pair_order"]),
            }
            for run in ordered
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-control-dir", action="append", type=Path, required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--candidate-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_name(args.output.name + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(
            f"refusing to overwrite text-control result or sidecar: {args.output}"
        )
    manifest_path = args.manifest.resolve()
    manifest = load_object(manifest_path)
    controls = manifest_controls(manifest, manifest_path)
    manifest_sha = sha256_file(manifest_path)
    matrix_path = MATRIX_PATH.resolve()
    matrix = load_object(matrix_path)
    cells = matrix_cells(matrix)
    matrix_sha = sha256_file(matrix_path)
    runs = [
        paired_control_record(
            path.resolve(),
            manifest,
            controls,
            manifest_sha,
            matrix,
            cells,
            matrix_sha,
        )
        for path in args.paired_control_dir
    ]
    helper = paired_identity_module()
    candidate = helper.candidate_identity(
        args.candidate_binary.resolve(), args.candidate_source_commit
    )
    result = aggregate(runs, controls, candidate)
    result["bindings"] = {
        "manifest": file_component(
            manifest_path,
            "benchmarks/fixtures/vl-text-decode-control-v0.1.0/manifest.json",
        ),
        "matrix": file_component(
            matrix_path,
            "benchmarks/fixtures/vl-performance-v0.1.0/comparable-matrix.json",
        ),
        "vl_request_capture": file_component(
            ROOT / "scripts/capture-vl-performance-request.py",
            "scripts/capture-vl-performance-request.py",
        ),
        "text_request_capture": file_component(
            ROOT / "scripts/capture-vl-text-decode-control.py",
            "scripts/capture-vl-text-decode-control.py",
        ),
        "candidate_launcher": file_component(
            ROOT / "scripts/run-native-vl-performance-candidate.sh",
            "scripts/run-native-vl-performance-candidate.sh",
        ),
        "paired_control_runner": file_component(
            ROOT / "scripts/run-vl-decode-control.sh",
            "scripts/run-vl-decode-control.sh",
        ),
        "summarizer": file_component(
            Path(__file__).resolve(),
            "scripts/summarize-vl-text-decode-controls.py",
        ),
    }
    result["command_template"] = (
        "python3 scripts/summarize-vl-text-decode-controls.py "
        "--paired-control-dir <pair-1> ... --paired-control-dir <pair-5> "
        "--manifest benchmarks/fixtures/vl-text-decode-control-v0.1.0/manifest.json "
        "--candidate-binary <exact-native-binary> "
        "--candidate-source-commit <embedded-source-commit> --output <result>"
    )
    result = seal_manifest(result)
    atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "complete": result["complete"],
                "qualified": result["qualified"],
                "cells": len(result["cells"]),
                "pairs": result["pair_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
