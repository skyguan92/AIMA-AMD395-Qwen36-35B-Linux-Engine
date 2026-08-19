#!/usr/bin/env python3
"""Qualify one G4 VL performance cell from fresh alternating pairs."""

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
import subprocess
from typing import Any, Mapping, Sequence


SCHEMA = "aima-amd395-qwen36/vl-performance-paired-cell/v1"
PAIR_ID = re.compile(r"^(.+)\.pair-([1-9][0-9]*)$")
STARTUP_LIMIT_MS = 44_900.0
THROUGHPUT_RATIOS = (
    "prefill_tps_candidate_over_reference",
    "vision_tps_candidate_over_reference",
    "decode_tps_candidate_over_reference",
)
LATENCY_RATIOS = (
    "ttft_candidate_over_reference",
    "total_candidate_over_reference",
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def diagnostic_module():
    path = Path(__file__).with_name("summarize-vl-performance-diagnostic.py")
    spec = importlib.util.spec_from_file_location("vl_pair_diagnostic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load VL diagnostic summarizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_pair(pair_dir: Path, diagnostic: Any) -> dict[str, Any]:
    summary = diagnostic.build_summary(pair_dir)
    reference = load_object(pair_dir / "reference/request.json")
    candidate = load_object(pair_dir / "candidate/request.json")
    health = load_object(pair_dir / "candidate/health.json")
    benchmark_id = reference.get("benchmark_id")
    if benchmark_id != candidate.get("benchmark_id") or not isinstance(
        benchmark_id, str
    ):
        raise ValueError(f"pair benchmark IDs differ: {pair_dir}")
    match = PAIR_ID.fullmatch(benchmark_id)
    if match is None:
        raise ValueError(f"pair benchmark ID is malformed: {benchmark_id}")
    pair_index = int(match.group(2))
    order = (pair_dir / "execution-order.txt").read_text(
        encoding="utf-8"
    ).strip()
    comparisons = summary.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError(f"pair comparisons are missing: {pair_dir}")
    ratios = {
        name: finite_float(comparisons.get(name), f"pair {pair_index} {name}")
        for name in comparisons
    }
    native = candidate.get("native_metrics")
    native_vl = native.get("vl") if isinstance(native, Mapping) else None
    if not isinstance(native_vl, Mapping):
        raise ValueError(f"candidate VL metrics are missing: {pair_dir}")
    return {
        "pair_index": pair_index,
        "benchmark_id": benchmark_id,
        "benchmark_base": match.group(1),
        "execution_order": order,
        "complete": summary.get("complete") is True,
        "pair_qualified": summary.get("qualified") is True,
        "all_pair_checks": all(summary.get("checks", {}).values()),
        "template_sha256": reference.get("request", {}).get(
            "template_sha256"
        ),
        "request_summary": reference.get("request", {}).get("summary"),
        "media": reference.get("request", {}).get("media"),
        "response_contract": {
            "content_sha256": reference.get("response", {}).get(
                "content_sha256"
            ),
            "finish_reason": reference.get("response", {}).get(
                "finish_reason"
            ),
            "usage": reference.get("response", {}).get("usage"),
        },
        "hostname": reference.get("host", {}).get("hostname"),
        "candidate_runtime": native.get("runtime"),
        "candidate_startup_ms": finite_float(
            health.get("command_to_ready_wall_ms"),
            f"pair {pair_index} candidate startup",
        ),
        "candidate_vision_warmup": {
            "completed": health.get("vision_warmup_completed"),
            "patches": health.get("vision_warmup_patches"),
            "visual_tokens": health.get("vision_warmup_visual_tokens"),
            "plan_cache_entries_at_ready": health.get(
                "vision_plan_cache_entries_at_ready"
            ),
            "plan_build_wall_ms": health.get(
                "vision_warmup_plan_build_wall_ms"
            ),
            "encode_wall_ms": health.get("vision_warmup_encode_wall_ms"),
        },
        "candidate_request_plan_cache_hit": native_vl.get(
            "vision_plan_cache_hit"
        ),
        "candidate_request_plan_build_wall_ms": native_vl.get(
            "vision_plan_build_wall_ms"
        ),
        "ratios": ratios,
        "measurements": summary.get("measurements"),
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    artifact_identity: Mapping[str, Any],
    minimum_pairs: int = 5,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: int(item["pair_index"]))
    ratio_names = tuple(ordered[0]["ratios"]) if ordered else ()
    medians = {
        name: float(
            statistics.median(float(item["ratios"][name]) for item in ordered)
        )
        for name in ratio_names
    }
    startup_values = [float(item["candidate_startup_ms"]) for item in ordered]
    startup_median = (
        float(statistics.median(startup_values)) if startup_values else None
    )
    expected_indices = list(range(1, len(ordered) + 1))
    invariants = (
        "benchmark_base",
        "template_sha256",
        "request_summary",
        "media",
        "response_contract",
        "hostname",
    )
    checks = {
        "minimum_five_pairs": len(ordered) >= minimum_pairs,
        "pair_indices_consecutive": [
            int(item["pair_index"]) for item in ordered
        ]
        == expected_indices,
        "execution_order_alternates": all(
            item["execution_order"]
            == (
                "reference candidate"
                if int(item["pair_index"]) % 2 == 1
                else "candidate reference"
            )
            for item in ordered
        ),
        "all_pairs_complete": all(item["complete"] for item in ordered),
        "all_pair_checks_pass": all(
            item["all_pair_checks"] for item in ordered
        ),
        "cross_pair_contract_exact": all(
            all(item.get(name) == ordered[0].get(name) for item in ordered)
            for name in invariants
        )
        if ordered
        else False,
        "candidate_runtime_native": all(
            str(item["candidate_runtime"]).startswith("native-resident-q")
            for item in ordered
        ),
        "vision_warmup_disclosed": all(
            item["candidate_vision_warmup"].get("completed") is True
            and item["candidate_vision_warmup"].get("patches") == 1024
            and item["candidate_vision_warmup"].get("visual_tokens") == 256
            and item["candidate_vision_warmup"].get(
                "plan_cache_entries_at_ready"
            )
            == 1
            for item in ordered
        ),
        "candidate_request_plan_prepared": all(
            item["candidate_request_plan_cache_hit"] is True
            and finite_float(
                item["candidate_request_plan_build_wall_ms"],
                "candidate request plan build",
            )
            == 0.0
            for item in ordered
        ),
        "artifact_identity_complete": bool(artifact_identity),
    }
    gates = {
        "ttft_paired_median_lte_reference": (
            medians.get("ttft_candidate_over_reference", math.inf) <= 1.0
        ),
        "total_paired_median_lte_reference": (
            medians.get("total_candidate_over_reference", math.inf) <= 1.0
        ),
        "prefill_paired_median_gte_reference": (
            medians.get("prefill_tps_candidate_over_reference", -math.inf)
            >= 1.0
        ),
        "vision_paired_median_gte_reference": (
            medians.get("vision_tps_candidate_over_reference", -math.inf)
            >= 1.0
        ),
        "candidate_startup_median_lte_44_9_seconds": (
            startup_median is not None
            and startup_median <= STARTUP_LIMIT_MS
        ),
    }
    if "decode_tps_candidate_over_reference" in medians:
        gates["decode_paired_median_gte_reference"] = (
            medians["decode_tps_candidate_over_reference"] >= 1.0
        )
    complete = all(checks.values())
    return {
        "schema": SCHEMA,
        "complete": complete,
        "qualified": complete and all(gates.values()),
        "scope": "one G4 performance cell; not the complete G4 matrix",
        "pair_count": len(ordered),
        "checks": checks,
        "gates": gates,
        "paired_medians": medians,
        "candidate_startup": {
            "measurements_ms": startup_values,
            "median_ms": startup_median,
            "limit_ms": STARTUP_LIMIT_MS,
        },
        "artifact_identity": dict(artifact_identity),
        "pairs": list(ordered),
    }


def candidate_identity(binary: Path, source_commit: str) -> dict[str, Any]:
    binary = binary.resolve()
    if not binary.is_file() or not source_commit:
        raise ValueError("candidate binary and source commit are required")
    if source_commit.encode("ascii") not in binary.read_bytes():
        raise ValueError("candidate source commit is not embedded in the binary")
    root = binary.parent
    required = (
        binary.name,
        "libaima-fmha-aotriton.so",
        "libaima-fmha-ck.so",
        "libaima-fmha-q16384-hybrid.so",
        "libaotriton_v2.so.0.11.1",
        "aima-vision-attention.hsaco",
    )
    files = []
    for name in required:
        path = root / name
        if not path.is_file():
            raise ValueError(f"candidate closure file is missing: {path}")
        files.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    images = sorted((root / "aotriton.images").rglob("*.aks2"))
    if len(images) != 1:
        raise ValueError("candidate closure must contain exactly one AOTriton image")
    image = images[0]
    files.append(
        {
            "path": image.relative_to(root).as_posix(),
            "bytes": image.stat().st_size,
            "sha256": file_sha256(image),
        }
    )
    return {"source_commit": source_commit, "files": files}


def reference_identity(python: Path) -> dict[str, Any]:
    launcher = python.absolute()
    resolved = launcher.resolve()
    if not launcher.is_file() or not resolved.is_file():
        raise ValueError("reference Python is missing")
    command = (
        "import importlib.metadata as m,json; d=m.distribution('vllm'); "
        "print(json.dumps({'version':d.version,'metadata':str(d._path/'METADATA')}))"
    )
    result = subprocess.run(
        [str(launcher), "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    metadata_path = Path(metadata["metadata"])
    return {
        "python_path": str(launcher),
        "python_resolved_path": str(resolved),
        "python_sha256": file_sha256(resolved),
        "vllm_version": metadata["version"],
        "vllm_metadata_sha256": file_sha256(metadata_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", action="append", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--candidate-source-commit", required=True)
    parser.add_argument("--reference-python", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite paired summary: {args.output}")
    model_index = args.model_dir.resolve() / "model.safetensors.index.json"
    if not model_index.is_file():
        raise SystemExit("model checkpoint index is missing")
    identity = {
        "candidate": candidate_identity(
            args.candidate_binary, args.candidate_source_commit
        ),
        "reference": reference_identity(args.reference_python),
        "model": {
            "checkpoint_index_path": str(model_index),
            "checkpoint_index_sha256": file_sha256(model_index),
        },
    }
    diagnostic = diagnostic_module()
    records = [collect_pair(path.resolve(), diagnostic) for path in args.pair_dir]
    result = aggregate_records(records, identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "complete": result["complete"],
                "qualified": result["qualified"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
