#!/usr/bin/env python3
"""Pair the standalone BF16 decode projection with the installed vLLM op.

PyTorch/vLLM are validation oracles only. The native executable does not load
or link either dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


# Keep the two previously qualified decode consumers and cover every unique
# [M, K] shape used by current vLLM's split linear-attention projections.
SHAPES = (
    (2048, 512),
    (2048, 4096),
    (8192, 2048),
    (4096, 2048),
    (32, 2048),
)
LAUNCHES_PER_SAMPLE = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-binary",
        type=Path,
        default=Path("build/native/aima-engine-native"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def native_probe(binary: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary.resolve()), "bf16-wvsplitk-probe"],
        check=False,
        capture_output=True,
        text=True,
    )
    require(completed.returncode == 0, completed.stderr or completed.stdout)
    value = json.loads(completed.stdout)
    require(isinstance(value, dict) and value.get("complete") is True,
            "native probe did not return a complete object")
    return value


def pattern(torch: Any, count: int, multiplier: int, increment: int,
            modulus: int, center: int, denominator: float) -> Any:
    indices = torch.arange(count, dtype=torch.int64, device="cuda")
    return (
        ((indices * multiplier + increment) % modulus - center)
        .to(torch.float32)
        .div_(denominator)
        .to(torch.bfloat16)
    )


def reference_cases() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from vllm import _custom_ops as ops

    properties = torch.cuda.get_device_properties(0)
    cu_count = int(properties.multi_processor_count)
    environment = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "vllm_rocm_op": str(torch.ops._rocm_C.wvSplitK),
        "device": properties.name,
        "cu_count": cu_count,
    }
    cases: list[dict[str, Any]] = []
    for m, k in SHAPES:
        weight = pattern(torch, m * k, 13, 7, 17, 8, 64.0).view(m, k)
        activation = pattern(torch, k, 11, 3, 19, 9, 32.0).view(1, k)
        output = None
        for _ in range(10):
            output = ops.wvSplitK(weight, activation, cu_count, None)
        torch.cuda.synchronize()
        measurements: list[float] = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(LAUNCHES_PER_SAMPLE):
                output = ops.wvSplitK(weight, activation, cu_count, None)
            stop.record()
            stop.synchronize()
            measurements.append(
                float(start.elapsed_time(stop)) / LAUNCHES_PER_SAMPLE
            )
        require(output is not None, "vLLM produced no output")
        payload = output.view(torch.int16).cpu().numpy().tobytes()
        cases.append(
            {
                "m": m,
                "n": 1,
                "k": k,
                "measured_ms": measurements,
                "median_ms": statistics.median(measurements),
                "launches_per_sample": LAUNCHES_PER_SAMPLE,
                "output_bf16_sha256": hashlib.sha256(payload).hexdigest(),
                "finite_elements": int(torch.isfinite(output).sum().item()),
                "expected_elements": m,
            }
        )
        del weight, activation, output
        torch.cuda.empty_cache()
    return environment, cases


def main() -> int:
    args = parse_args()
    native = native_probe(args.native_binary)
    environment, reference = reference_cases()
    native_by_shape = {(case["m"], case["k"]): case for case in native["cases"]}
    comparisons: list[dict[str, Any]] = []
    for case in reference:
        key = (case["m"], case["k"])
        native_case = native_by_shape[key]
        comparisons.append(
            {
                "m": key[0],
                "n": 1,
                "k": key[1],
                "output_bf16_sha256_equal": (
                    native_case["output_bf16_sha256"]
                    == case["output_bf16_sha256"]
                ),
                "native_median_ms": native_case["median_ms"],
                "vllm_median_ms": case["median_ms"],
                "native_to_vllm_time_ratio": (
                    native_case["median_ms"] / case["median_ms"]
                ),
            }
        )
    result = {
        "schema": "aima-amd395-qwen36/native-wvsplitk-paired-parity/v2",
        "complete": True,
        "scope": "decode_projection_provider_boundary_only",
        "native": native,
        "validation_environment": environment,
        "vllm_reference": reference,
        "comparison": comparisons,
        "decision": {
            "all_output_bf16_sha256_equal": all(
                item["output_bf16_sha256_equal"] for item in comparisons
            ),
            "all_native_time_ratios_lte_1_03": all(
                item["native_to_vllm_time_ratio"] <= 1.03
                for item in comparisons
            ),
        },
        "claims": {
            "full_native_inference_qualified": False,
            "full_context_matrix_qualified": False,
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if all(result["decision"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
