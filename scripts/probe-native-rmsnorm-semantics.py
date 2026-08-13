#!/usr/bin/env python3
"""Compare candidate RMSNorm lowerings against captured BF16 oracles."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open


CASES = (
    "image_local_png",
    "video_local_mp4",
    "multi_image",
    "multi_video",
    "mixed_image_video",
)
HIDDEN_SIZE = 2048
WEIGHT_NAME = "model.language_model.layers.0.input_layernorm.weight"


def load_weight(model_root: Path) -> torch.Tensor:
    with (model_root / "model.safetensors.index.json").open(
        encoding="utf-8"
    ) as stream:
        index = json.load(stream)
    shard = model_root / index["weight_map"][WEIGHT_NAME]
    with safe_open(shard, framework="pt", device="cpu") as stream:
        return stream.get_tensor(WEIGHT_NAME).to("cuda")


def load_bf16(path: Path, rows: int) -> torch.Tensor:
    raw = bytearray(path.read_bytes())
    return (
        torch.frombuffer(raw, dtype=torch.bfloat16)
        .clone()
        .reshape(rows, HIDDEN_SIZE)
        .to("cuda")
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    original_dtype = x.dtype
    value = x.to(torch.float32)
    scale = weight.to(torch.float32) + 1.0
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    value = value * torch.rsqrt(variance + 1e-6)
    value = value.to(scale.dtype) * scale
    return value.to(original_dtype)


def compare(
    label: str, actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, object]:
    torch.cuda.synchronize()
    exact = actual == expected
    exact_count = int(exact.sum().item())
    elements = expected.numel()
    delta = actual.float() - expected.float()
    relative_l2 = float(
        torch.linalg.vector_norm(delta)
        / torch.linalg.vector_norm(expected.float())
    )
    first_mismatch = None
    if exact_count != elements:
        index = int((~exact).flatten().nonzero()[0].item())
        first_mismatch = {
            "index": index,
            "expected": float(expected.flatten()[index]),
            "actual": float(actual.flatten()[index]),
        }
    return {
        "mode": label,
        "exact_elements": exact_count,
        "elements": elements,
        "relative_l2_error": relative_l2,
        "first_mismatch": first_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--diagnostic-oracle-root", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument(
        "--mode",
        action="append",
        choices=("eager", "inductor-static", "inductor-dynamic"),
    )
    args = parser.parse_args()

    weight = load_weight(args.model_root.resolve())
    candidate_map: dict[str, Callable[..., torch.Tensor]] = {
        "eager": rms_norm,
        "inductor-static": torch.compile(
            rms_norm, backend="inductor", fullgraph=True, dynamic=False
        ),
        "inductor-dynamic": torch.compile(
            rms_norm, backend="inductor", fullgraph=True, dynamic=True
        ),
    }
    selected_modes = args.mode or list(candidate_map)
    candidates = tuple((name, candidate_map[name]) for name in selected_modes)
    for case in args.case or CASES:
        input_path = (
            args.worktree
            / "benchmarks/oracles/vl-v0.1.0"
            / case
            / "boundaries/injected_embeddings.bin"
        )
        expected_path = (
            args.diagnostic_oracle_root / case / "components/input_norm.bin"
        )
        rows = input_path.stat().st_size // (HIDDEN_SIZE * 2)
        value = load_bf16(input_path, rows)
        expected = load_bf16(expected_path, rows)
        results = [
            compare(label, function(value, weight), expected)
            for label, function in candidates
        ]
        print(
            json.dumps(
                {"case": case, "rows": rows, "results": results},
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
