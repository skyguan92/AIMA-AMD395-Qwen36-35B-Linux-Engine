#!/usr/bin/env python3
"""Compare candidate layer-0 GDN projection packings with frozen oracles."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


CASES = (
    "image_local_png",
    "video_local_mp4",
    "multi_image",
    "multi_video",
    "mixed_image_video",
)
HIDDEN_SIZE = 2048
PREFIX = "model.language_model.layers.0.linear_attn."
WEIGHT_NAMES = {
    "qkv": PREFIX + "in_proj_qkv.weight",
    "z": PREFIX + "in_proj_z.weight",
    "a": PREFIX + "in_proj_a.weight",
    "b": PREFIX + "in_proj_b.weight",
}


def load_weights(model_root: Path) -> dict[str, torch.Tensor]:
    with (model_root / "model.safetensors.index.json").open(
        encoding="utf-8"
    ) as stream:
        weight_map = json.load(stream)["weight_map"]
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for label, name in WEIGHT_NAMES.items():
        by_shard.setdefault(weight_map[name], []).append((label, name))
    weights: dict[str, torch.Tensor] = {}
    for shard_name, entries in by_shard.items():
        with safe_open(
            model_root / shard_name, framework="pt", device="cpu"
        ) as stream:
            for label, name in entries:
                weights[label] = stream.get_tensor(name).to("cuda")
    return weights


def load_bf16(path: Path, columns: int) -> torch.Tensor:
    raw = bytearray(path.read_bytes())
    return (
        torch.frombuffer(raw, dtype=torch.bfloat16)
        .clone()
        .reshape(-1, columns)
        .to("cuda")
    )


def compare(
    mode: str, label: str, actual: torch.Tensor, expected: torch.Tensor
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
        "mode": mode,
        "label": label,
        "exact_elements": exact_count,
        "elements": elements,
        "relative_l2_error": relative_l2,
        "first_mismatch": first_mismatch,
    }


def exact_dot_analysis(
    value: torch.Tensor,
    weight: torch.Tensor,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, object] | None:
    mismatch = actual != expected
    if not bool(mismatch.any().item()):
        return None
    index = int(mismatch.flatten().nonzero()[0].item())
    row, column = divmod(index, expected.shape[1])
    dot = (value[row].double() * weight[column].double()).sum()
    return {
        "index": index,
        "row": row,
        "column": column,
        "float64_dot": float(dot.item()),
        "float64_dot_rounded_bf16": float(dot.bfloat16().float().item()),
        "expected": float(expected.flatten()[index]),
        "actual": float(actual.flatten()[index]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--diagnostic-oracle-root", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()

    weights = load_weights(args.model_root.resolve())
    official_qkvz_weight = torch.cat((weights["qkv"], weights["z"]), dim=0)
    official_ba_weight = torch.cat((weights["b"], weights["a"]), dim=0)
    native_fused_weight = torch.cat(
        (weights["qkv"], weights["z"], weights["a"], weights["b"]),
        dim=0,
    )

    for case in args.case or CASES:
        root = args.diagnostic_oracle_root.resolve() / case / "components"
        value = load_bf16(root / "input_norm.bin", HIDDEN_SIZE)
        expected = {
            "qkvz": load_bf16(root / "gdn_qkvz_projection.bin", 12288),
            "a": load_bf16(root / "gdn_a_projection.bin", 32),
            "b": load_bf16(root / "gdn_b_projection.bin", 32),
        }

        official_qkvz = F.linear(value, official_qkvz_weight)
        official_ba = F.linear(value, official_ba_weight)
        official_b, official_a = official_ba.chunk(2, dim=-1)
        native_fused = F.linear(value, native_fused_weight)
        native_qkvz = native_fused[:, :12288]
        native_a = native_fused[:, 12288:12320]
        native_b = native_fused[:, 12320:]
        individual_a = F.linear(value, weights["a"])
        individual_b = F.linear(value, weights["b"])

        padded = torch.zeros(
            (1024, HIDDEN_SIZE), dtype=value.dtype, device=value.device
        )
        padded[: value.shape[0]].copy_(value)
        padded_official_qkvz = F.linear(padded, official_qkvz_weight)[
            : value.shape[0]
        ]
        padded_official_ba = F.linear(padded, official_ba_weight)[
            : value.shape[0]
        ]
        padded_official_b, padded_official_a = padded_official_ba.chunk(
            2, dim=-1
        )
        padded_native_fused = F.linear(padded, native_fused_weight)[
            : value.shape[0]
        ]
        padded_native_qkvz = padded_native_fused[:, :12288]
        padded_native_a = padded_native_fused[:, 12288:12320]
        padded_native_b = padded_native_fused[:, 12320:]
        padded_individual_a = F.linear(padded, weights["a"])[
            : value.shape[0]
        ]
        padded_individual_b = F.linear(padded, weights["b"])[
            : value.shape[0]
        ]

        results = [
            compare("official-merged", "qkvz", official_qkvz, expected["qkvz"]),
            compare("official-merged", "a", official_a, expected["a"]),
            compare("official-merged", "b", official_b, expected["b"]),
            compare("native-fused", "qkvz", native_qkvz, expected["qkvz"]),
            compare("native-fused", "a", native_a, expected["a"]),
            compare("native-fused", "b", native_b, expected["b"]),
            compare("individual", "a", individual_a, expected["a"]),
            compare("individual", "b", individual_b, expected["b"]),
            compare(
                "official-merged-padded",
                "qkvz",
                padded_official_qkvz,
                expected["qkvz"],
            ),
            compare(
                "official-merged-padded",
                "a",
                padded_official_a,
                expected["a"],
            ),
            compare(
                "official-merged-padded",
                "b",
                padded_official_b,
                expected["b"],
            ),
            compare(
                "native-fused-padded",
                "qkvz",
                padded_native_qkvz,
                expected["qkvz"],
            ),
            compare(
                "native-fused-padded", "a", padded_native_a, expected["a"]
            ),
            compare(
                "native-fused-padded", "b", padded_native_b, expected["b"]
            ),
            compare(
                "individual-padded", "a", padded_individual_a, expected["a"]
            ),
            compare(
                "individual-padded", "b", padded_individual_b, expected["b"]
            ),
        ]
        for padded_rows in (64, 128, 256, 512, 1024):
            if padded_rows < value.shape[0]:
                continue
            bucket_value = torch.zeros(
                (padded_rows, HIDDEN_SIZE),
                dtype=value.dtype,
                device=value.device,
            )
            bucket_value[: value.shape[0]].copy_(value)
            bucket_b = F.linear(bucket_value, weights["b"])[
                : value.shape[0]
            ]
            results.append(
                compare(
                    f"individual-bucket-{padded_rows}",
                    "b",
                    bucket_b,
                    expected["b"],
                )
            )
        print(
            json.dumps(
                {
                    "case": case,
                    "rows": value.shape[0],
                    "b_exact_dot_analysis": exact_dot_analysis(
                        value,
                        weights["b"],
                        padded_individual_b,
                        expected["b"],
                    ),
                    "results": results,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
