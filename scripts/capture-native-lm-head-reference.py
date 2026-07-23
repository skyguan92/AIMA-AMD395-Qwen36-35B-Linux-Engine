#!/usr/bin/env python3
"""Capture byte-exact qualified LM-head quantization reference hashes.

This is a validation-only tool.  It intentionally uses the qualified Torch
implementation and is never copied into the native runtime bundle.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any, chunk_rows: int = 8192) -> str:
    digest = hashlib.sha256()
    rows = int(tensor.shape[0])
    for start in range(0, rows, chunk_rows):
        chunk = tensor[start : start + chunk_rows].contiguous().cpu()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sample_values(tensor: Any) -> list[dict[str, Any]]:
    rows = int(tensor.shape[0])
    columns = int(tensor.shape[1]) if tensor.ndim == 2 else 1
    positions = {
        (0, 0),
        (0, columns - 1),
        (1, 1 if columns > 1 else 0),
        (rows // 2, columns // 2),
        (rows - 2, columns - 2 if columns > 1 else 0),
        (rows - 1, columns - 1),
    }
    result: list[dict[str, Any]] = []
    for row, column in sorted(positions):
        value = tensor[row, column] if tensor.ndim == 2 else tensor[row]
        result.append({"row": row, "column": column, "value": value.item()})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-scales", type=Path)
    parser.add_argument("--dump-residual-l2", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    global torch
    import torch
    from safetensors import safe_open

    model_dir = args.model_dir.resolve()
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index["weight_map"]["lm_head.weight"]
    shard_path = model_dir / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as stream:
        weight = stream.get_tensor("lm_head.weight").to(device=args.device)
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (248320, 2048):
        raise RuntimeError(f"unexpected LM-head tensor: {weight.dtype} {tuple(weight.shape)}")

    started = time.perf_counter()
    rows, _ = weight.shape
    q_weight = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty(rows, device=weight.device, dtype=torch.float32)
    residual_l2 = torch.empty(rows, device=weight.device, dtype=torch.float32)
    chunk_rows = 8192
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        source = weight[start:end].float()
        chunk_scales = source.abs().amax(dim=1).clamp_min(1e-12) / 127.0
        quantized = torch.round(source / chunk_scales[:, None]).clamp(-127, 127).to(torch.int8)
        residual = source - quantized.float() * chunk_scales[:, None]
        q_weight[start:end].copy_(quantized)
        scales[start:end].copy_(chunk_scales)
        residual_l2[start:end].copy_(torch.linalg.vector_norm(residual, dim=1))
    torch.cuda.synchronize()
    quantize_ms = (time.perf_counter() - started) * 1000.0
    if args.dump_scales is not None:
        dump_scales = args.dump_scales.resolve()
        dump_scales.parent.mkdir(parents=True, exist_ok=True)
        dump_scales.write_bytes(scales.contiguous().cpu().numpy().tobytes())
    if args.dump_residual_l2 is not None:
        dump_residual = args.dump_residual_l2.resolve()
        dump_residual.parent.mkdir(parents=True, exist_ok=True)
        dump_residual.write_bytes(residual_l2.contiguous().cpu().numpy().tobytes())

    hash_started = time.perf_counter()
    payload = {
        "schema": "aima-amd395-qwen36/qualified-lm-head-int8-reference/v1",
        "complete": True,
        "model": "Qwen3.6-35B-A3B-BF16",
        "checkpoint_index_sha256": sha256_file(index_path),
        "source_shard": shard_name,
        "source_shard_sha256": sha256_file(shard_path),
        "source_shape": list(weight.shape),
        "source_dtype": str(weight.dtype),
        "quantization": {
            "owner": "qualified production Torch implementation",
            "row_scale": "clamp_min(max(abs(row)), 1e-12) / 127",
            "rounding": "torch.round then clamp(-127,127) then int8",
            "residual_l2": "torch.linalg.vector_norm(source - q.float * scale, dim=1)",
            "chunk_rows": chunk_rows,
        },
        "outputs": {
            "q_weight": {
                "shape": list(q_weight.shape),
                "dtype": str(q_weight.dtype),
                "bytes": q_weight.numel() * q_weight.element_size(),
                "sha256": tensor_sha256(q_weight),
                "samples": sample_values(q_weight),
            },
            "scales": {
                "shape": list(scales.shape),
                "dtype": str(scales.dtype),
                "bytes": scales.numel() * scales.element_size(),
                "sha256": tensor_sha256(scales),
                "samples": sample_values(scales),
            },
            "residual_l2": {
                "shape": list(residual_l2.shape),
                "dtype": str(residual_l2.dtype),
                "bytes": residual_l2.numel() * residual_l2.element_size(),
                "sha256": tensor_sha256(residual_l2),
                "samples": sample_values(residual_l2),
            },
        },
        "timing": {
            "quantize_ms": quantize_ms,
            "hash_ms": (time.perf_counter() - hash_started) * 1000.0,
            "role": "validation_only_not_performance_evidence",
        },
        "environment": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "outputs": payload["outputs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
