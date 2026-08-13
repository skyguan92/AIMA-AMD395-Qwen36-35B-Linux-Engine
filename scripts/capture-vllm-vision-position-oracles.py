#!/usr/bin/env python3
"""Capture Qwen3.6 vision position interpolation from the frozen vLLM."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any


SCHEMA = "aima-amd395-qwen36/vision-position-oracle/v1"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
CONFIG_SHA256 = "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
INDEX_SHA256 = "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
VLLM_VERSION = "0.19.1rc1.dev300+g29e5d1020.rocm721"
TORCH_VERSION = "2.10.0+git8514f05"
TRANSFORMERS_VERSION = "4.57.6"
VLLM_SOURCE_SHA256 = "8ba3592a0fb481a959d6952af25a721cfaeab966558ac11214304e5cf7524d1a"
TENSOR_NAME = "model.visual.pos_embed.weight"
CASES = {
    "image_16x16": (1, 16, 16),
    "image_12x32": (1, 12, 32),
    "video_2x8x8": (2, 8, 8),
    "video_2x8x12": (2, 8, 12),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def tensor_from_header(model_dir: Path, shard_name: str) -> tuple[int, int]:
    shard = model_dir / shard_name
    with shard.open("rb") as stream:
        encoded_length = stream.read(8)
        if len(encoded_length) != 8:
            raise RuntimeError("position embedding shard is truncated")
        header_length = struct.unpack("<Q", encoded_length)[0]
        header = json.loads(stream.read(header_length))
    tensor = header.get(TENSOR_NAME)
    if not isinstance(tensor, dict) or tensor.get("dtype") != "BF16" or tensor.get(
        "shape"
    ) != [2304, 1152]:
        raise RuntimeError("position embedding tensor geometry is invalid")
    offsets = tensor.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise RuntimeError("position embedding tensor offsets are invalid")
    return 8 + header_length + int(offsets[0]), int(offsets[1]) - int(offsets[0])


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def capture(model_dir: Path, output_dir: Path) -> dict[str, Any]:
    import torch
    from vllm.model_executor.models import qwen3_vl

    versions = {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "vllm": importlib.metadata.version("vllm"),
    }
    expected_versions = {
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "vllm": VLLM_VERSION,
    }
    if versions != expected_versions:
        raise RuntimeError("reference package versions differ from the frozen contract")
    source_path = Path(inspect.getfile(qwen3_vl)).resolve()
    if sha256_file(source_path) != VLLM_SOURCE_SHA256:
        raise RuntimeError("qwen3_vl source differs from the frozen reference")

    model_dir = model_dir.resolve()
    if sha256_file(model_dir / "config.json") != CONFIG_SHA256:
        raise RuntimeError("model config differs from the frozen contract")
    index_path = model_dir / "model.safetensors.index.json"
    if sha256_file(index_path) != INDEX_SHA256:
        raise RuntimeError("checkpoint index differs from the frozen contract")
    index = load_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or TENSOR_NAME not in weight_map:
        raise RuntimeError("checkpoint index has no position embedding")
    shard_name = str(weight_map[TENSOR_NAME])
    offset, payload_bytes = tensor_from_header(model_dir, shard_name)
    if payload_bytes != 2304 * 1152 * 2:
        raise RuntimeError("position embedding payload bytes are invalid")
    with (model_dir / shard_name).open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(payload_bytes)
    if len(raw) != payload_bytes:
        raise RuntimeError("position embedding payload is truncated")
    weight = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(2304, 1152)
    weight = weight.to("cuda")

    output_dir.mkdir(parents=True, exist_ok=False)
    cases: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case_id, (temporal, height, width) in CASES.items():
            value = qwen3_vl.pos_embed_interpolate_native(
                weight, temporal, height, width, 48, 2, torch.bfloat16
            ).contiguous()
            torch.cuda.synchronize()
            payload = value.view(torch.uint16).cpu().numpy().tobytes()
            relative = f"{case_id}.bin"
            write_atomic(output_dir / relative, payload)
            cases.append(
                {
                    "case_id": case_id,
                    "grid_thw": [temporal, height, width],
                    "shape": list(value.shape),
                    "dtype": "bfloat16",
                    "bytes": len(payload),
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return {
        "schema": SCHEMA,
        "complete": True,
        "model": {
            "revision": MODEL_REVISION,
            "config_sha256": CONFIG_SHA256,
            "checkpoint_index_sha256": INDEX_SHA256,
            "position_tensor": TENSOR_NAME,
            "position_tensor_shard": shard_name,
            "position_tensor_sha256": hashlib.sha256(raw).hexdigest(),
        },
        "runtime": versions,
        "reference": {
            "function": "vllm.model_executor.models.qwen3_vl.pos_embed_interpolate_native",
            "source_sha256": VLLM_SOURCE_SHA256,
            "num_grid_per_side": 48,
            "spatial_merge_size": 2,
            "device": "cuda",
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.model_dir, args.output_dir.resolve())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    write_atomic(args.output_dir.resolve() / "manifest.json", encoded.encode())
    print(args.output_dir.resolve() / "manifest.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"capture vision position oracle: {error}", file=sys.stderr)
        raise SystemExit(1)
