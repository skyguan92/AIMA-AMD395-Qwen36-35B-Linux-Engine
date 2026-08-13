#!/usr/bin/env python3
"""Capture the frozen Qwen3.6 visual tensor layout from Safetensors headers."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any


SCHEMA = "aima-amd395-qwen36/native-visual-weight-manifest/v1"
MODEL_ID = "Qwen3.6-35B-A3B-BF16"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
CONFIG_SHA256 = "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
CHECKPOINT_INDEX_SHA256 = (
    "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
)
EXPECTED_TENSOR_COUNT = 333
EXPECTED_PAYLOAD_BYTES = 893_142_496
VISUAL_PREFIX = "model.visual."


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


def load_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        encoded_length = stream.read(8)
        if len(encoded_length) != 8:
            raise RuntimeError(f"truncated Safetensors length: {path}")
        header_length = struct.unpack("<Q", encoded_length)[0]
        if header_length == 0 or header_length > 128 * 1024 * 1024:
            raise RuntimeError(f"invalid Safetensors header length: {path}")
        encoded = stream.read(header_length)
        if len(encoded) != header_length:
            raise RuntimeError(f"truncated Safetensors header: {path}")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise RuntimeError(f"Safetensors header is not an object: {path}")
    return 8 + header_length, value


def checksum_payloads(
    model_dir: Path, entries: list[dict[str, Any]]
) -> tuple[int, int]:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - capture-host contract
        raise RuntimeError(
            "capture requires NumPy for the byte-exact uint64 checksum"
        ) from error

    payload_xor = np.uint64(0)
    payload_sum = np.uint64(0)
    open_shard: Path | None = None
    stream = None
    try:
        for entry in sorted(
            entries,
            key=lambda item: (item["source_shard"], item["source_offset_bytes"]),
        ):
            shard = model_dir / entry["source_shard"]
            if shard != open_shard:
                if stream is not None:
                    stream.close()
                stream = shard.open("rb")
                open_shard = shard
            assert stream is not None
            stream.seek(entry["source_offset_bytes"])
            remaining = int(entry["payload_bytes"])
            while remaining:
                block = stream.read(min(32 * 1024 * 1024, remaining))
                if not block:
                    raise RuntimeError(f"truncated visual tensor payload: {shard}")
                if len(block) % 8:
                    raise RuntimeError("visual tensor checksum block is not uint64 aligned")
                words = np.frombuffer(block, dtype="<u8")
                payload_xor = np.bitwise_xor(payload_xor, np.bitwise_xor.reduce(words))
                payload_sum = np.add(payload_sum, words.sum(dtype=np.uint64))
                remaining -= len(block)
    finally:
        if stream is not None:
            stream.close()
    return int(payload_xor), int(payload_sum)


def capture(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if sha256_file(config_path) != CONFIG_SHA256:
        raise RuntimeError("model config SHA-256 differs from the frozen VL contract")
    if sha256_file(index_path) != CHECKPOINT_INDEX_SHA256:
        raise RuntimeError("checkpoint index SHA-256 differs from the frozen VL contract")

    index = load_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("checkpoint index has no weight_map object")
    names = sorted(
        str(name) for name in weight_map if str(name).startswith(VISUAL_PREFIX)
    )
    if len(names) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError("visual tensor count differs from the frozen VL contract")

    shard_names = sorted({str(weight_map[name]) for name in names})
    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    source_files: dict[str, dict[str, Any]] = {}
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        headers[shard_name] = load_safetensors_header(shard_path)
        source_files[shard_name] = {
            "bytes": shard_path.stat().st_size,
            "sha256": sha256_file(shard_path),
        }

    entries: list[dict[str, Any]] = []
    payload_total = 0
    for name in names:
        shard_name = str(weight_map[name])
        data_base, header = headers[shard_name]
        tensor = header.get(name)
        if not isinstance(tensor, dict):
            raise RuntimeError(f"checkpoint header is missing visual tensor: {name}")
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if dtype != "BF16" or not isinstance(shape, list) or not 1 <= len(shape) <= 5:
            raise RuntimeError(f"unsupported visual tensor contract: {name}")
        dimensions = [int(value) for value in shape]
        if any(value <= 0 or value > 0xFFFFFFFF for value in dimensions):
            raise RuntimeError(f"invalid visual tensor shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"invalid visual tensor offsets: {name}")
        payload_bytes = math.prod(dimensions) * 2
        if int(offsets[1]) - int(offsets[0]) != payload_bytes:
            raise RuntimeError(f"visual tensor payload differs from shape: {name}")
        source_offset = data_base + int(offsets[0])
        shard_bytes = int(source_files[shard_name]["bytes"])
        if source_offset < data_base or source_offset + payload_bytes > shard_bytes:
            raise RuntimeError(f"visual tensor exceeds its checkpoint shard: {name}")
        entries.append(
            {
                "dtype": dtype,
                "name": name,
                "payload_bytes": payload_bytes,
                "shape": dimensions,
                "source_offset_bytes": source_offset,
                "source_shard": shard_name,
            }
        )
        payload_total += payload_bytes
    if payload_total != EXPECTED_PAYLOAD_BYTES:
        raise RuntimeError("visual payload bytes differ from the frozen VL contract")

    payload_xor, payload_sum = checksum_payloads(model_dir, entries)
    names_digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
    geometry_digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "complete": True,
        "model": {
            "config_sha256": CONFIG_SHA256,
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "checkpoint_index_sha256": CHECKPOINT_INDEX_SHA256,
        },
        "source_files": source_files,
        "layout": {
            "active_names_sha256": names_digest,
            "geometry_sha256": geometry_digest,
            "payload_bytes": payload_total,
            "payload_sum_u64": f"0x{payload_sum:016x}",
            "payload_xor_u64": f"0x{payload_xor:016x}",
            "tensor_count": len(entries),
        },
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.model_dir)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(output)
    print(f"{output}: {len(result['entries'])} tensors, {result['layout']['payload_bytes']} bytes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"capture native visual layout: {error}", file=sys.stderr)
        raise SystemExit(1)
