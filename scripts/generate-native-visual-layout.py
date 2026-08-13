#!/usr/bin/env python3
"""Generate the compile-time native visual tensor layout."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "engine" / "native-visual-weight-manifest.json"
DEFAULT_OUTPUT = ROOT / "native" / "generated" / "visual_model_layout.h"
SCHEMA = "aima-amd395-qwen36/native-visual-weight-manifest/v1"
MODEL_ID = "Qwen3.6-35B-A3B-BF16"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
CONFIG_SHA256 = "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
CHECKPOINT_INDEX_SHA256 = (
    "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
)
EXPECTED_TENSOR_COUNT = 333
EXPECTED_PAYLOAD_BYTES = 893_142_496
EXPECTED_MANIFEST_SHA256 = (
    "abc5b3a0cc0881ba2d3e815b472eebe3404a6e3bc6438a430faccfbe8093c0aa"
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def expected_shapes() -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {
        "model.visual.merger.linear_fc1.bias": [4608],
        "model.visual.merger.linear_fc1.weight": [4608, 4608],
        "model.visual.merger.linear_fc2.bias": [2048],
        "model.visual.merger.linear_fc2.weight": [2048, 4608],
        "model.visual.merger.norm.bias": [1152],
        "model.visual.merger.norm.weight": [1152],
        "model.visual.patch_embed.proj.bias": [1152],
        "model.visual.patch_embed.proj.weight": [1152, 3, 2, 16, 16],
        "model.visual.pos_embed.weight": [2304, 1152],
    }
    block_suffixes = {
        "attn.proj.bias": [1152],
        "attn.proj.weight": [1152, 1152],
        "attn.qkv.bias": [3456],
        "attn.qkv.weight": [3456, 1152],
        "mlp.linear_fc1.bias": [4304],
        "mlp.linear_fc1.weight": [4304, 1152],
        "mlp.linear_fc2.bias": [1152],
        "mlp.linear_fc2.weight": [1152, 4304],
        "norm1.bias": [1152],
        "norm1.weight": [1152],
        "norm2.bias": [1152],
        "norm2.weight": [1152],
    }
    for block in range(27):
        for suffix, shape in block_suffixes.items():
            shapes[f"model.visual.blocks.{block}.{suffix}"] = shape
    return shapes


def parse_u64(value: Any, label: str) -> int:
    try:
        parsed = int(str(value), 0)
    except ValueError as error:
        raise RuntimeError(f"invalid {label}") from error
    if parsed < 0 or parsed > 0xFFFFFFFFFFFFFFFF:
        raise RuntimeError(f"{label} is outside uint64")
    return parsed


def render(manifest_path: Path) -> str:
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("native visual manifest SHA-256 differs from the frozen contract")
    manifest = load_object(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise RuntimeError("native visual weight manifest is incomplete or unsupported")
    model = manifest.get("model")
    layout = manifest.get("layout")
    entries = manifest.get("entries")
    sources = manifest.get("source_files")
    if not all(isinstance(item, dict) for item in (model, layout, sources)):
        raise RuntimeError("native visual manifest metadata is incomplete")
    if not isinstance(entries, list):
        raise RuntimeError("native visual manifest entries are absent")
    assert isinstance(model, dict) and isinstance(layout, dict) and isinstance(sources, dict)
    identities = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "config_sha256": CONFIG_SHA256,
        "checkpoint_index_sha256": CHECKPOINT_INDEX_SHA256,
    }
    for key, expected in identities.items():
        if model.get(key) != expected:
            raise RuntimeError(f"native visual model {key} differs from the frozen contract")
    if len(entries) != EXPECTED_TENSOR_COUNT or int(layout.get("tensor_count", -1)) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError("native visual tensor count differs from the frozen contract")

    frozen_shapes = expected_shapes()
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    payload_total = 0
    shard_names = sorted(str(name) for name in sources)
    if shard_names != [
        "model-00001-of-00026.safetensors",
        "model-00002-of-00026.safetensors",
    ]:
        raise RuntimeError("native visual shard set differs from the frozen contract")
    shard_indices = {name: index for index, name in enumerate(shard_names)}
    for shard_name in shard_names:
        source = sources[shard_name]
        if not isinstance(source, dict) or int(source.get("bytes", 0)) <= 0 or not re.fullmatch(
            r"[0-9a-f]{64}", str(source.get("sha256", ""))
        ):
            raise RuntimeError(f"invalid visual source identity: {shard_name}")

    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"visual manifest entry {position} is not an object")
        name = str(entry.get("name", ""))
        shape = entry.get("shape")
        shard = str(entry.get("source_shard", ""))
        offset = int(entry.get("source_offset_bytes", -1))
        payload = int(entry.get("payload_bytes", 0))
        if name in names or name not in frozen_shapes or shape != frozen_shapes[name]:
            raise RuntimeError(f"unexpected native visual tensor geometry: {name}")
        if entry.get("dtype") != "BF16" or shard not in shard_indices or offset < 0:
            raise RuntimeError(f"invalid native visual tensor source: {name}")
        dimensions = [int(value) for value in shape]
        if payload != math.prod(dimensions) * 2:
            raise RuntimeError(f"invalid native visual tensor payload: {name}")
        if offset + payload > int(sources[shard]["bytes"]):
            raise RuntimeError(f"native visual tensor exceeds source shard: {name}")
        names.add(name)
        payload_total += payload
        normalized.append(
            {
                "name": name,
                "shard": shard_indices[shard],
                "offset": offset,
                "bytes": payload,
                "rank": len(dimensions),
                "shape": dimensions + [1] * (5 - len(dimensions)),
            }
        )
    if names != set(frozen_shapes):
        raise RuntimeError("native visual tensor names do not close the frozen architecture")
    if payload_total != EXPECTED_PAYLOAD_BYTES or int(layout.get("payload_bytes", -1)) != EXPECTED_PAYLOAD_BYTES:
        raise RuntimeError("native visual payload bytes differ from the frozen contract")
    names_digest = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
    if layout.get("active_names_sha256") != names_digest:
        raise RuntimeError("native visual name digest differs from the manifest")

    payload_xor = parse_u64(layout.get("payload_xor_u64"), "visual payload XOR")
    payload_sum = parse_u64(layout.get("payload_sum_u64"), "visual payload sum")
    lines = [
        "// Generated by scripts/generate-native-visual-layout.py. Do not edit.",
        "// SPDX-License-Identifier: Apache-2.0",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace aima::generated::visual {",
        "",
        "struct TensorSpec {",
        "  const char* name;",
        "  std::uint32_t shard_index;",
        "  std::uint64_t source_offset_bytes;",
        "  std::uint64_t payload_bytes;",
        "  std::uint8_t rank;",
        "  std::array<std::uint32_t, 5> shape;",
        "};",
        "",
        f"inline constexpr char kModelId[] = {cpp_string(MODEL_ID)};",
        f"inline constexpr char kModelRevision[] = {cpp_string(MODEL_REVISION)};",
        f"inline constexpr char kModelConfigSha256[] = {cpp_string(CONFIG_SHA256)};",
        f"inline constexpr char kCheckpointIndexSha256[] = {cpp_string(CHECKPOINT_INDEX_SHA256)};",
        f"inline constexpr char kManifestSha256[] = {cpp_string(manifest_sha256)};",
        f"inline constexpr std::uint64_t kPayloadBytes = {payload_total}ULL;",
        f"inline constexpr std::uint64_t kExpectedPayloadXor = {payload_xor}ULL;",
        f"inline constexpr std::uint64_t kExpectedPayloadSum = {payload_sum}ULL;",
        "",
        f"inline constexpr std::array<const char*, {len(shard_names)}> kShardNames = {{{{",
    ]
    lines.extend(f"  {cpp_string(name)}," for name in shard_names)
    lines.extend(["}};", "", f"inline constexpr std::array<TensorSpec, {len(normalized)}> kTensorSpecs = {{{{"])
    for entry in normalized:
        shape = ", ".join(str(value) for value in entry["shape"])
        lines.append(
            "  TensorSpec{"
            f"{cpp_string(entry['name'])}, {entry['shard']}, {entry['offset']}ULL, "
            f"{entry['bytes']}ULL, {entry['rank']}, {{{{{shape}}}}}"
            "},"
        )
    lines.extend(["}};", "", "}  // namespace aima::generated::visual", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render(args.manifest.resolve())
    output = args.output.resolve()
    if args.check:
        actual = output.read_text(encoding="utf-8") if output.is_file() else None
        if actual != expected:
            print(f"generated native visual layout is stale: {output}", file=sys.stderr)
            return 1
        print(f"native visual layout: PASS ({len(expected.encode())} bytes)")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(expected, encoding="utf-8")
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
