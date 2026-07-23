#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors
"""Preload qualified Safetensors shards into the resident engine cache."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any


ENVIRONMENT = {
    "plan": "AIMA_DIRECT_CHECKPOINT_PLAN",
    "loader": "AIMA_DIRECT_CHECKPOINT_LOADER",
    "native_report": "AIMA_DIRECT_CHECKPOINT_NATIVE_REPORT",
    "report": "AIMA_DIRECT_CHECKPOINT_REPORT",
    "index_sha256": "AIMA_DIRECT_CHECKPOINT_INDEX_SHA256",
    "expected_xor": "AIMA_DIRECT_CHECKPOINT_EXPECTED_XOR",
    "expected_sum": "AIMA_DIRECT_CHECKPOINT_EXPECTED_SUM",
    "payload_bytes": "AIMA_DIRECT_CHECKPOINT_PAYLOAD_BYTES",
    "tensor_count": "AIMA_DIRECT_CHECKPOINT_TENSOR_COUNT",
    "chunk_bytes": "AIMA_DIRECT_CHECKPOINT_CHUNK_BYTES",
    "workers": "AIMA_DIRECT_CHECKPOINT_WORKERS",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def required_environment() -> dict[str, str]:
    values = {key: os.environ.get(name) for key, name in ENVIRONMENT.items()}
    missing = [ENVIRONMENT[key] for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"direct checkpoint environment is incomplete: {missing}")
    return {key: str(value) for key, value in values.items()}


def positive_int(value: str, label: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be an integer") from exc
    if parsed <= 0:
        raise RuntimeError(f"{label} must be positive")
    return parsed


def preload_from_environment(
    *,
    engine: Any,
    model_dir: Path,
    device: str,
) -> dict[str, Any] | None:
    """Load all raw checkpoint tensors when the direct path is configured."""

    if not os.environ.get(ENVIRONMENT["plan"]):
        return None
    values = required_environment()
    plan_path = Path(values["plan"]).expanduser().resolve()
    loader_path = Path(values["loader"]).expanduser().resolve()
    native_report_path = Path(values["native_report"]).expanduser().resolve()
    report_path = Path(values["report"]).expanduser().resolve()
    model_path = model_dir.expanduser().resolve()
    if not plan_path.is_file():
        raise RuntimeError(f"direct checkpoint plan does not exist: {plan_path}")
    if not loader_path.is_file():
        raise RuntimeError(f"direct checkpoint loader does not exist: {loader_path}")

    plan_sha256 = sha256_file(plan_path)
    loader_sha256 = sha256_file(loader_path)
    state_key = f"{model_path}:{plan_sha256}:{loader_sha256}:{device}"
    retained_states = getattr(engine, "_AIMA_DIRECT_CHECKPOINT_STATES", {})
    retained = retained_states.get(state_key)
    if retained is not None:
        report = dict(retained["report"])
        report["reused"] = True
        return report

    started = time.perf_counter()
    plan = load_json(plan_path)
    entries = plan.get("entries")
    if plan.get("complete") is not True or not isinstance(entries, list):
        raise RuntimeError("direct checkpoint plan is incomplete")
    expected_tensor_count = positive_int(values["tensor_count"], "tensor count")
    expected_payload_bytes = positive_int(values["payload_bytes"], "payload bytes")
    chunk_bytes = positive_int(values["chunk_bytes"], "chunk bytes")
    worker_count = positive_int(values["workers"], "worker count")
    expected_xor = positive_int(values["expected_xor"], "expected XOR")
    expected_sum = positive_int(values["expected_sum"], "expected sum")
    if len(entries) != expected_tensor_count:
        raise RuntimeError(
            f"direct checkpoint tensor count mismatch: {len(entries)} != {expected_tensor_count}"
        )
    if plan.get("layout", {}).get("payload_bytes") != expected_payload_bytes:
        raise RuntimeError("direct checkpoint payload total does not match the release contract")

    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise RuntimeError("model directory is missing config.json or model.safetensors.index.json")
    config_sha256 = sha256_file(config_path)
    index_sha256 = sha256_file(index_path)
    if index_sha256 != values["index_sha256"]:
        raise RuntimeError(
            f"checkpoint index mismatch: expected {values['index_sha256']}, got {index_sha256}"
        )
    if (
        plan.get("inputs", {}).get("checkpoint_index", {}).get("sha256")
        != index_sha256
    ):
        raise RuntimeError("direct checkpoint plan is bound to a different checkpoint index")
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("checkpoint index does not contain a weight_map object")

    entry_names: set[str] = set()
    shard_names: set[str] = set()
    payload_total = 0
    normalized_entries: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"direct checkpoint entry {position} is not an object")
        name = entry.get("name")
        shard = entry.get("source_shard")
        shape = entry.get("shape")
        if not isinstance(name, str) or not name or name in entry_names:
            raise RuntimeError(f"invalid or duplicate tensor name at entry {position}")
        if (
            not isinstance(shard, str)
            or not shard
            or Path(shard).name != shard
            or weight_map.get(name) != shard
        ):
            raise RuntimeError(f"checkpoint shard mapping mismatch for tensor {name}")
        if entry.get("dtype") != "BF16" or not isinstance(shape, list) or not shape:
            raise RuntimeError(f"unsupported tensor contract for {name}")
        dimensions = [int(value) for value in shape]
        if any(value <= 0 for value in dimensions):
            raise RuntimeError(f"invalid tensor shape for {name}")
        payload_bytes = int(entry.get("payload_bytes", 0))
        source_offset = int(entry.get("source_offset_bytes", -1))
        if payload_bytes != math.prod(dimensions) * 2 or source_offset < 0:
            raise RuntimeError(f"tensor byte geometry mismatch for {name}")
        entry_names.add(name)
        shard_names.add(shard)
        payload_total += payload_bytes
        normalized_entries.append(
            {
                "name": name,
                "shape": tuple(dimensions),
                "source_shard": shard,
                "source_offset": source_offset,
                "payload_bytes": payload_bytes,
            }
        )
    if not entry_names.issubset(weight_map):
        raise RuntimeError("direct checkpoint plan contains tensors absent from checkpoint weight_map")
    if payload_total != expected_payload_bytes:
        raise RuntimeError("direct checkpoint entry payload sum mismatch")

    shard_order = sorted(shard_names)
    shard_paths = [model_path / name for name in shard_order]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"checkpoint shards are missing: {missing[:4]}")
    shard_sizes = [path.stat().st_size for path in shard_paths]
    shard_indices = {name: index for index, name in enumerate(shard_order)}
    for entry in normalized_entries:
        shard_size = shard_sizes[shard_indices[entry["source_shard"]]]
        if entry["source_offset"] + entry["payload_bytes"] > shard_size:
            raise RuntimeError(f"tensor payload exceeds checkpoint shard: {entry['name']}")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for direct checkpoint loading") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("direct checkpoint loading requires the ROCm CUDA compatibility API")

    raw_prefix = engine.engine_cache_key(
        {
            "cache_scope": "raw_model_weights",
            "model_dir": str(model_path),
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "device": device,
        }
    )
    cache = engine._ENGINE_TENSOR_CACHE
    raw_entries_before = sum(1 for key in cache if key.startswith(f"{raw_prefix}:raw:"))
    if raw_entries_before != 0:
        raise RuntimeError("direct checkpoint loader requires an empty raw tensor cache")

    allocation_started = time.perf_counter()
    tensors = [
        torch.empty(entry["shape"], dtype=torch.bfloat16, device=device)
        for entry in normalized_entries
    ]
    torch.cuda.synchronize()
    allocation_ms = (time.perf_counter() - allocation_started) * 1000.0
    destination_pointers = [int(tensor.data_ptr()) for tensor in tensors]
    if len(set(destination_pointers)) != expected_tensor_count:
        raise RuntimeError("direct checkpoint allocations do not have unique device pointers")
    if any(int(tensor.numel()) * int(tensor.element_size()) != entry["payload_bytes"]
           for tensor, entry in zip(tensors, normalized_entries)):
        raise RuntimeError("direct checkpoint Torch allocation geometry mismatch")

    encoded_paths = [str(path.resolve()).encode() for path in shard_paths]
    path_array = (ctypes.c_char_p * len(encoded_paths))(*encoded_paths)
    size_array = (ctypes.c_uint64 * len(shard_sizes))(*shard_sizes)
    tensor_shard_array = (ctypes.c_uint32 * expected_tensor_count)(
        *[shard_indices[entry["source_shard"]] for entry in normalized_entries]
    )
    source_offset_array = (ctypes.c_uint64 * expected_tensor_count)(
        *[entry["source_offset"] for entry in normalized_entries]
    )
    payload_array = (ctypes.c_uint64 * expected_tensor_count)(
        *[entry["payload_bytes"] for entry in normalized_entries]
    )
    pointer_array = (ctypes.c_uint64 * expected_tensor_count)(*destination_pointers)
    native_report_path.parent.mkdir(parents=True, exist_ok=True)
    library = ctypes.CDLL(str(loader_path))
    function = library.torch_owned_safetensors_tensor_scatter_ingest
    function.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_char_p,
    ]
    function.restype = ctypes.c_int
    native_started = time.perf_counter()
    return_code = int(
        function(
            path_array,
            size_array,
            ctypes.c_size_t(len(shard_paths)),
            tensor_shard_array,
            source_offset_array,
            payload_array,
            pointer_array,
            ctypes.c_size_t(expected_tensor_count),
            ctypes.c_size_t(chunk_bytes),
            ctypes.c_size_t(worker_count),
            ctypes.c_uint64(expected_xor),
            ctypes.c_uint64(expected_sum),
            str(native_report_path).encode(),
        )
    )
    torch.cuda.synchronize()
    native_elapsed_ms = (time.perf_counter() - native_started) * 1000.0
    native = load_json(native_report_path)
    native_checks = {
        "return_zero": return_code == 0,
        "complete": native.get("complete") is True,
        "shard_count": native.get("shard_count") == len(shard_paths),
        "tensor_count": native.get("tensor_count") == expected_tensor_count,
        "payload_bytes": native.get("payload_bytes") == expected_payload_bytes,
        "scheduled_payload_bytes": native.get("scheduled_payload_bytes")
        == expected_payload_bytes,
        "unique_destinations": native.get("unique_destination_pointers")
        == expected_tensor_count,
        "pointer_types_device": native.get("all_pointer_types_device") is True,
        "pointers_match": native.get("all_device_pointers_match") is True,
        "gpu_checksum": native.get("gpu_payload_checksum_equal") is True,
        "destination_not_freed": native.get("destination_freed_by_native") is False,
        "cleanup_complete": native.get("cleanup_complete") is True,
    }
    if not all(native_checks.values()):
        raise RuntimeError(f"direct checkpoint native loader failed: {native_checks}")

    bind_started = time.perf_counter()
    cache_keys = [f"{raw_prefix}:raw:{entry['name']}" for entry in normalized_entries]
    for key, tensor in zip(cache_keys, tensors):
        if key in cache:
            raise RuntimeError(f"duplicate direct checkpoint cache key: {key}")
        cache[key] = tensor
    bind_ms = (time.perf_counter() - bind_started) * 1000.0
    raw_entries_after = sum(1 for key in cache if key.startswith(f"{raw_prefix}:raw:"))
    if raw_entries_after != expected_tensor_count:
        raise RuntimeError("direct checkpoint cache population count mismatch")

    report = {
        "schema": "aima-amd395-qwen36/direct-checkpoint-loader/v1",
        "active": True,
        "complete": True,
        "reused": False,
        "provider": "native_safetensors_odirect_pinned_h2d_scatter",
        "model_dir": str(model_path),
        "checkpoint_index_sha256": index_sha256,
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "library": str(loader_path),
        "library_sha256": loader_sha256,
        "source_shard_count": len(shard_paths),
        "source_shard_bytes": sum(shard_sizes),
        "tensor_count": expected_tensor_count,
        "payload_bytes": expected_payload_bytes,
        "extra_weight_copy_bytes": 0,
        "allocation_ms": allocation_ms,
        "native_elapsed_ms": native_elapsed_ms,
        "bind_ms": bind_ms,
        "preload_wall_time_ms": (time.perf_counter() - started) * 1000.0,
        "raw_cache_entries_before": raw_entries_before,
        "raw_cache_entries_after": raw_entries_after,
        "native_return_code": return_code,
        "native_checks": native_checks,
        "native": native,
    }
    write_json(report_path, report)
    retained_states[state_key] = {"tensors": tensors, "report": report}
    engine._AIMA_DIRECT_CHECKPOINT_STATES = retained_states
    return report
