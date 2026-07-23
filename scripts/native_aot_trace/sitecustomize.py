"""Capture the exact Triton AOT closure exercised by a qualified runtime.

This module is loaded through ``PYTHONPATH`` only during an export run.  The
default mode records one structural launch per compiled kernel.  An explicit
executor-migration mode can retain every launch and device pointer so the
native schedule can be reconstructed without changing normal runtime
semantics.  The final native product neither imports nor packages this module.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any


TRACE_PATH = os.environ.get("AIMA_AOT_TRACE_JSONL")
TRACE_ALL_LAUNCHES = os.environ.get("AIMA_AOT_TRACE_ALL_LAUNCHES") == "1"
TRACE_POINTERS = os.environ.get("AIMA_AOT_TRACE_POINTERS") == "1"
TENSOR_REGISTRY_PATH = os.environ.get("AIMA_AOT_TENSOR_REGISTRY_JSON")
TENSOR_REGISTRY_PHASE = os.environ.get("AIMA_AOT_TENSOR_REGISTRY_PHASE", "decode")
TENSOR_REGISTRY_TOKENS = os.environ.get("AIMA_AOT_TENSOR_REGISTRY_TOKENS")
ORACLE_DIR = os.environ.get("AIMA_NATIVE_ORACLE_DIR")
ORACLE_ALL_LAYERS = os.environ.get("AIMA_NATIVE_ORACLE_ALL_LAYERS") == "1"
ORACLE_PHASE = os.environ.get("AIMA_NATIVE_ORACLE_PHASE", "decode")
ORACLE_TOKENS = int(os.environ.get("AIMA_NATIVE_ORACLE_TOKENS", "1"))
ORACLE_MAX_BYTES = int(os.environ.get("AIMA_NATIVE_ORACLE_MAX_BYTES", str(3 * 1024 * 1024)))
ORACLE_LARGE_TENSOR_TAIL_ROWS = max(
    0, int(os.environ.get("AIMA_NATIVE_ORACLE_LARGE_TENSOR_TAIL_ROWS", "0"))
)
ORACLE_LARGE_TENSOR_TAIL_AXIS_SIZE = max(
    0,
    int(
        os.environ.get(
            "AIMA_NATIVE_ORACLE_LARGE_TENSOR_TAIL_AXIS_SIZE", "0"
        )
    ),
)
ORACLE_CAPTURE_LAUNCH_ARGUMENTS = (
    os.environ.get("AIMA_NATIVE_ORACLE_CAPTURE_LAUNCH_ARGUMENTS", "1") == "1"
)
ORACLE_FINAL_LOGITS_OCCURRENCE = max(
    1, int(os.environ.get("AIMA_NATIVE_ORACLE_FINAL_LOGITS_OCCURRENCE", "1"))
)
ORACLE_CAPTURE_DECODE_FINAL_LOGITS = (
    os.environ.get("AIMA_NATIVE_ORACLE_CAPTURE_DECODE_FINAL_LOGITS") == "1"
)
ORACLE_CAPTURE_FULL_ATTENTION_PRE_GATE = (
    os.environ.get("AIMA_NATIVE_ORACLE_CAPTURE_FULL_ATTENTION_PRE_GATE") == "1"
)
_ORACLE_MOE_SUM_COORDINATE_TEXT = os.environ.get(
    "AIMA_NATIVE_ORACLE_MOE_SUM_COORDINATE", ""
).strip()
ORACLE_MOE_SUM_COORDINATE = (
    tuple(int(item) for item in _ORACLE_MOE_SUM_COORDINATE_TEXT.split(","))
    if _ORACLE_MOE_SUM_COORDINATE_TEXT
    else None
)
if ORACLE_MOE_SUM_COORDINATE is not None and len(ORACLE_MOE_SUM_COORDINATE) != 3:
    raise RuntimeError(
        "AIMA_NATIVE_ORACLE_MOE_SUM_COORDINATE must be layer,token,hidden"
    )
ORACLE_DECODE_FINAL_LOGITS_OCCURRENCE = max(
    1,
    int(
        os.environ.get(
            "AIMA_NATIVE_ORACLE_DECODE_FINAL_LOGITS_OCCURRENCE", "3"
        )
    ),
)
_ORACLE_LABEL_TEXT = os.environ.get("AIMA_NATIVE_ORACLE_LABELS", "").strip()
ORACLE_LABELS = (
    {item.strip() for item in _ORACLE_LABEL_TEXT.split(",") if item.strip()}
    if _ORACLE_LABEL_TEXT
    else None
)
_LOCK = threading.Lock()
_SEEN: set[str] = set()
_SEQUENCE = 0
_REGISTRY_WRITTEN = False
_ORACLE_ACTIVE = False
_ORACLE_FINISHED = False
_ORACLE_FINAL_LOGITS_CAPTURED = False
_ORACLE_DECODE_FINAL_LOGITS_CAPTURED = False
_ORACLE_FINAL_LOGITS_SEEN = 0
_ORACLE_LAUNCH_SEQUENCE = 0
_ORACLE_RECORD_SEQUENCE = 0
_ORACLE_ACTIVE_LAYERS: set[int] = set()
_ORACLE_FINISHED_LAYERS: set[int] = set()
_ORACLE_LAYER_LAUNCH_SEQUENCES: dict[int, int] = {}
_ORACLE_FULL_ATTENTION_PRE_GATE_CAPTURED: set[int] = set()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append(record: dict[str, Any]) -> None:
    if not TRACE_PATH:
        return
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    destination = Path(TRACE_PATH).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()


def _oracle_append(record: dict[str, Any]) -> None:
    if not ORACLE_DIR:
        return
    destination = Path(ORACLE_DIR).expanduser().resolve() / "oracle.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with _LOCK:
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()


def _oracle_tensor(label: str, value: Any, **context: Any) -> None:
    global _ORACLE_RECORD_SEQUENCE
    if ORACLE_LABELS is not None and label not in ORACLE_LABELS:
        return
    try:
        tensor = _tensor_record(value)
        if tensor is None:
            return
        logical_bytes = int(value.numel()) * int(value.element_size())
        source_dimensions = [int(item) for item in value.shape]
        tail_axis_size = ORACLE_TOKENS
        if ORACLE_LARGE_TENSOR_TAIL_AXIS_SIZE in source_dimensions:
            tail_axis_size = ORACLE_LARGE_TENSOR_TAIL_AXIS_SIZE
        token_axis = (
            source_dimensions.index(tail_axis_size)
            if tail_axis_size in source_dimensions
            else -1
        )
        if (
            ORACLE_LARGE_TENSOR_TAIL_ROWS
            and int(value.ndim) >= 1
            and (
                token_axis >= 0
                or (
                    int(value.ndim) == 1
                    and int(value.numel()) % ORACLE_TOKENS == 0
                )
            )
        ):
            source_shape = [int(item) for item in value.shape]
            tail_rows = min(ORACLE_LARGE_TENSOR_TAIL_ROWS, tail_axis_size)
            if int(value.ndim) == 1:
                row_width = int(value.numel()) // ORACLE_TOKENS
                value = value[-tail_rows * row_width :].reshape(
                    tail_rows, row_width
                )
            else:
                slices = [slice(None)] * int(value.ndim)
                slices[token_axis] = slice(-tail_rows, None)
                value = value[tuple(slices)]
            context = {
                **context,
                "source_shape": source_shape,
                "slice": "tail_rows",
                "tail_rows": tail_rows,
            }
            logical_bytes = int(value.numel()) * int(value.element_size())
        # The native loader is the source of truth for model parameters.  The
        # validation oracle only needs state and intermediate values; this cap
        # prevents accidental checkpoint duplication while retaining the 2 MiB
        # recurrent state.
        large_full_attention_state = (
            "return-full_attention-k_cache" in label
            or "return-full_attention-v_cache" in label
        )
        maximum_bytes = max(20 * 1024 * 1024, ORACLE_MAX_BYTES) if large_full_attention_state else ORACLE_MAX_BYTES
        if logical_bytes > maximum_bytes:
            return
        import torch

        torch.cuda.synchronize()
        payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
        with _LOCK:
            sequence = _ORACLE_RECORD_SEQUENCE
            _ORACLE_RECORD_SEQUENCE += 1
        filename = f"{sequence:04d}-{safe_label}.bin"
        destination = Path(ORACLE_DIR).expanduser().resolve() / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        _oracle_append(
            {
                "event": "native_layer_oracle_tensor",
                "sequence": sequence,
                "label": label,
                "file": filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": [int(item) for item in value.shape],
                "stride": [int(item) for item in value.stride()],
                "dtype": str(value.dtype),
                **context,
            }
        )
    except Exception as exc:
        _oracle_append(
            {
                "event": "native_layer_oracle_error",
                "label": label,
                "error": repr(exc),
                **context,
            }
        )


def _oracle_runtime_context() -> dict[str, Any] | None:
    frame = sys._getframe(1)
    layer_index = -1
    while frame is not None:
        if layer_index < 0 and isinstance(frame.f_locals.get("layer_index"), int):
            layer_index = int(frame.f_locals["layer_index"])
        if frame.f_code.co_name == "run_with_torch":
            mode = str(frame.f_locals.get("mode", ""))
            tokens = int(frame.f_locals.get("tokens", 0))
            return {"mode": mode, "tokens": tokens, "layer_index": layer_index}
        frame = frame.f_back
    return None


def _oracle_profile(frame: Any, event: str, value: Any) -> Any:
    global _ORACLE_FINISHED, _ORACLE_FINAL_LOGITS_CAPTURED
    global _ORACLE_DECODE_FINAL_LOGITS_CAPTURED, _ORACLE_FINAL_LOGITS_SEEN
    filename = frame.f_code.co_filename
    function = frame.f_code.co_name
    if (
        event == "call"
        and function == "moe_sum"
        and filename.endswith("/vllm/_custom_ops.py")
        and ORACLE_MOE_SUM_COORDINATE is not None
    ):
        target_layer, target_token, target_hidden = ORACLE_MOE_SUM_COORDINATE
        runtime = frame
        layer_index = -1
        mode = ""
        tokens = 0
        while runtime is not None:
            if layer_index < 0 and isinstance(
                runtime.f_locals.get("layer_index"), int
            ):
                layer_index = int(runtime.f_locals["layer_index"])
            if runtime.f_code.co_name == "run_with_torch":
                mode = str(runtime.f_locals.get("mode", ""))
                tokens = int(runtime.f_locals.get("tokens", 0))
                break
            runtime = runtime.f_back
        if (
            layer_index == target_layer
            and mode == ORACLE_PHASE
            and tokens == ORACLE_TOKENS
        ):
            expert_output = frame.f_locals.get("input")
            if expert_output is not None:
                _oracle_tensor(
                    f"layer-{layer_index:03d}-intermediate-moe_sum-expert_values-"
                    f"token-{target_token:06d}-hidden-{target_hidden:04d}",
                    expert_output[target_token, :, target_hidden].contiguous(),
                    function=function,
                    layer_index=layer_index,
                    token_index=target_token,
                    hidden_index=target_hidden,
                )
        return None
    selected = {
        "linear_attention",
        "full_attention",
        "shared_expert",
        "native_moe_consumer_routed_moe",
        "layer_body",
        "final_logits",
    }
    if "/benchmarks/shape-lab/four_layer_mini_engine.py" not in filename:
        return None
    if event == "call":
        return _oracle_profile if function in selected else None
    if event == "return" and function == "final_logits":
        runtime = frame
        mode = ""
        tokens = 0
        while runtime is not None:
            if runtime.f_code.co_name == "run_with_torch":
                mode = str(runtime.f_locals.get("mode", ""))
                tokens = int(runtime.f_locals.get("tokens", 0))
                break
            runtime = runtime.f_back
        if (mode == ORACLE_PHASE and tokens == ORACLE_TOKENS):
            _ORACLE_FINAL_LOGITS_SEEN += 1
        primary_capture = (
            not _ORACLE_FINAL_LOGITS_CAPTURED
            and _ORACLE_FINAL_LOGITS_SEEN == ORACLE_FINAL_LOGITS_OCCURRENCE
        )
        decode_capture = (
            ORACLE_CAPTURE_DECODE_FINAL_LOGITS
            and not _ORACLE_DECODE_FINAL_LOGITS_CAPTURED
            and _ORACLE_FINAL_LOGITS_SEEN
            == ORACLE_DECODE_FINAL_LOGITS_OCCURRENCE
            and not primary_capture
        )
        if mode == ORACLE_PHASE and tokens == ORACLE_TOKENS and (
            primary_capture or decode_capture
        ):
            label_prefix = (
                "return-final_logits"
                if primary_capture
                else "return-decode-final_logits"
            )
            final_hidden = frame.f_locals.get("final_hidden")
            if final_hidden is not None:
                _oracle_tensor(
                    f"{label_prefix}-final_hidden",
                    final_hidden,
                    function=function,
                )
            if value is not None:
                _oracle_tensor(
                    f"{label_prefix}-output",
                    value,
                    function=function,
                )
            if primary_capture:
                _ORACLE_FINAL_LOGITS_CAPTURED = True
            else:
                _ORACLE_DECODE_FINAL_LOGITS_CAPTURED = True
        return _oracle_profile
    layer_index = int(frame.f_locals.get("layer_index", -1))
    all_layers_active = (
        ORACLE_ALL_LAYERS
        and layer_index in _ORACLE_ACTIVE_LAYERS
        and layer_index not in _ORACLE_FINISHED_LAYERS
    )
    legacy_active = (
        not ORACLE_ALL_LAYERS
        and _ORACLE_ACTIVE
        and not _ORACLE_FINISHED
        and layer_index == 0
    )
    if (
        event == "line"
        and function == "full_attention"
        and ORACLE_CAPTURE_FULL_ATTENTION_PRE_GATE
        and (all_layers_active or legacy_active)
        and layer_index not in _ORACLE_FULL_ATTENTION_PRE_GATE_CAPTURED
    ):
        # The first line event after the SDPA/grouped-BMM assignment observes
        # `attn_out` before the Python path overwrites it with sigmoid gating.
        # This is qualification-only and keeps the frozen engine source intact.
        attention = frame.f_locals.get("attn_out")
        if attention is not None:
            prefix = f"layer-{layer_index:03d}-" if ORACLE_ALL_LAYERS else ""
            _oracle_tensor(
                f"{prefix}intermediate-full_attention-attn_pre_gate",
                attention,
                function=function,
                layer_index=layer_index,
            )
            _ORACLE_FULL_ATTENTION_PRE_GATE_CAPTURED.add(layer_index)
        return _oracle_profile
    if event != "return" or function not in selected or not (
        all_layers_active or legacy_active
    ):
        return _oracle_profile

    prefix = f"layer-{layer_index:03d}-" if ORACLE_ALL_LAYERS else ""

    names = {
        "linear_attention": (
            "inp",
            "mixed_qkv",
            "conv_qkv",
            "core_attn_out",
            "final_state",
            "variance",
            "gated_out",
        ),
        "full_attention": (
            "inp",
            "qkv",
            "q_gate",
            "q",
            "gate",
            "k",
            "v",
            "k_cache",
            "v_cache",
            "q_grouped",
            "k_grouped",
            "v_grouped",
            "scores",
            "probs",
            "attn_grouped",
            "attn_out",
        ),
        "shared_expert": (
            "inp",
            "shared_input",
            "shared_gate",
            "gate",
            "up",
            "activated",
            "shared_out",
        ),
        "native_moe_consumer_routed_moe": (
            "inp",
            "scores",
            "indices",
            "routing",
            "topk_ids",
        ),
        "layer_body": (
            "inp",
            "h1",
            "attn_out",
            "after_attn",
            "h2",
            "scores",
            "indices",
            "shared_out",
            "moe_out",
        ),
    }[function]
    for name in names:
        candidate = frame.f_locals.get(name)
        if candidate is not None:
            _oracle_tensor(
                f"{prefix}return-{function}-{name}",
                candidate,
                function=function,
                layer_index=layer_index,
            )
    if function == "layer_body" and isinstance(value, tuple) and value:
        _oracle_tensor(
            f"{prefix}return-layer_body-output",
            value[0],
            function=function,
            layer_index=layer_index,
        )
        if ORACLE_ALL_LAYERS:
            _ORACLE_FINISHED_LAYERS.add(layer_index)
        else:
            _ORACLE_FINISHED = True
        _oracle_append(
            {"event": "native_layer_oracle_complete", "layer_index": layer_index}
        )
    elif function == "native_moe_consumer_routed_moe" and value is not None:
        _oracle_tensor(
            f"{prefix}return-layer_body-routed_moe",
            value,
            function=function,
            layer_index=layer_index,
        )
    elif value is not None:
        _oracle_tensor(
            f"{prefix}return-{function}-output",
            value,
            function=function,
            layer_index=layer_index,
        )
    return _oracle_profile


def _tensor_record(value: Any) -> dict[str, Any] | None:
    if not all(hasattr(value, attr) for attr in ("shape", "dtype", "device", "stride")):
        return None
    try:
        stride = value.stride() if callable(value.stride) else value.stride
        record = {
            "kind": "tensor",
            "shape": [int(item) for item in value.shape],
            "stride": [int(item) for item in stride],
            "dtype": str(value.dtype),
            "device": str(value.device),
            "element_size": int(value.element_size()),
        }
        if TRACE_POINTERS:
            storage = value.untyped_storage()
            record.update(
                {
                    "data_ptr": int(value.data_ptr()),
                    "storage_data_ptr": int(storage.data_ptr()),
                    "storage_nbytes": int(storage.nbytes()),
                    "storage_offset": int(value.storage_offset()),
                    "logical_nbytes": int(value.numel()) * int(value.element_size()),
                }
            )
        return record
    except Exception:
        return {"kind": "tensor", "repr": repr(value)[:256]}


def _argument_record(name: str, value: Any, abi_type: str | None) -> dict[str, Any]:
    tensor = _tensor_record(value)
    if tensor is not None:
        return {"name": name, "abi_type": abi_type, **tensor}
    if abi_type and abi_type.startswith("*"):
        return {
            "name": name,
            "abi_type": abi_type,
            "kind": "device_pointer",
            "is_null": value is None or value == 0,
        }
    return {
        "name": name,
        "abi_type": abi_type,
        "kind": "scalar",
        "value": _json_value(value),
    }


def _source_signature(kernel: Any) -> dict[str, str]:
    source = getattr(kernel, "src", None)
    signature = getattr(source, "signature", None)
    if not isinstance(signature, dict):
        return {}
    return {str(name): str(value) for name, value in signature.items()}


def _cache_files(kernel: Any) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for path_value in getattr(kernel, "metadata_group", {}).values():
        path = Path(path_value)
        if path.is_file() and path.suffix in {
            ".amdgcn",
            ".hsaco",
            ".json",
            ".llir",
            ".source",
            ".ttgir",
            ".ttir",
        }:
            paths.add(path.resolve())
    return [
        {
            "name": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(paths)
    ]


def _metadata_record(kernel: Any) -> dict[str, Any]:
    metadata = getattr(kernel, "metadata", None)
    target = getattr(metadata, "target", None)
    return {
        "name": str(getattr(metadata, "name", getattr(kernel, "name", "unknown"))),
        "num_warps": int(getattr(metadata, "num_warps", 0)),
        "num_ctas": int(getattr(metadata, "num_ctas", 1)),
        "num_stages": int(getattr(metadata, "num_stages", 0)),
        "shared": int(getattr(metadata, "shared", 0)),
        "warp_size": int(getattr(metadata, "warp_size", getattr(target, "warp_size", 0))),
        "launch_cooperative_grid": bool(
            getattr(metadata, "launch_cooperative_grid", False)
        ),
        "target": {
            "backend": str(getattr(target, "backend", "")),
            "arch": str(getattr(target, "arch", "")),
            "warp_size": int(getattr(target, "warp_size", 0)),
        },
    }


def _source_path(path: str) -> str:
    for marker in ("/benchmarks/shape-lab/", "/site-packages/"):
        if marker in path:
            prefix = "benchmarks/shape-lab/" if marker.startswith("/benchmarks") else ""
            return prefix + path.split(marker, 1)[1]
    return Path(path).name


def _caller_context() -> list[dict[str, Any]]:
    selected_names = {
        "layer_index",
        "model_layer",
        "mode",
        "tokens",
        "position_start",
        "position_end",
        "variant",
        "stage",
    }
    records: list[dict[str, Any]] = []
    frame = sys._getframe(1)
    while frame is not None and len(records) < 24:
        filename = frame.f_code.co_filename
        if "/benchmarks/shape-lab/" in filename or "/site-packages/vllm/" in filename:
            scalars = {
                name: _json_value(frame.f_locals[name])
                for name in selected_names
                if name in frame.f_locals
                and isinstance(frame.f_locals[name], (bool, int, float, str, type(None)))
            }
            records.append(
                {
                    "function": frame.f_code.co_name,
                    "source": _source_path(filename),
                    "line": int(frame.f_lineno),
                    "scalars": scalars,
                }
            )
        frame = frame.f_back
    return records


def _maybe_dump_tensor_registry() -> None:
    global _REGISTRY_WRITTEN
    if _REGISTRY_WRITTEN or not TENSOR_REGISTRY_PATH:
        return
    frame = sys._getframe(1)
    owner = None
    while frame is not None:
        if frame.f_code.co_name == "run_with_torch":
            owner = frame
            break
        frame = frame.f_back
    if owner is None:
        return
    local_values = owner.f_locals
    phase = str(local_values.get("mode", ""))
    if TENSOR_REGISTRY_PHASE and phase != TENSOR_REGISTRY_PHASE:
        return
    tokens = int(local_values.get("tokens", 0))
    if TENSOR_REGISTRY_TOKENS is not None and tokens != int(TENSOR_REGISTRY_TOKENS):
        return
    if phase == "decode" and int(local_values.get("tokens", 0)) != 1:
        return

    tensors: list[dict[str, Any]] = []
    visited_containers: set[int] = set()

    def visit(name: str, value: Any) -> None:
        tensor = _tensor_record(value)
        if tensor is not None:
            tensors.append({"name": name, **tensor})
            return
        if isinstance(value, dict):
            identity = id(value)
            if identity in visited_containers:
                return
            visited_containers.add(identity)
            for key, child in value.items():
                visit(f"{name}.{key}", child)
            return
        if isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in visited_containers:
                return
            visited_containers.add(identity)
            for index, child in enumerate(value):
                visit(f"{name}.{index}", child)

    # Prefer the durable owner dictionaries over loop-local aliases.  The
    # latter are reassigned on every layer and, on the final layer, may point at
    # the same storage as ``layer_weights.39``.  Visiting the owners first keeps
    # stable, layer-qualified names in the registry without depending on local
    # insertion order.  The visited-container guard then suppresses aliases.
    owner_names = ("layer_weights", "global_tensors")
    for name in owner_names:
        if name in local_values:
            visit(name, local_values[name])
    for name, value in local_values.items():
        if name not in owner_names:
            visit(str(name), value)
    payload = {
        "schema": "aima-amd395-qwen36/native-tensor-registry/v1",
        "capture": {
            "phase": phase,
            "tokens": tokens,
            "position_start": int(local_values.get("position_start", 0)),
            "position_end": int(local_values.get("position_end", 0)),
        },
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    destination = Path(TENSOR_REGISTRY_PATH).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    _REGISTRY_WRITTEN = True


def _install() -> None:
    if not TRACE_PATH and not ORACLE_DIR:
        return
    try:
        from triton.runtime import autotuner
        from triton.runtime import driver
        from triton.runtime import jit
    except Exception as exc:
        _append({"event": "trace_install_error", "error": repr(exc)})
        return

    original = jit.JITFunction.run
    original_autotuner = autotuner.Autotuner.run

    @functools.wraps(original)
    def traced_run(self: Any, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
        global _SEQUENCE, _ORACLE_ACTIVE, _ORACLE_LAUNCH_SEQUENCE
        kernel = original(self, *args, grid=grid, warmup=warmup, **kwargs)
        if warmup or kernel is None:
            return kernel
        try:
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            _maybe_dump_tensor_registry()
            device = driver.active.get_current_device()
            _, _, _, _, binder = self.device_caches[device]
            bound_args, _, _ = binder(*args, **kwargs)
            oracle_context = _oracle_runtime_context() if ORACLE_DIR else None
            if (
                oracle_context is not None
                and oracle_context["mode"] == ORACLE_PHASE
                and oracle_context["tokens"] == ORACLE_TOKENS
                and (ORACLE_ALL_LAYERS or not _ORACLE_FINISHED)
            ):
                oracle_launch_sequence = _ORACLE_LAUNCH_SEQUENCE
                _ORACLE_LAUNCH_SEQUENCE += 1
                oracle_layer_index = int(oracle_context.get("layer_index", -1))
                if ORACLE_ALL_LAYERS and oracle_layer_index >= 0:
                    _ORACLE_ACTIVE_LAYERS.add(oracle_layer_index)
                    local_launch_sequence = _ORACLE_LAYER_LAUNCH_SEQUENCES.get(
                        oracle_layer_index, 0
                    )
                    _ORACLE_LAYER_LAUNCH_SEQUENCES[oracle_layer_index] = (
                        local_launch_sequence + 1
                    )
                else:
                    _ORACLE_ACTIVE = True
                    local_launch_sequence = oracle_launch_sequence
                capture_launch = (
                    ORACLE_ALL_LAYERS
                    and oracle_layer_index >= 0
                    and 0 <= local_launch_sequence < 13
                ) or (
                    not ORACLE_ALL_LAYERS
                    and oracle_launch_sequence < (13 if ORACLE_PHASE == "prefill" else 10)
                )
                if capture_launch and ORACLE_CAPTURE_LAUNCH_ARGUMENTS:
                    prefix = (
                        f"layer-{oracle_layer_index:03d}-"
                        if ORACLE_ALL_LAYERS
                        else ""
                    )
                    for name, value in bound_args.items():
                        _oracle_tensor(
                            f"{prefix}launch-{local_launch_sequence:03d}-{name}",
                            value,
                            launch_sequence=oracle_launch_sequence,
                            local_launch_sequence=local_launch_sequence,
                            layer_index=oracle_layer_index,
                            kernel_hash=str(getattr(kernel, "hash", "")),
                            kernel_symbol=str(
                                getattr(
                                    getattr(kernel, "metadata", None),
                                    "name",
                                    getattr(kernel, "name", "unknown"),
                                )
                            ),
                            argument=str(name),
                        )
            actual_grid = grid(bound_args) if callable(grid) else grid
            actual_grid = tuple(actual_grid) + (1,) * (3 - len(actual_grid))
            signature = _source_signature(kernel)
            arguments = [
                _argument_record(
                    str(name),
                    value,
                    signature.get(str(name)),
                )
                for name, value in bound_args.items()
            ]
            structural_arguments = [
                {
                    key: value
                    for key, value in argument.items()
                    if key
                    not in {
                        "value",
                        "data_ptr",
                        "storage_data_ptr",
                        "storage_nbytes",
                        "storage_offset",
                        "logical_nbytes",
                    }
                }
                for argument in arguments
            ]
            identity = json.dumps(
                {
                    "hash": str(getattr(kernel, "hash", "")),
                    "grid": actual_grid,
                    "arguments": structural_arguments,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            identity_sha256 = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            with _LOCK:
                first_structural_observation = identity_sha256 not in _SEEN
                if not first_structural_observation and not TRACE_ALL_LAUNCHES:
                    return kernel
                _SEEN.add(identity_sha256)
                sequence = _SEQUENCE
                _SEQUENCE += 1
            _append(
                {
                    "event": "triton_launch",
                    "sequence": sequence,
                    "captured_unix_ns": time.time_ns(),
                    "captured_monotonic_ns": time.monotonic_ns(),
                    "first_structural_observation": first_structural_observation,
                    "identity_sha256": identity_sha256,
                    "kernel_hash": str(getattr(kernel, "hash", "")),
                    "python_qualname": str(getattr(self, "__qualname__", "")),
                    "caller_context": _caller_context(),
                    "grid": [int(item) for item in actual_grid],
                    "metadata": _metadata_record(kernel),
                    "signature": signature,
                    "arguments": arguments,
                    "cache_files": _cache_files(kernel) if first_structural_observation else [],
                }
            )
        except Exception as exc:
            _append(
                {
                    "event": "trace_launch_error",
                    "python_qualname": str(getattr(self, "__qualname__", "")),
                    "error": repr(exc),
                }
            )
        return kernel

    jit.JITFunction.run = traced_run

    @functools.wraps(original_autotuner)
    def traced_autotuner_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_autotuner(self, *args, **kwargs)
        try:
            best = self.best_config
            timings = getattr(self, "configs_timings", {})
            _append(
                {
                    "event": "triton_autotune_selection",
                    "captured_unix_ns": time.time_ns(),
                    "python_qualname": str(getattr(self.base_fn, "__qualname__", "")),
                    "key_names": [str(item) for item in self.keys],
                    "arguments": [
                        _argument_record(str(name), value, None)
                        for name, value in zip(self.arg_names, args)
                    ],
                    "best_config": _json_value(best.all_kwargs()),
                    "config_timings_ms": [
                        {
                            "config": _json_value(config.all_kwargs()),
                            "quantiles": _json_value(quantiles),
                        }
                        for config, quantiles in timings.items()
                    ],
                }
            )
        except Exception as exc:
            _append(
                {
                    "event": "trace_autotune_error",
                    "python_qualname": str(getattr(self.base_fn, "__qualname__", "")),
                    "error": repr(exc),
                }
            )
        return result

    autotuner.Autotuner.run = traced_autotuner_run
    if ORACLE_DIR:
        sys.settrace(_oracle_profile)
        threading.settrace(_oracle_profile)
        _oracle_append(
            {
                "event": "native_layer_oracle_installed",
                "scope": (
                    "all_q8192_decode_layers"
                    if ORACLE_ALL_LAYERS
                    else f"first_q8192_{ORACLE_PHASE}_linear_attention_layer"
                ),
            }
        )
    _append(
        {
            "event": "trace_installed",
            "pid": os.getpid(),
            "all_launches": TRACE_ALL_LAUNCHES,
            "device_pointers": TRACE_POINTERS,
        }
    )


_install()
