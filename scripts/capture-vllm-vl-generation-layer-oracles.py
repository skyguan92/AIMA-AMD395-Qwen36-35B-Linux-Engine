#!/usr/bin/env python3
"""Capture fixed-vLLM decode-layer rows at the two VL generation drifts."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_generation_layer_oracle import (  # noqa: E402
    BOUNDARY_NAMES,
    FIRST_DECODE_LINEAR_OUTPUT_INDEX,
    GENERATION_LAYER_ORACLE_SCHEMA,
    HIDDEN_SIZE,
    LAYER0_TAIL_BOUNDARY_SPECS,
    LINEAR_ATTENTION_BOUNDARY_SPECS,
    validate_generation_layer_oracle_manifest,
)
from aima_engine.vl_generation_oracle import (  # noqa: E402
    CASE_CONTRACTS,
    CASE_ORDER,
    MODEL_VOCABULARY_SIZE,
    validate_generation_oracle_manifest,
)
from aima_engine.vl_oracle import (  # noqa: E402
    canonical_int_list_sha256,
    write_raw_tensor,
)
from aima_engine.vl_reference import (  # noqa: E402
    MODEL_REVISION,
    PINNED_PACKAGES,
    atomic_json,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
)


GENERATION_CAPTURE = ROOT / "scripts/capture-vllm-vl-generation-oracles.py"
STATE_ATTRIBUTE = "_aima_vl_generation_layer_oracle_state"
GENERATION_LAYER_DIAGNOSTIC_SCHEMA = (
    "aima-amd395-qwen36/vl-generation-layer-diagnostic/v1"
)
FULL_ATTENTION_LAYER = 3
LINEAR_ATTENTION_LAYER = 0
FIRST_DIVERGENCE_FULL_ATTENTION_LAYERS = {
    "tool_forced_image": 11,
    "tool_auto_image": 3,
}
FIRST_DIVERGENCE_LINEAR_ATTENTION_LAYERS = {
    "tool_forced_image": 5,
    "tool_auto_image": 10,
}
FULL_ATTENTION_DECODE_COMPONENT_NAMES = (
    "query",
    "key_cache",
    "value_cache",
    "block_table",
    "sequence_lengths",
    "query_starts",
    "k_descale",
    "v_descale",
    "output",
)
FULL_ATTENTION_PROJECTION_COMPONENT_NAMES = (
    "qkv_projection",
    *FULL_ATTENTION_DECODE_COMPONENT_NAMES,
    "gated_attention",
    "projected_attention",
    "attention_residual",
    "post_attention_norm",
    *(
        name
        for name in LAYER0_TAIL_BOUNDARY_SPECS
        if name not in {"attention_residual", "post_attention_norm"}
    ),
)


def parse_case_linear_attention_layers(
    values: list[str] | None,
) -> dict[str, int] | None:
    """Parse an exact two-case diagnostic linear-layer selection."""

    if not values:
        return None
    layers: dict[str, int] = {}
    for value in values:
        case_id, separator, layer_text = value.partition("=")
        if separator != "=" or case_id not in CASE_ORDER:
            raise ValueError(
                "diagnostic linear-attention layer must be CASE_ID=LAYER"
            )
        if case_id in layers:
            raise ValueError(
                f"duplicate diagnostic linear-attention case: {case_id}"
            )
        try:
            layer_index = int(layer_text)
        except ValueError as error:
            raise ValueError(
                f"diagnostic linear-attention layer is not an integer: {value}"
            ) from error
        if layer_index < 0 or layer_index >= 40 or layer_index % 4 == 3:
            raise ValueError(
                f"diagnostic linear-attention layer is unsupported: {value}"
            )
        layers[case_id] = layer_index
    missing = [case_id for case_id in CASE_ORDER if case_id not in layers]
    if missing:
        raise ValueError(
            "diagnostic linear-attention layer mapping is incomplete: "
            + ", ".join(missing)
        )
    return {case_id: layers[case_id] for case_id in CASE_ORDER}


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load capture dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _first_tensor(value: Any) -> Any | None:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _shape_elements(shape: tuple[int, ...]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def _find_model_root(model: Any) -> Any:
    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, "visual") and hasattr(candidate, "language_model"):
            return candidate
        for name in ("model", "module"):
            child = getattr(candidate, name, None)
            if child is not None:
                queue.append(child)
    raise RuntimeError("could not locate the multimodal model root")


def _remove_hooks(root: Any, state: dict[str, Any]) -> None:
    for handle in state.get("handles", []):
        handle.remove()
    state["handles"] = []
    gdn_module = state.get("gdn_module")
    if gdn_module is not None:
        gdn_module.causal_conv1d_update = state[
            "original_causal_conv1d_update"
        ]
        gdn_module.fused_recurrent_gated_delta_rule_packed_decode = state[
            "original_packed_decode"
        ]
    router = state.get("router")
    if router is not None:
        router.select_experts = state["original_router_select_experts"]
    full_attention_router = state.get("full_attention_router")
    if full_attention_router is not None:
        full_attention_router.select_experts = state[
            "original_full_attention_router_select_experts"
        ]
    fused_moe_module = state.get("fused_moe_module")
    if fused_moe_module is not None:
        fused_moe_module.apply_moe_activation = state[
            "original_apply_moe_activation"
        ]
    modular_moe_module = state.get("modular_moe_module")
    if modular_moe_module is not None:
        modular_moe_module.apply_moe_activation = state[
            "original_modular_apply_moe_activation"
        ]
    custom_ops = state.get("custom_ops")
    if custom_ops is not None:
        custom_ops.moe_sum = state["original_moe_sum"]
    triton_attention_module = state.get("triton_attention_module")
    if triton_attention_module is not None:
        triton_attention_module.unified_attention = state[
            "original_unified_attention"
        ]
    if hasattr(root, STATE_ATTRIBUTE):
        delattr(root, STATE_ATTRIBUTE)


class InstallGenerationLayerHooks:
    """Serializable worker hook for one exact generated-token boundary set."""

    def __init__(
        self,
        *,
        case_id: str,
        output_index: int,
        linear_attention_layer_index: int = LINEAR_ATTENTION_LAYER,
        full_attention_layer_index: int = FULL_ATTENTION_LAYER,
        capture_full_attention_projection: bool = False,
    ) -> None:
        self.case_id = case_id
        self.output_index = output_index
        self.linear_attention_layer_index = linear_attention_layer_index
        self.full_attention_layer_index = full_attention_layer_index
        self.capture_full_attention_projection = (
            capture_full_attention_projection
        )

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        if self.output_index <= 0:
            raise RuntimeError("decode layer capture requires output index > 0")
        if (
            self.linear_attention_layer_index < 0
            or self.linear_attention_layer_index >= 40
            or self.linear_attention_layer_index % 4 == 3
        ):
            raise RuntimeError("decode linear-attention layer is unsupported")
        if (
            self.full_attention_layer_index < 0
            or self.full_attention_layer_index >= 40
            or self.full_attention_layer_index % 4 != 3
        ):
            raise RuntimeError("decode full-attention layer is unsupported")
        root = _find_model_root(model)
        previous = getattr(root, STATE_ATTRIBUTE, None)
        if isinstance(previous, dict):
            _remove_hooks(root, previous)
        language = root.language_model
        if len(language.model.layers) != 40:
            raise RuntimeError("language layer count differs from the frozen model")
        state: dict[str, Any] = {
            "case_id": self.case_id,
            "target_output_index": self.output_index,
            "captures": {},
            "first_decode_captures": {},
            "linear_captures": {},
            "first_decode_linear_captures": {},
            "layer0_tail_captures": {},
            "first_decode_layer0_tail_captures": {},
            "full_attention_captures": {},
            "first_decode_full_attention_captures": {},
            "full_attention_metadata": {},
            "first_decode_full_attention_metadata": {},
            "linear_singleton_calls": 0,
            "linear_capture_kind": None,
            "linear_attention_layer_index": (
                self.linear_attention_layer_index
            ),
            "full_attention_singleton_calls": 0,
            "full_attention_capture_kind": None,
            "full_attention_layer_index": self.full_attention_layer_index,
            "full_attention_component_names": (
                FULL_ATTENTION_PROJECTION_COMPONENT_NAMES
                if self.capture_full_attention_projection
                else FULL_ATTENTION_DECODE_COMPONENT_NAMES
            ),
            "boundary_singleton_calls": {name: 0 for name in BOUNDARY_NAMES},
            "logits_prefill_calls": 0,
            "logits_decode_calls": 0,
            "captured_logits_output_index": None,
            "first_decode_logits": None,
            "target_logits": None,
            "target_logits_sha256": None,
            "target_logits_top1_token_id": None,
            "handles": [],
        }

        linear_layer = language.model.layers[
            self.linear_attention_layer_index
        ]
        if getattr(linear_layer, "layer_type", None) != "linear_attention":
            raise RuntimeError(
                "generation diagnostic layer is not linear attention"
            )
        linear = linear_layer.linear_attn
        mlp = linear_layer.mlp
        shared_expert = mlp.shared_expert
        if shared_expert is None:
            raise RuntimeError(
                "generation diagnostic linear shared expert is missing"
            )
        router = mlp.experts.router
        full_attention_layer = language.model.layers[
            self.full_attention_layer_index
        ]
        if (
            getattr(full_attention_layer, "layer_type", None)
            != "full_attention"
        ):
            raise RuntimeError(
                "generation diagnostic layer is not full attention"
            )
        full_attention_module = full_attention_layer.self_attn
        full_attention_mlp = full_attention_layer.mlp
        full_attention_shared_expert = full_attention_mlp.shared_expert
        if full_attention_shared_expert is None:
            raise RuntimeError(
                "generation diagnostic full-attention shared expert is missing"
            )
        full_attention_router = full_attention_mlp.experts.router
        gdn_module = __import__(
            "vllm.model_executor.layers.mamba.gdn_linear_attn",
            fromlist=["causal_conv1d_update"],
        )
        fused_moe_module = __import__(
            "vllm.model_executor.layers.fused_moe.fused_moe",
            fromlist=["apply_moe_activation"],
        )
        modular_moe_module = __import__(
            "vllm.model_executor.layers.fused_moe.modular_kernel",
            fromlist=["apply_moe_activation"],
        )
        custom_ops = __import__("vllm._custom_ops", fromlist=["moe_sum"])
        triton_attention_module = __import__(
            "vllm.v1.attention.backends.triton_attn",
            fromlist=["unified_attention"],
        )
        state.update(
            {
                "gdn_module": gdn_module,
                "original_causal_conv1d_update": (
                    gdn_module.causal_conv1d_update
                ),
                "original_packed_decode": (
                    gdn_module.fused_recurrent_gated_delta_rule_packed_decode
                ),
                "linear_conv_weight_pointer": linear.conv1d.weight.data_ptr(),
                "linear_a_log_pointer": linear.A_log.data_ptr(),
                "router": router,
                "original_router_select_experts": router.select_experts,
                "full_attention_router": full_attention_router,
                "original_full_attention_router_select_experts": (
                    full_attention_router.select_experts
                ),
                "fused_moe_module": fused_moe_module,
                "original_apply_moe_activation": (
                    fused_moe_module.apply_moe_activation
                ),
                "modular_moe_module": modular_moe_module,
                "original_modular_apply_moe_activation": (
                    modular_moe_module.apply_moe_activation
                ),
                "custom_ops": custom_ops,
                "original_moe_sum": custom_ops.moe_sum,
                "triton_attention_module": triton_attention_module,
                "original_unified_attention": (
                    triton_attention_module.unified_attention
                ),
            }
        )

        def capture_linear(name: str, value: Any) -> None:
            capture_kind = state["linear_capture_kind"]
            if capture_kind is None:
                return
            captures = (
                state["linear_captures"]
                if capture_kind == "target"
                else state["first_decode_linear_captures"]
            )
            if name in captures:
                return
            tensor = _first_tensor(value)
            if tensor is None:
                raise RuntimeError(
                    f"generation linear-attention boundary has no tensor: {name}"
                )
            expected_shape, expected_dtype, _element_size = (
                LINEAR_ATTENTION_BOUNDARY_SPECS[name]
            )
            expected_elements = 1
            for dimension in expected_shape:
                expected_elements *= dimension
            if tensor.numel() != expected_elements or str(tensor.dtype) != expected_dtype:
                raise RuntimeError(
                    "generation linear-attention boundary geometry changed: "
                    f"{name}/{list(tensor.shape)}/{tensor.dtype}"
                )
            captures[name] = (
                tensor.detach().reshape(expected_shape).contiguous().cpu()
            )

        def capture_layer0_tail(name: str, value: Any) -> None:
            capture_kind = state["linear_capture_kind"]
            if capture_kind is None:
                return
            captures = (
                state["layer0_tail_captures"]
                if capture_kind == "target"
                else state["first_decode_layer0_tail_captures"]
            )
            if name in captures:
                return
            tensor = _first_tensor(value)
            if tensor is None:
                raise RuntimeError(
                    f"generation layer-0 tail boundary has no tensor: {name}"
                )
            expected_shape, expected_dtype, _element_size = (
                LAYER0_TAIL_BOUNDARY_SPECS[name]
            )
            expected_elements = 1
            for dimension in expected_shape:
                expected_elements *= dimension
            if tensor.numel() != expected_elements or str(tensor.dtype) != expected_dtype:
                raise RuntimeError(
                    "generation layer-0 tail boundary geometry changed: "
                    f"{name}/{list(tensor.shape)}/{tensor.dtype}"
                )
            captures[name] = (
                tensor.detach().reshape(expected_shape).contiguous().cpu()
            )

        def capture_full_attention(name: str, value: Any) -> None:
            capture_kind = state["full_attention_capture_kind"]
            if capture_kind is None:
                return
            captures = (
                state["full_attention_captures"]
                if capture_kind == "target"
                else state["first_decode_full_attention_captures"]
            )
            if name in captures:
                return
            tensor = _first_tensor(value)
            if tensor is None:
                raise RuntimeError(
                    f"generation full-attention boundary has no tensor: {name}"
                )
            expected_shape = None
            if name in LAYER0_TAIL_BOUNDARY_SPECS:
                shape, dtype_name, _element_size = (
                    LAYER0_TAIL_BOUNDARY_SPECS[name]
                )
                expected_shape = tuple(shape)
                expected_dtype = {
                    "torch.bfloat16": torch.bfloat16,
                    "torch.float32": torch.float32,
                    "torch.int32": torch.int32,
                }[dtype_name]
            else:
                expected_dtype = (
                    torch.int32
                    if name
                    in {"block_table", "sequence_lengths", "query_starts"}
                    else (
                        torch.float32
                        if name.endswith("_descale")
                        else torch.bfloat16
                    )
                )
            if tensor.dtype != expected_dtype or (
                expected_shape is not None
                and tensor.numel() != _shape_elements(expected_shape)
            ):
                raise RuntimeError(
                    "generation full-attention boundary geometry changed: "
                    f"{name}/{list(tensor.shape)}/{tensor.dtype}"
                )
            if expected_shape is not None:
                tensor = tensor.reshape(expected_shape)
            captures[name] = tensor.detach().contiguous().cpu()

        def full_attention_qkv_projection_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            if (
                not self.capture_full_attention_projection
                or state["full_attention_capture_kind"] is None
            ):
                return
            tensor = _first_tensor(output)
            if (
                tensor is None
                or tensor.ndim != 2
                or tensor.shape != (1, 9_216)
            ):
                raise RuntimeError(
                    "generation full-attention QKV projection geometry changed"
                )
            capture_full_attention("qkv_projection", tensor)

        def full_attention_o_proj_pre_hook(
            _module: Any, args: Any
        ) -> None:
            if (
                not self.capture_full_attention_projection
                or state["full_attention_capture_kind"] is None
            ):
                return
            tensor = _first_tensor(args)
            if tensor is None or tensor.shape != (1, 4_096):
                raise RuntimeError(
                    "generation full-attention gated output geometry changed"
                )
            capture_full_attention("gated_attention", tensor)

        def full_attention_o_proj_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            if (
                not self.capture_full_attention_projection
                or state["full_attention_capture_kind"] is None
            ):
                return
            tensor = _first_tensor(output)
            if tensor is None or tensor.shape != (1, HIDDEN_SIZE):
                raise RuntimeError(
                    "generation full-attention projected output geometry changed"
                )
            capture_full_attention("projected_attention", tensor)

        def full_attention_post_attention_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            if (
                not self.capture_full_attention_projection
                or state["full_attention_capture_kind"] is None
            ):
                return
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError(
                    "generation full-attention post-attention norm contract changed"
                )
            post_attention_norm = _first_tensor(output[0])
            attention_residual = _first_tensor(output[1])
            if (
                post_attention_norm is None
                or post_attention_norm.shape != (1, HIDDEN_SIZE)
                or attention_residual is None
                or attention_residual.shape != (1, HIDDEN_SIZE)
            ):
                raise RuntimeError(
                    "generation full-attention residual geometry changed"
                )
            capture_full_attention("attention_residual", attention_residual)
            capture_full_attention("post_attention_norm", post_attention_norm)

        def full_attention_input_norm_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            tensor = _first_tensor(output)
            if tensor is None or tensor.ndim != 2 or tensor.shape[-1] != HIDDEN_SIZE:
                raise RuntimeError(
                    "generation full-attention input norm geometry changed"
                )
            if tensor.shape[0] != 1:
                return
            state["full_attention_singleton_calls"] += 1
            decode_call = state["full_attention_singleton_calls"]
            if decode_call == FIRST_DECODE_LINEAR_OUTPUT_INDEX:
                state["full_attention_capture_kind"] = "first_decode"
            elif decode_call == self.output_index:
                state["full_attention_capture_kind"] = "target"
            else:
                state["full_attention_capture_kind"] = None

        def full_attention_output_hook(
            _module: Any, _args: Any, _output: Any
        ) -> None:
            state["full_attention_capture_kind"] = None

        def instrumented_unified_attention(*args: Any, **kwargs: Any) -> Any:
            result = state["original_unified_attention"](*args, **kwargs)
            capture_kind = state["full_attention_capture_kind"]
            if capture_kind is None:
                return result

            def argument(index: int, name: str) -> Any:
                return kwargs.get(name, args[index] if len(args) > index else None)

            query = argument(0, "q")
            key_cache = argument(1, "k")
            value_cache = argument(2, "v")
            output = argument(3, "out")
            query_starts = argument(4, "cu_seqlens_q")
            max_seqlen_q = argument(5, "max_seqlen_q")
            sequence_lengths = argument(6, "seqused_k")
            max_seqlen_k = argument(7, "max_seqlen_k")
            softmax_scale = argument(8, "softmax_scale")
            causal = argument(9, "causal")
            window_size = argument(10, "window_size")
            block_table = argument(11, "block_table")
            softcap = argument(12, "softcap")
            q_descale = argument(13, "q_descale")
            k_descale = argument(14, "k_descale")
            v_descale = argument(15, "v_descale")
            sequence_threshold_3d = argument(16, "seq_threshold_3D")
            softmax_segments = argument(17, "num_par_softmax_segments")
            segment_output = argument(18, "softmax_segm_output")
            segment_max = argument(19, "softmax_segm_max")
            segment_expsum = argument(20, "softmax_segm_expsum")
            tensors = (
                query,
                key_cache,
                value_cache,
                output,
                query_starts,
                sequence_lengths,
                block_table,
                k_descale,
                v_descale,
                segment_output,
                segment_max,
                segment_expsum,
            )
            if not all(isinstance(value, torch.Tensor) for value in tensors):
                raise RuntimeError(
                    "generation unified-attention tensor ABI changed"
                )
            if (
                query.shape != (1, 16, 256)
                or output.shape != query.shape
                or key_cache.ndim != 4
                or value_cache.shape != key_cache.shape
                or key_cache.shape[2:] != (2, 256)
                or block_table.ndim != 2
                or block_table.shape[0] != 1
                or sequence_lengths.numel() != 1
                or query_starts.numel() != 2
                or q_descale is not None
                or k_descale.shape != (1, 2)
                or v_descale.shape != (1, 2)
                or k_descale.dtype != torch.float32
                or v_descale.dtype != torch.float32
                or not isinstance(sequence_threshold_3d, int)
                or not isinstance(softmax_segments, int)
                or sequence_threshold_3d <= 0
                or softmax_segments != 16
                or segment_output.shape
                != (sequence_threshold_3d, 16, 16, 256)
                or segment_max.shape
                != (sequence_threshold_3d, 16, 16)
                or segment_expsum.shape != segment_max.shape
                or segment_output.dtype != torch.float32
                or segment_max.dtype != torch.float32
                or segment_expsum.dtype != torch.float32
            ):
                raise RuntimeError(
                    "generation unified-attention singleton geometry changed"
                )
            sequence_length = int(sequence_lengths.reshape(-1)[0].item())
            block_size = int(key_cache.shape[1])
            logical_blocks = (sequence_length + block_size - 1) // block_size
            physical_blocks = block_table[0, :logical_blocks].to(torch.int64)
            if (
                logical_blocks == 0
                or physical_blocks.numel() != logical_blocks
                or int(physical_blocks.min().item()) < 0
                or int(physical_blocks.max().item()) >= key_cache.shape[0]
            ):
                raise RuntimeError(
                    "generation unified-attention block table changed"
                )
            compact_key = key_cache.index_select(0, physical_blocks)
            compact_value = value_cache.index_select(0, physical_blocks)
            identity_table = torch.arange(
                logical_blocks, device=block_table.device, dtype=torch.int32
            ).reshape(1, logical_blocks)
            capture_full_attention("query", query)
            capture_full_attention("key_cache", compact_key)
            capture_full_attention("value_cache", compact_value)
            capture_full_attention("block_table", identity_table)
            capture_full_attention("sequence_lengths", sequence_lengths)
            capture_full_attention("query_starts", query_starts)
            capture_full_attention("k_descale", k_descale)
            capture_full_attention("v_descale", v_descale)
            capture_full_attention("output", output)
            metadata = {
                "layer_index": self.full_attention_layer_index,
                "sequence_length": sequence_length,
                "block_size": block_size,
                "logical_blocks": logical_blocks,
                "query_heads": int(query.shape[1]),
                "kv_heads": int(key_cache.shape[2]),
                "head_size": int(query.shape[2]),
                "max_seqlen_q": int(max_seqlen_q),
                "max_seqlen_k": int(max_seqlen_k),
                "softmax_scale": float(softmax_scale),
                "causal": bool(causal),
                "window_size": [int(value) for value in window_size],
                "softcap": float(softcap),
                "sequence_threshold_3d": int(sequence_threshold_3d),
                "softmax_segments": int(softmax_segments),
                "attention_path": "segmented_3d_plus_reduce",
                "key_cache_source_stride": [
                    int(value) for value in key_cache.stride()
                ],
                "value_cache_source_stride": [
                    int(value) for value in value_cache.stride()
                ],
                "physical_block_ids": [
                    int(value) for value in physical_blocks.detach().cpu().tolist()
                ],
            }
            destination = (
                state["full_attention_metadata"]
                if capture_kind == "target"
                else state["first_decode_full_attention_metadata"]
            )
            if destination and destination != metadata:
                raise RuntimeError(
                    "generation unified-attention metadata changed within a decode"
                )
            destination.update(metadata)
            return result

        def linear_input_norm_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            tensor = _first_tensor(output)
            if tensor is None or tensor.ndim != 2 or tensor.shape[-1] != HIDDEN_SIZE:
                raise RuntimeError(
                    "generation linear input norm geometry changed"
                )
            if tensor.shape[0] != 1:
                return
            state["linear_singleton_calls"] += 1
            decode_call = state["linear_singleton_calls"]
            if decode_call == FIRST_DECODE_LINEAR_OUTPUT_INDEX:
                state["linear_capture_kind"] = "first_decode"
            elif decode_call == self.output_index:
                state["linear_capture_kind"] = "target"
            else:
                state["linear_capture_kind"] = None
            capture_linear("input_norm", tensor)

        def qkvz_projection_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            if state["linear_capture_kind"] is None:
                return
            tensor = _first_tensor(output)
            if tensor is None or tensor.shape != (1, 12_288):
                raise RuntimeError("generation QKVZ projection geometry changed")
            capture_linear("qkv_projection", tensor[:, :8_192])
            capture_linear("z_projection", tensor[:, 8_192:])

        def ba_projection_hook(_module: Any, _args: Any, output: Any) -> None:
            if state["linear_capture_kind"] is None:
                return
            tensor = _first_tensor(output)
            if tensor is None or tensor.shape != (1, 64):
                raise RuntimeError("generation BA projection geometry changed")
            b, a = tensor.chunk(2, dim=-1)
            capture_linear("a_projection", a)
            capture_linear("b_projection", b)

        def gated_norm_hook(_module: Any, _args: Any, output: Any) -> None:
            capture_linear("gated_norm", output)

        def attention_output_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            capture_linear("attention_output", output)

        def post_attention_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError(
                    "generation post-attention norm contract changed"
                )
            capture_layer0_tail("post_attention_norm", output[0])
            capture_layer0_tail("attention_residual", output[1])

        def experts_hook(_module: Any, _args: Any, output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError(
                    "generation linear experts contract changed"
                )
            capture_layer0_tail("shared_moe_output", output[0])
            capture_layer0_tail("routed_moe_output", output[1])

        def router_logits_hook(_module: Any, _args: Any, output: Any) -> None:
            capture_layer0_tail("router_logits", output)

        def shared_gate_hook(_module: Any, _args: Any, output: Any) -> None:
            capture_layer0_tail("shared_gate_logits", output)

        def shared_gate_up_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            capture_layer0_tail("shared_gate_up_projection", output)

        def shared_activation_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            capture_layer0_tail("shared_activation", output)

        def shared_down_hook(_module: Any, _args: Any, output: Any) -> None:
            capture_layer0_tail("shared_down_projection", output)

        def mlp_hook(_module: Any, _args: Any, output: Any) -> None:
            capture_layer0_tail("combined_moe_output", output)

        def full_attention_experts_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError(
                    "generation full-attention experts contract changed"
                )
            capture_full_attention("shared_moe_output", output[0])
            capture_full_attention("routed_moe_output", output[1])

        def full_attention_router_logits_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            capture_full_attention("router_logits", output)

        def full_attention_shared_gate_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            capture_full_attention("shared_gate_logits", output)

        def full_attention_shared_gate_up_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            capture_full_attention("shared_gate_up_projection", output)

        def full_attention_shared_activation_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            capture_full_attention("shared_activation", output)

        def full_attention_shared_down_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            capture_full_attention("shared_down_projection", output)

        def full_attention_mlp_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            capture_full_attention("combined_moe_output", output)

        def linear_output_hook(_module: Any, _args: Any, _output: Any) -> None:
            state["linear_capture_kind"] = None

        def instrumented_causal_conv1d_update(
            *args: Any, **kwargs: Any
        ) -> Any:
            weight = args[2] if len(args) > 2 else kwargs.get("weight")
            is_selected_linear_layer = (
                isinstance(weight, torch.Tensor)
                and weight.data_ptr() == state["linear_conv_weight_pointer"]
            )
            if (
                state["linear_capture_kind"] is None
                or not is_selected_linear_layer
            ):
                return state["original_causal_conv1d_update"](*args, **kwargs)
            conv_state = args[1] if len(args) > 1 else kwargs.get("conv_state")
            indices = kwargs.get("conv_state_indices")
            if not isinstance(conv_state, torch.Tensor) or not isinstance(
                indices, torch.Tensor
            ) or indices.numel() != 1:
                raise RuntimeError(
                    "generation linear conv state geometry changed"
                )
            state_index = int(indices.reshape(-1)[0].item())
            capture_linear("conv_state_before", conv_state[state_index])
            output = state["original_causal_conv1d_update"](*args, **kwargs)
            capture_linear("post_conv_mixed_qkv", output)
            capture_linear("conv_state_after", conv_state[state_index])
            return output

        def instrumented_packed_decode(*args: Any, **kwargs: Any) -> Any:
            a_log = args[3] if len(args) > 3 else kwargs.get("A_log")
            is_selected_linear_layer = (
                isinstance(a_log, torch.Tensor)
                and a_log.data_ptr() == state["linear_a_log_pointer"]
            )
            if (
                state["linear_capture_kind"] is None
                or not is_selected_linear_layer
            ):
                return state["original_packed_decode"](*args, **kwargs)
            initial_state = (
                args[6] if len(args) > 6 else kwargs.get("initial_state")
            )
            output = args[7] if len(args) > 7 else kwargs.get("out")
            indices = (
                args[8] if len(args) > 8 else kwargs.get("ssm_state_indices")
            )
            if not isinstance(initial_state, torch.Tensor) or not isinstance(
                output, torch.Tensor
            ) or not isinstance(indices, torch.Tensor) or indices.numel() != 1:
                raise RuntimeError(
                    "generation linear recurrent state geometry changed"
                )
            state_index = int(indices.reshape(-1)[0].item())
            capture_linear(
                "recurrent_state_before", initial_state[state_index]
            )
            result = state["original_packed_decode"](*args, **kwargs)
            capture_linear("recurrent_output", output)
            capture_linear(
                "recurrent_state_after", initial_state[state_index]
            )
            return result

        def instrumented_router_select_experts(
            *args: Any, **kwargs: Any
        ) -> Any:
            output = state["original_router_select_experts"](*args, **kwargs)
            if state["linear_capture_kind"] is None:
                return output
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("generation linear router contract changed")
            weights, indices = output
            if not isinstance(weights, torch.Tensor) or not isinstance(
                indices, torch.Tensor
            ):
                raise RuntimeError(
                    "generation linear router indices unavailable"
                )
            capture_layer0_tail("router_weights", weights)
            capture_layer0_tail("router_indices", indices.to(torch.int32))
            return output

        def instrumented_full_attention_router_select_experts(
            *args: Any, **kwargs: Any
        ) -> Any:
            output = state[
                "original_full_attention_router_select_experts"
            ](*args, **kwargs)
            if state["full_attention_capture_kind"] is None:
                return output
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError(
                    "generation full-attention router contract changed"
                )
            weights, indices = output
            if not isinstance(weights, torch.Tensor) or not isinstance(
                indices, torch.Tensor
            ):
                raise RuntimeError(
                    "generation full-attention router indices unavailable"
                )
            capture_full_attention("router_weights", weights)
            capture_full_attention("router_indices", indices.to(torch.int32))
            return output

        def instrumented_apply_moe_activation(
            activation: Any, activation_output: Any, activation_input: Any
        ) -> Any:
            capture_layer0_tail(
                "routed_gate_up_projection", activation_input
            )
            capture_full_attention(
                "routed_gate_up_projection", activation_input
            )
            result = state["original_apply_moe_activation"](
                activation, activation_output, activation_input
            )
            capture_layer0_tail("routed_activation", activation_output)
            capture_full_attention("routed_activation", activation_output)
            return result

        def instrumented_moe_sum(moe_input: Any, moe_output: Any) -> Any:
            capture_layer0_tail(
                "routed_weighted_expert_outputs", moe_input
            )
            capture_full_attention(
                "routed_weighted_expert_outputs", moe_input
            )
            return state["original_moe_sum"](moe_input, moe_output)

        handles = state["handles"]
        handles.append(
            linear_layer.input_layernorm.register_forward_hook(
                linear_input_norm_hook
            )
        )
        handles.append(
            linear.in_proj_qkvz.register_forward_hook(qkvz_projection_hook)
        )
        handles.append(
            linear.in_proj_ba.register_forward_hook(ba_projection_hook)
        )
        handles.append(linear.norm.register_forward_hook(gated_norm_hook))
        handles.append(
            linear.out_proj.register_forward_hook(attention_output_hook)
        )
        handles.append(
            linear_layer.post_attention_layernorm.register_forward_hook(
                post_attention_hook
            )
        )
        handles.append(
            mlp.shared_expert_gate.register_forward_hook(shared_gate_hook)
        )
        handles.append(mlp.gate.register_forward_hook(router_logits_hook))
        handles.append(
            shared_expert.gate_up_proj.register_forward_hook(
                shared_gate_up_hook
            )
        )
        handles.append(
            shared_expert.act_fn.register_forward_hook(
                shared_activation_hook
            )
        )
        handles.append(
            shared_expert.down_proj.register_forward_hook(shared_down_hook)
        )
        handles.append(mlp.experts.register_forward_hook(experts_hook))
        handles.append(mlp.register_forward_hook(mlp_hook))
        handles.append(linear_layer.register_forward_hook(linear_output_hook))
        handles.append(
            full_attention_layer.input_layernorm.register_forward_hook(
                full_attention_input_norm_hook
            )
        )
        if self.capture_full_attention_projection:
            handles.append(
                full_attention_module.qkv_proj.register_forward_hook(
                    full_attention_qkv_projection_hook
                )
            )
            handles.append(
                full_attention_module.o_proj.register_forward_pre_hook(
                    full_attention_o_proj_pre_hook
                )
            )
            handles.append(
                full_attention_module.o_proj.register_forward_hook(
                    full_attention_o_proj_hook
                )
            )
            handles.append(
                full_attention_layer.post_attention_layernorm.register_forward_hook(
                    full_attention_post_attention_hook
                )
            )
            handles.append(
                full_attention_mlp.shared_expert_gate.register_forward_hook(
                    full_attention_shared_gate_hook
                )
            )
            handles.append(
                full_attention_mlp.gate.register_forward_hook(
                    full_attention_router_logits_hook
                )
            )
            handles.append(
                full_attention_shared_expert.gate_up_proj.register_forward_hook(
                    full_attention_shared_gate_up_hook
                )
            )
            handles.append(
                full_attention_shared_expert.act_fn.register_forward_hook(
                    full_attention_shared_activation_hook
                )
            )
            handles.append(
                full_attention_shared_expert.down_proj.register_forward_hook(
                    full_attention_shared_down_hook
                )
            )
            handles.append(
                full_attention_mlp.experts.register_forward_hook(
                    full_attention_experts_hook
                )
            )
            handles.append(
                full_attention_mlp.register_forward_hook(
                    full_attention_mlp_hook
                )
            )
        handles.append(
            full_attention_layer.register_forward_hook(
                full_attention_output_hook
            )
        )
        gdn_module.causal_conv1d_update = instrumented_causal_conv1d_update
        gdn_module.fused_recurrent_gated_delta_rule_packed_decode = (
            instrumented_packed_decode
        )
        router.select_experts = instrumented_router_select_experts
        full_attention_router.select_experts = (
            instrumented_full_attention_router_select_experts
        )
        fused_moe_module.apply_moe_activation = (
            instrumented_apply_moe_activation
        )
        modular_moe_module.apply_moe_activation = (
            instrumented_apply_moe_activation
        )
        custom_ops.moe_sum = instrumented_moe_sum
        triton_attention_module.unified_attention = (
            instrumented_unified_attention
        )

        def capture_boundary(name: str, value: Any) -> None:
            tensor = _first_tensor(value)
            if tensor is None or tensor.ndim != 2:
                raise RuntimeError(f"generation boundary is not a matrix: {name}")
            if tensor.shape[-1] != HIDDEN_SIZE:
                raise RuntimeError(
                    f"generation boundary hidden size changed: {name}"
                )
            if tensor.shape[0] != 1:
                return
            state["boundary_singleton_calls"][name] += 1
            decode_call = state["boundary_singleton_calls"][name]
            if (
                decode_call == FIRST_DECODE_LINEAR_OUTPUT_INDEX
                and name not in state["first_decode_captures"]
            ):
                state["first_decode_captures"][name] = (
                    tensor[-1].detach().contiguous().cpu()
                )
            if decode_call == self.output_index and name not in state["captures"]:
                state["captures"][name] = (
                    tensor[-1].detach().contiguous().cpu()
                )

        def next_input_norm_hook(next_layer: int):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                if not isinstance(output, (tuple, list)) or len(output) != 2:
                    raise RuntimeError(
                        f"language layer {next_layer} input norm differs"
                    )
                capture_boundary(
                    f"layer_{next_layer - 1:03d}_output", output[1]
                )

            return hook

        def final_norm_hook(_module: Any, _args: Any, output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("language final norm residual contract differs")
            capture_boundary("language_final_norm", output[0])
            capture_boundary("layer_039_output", output[1])

        def logits_hook(_module: Any, _args: Any, output: Any) -> None:
            logits = _first_tensor(output)
            if logits is None or logits.ndim != 2:
                return
            if logits.shape[1] != MODEL_VOCABULARY_SIZE:
                raise RuntimeError(
                    f"generation vocabulary changed: {list(logits.shape)}"
                )
            if logits.shape[0] > 1:
                state["logits_prefill_calls"] += 1
                return
            if logits.shape[0] != 1:
                return
            state["logits_decode_calls"] += 1
            output_index = state["logits_decode_calls"] - 1
            logits_row = logits[-1].detach().float().contiguous().cpu()
            if (
                output_index == FIRST_DECODE_LINEAR_OUTPUT_INDEX
                and state["first_decode_logits"] is None
            ):
                state["first_decode_logits"] = logits_row
            if output_index != self.output_index:
                return
            target = logits_row
            state["captured_logits_output_index"] = output_index
            state["target_logits"] = target
            state["target_logits_sha256"] = hashlib.sha256(
                target.numpy().tobytes(order="C")
            ).hexdigest()
            state["target_logits_top1_token_id"] = int(
                torch.argmax(target).item()
            )

        for next_layer in range(1, 40):
            handles.append(
                language.model.layers[
                    next_layer
                ].input_layernorm.register_forward_hook(
                    next_input_norm_hook(next_layer)
                )
            )
        handles.append(language.model.norm.register_forward_hook(final_norm_hook))
        handles.append(
            language.logits_processor.register_forward_hook(logits_hook)
        )
        setattr(root, STATE_ATTRIBUTE, state)
        return {
            "installed": True,
            "target_output_index": self.output_index,
            "linear_attention_layer_index": (
                self.linear_attention_layer_index
            ),
            "full_attention_layer_index": self.full_attention_layer_index,
            "full_attention_projection_captured": (
                self.capture_full_attention_projection
            ),
        }


class FinalizeGenerationLayerHooks:
    def __init__(self, *, output_root: str) -> None:
        self.output_root = output_root

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = _find_model_root(model)
        state = getattr(root, STATE_ATTRIBUTE, None)
        if not isinstance(state, dict):
            raise RuntimeError("generation layer hooks were not installed")
        captures = state["captures"]
        missing = set(BOUNDARY_NAMES) - set(captures)
        if missing:
            _remove_hooks(root, state)
            raise RuntimeError(
                "missing target decode boundaries: " + ", ".join(sorted(missing))
            )
        first_decode_captures = state["first_decode_captures"]
        missing_first_decode = set(BOUNDARY_NAMES) - set(first_decode_captures)
        if missing_first_decode:
            _remove_hooks(root, state)
            raise RuntimeError(
                "missing first-decode boundaries: "
                + ", ".join(sorted(missing_first_decode))
            )
        if state["captured_logits_output_index"] != state["target_output_index"]:
            _remove_hooks(root, state)
            raise RuntimeError("target logits and decode boundaries were not aligned")
        linear_captures = state["linear_captures"]
        missing_linear = set(LINEAR_ATTENTION_BOUNDARY_SPECS) - set(
            linear_captures
        )
        if missing_linear:
            _remove_hooks(root, state)
            raise RuntimeError(
                "missing target layer-0 linear-attention boundaries: "
                + ", ".join(sorted(missing_linear))
            )
        first_decode_linear_captures = state[
            "first_decode_linear_captures"
        ]
        missing_first_decode_linear = set(
            LINEAR_ATTENTION_BOUNDARY_SPECS
        ) - set(first_decode_linear_captures)
        if missing_first_decode_linear:
            _remove_hooks(root, state)
            raise RuntimeError(
                "missing first-decode layer-0 linear-attention boundaries: "
                + ", ".join(sorted(missing_first_decode_linear))
            )
        layer0_tail_captures = state["layer0_tail_captures"]
        missing_layer0_tail = set(LAYER0_TAIL_BOUNDARY_SPECS) - set(
            layer0_tail_captures
        )
        if missing_layer0_tail:
            _remove_hooks(root, state)
            raise RuntimeError(
                "missing target layer-0 tail boundaries: "
                + ", ".join(sorted(missing_layer0_tail))
            )
        first_decode_layer0_tail_captures = state[
            "first_decode_layer0_tail_captures"
        ]
        missing_first_decode_layer0_tail = set(
            LAYER0_TAIL_BOUNDARY_SPECS
        ) - set(first_decode_layer0_tail_captures)
        if missing_first_decode_layer0_tail:
            _remove_hooks(root, state)
            raise RuntimeError(
                "missing first-decode layer-0 tail boundaries: "
                + ", ".join(sorted(missing_first_decode_layer0_tail))
            )
        full_attention_captures = state["full_attention_captures"]
        full_attention_component_names = state[
            "full_attention_component_names"
        ]
        missing_full_attention = set(full_attention_component_names) - set(
            full_attention_captures
        )
        if missing_full_attention:
            _remove_hooks(root, state)
            raise RuntimeError(
                "missing target unified-attention boundaries: "
                + ", ".join(sorted(missing_full_attention))
            )
        first_decode_full_attention_captures = state[
            "first_decode_full_attention_captures"
        ]
        missing_first_decode_full_attention = set(
            full_attention_component_names
        ) - set(first_decode_full_attention_captures)
        if missing_first_decode_full_attention:
            _remove_hooks(root, state)
            raise RuntimeError(
                "missing first-decode unified-attention boundaries: "
                + ", ".join(sorted(missing_first_decode_full_attention))
            )
        if not state["full_attention_metadata"] or not state[
            "first_decode_full_attention_metadata"
        ]:
            _remove_hooks(root, state)
            raise RuntimeError(
                "generation unified-attention metadata is missing"
            )
        target_call = state["target_output_index"]
        capture_calls = {
            state["boundary_singleton_calls"][name] for name in BOUNDARY_NAMES
        }
        if capture_calls != {target_call}:
            _remove_hooks(root, state)
            raise RuntimeError("decode boundary singleton counts are inconsistent")
        if state["full_attention_singleton_calls"] != target_call:
            _remove_hooks(root, state)
            raise RuntimeError(
                "unified-attention decode singleton count is inconsistent"
            )
        target_logits = state.get("target_logits")
        first_decode_logits = state.get("first_decode_logits")
        if not isinstance(target_logits, torch.Tensor) or not isinstance(
            first_decode_logits, torch.Tensor
        ):
            _remove_hooks(root, state)
            raise RuntimeError("generation logits boundary capture is incomplete")

        output_root = Path(self.output_root)
        case_id = state["case_id"]
        case_root = output_root / case_id

        def write_layer_boundary_set(
            directory: str | None,
            values: dict[str, Any],
            target_decode_call: int,
        ) -> dict[str, Any]:
            relative_root = case_id
            if directory is not None:
                relative_root = f"{case_id}/{directory}"
            boundary_components = {
                name: write_raw_tensor(
                    output_root,
                    f"{relative_root}/components/{name}",
                    values[name],
                )
                for name in BOUNDARY_NAMES
            }
            lines = [
                json.dumps(
                    {
                        "event": "native_layer_oracle_tensor",
                        "label": name,
                        "file": Path(boundary_components[name]["path"])
                        .relative_to(relative_root)
                        .as_posix(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for name in BOUNDARY_NAMES
            ]
            boundary_root = output_root / relative_root
            boundary_ledger = boundary_root / "oracle.jsonl"
            boundary_ledger.write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            return {
                "target_layer_decode_call": target_decode_call,
                "components": boundary_components,
                "oracle_jsonl": file_component(
                    boundary_ledger, f"{relative_root}/oracle.jsonl"
                ),
            }

        target_boundaries = write_layer_boundary_set(
            None, captures, state["target_output_index"]
        )
        first_decode = write_layer_boundary_set(
            "first-decode",
            first_decode_captures,
            FIRST_DECODE_LINEAR_OUTPUT_INDEX,
        )
        target_logits_component = write_raw_tensor(
            output_root, f"{case_id}/target-logits", target_logits
        )
        first_decode_logits_component = write_raw_tensor(
            output_root,
            f"{case_id}/first-decode-logits",
            first_decode_logits,
        )

        def write_tensor_boundary_set(
            directory: str,
            values: dict[str, Any],
            target_decode_call: int,
            specs: Mapping[str, Any],
        ) -> dict[str, Any]:
            boundary_components = {
                name: write_raw_tensor(
                    output_root,
                    f"{case_id}/{directory}/components/{name}",
                    values[name],
                )
                for name in specs
            }
            lines = [
                json.dumps(
                    {
                        "event": "native_layer_oracle_tensor",
                        "label": name,
                        "file": Path(boundary_components[name]["path"])
                        .relative_to(f"{case_id}/{directory}")
                        .as_posix(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for name in specs
            ]
            boundary_ledger = case_root / directory / "oracle.jsonl"
            boundary_ledger.write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            return {
                "target_decode_call": target_decode_call,
                "components": boundary_components,
                "oracle_jsonl": file_component(
                    boundary_ledger,
                    f"{case_id}/{directory}/oracle.jsonl",
                ),
            }

        linear_attention = write_tensor_boundary_set(
            "linear",
            linear_captures,
            target_call,
            LINEAR_ATTENTION_BOUNDARY_SPECS,
        )
        linear_attention["metadata"] = {
            "layer_index": state["linear_attention_layer_index"]
        }
        first_decode_linear_attention = write_tensor_boundary_set(
            "first-decode-linear",
            first_decode_linear_captures,
            FIRST_DECODE_LINEAR_OUTPUT_INDEX,
            LINEAR_ATTENTION_BOUNDARY_SPECS,
        )
        first_decode_linear_attention["metadata"] = {
            "layer_index": state["linear_attention_layer_index"]
        }
        layer0_tail = write_tensor_boundary_set(
            "layer0-tail",
            layer0_tail_captures,
            target_call,
            LAYER0_TAIL_BOUNDARY_SPECS,
        )
        layer0_tail["metadata"] = {
            "layer_index": state["linear_attention_layer_index"]
        }
        first_decode_layer0_tail = write_tensor_boundary_set(
            "first-decode-layer0-tail",
            first_decode_layer0_tail_captures,
            FIRST_DECODE_LINEAR_OUTPUT_INDEX,
            LAYER0_TAIL_BOUNDARY_SPECS,
        )
        first_decode_layer0_tail["metadata"] = {
            "layer_index": state["linear_attention_layer_index"]
        }
        full_attention = write_tensor_boundary_set(
            "full-attention",
            full_attention_captures,
            target_call,
            full_attention_component_names,
        )
        full_attention["metadata"] = state["full_attention_metadata"]
        first_decode_full_attention = write_tensor_boundary_set(
            "first-decode-full-attention",
            first_decode_full_attention_captures,
            FIRST_DECODE_LINEAR_OUTPUT_INDEX,
            full_attention_component_names,
        )
        first_decode_full_attention["metadata"] = state[
            "first_decode_full_attention_metadata"
        ]
        result = {
            **target_boundaries,
            "logits_prefill_calls": state["logits_prefill_calls"],
            "logits_decode_calls": state["logits_decode_calls"],
            "captured_logits_output_index": state[
                "captured_logits_output_index"
            ],
            "target_logits_sha256": state["target_logits_sha256"],
            "target_logits_top1_token_id": state[
                "target_logits_top1_token_id"
            ],
            "target_logits_component": target_logits_component,
            "first_decode_logits_output_index": (
                FIRST_DECODE_LINEAR_OUTPUT_INDEX
            ),
            "first_decode_logits_top1_token_id": int(
                torch.argmax(first_decode_logits).item()
            ),
            "first_decode_logits_component": first_decode_logits_component,
            "first_decode": first_decode,
            "linear_attention": linear_attention,
            "first_decode_linear_attention": first_decode_linear_attention,
            "layer0_tail": layer0_tail,
            "first_decode_layer0_tail": first_decode_layer0_tail,
            "full_attention": full_attention,
            "first_decode_full_attention": first_decode_full_attention,
        }
        _remove_hooks(root, state)
        return result


class RemoveGenerationLayerHooks:
    def __call__(self, model: Any) -> bool:
        root = _find_model_root(model)
        state = getattr(root, STATE_ATTRIBUTE, None)
        if not isinstance(state, dict):
            return False
        _remove_hooks(root, state)
        return True


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import cloudpickle
    import torch
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import StructuredOutputsParams

    explicit_linear_attention_layers = parse_case_linear_attention_layers(
        args.diagnostic_linear_attention_layer
    )
    if (
        explicit_linear_attention_layers is not None
        and args.first_divergence_linear_attention
    ):
        raise ValueError(
            "explicit and first-divergence linear-attention layers are "
            "mutually exclusive"
        )
    selected_linear_attention_layers = (
        explicit_linear_attention_layers
        if explicit_linear_attention_layers is not None
        else (
            FIRST_DIVERGENCE_LINEAR_ATTENTION_LAYERS
            if args.first_divergence_linear_attention
            else None
        )
    )

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"generation layer oracle root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source = git_identity(ROOT)
    if source["dirty"]:
        raise ValueError("generation layer capture source must be a clean commit")

    generation = load_module(
        GENERATION_CAPTURE, "aima_vl_generation_layer_base_capture"
    )
    base = generation.load_module(
        generation.BASE_CAPTURE, "aima_vl_generation_layer_vl_base_capture"
    )
    probe = generation.load_module(
        generation.CAPABILITY_PROBE,
        "aima_vl_generation_layer_capability_probe",
    )
    fixture_root = args.fixture_root.resolve()
    reference, _launch, _processor = base._require_inputs(
        reference_path=args.reference_manifest,
        launch_path=args.launch_config,
        processor_path=args.processor_probe,
        fixture_root=fixture_root,
    )
    if socket.gethostname() != reference.get("host", {}).get("hostname"):
        raise ValueError("generation layer host differs from the frozen reference")

    capability = load_json_object(args.capability_manifest)
    capability_errors = generation.validate_capability_manifest(capability)
    if capability_errors:
        raise ValueError(
            "invalid capability manifest:\n- " + "\n- ".join(capability_errors)
        )
    render = load_json_object(args.api_render_manifest)
    render_errors = generation.validate_api_render_manifest(render)
    if render_errors:
        raise ValueError(
            "invalid API render manifest:\n- " + "\n- ".join(render_errors)
        )
    generation_oracle_path = args.generation_oracle_manifest.resolve()
    generation_oracle_root = args.generation_oracle_root.resolve()
    generation_oracle = load_json_object(generation_oracle_path)
    generation_errors = validate_generation_oracle_manifest(
        generation_oracle, oracle_root=generation_oracle_root
    )
    if generation_errors:
        raise ValueError(
            "invalid generation oracle:\n- " + "\n- ".join(generation_errors)
        )
    generation_by_id = {
        case["case_id"]: case for case in generation_oracle["cases"]
    }
    capability_by_id = {case["case_id"]: case for case in capability["cases"]}
    render_by_id = {case["case_id"]: case for case in render["cases"]}

    versions = base._runtime_versions()
    for name, expected in PINNED_PACKAGES.items():
        actual = versions.get(name)
        if not isinstance(actual, str) or not (
            actual == expected or actual.startswith(expected + ".")
        ):
            raise ValueError(f"runtime pin mismatch for {name}: {actual!r}")

    fixtures = probe.Fixtures(fixture_root, "http://127.0.0.1:9")
    specs = {
        spec["case_id"]: spec
        for spec in probe.build_cases(fixtures, "qwen36-vl-reference")
        if spec["case_id"] in CASE_ORDER
    }
    if tuple(case_id for case_id in CASE_ORDER if case_id in specs) != CASE_ORDER:
        raise RuntimeError("generation layer tool case set changed")

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    llm_kwargs = base._llm_kwargs(args.model_dir.resolve(), fixture_root)
    llm_kwargs["skip_mm_profiling"] = True
    llm = LLM(**llm_kwargs)
    print(json.dumps({"event": "engine_ready"}, sort_keys=True), flush=True)
    cases: list[dict[str, Any]] = []
    try:
        for case_id in CASE_ORDER:
            contract = CASE_CONTRACTS[case_id]
            generation_case = generation_by_id[case_id]
            spec = specs[case_id]
            payload = generation.canonical_round_trip(spec["payload"])
            openai_request = ChatCompletionRequest.model_validate(payload)
            tool_dicts = (
                [tool.model_dump() for tool in openai_request.tools]
                if openai_request.tools is not None
                else None
            )
            normalized_request = probe.recursive_replace(
                spec["payload"], spec["replacements"]
            )
            if normalized_request != capability_by_id[case_id].get("request"):
                raise RuntimeError(f"generation layer request drifted: {case_id}")

            llm.reset_mm_cache()
            llm.llm_engine.reset_encoder_cache()
            engine_input = llm._preprocess_chat_one(
                openai_request.messages,
                chat_template_content_format="string",
                chat_template_kwargs=openai_request.chat_template_kwargs,
                tools=tool_dicts,
            )
            prompt_token_ids = [
                int(token) for token in engine_input["prompt_token_ids"]
            ]
            frozen_render = render_by_id[case_id]
            if prompt_token_ids != frozen_render.get("prompt_token_ids"):
                raise RuntimeError(
                    f"generation layer prompt differs from render: {case_id}"
                )

            structured = None
            if contract["structured"]:
                structured_record = frozen_render.get("structured_outputs")
                schema = (
                    structured_record.get("json")
                    if isinstance(structured_record, dict)
                    else None
                )
                if not isinstance(schema, dict):
                    raise RuntimeError("forced tool render lost its JSON schema")
                structured = StructuredOutputsParams(json=schema)
            target_index = (
                args.diagnostic_output_index
                if args.diagnostic_output_index is not None
                else contract["divergence_output_index"]
            )
            if target_index >= len(
                generation_case["generation"]["output_token_ids"]
            ):
                raise RuntimeError(
                    f"generation layer diagnostic index exceeds oracle: {case_id}"
                )
            sampling = SamplingParams(
                temperature=0,
                max_tokens=target_index + 1,
                prompt_logprobs=1,
                seed=0,
                structured_outputs=structured,
            )
            full_attention_layer_index = (
                FIRST_DIVERGENCE_FULL_ATTENTION_LAYERS[case_id]
                if args.first_divergence_full_attention
                else FULL_ATTENTION_LAYER
            )
            linear_attention_layer_index = (
                selected_linear_attention_layers[case_id]
                if selected_linear_attention_layers is not None
                else LINEAR_ATTENTION_LAYER
            )

            def install_callable(model: Any) -> dict[str, Any]:
                return InstallGenerationLayerHooks(
                    case_id=case_id,
                    output_index=target_index,
                    linear_attention_layer_index=(
                        linear_attention_layer_index
                    ),
                    full_attention_layer_index=full_attention_layer_index,
                    capture_full_attention_projection=(
                        args.first_divergence_full_attention
                    ),
                )(model)

            def finalize_callable(model: Any) -> dict[str, Any]:
                return FinalizeGenerationLayerHooks(
                    output_root=str(output_root)
                )(model)

            def cleanup_callable(model: Any) -> bool:
                return RemoveGenerationLayerHooks()(model)

            installation = llm.apply_model(install_callable)
            try:
                outputs = llm._render_and_run_requests(
                    prompts=iter((engine_input,)),
                    params=[sampling],
                    output_type=RequestOutput,
                    use_tqdm=False,
                )
                finalization = llm.apply_model(finalize_callable)
            except BaseException:
                llm.apply_model(cleanup_callable)
                raise
            if (
                len(outputs) != 1
                or len(outputs[0].outputs) != 1
                or len(installation) != 1
                or len(finalization) != 1
            ):
                raise RuntimeError(
                    "generation layer capture requires one request and TP=1"
                )

            output_token_ids = [
                int(token) for token in outputs[0].outputs[0].token_ids
            ]
            expected_output_ids = generation_case["generation"][
                "output_token_ids"
            ][: target_index + 1]
            if output_token_ids != expected_output_ids:
                raise RuntimeError(
                    f"generation layer target prefix changed: {case_id}"
                )
            record = finalization[0]
            if record["target_logits_component"]["sha256"] != record[
                "target_logits_sha256"
            ]:
                raise RuntimeError(
                    f"generation layer target logits artifact changed: {case_id}"
                )
            if args.diagnostic_output_index is None:
                expected_logits_sha256 = generation_case[
                    "reference_logits"
                ]["component"]["sha256"]
                if record["target_logits_sha256"] != expected_logits_sha256:
                    raise RuntimeError(
                        f"generation layer target logits changed: {case_id}"
                    )
                if record["target_logits_top1_token_id"] != contract[
                    "reference_token_id"
                ]:
                    raise RuntimeError(
                        f"generation layer target top1 changed: {case_id}"
                    )
            cases.append(
                {
                    "case_id": case_id,
                    "passed": True,
                    "prompt_tokens": len(prompt_token_ids),
                    "prompt_token_ids_sha256": canonical_int_list_sha256(
                        prompt_token_ids
                    ),
                    "target_output_index": target_index,
                    "target_token_id": output_token_ids[-1],
                    "target_prefix_token_ids_sha256": (
                        canonical_int_list_sha256(output_token_ids)
                    ),
                    **record,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "case_complete",
                        "case_id": case_id,
                        "target_output_index": target_index,
                        "boundaries": len(record["components"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    diagnostic = (
        args.diagnostic_output_index is not None
        or args.first_divergence_full_attention
        or selected_linear_attention_layers is not None
    )
    scope = (
        "two-fixed-vllm-vl-decode-layer-and-selected-linear-attention-plus-"
        "tail-routed-moe-and-selected-full-attention-qkv-diagnostic-"
        "boundary-sets"
        if diagnostic
        else "two-fixed-vllm-vl-target-decode-layer-plus-target-and-first-"
        "decode-layer0-linear-attention-tail-and-layer3-unified-attention-"
        "boundary-sets"
    )
    decision = (
        {
            "two_diagnostic_prefixes_exact": len(cases) == 2,
            "two_diagnostic_logits_captured": len(cases) == 2,
            "two_diagnostic_decode_boundary_sets_captured": len(cases) == 2,
            "two_diagnostic_layer0_linear_attention_boundary_sets_captured": (
                len(cases) == 2
            ),
            "two_diagnostic_selected_linear_attention_sets_captured": (
                selected_linear_attention_layers is not None
                and len(cases) == 2
            ),
            "two_diagnostic_layer0_tail_boundary_sets_captured": (
                len(cases) == 2
            ),
            "two_diagnostic_first_decode_layer0_tail_boundary_sets_captured": (
                len(cases) == 2
            ),
            "two_diagnostic_routed_moe_stage_sets_captured": len(cases) == 2,
            "two_diagnostic_layer3_unified_attention_sets_captured": (
                len(cases) == 2
            ),
            "two_diagnostic_first_decode_layer3_unified_attention_sets_captured": (
                len(cases) == 2
            ),
            "two_diagnostic_selected_full_attention_qkv_sets_captured": (
                args.first_divergence_full_attention and len(cases) == 2
            ),
            "promotion_oracle": False,
            "g1_passed": False,
            "g2_passed": False,
            "g3_passed": False,
            "g4_passed": False,
            "g5_passed": False,
        }
        if diagnostic
        else {
            "two_target_prefixes_exact": len(cases) == 2,
            "two_target_logits_bound": len(cases) == 2,
            "two_decode_boundary_sets_captured": len(cases) == 2,
            "two_layer0_linear_attention_boundary_sets_captured": (
                len(cases) == 2
            ),
            "two_first_decode_layer0_linear_attention_boundary_sets_captured": (
                len(cases) == 2
            ),
            "two_layer0_tail_boundary_sets_captured": len(cases) == 2,
            "two_first_decode_layer0_tail_boundary_sets_captured": (
                len(cases) == 2
            ),
            "two_routed_moe_stage_sets_captured": len(cases) == 2,
            "two_layer3_unified_attention_sets_captured": len(cases) == 2,
            "two_first_decode_layer3_unified_attention_sets_captured": (
                len(cases) == 2
            ),
            "g1_passed": False,
            "g2_passed": False,
            "g3_passed": False,
            "g4_passed": False,
            "g5_passed": False,
        }
    )
    manifest = seal_manifest(
        {
            "schema": (
                GENERATION_LAYER_DIAGNOSTIC_SCHEMA
                if diagnostic
                else GENERATION_LAYER_ORACLE_SCHEMA
            ),
            "captured_at": base.utc_now(),
            "complete": True,
            "qualified_for_decode_attribution": True,
            "scope": scope,
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in (
                        Path(__file__).resolve(),
                        ROOT / "aima_engine/vl_generation_layer_oracle.py",
                        GENERATION_CAPTURE,
                        generation.BASE_CAPTURE,
                        generation.CAPABILITY_PROBE,
                    )
                ],
            },
            "host": {"label": args.host_label, "hostname": socket.gethostname()},
            "model": {
                "repository": "Qwen/Qwen3.6-35B-A3B",
                "revision": MODEL_REVISION,
                "dtype": "bfloat16",
            },
            "runtime": {
                "packages": versions,
                "python_version": sys.version.split()[0],
            },
            "generation_oracle": file_component(
                generation_oracle_path,
                "benchmarks/results/vl-generation-oracle-v0.1.0.json",
            ),
            "capture_control_plane": {
                "vllm_allow_insecure_serialization": True,
                "skip_mm_profiling": True,
                "maximum_tokens_per_case": "target_output_index_plus_one",
                "diagnostic_output_index": args.diagnostic_output_index,
                "first_divergence_full_attention": (
                    args.first_divergence_full_attention
                ),
                "first_divergence_full_attention_layers": (
                    FIRST_DIVERGENCE_FULL_ATTENTION_LAYERS
                    if args.first_divergence_full_attention
                    else None
                ),
                "first_divergence_linear_attention": (
                    args.first_divergence_linear_attention
                ),
                "first_divergence_linear_attention_layers": (
                    FIRST_DIVERGENCE_LINEAR_ATTENTION_LAYERS
                    if args.first_divergence_linear_attention
                    else None
                ),
                "diagnostic_linear_attention_layers": (
                    explicit_linear_attention_layers
                ),
                "selected_linear_attention_layers": (
                    selected_linear_attention_layers
                ),
                "layer3_unified_attention_compact_identity_block_table": True,
                "product_runtime_dependency": False,
            },
            "oracle_root": args.oracle_root_label,
            "cases": cases,
            "decision": decision,
        }
    )
    errors = (
        []
        if diagnostic
        else validate_generation_layer_oracle_manifest(
            manifest, oracle_root=output_root
        )
    )
    if errors:
        raise RuntimeError(
            "generation layer oracle validation failed:\n- "
            + "\n- ".join(errors)
        )
    atomic_json(args.output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--processor-probe", type=Path, required=True)
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument("--api-render-manifest", type=Path, required=True)
    parser.add_argument("--generation-oracle-manifest", type=Path, required=True)
    parser.add_argument("--generation-oracle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-output-index",
        type=int,
        choices=range(1, 1024),
        metavar="1..1023",
        help="capture a shared non-promotion decode index for attribution",
    )
    parser.add_argument(
        "--first-divergence-full-attention",
        action="store_true",
        help=(
            "capture QKV and attention state at each case's first known "
            "divergent full-attention layer"
        ),
    )
    parser.add_argument(
        "--first-divergence-linear-attention",
        action="store_true",
        help=(
            "capture the resident state boundaries at each case's first "
            "known divergent linear-attention layer"
        ),
    )
    parser.add_argument(
        "--diagnostic-linear-attention-layer",
        action="append",
        default=[],
        metavar="CASE_ID=LAYER",
        help=(
            "select one validated linear-attention observer layer per fixed "
            "case; repeat for both cases"
        ),
    )
    parser.add_argument(
        "--oracle-root-label",
        default="benchmarks/oracles/vl-generation-layer-v0.1.0",
    )
    parser.add_argument("--host-label", default="amd395")
    return parser.parse_args()


def main() -> int:
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
        raise ValueError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required for isolated "
            "offline qualification hooks"
        )
    args = parse_args()
    manifest = capture(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(manifest["cases"]),
                "qualified": manifest["qualified_for_decode_attribution"],
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
