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
    fused_moe_module = state.get("fused_moe_module")
    if fused_moe_module is not None:
        fused_moe_module.apply_moe_activation = state[
            "original_apply_moe_activation"
        ]
    custom_ops = state.get("custom_ops")
    if custom_ops is not None:
        custom_ops.moe_sum = state["original_moe_sum"]
    if hasattr(root, STATE_ATTRIBUTE):
        delattr(root, STATE_ATTRIBUTE)


class InstallGenerationLayerHooks:
    """Serializable worker hook for one exact generated-token boundary set."""

    def __init__(self, *, case_id: str, output_index: int) -> None:
        self.case_id = case_id
        self.output_index = output_index

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        if self.output_index <= 0:
            raise RuntimeError("decode layer capture requires output index > 0")
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
            "linear_singleton_calls": 0,
            "linear_capture_kind": None,
            "boundary_singleton_calls": {name: 0 for name in BOUNDARY_NAMES},
            "logits_prefill_calls": 0,
            "logits_decode_calls": 0,
            "captured_logits_output_index": None,
            "target_logits_sha256": None,
            "target_logits_top1_token_id": None,
            "handles": [],
        }

        layer0 = language.model.layers[0]
        if getattr(layer0, "layer_type", None) != "linear_attention":
            raise RuntimeError("generation language layer 0 is not linear attention")
        linear = layer0.linear_attn
        mlp = layer0.mlp
        shared_expert = mlp.shared_expert
        if shared_expert is None:
            raise RuntimeError("generation layer-0 shared expert is missing")
        router = mlp.experts.router
        gdn_module = __import__(
            "vllm.model_executor.layers.mamba.gdn_linear_attn",
            fromlist=["causal_conv1d_update"],
        )
        fused_moe_module = __import__(
            "vllm.model_executor.layers.fused_moe.fused_moe",
            fromlist=["apply_moe_activation"],
        )
        custom_ops = __import__("vllm._custom_ops", fromlist=["moe_sum"])
        state.update(
            {
                "gdn_module": gdn_module,
                "original_causal_conv1d_update": (
                    gdn_module.causal_conv1d_update
                ),
                "original_packed_decode": (
                    gdn_module.fused_recurrent_gated_delta_rule_packed_decode
                ),
                "layer0_conv_weight_pointer": linear.conv1d.weight.data_ptr(),
                "layer0_a_log_pointer": linear.A_log.data_ptr(),
                "router": router,
                "original_router_select_experts": router.select_experts,
                "fused_moe_module": fused_moe_module,
                "original_apply_moe_activation": (
                    fused_moe_module.apply_moe_activation
                ),
                "custom_ops": custom_ops,
                "original_moe_sum": custom_ops.moe_sum,
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

        def layer0_input_norm_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            tensor = _first_tensor(output)
            if tensor is None or tensor.ndim != 2 or tensor.shape[-1] != HIDDEN_SIZE:
                raise RuntimeError("generation layer-0 input norm geometry changed")
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
                raise RuntimeError("generation layer-0 experts contract changed")
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

        def layer0_output_hook(_module: Any, _args: Any, _output: Any) -> None:
            state["linear_capture_kind"] = None

        def instrumented_causal_conv1d_update(
            *args: Any, **kwargs: Any
        ) -> Any:
            weight = args[2] if len(args) > 2 else kwargs.get("weight")
            is_layer0 = (
                isinstance(weight, torch.Tensor)
                and weight.data_ptr() == state["layer0_conv_weight_pointer"]
            )
            if state["linear_capture_kind"] is None or not is_layer0:
                return state["original_causal_conv1d_update"](*args, **kwargs)
            conv_state = args[1] if len(args) > 1 else kwargs.get("conv_state")
            indices = kwargs.get("conv_state_indices")
            if not isinstance(conv_state, torch.Tensor) or not isinstance(
                indices, torch.Tensor
            ) or indices.numel() != 1:
                raise RuntimeError("generation layer-0 conv state geometry changed")
            state_index = int(indices.reshape(-1)[0].item())
            capture_linear("conv_state_before", conv_state[state_index])
            output = state["original_causal_conv1d_update"](*args, **kwargs)
            capture_linear("post_conv_mixed_qkv", output)
            capture_linear("conv_state_after", conv_state[state_index])
            return output

        def instrumented_packed_decode(*args: Any, **kwargs: Any) -> Any:
            a_log = args[3] if len(args) > 3 else kwargs.get("A_log")
            is_layer0 = (
                isinstance(a_log, torch.Tensor)
                and a_log.data_ptr() == state["layer0_a_log_pointer"]
            )
            if state["linear_capture_kind"] is None or not is_layer0:
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
                    "generation layer-0 recurrent state geometry changed"
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
                raise RuntimeError("generation layer-0 router contract changed")
            weights, indices = output
            if not isinstance(weights, torch.Tensor) or not isinstance(
                indices, torch.Tensor
            ):
                raise RuntimeError("generation layer-0 router indices unavailable")
            capture_layer0_tail("router_weights", weights)
            capture_layer0_tail("router_indices", indices.to(torch.int32))
            return output

        def instrumented_apply_moe_activation(
            activation: Any, activation_output: Any, activation_input: Any
        ) -> Any:
            capture_layer0_tail(
                "routed_gate_up_projection", activation_input
            )
            result = state["original_apply_moe_activation"](
                activation, activation_output, activation_input
            )
            capture_layer0_tail("routed_activation", activation_output)
            return result

        def instrumented_moe_sum(moe_input: Any, moe_output: Any) -> Any:
            capture_layer0_tail(
                "routed_weighted_expert_outputs", moe_input
            )
            return state["original_moe_sum"](moe_input, moe_output)

        handles = state["handles"]
        handles.append(
            layer0.input_layernorm.register_forward_hook(
                layer0_input_norm_hook
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
            layer0.post_attention_layernorm.register_forward_hook(
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
        handles.append(layer0.register_forward_hook(layer0_output_hook))
        gdn_module.causal_conv1d_update = instrumented_causal_conv1d_update
        gdn_module.fused_recurrent_gated_delta_rule_packed_decode = (
            instrumented_packed_decode
        )
        router.select_experts = instrumented_router_select_experts
        fused_moe_module.apply_moe_activation = (
            instrumented_apply_moe_activation
        )
        custom_ops.moe_sum = instrumented_moe_sum

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
            if output_index != self.output_index:
                return
            target = logits[-1].detach().float().contiguous().cpu()
            state["captured_logits_output_index"] = output_index
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
        return {"installed": True, "target_output_index": self.output_index}


class FinalizeGenerationLayerHooks:
    def __init__(self, *, output_root: str) -> None:
        self.output_root = output_root

    def __call__(self, model: Any) -> dict[str, Any]:
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
        target_call = state["target_output_index"]
        capture_calls = {
            state["boundary_singleton_calls"][name] for name in BOUNDARY_NAMES
        }
        if capture_calls != {target_call}:
            _remove_hooks(root, state)
            raise RuntimeError("decode boundary singleton counts are inconsistent")

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
        first_decode_linear_attention = write_tensor_boundary_set(
            "first-decode-linear",
            first_decode_linear_captures,
            FIRST_DECODE_LINEAR_OUTPUT_INDEX,
            LINEAR_ATTENTION_BOUNDARY_SPECS,
        )
        layer0_tail = write_tensor_boundary_set(
            "layer0-tail",
            layer0_tail_captures,
            target_call,
            LAYER0_TAIL_BOUNDARY_SPECS,
        )
        first_decode_layer0_tail = write_tensor_boundary_set(
            "first-decode-layer0-tail",
            first_decode_layer0_tail_captures,
            FIRST_DECODE_LINEAR_OUTPUT_INDEX,
            LAYER0_TAIL_BOUNDARY_SPECS,
        )
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
            "first_decode": first_decode,
            "linear_attention": linear_attention,
            "first_decode_linear_attention": first_decode_linear_attention,
            "layer0_tail": layer0_tail,
            "first_decode_layer0_tail": first_decode_layer0_tail,
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

            def install_callable(model: Any) -> dict[str, Any]:
                return InstallGenerationLayerHooks(
                    case_id=case_id, output_index=target_index
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

    diagnostic = args.diagnostic_output_index is not None
    scope = (
        "two-fixed-vllm-vl-shared-output-index-layer-and-layer0-linear-"
        "attention-plus-tail-and-routed-moe-diagnostic-boundary-sets"
        if diagnostic
        else "two-fixed-vllm-vl-target-decode-layer-plus-target-and-first-"
        "decode-layer0-linear-attention-boundary-sets"
    )
    decision = (
        {
            "two_diagnostic_prefixes_exact": len(cases) == 2,
            "two_diagnostic_logits_captured": len(cases) == 2,
            "two_diagnostic_decode_boundary_sets_captured": len(cases) == 2,
            "two_diagnostic_layer0_linear_attention_boundary_sets_captured": (
                len(cases) == 2
            ),
            "two_diagnostic_layer0_tail_boundary_sets_captured": (
                len(cases) == 2
            ),
            "two_diagnostic_first_decode_layer0_tail_boundary_sets_captured": (
                len(cases) == 2
            ),
            "two_diagnostic_routed_moe_stage_sets_captured": len(cases) == 2,
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
