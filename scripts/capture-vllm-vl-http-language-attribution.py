#!/usr/bin/env python3
"""Capture sparse HTTP VL language drift-attribution boundaries."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_HOOKS = ROOT / "scripts/capture-vllm-vl-language-layer-outputs.py"
SPEC = importlib.util.spec_from_file_location(
    "aima_vl_http_language_attribution_base", BASE_HOOKS
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the frozen all-layer language hooks")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)
prefix = base.prefix


STATE_ATTRIBUTE = "_aima_vl_http_language_attribution_state"
ATTENTION_DIAGNOSTIC_LAYER = 1
DETAILED_ROUTER_LAYERS = (21, 30, 31)
OUTPUT_COMPONENTS = {f"layer_{layer:03d}_output" for layer in range(40)}
ROUTER_COMPONENTS = {
    f"layer_{layer:03d}_router_indices" for layer in range(40)
}
ATTENTION_DIAGNOSTIC_COMPONENTS = {
    f"layer_{ATTENTION_DIAGNOSTIC_LAYER:03d}_{suffix}"
    for suffix in (
        "attention_input",
        "attention_output",
        "attention_residual",
        "post_attention_norm",
        "fused_input_projection",
        "a_projection",
        "b_projection",
        "convolution",
        "gdn_q",
        "gdn_k",
        "gdn_v",
        "gdn_g",
        "gdn_beta",
        "gdn_g_cumsum",
        "gdn_chunk_matrix",
        "gdn_chunk_matrix_inverse",
        "gdn_w",
        "gdn_u",
        "gdn_chunk_state",
        "gdn_v_new",
        "gdn_final_state",
        "gdn_core",
        "gdn_gated",
    )
}
DETAILED_ROUTER_COMPONENTS = {
    f"layer_{layer:03d}_{suffix}"
    for layer in DETAILED_ROUTER_LAYERS
    for suffix in ("router_logits", "router_scores", "router_weights")
}
REQUIRED_COMPONENTS = (
    OUTPUT_COMPONENTS
    | ROUTER_COMPONENTS
    | ATTENTION_DIAGNOSTIC_COMPONENTS
    | DETAILED_ROUTER_COMPONENTS
    | {
        "language_final_norm",
        "layer_000_combined_moe_output",
        "layer_003_attention_input",
    }
)


def component_name(layer: int, suffix: str) -> str:
    return f"layer_{layer:03d}_{suffix}"


def oracle_labels() -> dict[str, str]:
    labels: dict[str, str] = {"language-final-norm": "language_final_norm"}
    for layer in range(40):
        layer_prefix = f"layer-{layer:03d}-"
        labels[layer_prefix + "return-layer_body-output"] = component_name(
            layer, "output"
        )
        labels[layer_prefix + "return-layer_body-router_indices"] = (
            component_name(layer, "router_indices")
        )
    attention_prefix = f"layer-{ATTENTION_DIAGNOSTIC_LAYER:03d}-"
    labels[attention_prefix + "launch-000-out"] = component_name(
        ATTENTION_DIAGNOSTIC_LAYER, "attention_input"
    )
    labels[attention_prefix + "return-linear_attention-output"] = (
        component_name(ATTENTION_DIAGNOSTIC_LAYER, "attention_output")
    )
    labels[attention_prefix + "launch-009-residual_out"] = component_name(
        ATTENTION_DIAGNOSTIC_LAYER, "attention_residual"
    )
    labels[attention_prefix + "launch-009-norm_out"] = component_name(
        ATTENTION_DIAGNOSTIC_LAYER, "post_attention_norm"
    )
    labels[attention_prefix + "diagnostic-fused-input"] = component_name(
        ATTENTION_DIAGNOSTIC_LAYER, "fused_input_projection"
    )
    for label_suffix, component_suffix in (
        ("diagnostic-a", "a_projection"),
        ("diagnostic-b", "b_projection"),
        ("diagnostic-conv", "convolution"),
        ("diagnostic-q", "gdn_q"),
        ("diagnostic-k", "gdn_k"),
        ("diagnostic-v", "gdn_v"),
        ("diagnostic-g", "gdn_g"),
        ("diagnostic-beta", "gdn_beta"),
    ):
        labels[attention_prefix + label_suffix] = component_name(
            ATTENTION_DIAGNOSTIC_LAYER, component_suffix
        )
    labels[attention_prefix + "launch-008-o"] = component_name(
        ATTENTION_DIAGNOSTIC_LAYER, "gdn_core"
    )
    for label_suffix, component_suffix in (
        ("diagnostic-g-cumsum", "gdn_g_cumsum"),
        ("diagnostic-chunk-matrix", "gdn_chunk_matrix"),
        ("diagnostic-chunk-matrix-inverse", "gdn_chunk_matrix_inverse"),
        ("diagnostic-w", "gdn_w"),
        ("diagnostic-u", "gdn_u"),
        ("diagnostic-chunk-state", "gdn_chunk_state"),
        ("diagnostic-v-new", "gdn_v_new"),
        ("diagnostic-final-state", "gdn_final_state"),
    ):
        labels[attention_prefix + label_suffix] = component_name(
            ATTENTION_DIAGNOSTIC_LAYER, component_suffix
        )
    labels[
        attention_prefix + "return-linear_attention-gated_out"
    ] = component_name(ATTENTION_DIAGNOSTIC_LAYER, "gdn_gated")
    for layer in DETAILED_ROUTER_LAYERS:
        router_prefix = f"layer-{layer:03d}-return-layer_body-"
        for suffix in ("router_logits", "router_scores", "router_weights"):
            labels[router_prefix + suffix] = component_name(layer, suffix)
    return labels


def remove_hooks(root: Any, state: dict[str, Any]) -> None:
    gdn_module = state.get("gdn_module")
    if gdn_module is not None:
        gdn_module.causal_conv1d_fn = state["original_causal_conv1d_fn"]
        gdn_module.fused_post_conv_prep = state[
            "original_fused_post_conv_prep"
        ]
    chunk_module = state.get("chunk_module")
    if chunk_module is not None:
        for name, original in state["original_chunk_functions"].items():
            setattr(chunk_module, name, original)
    prefix._remove_hooks(root, state)
    if hasattr(root, STATE_ATTRIBUTE):
        delattr(root, STATE_ATTRIBUTE)


class InstallLanguageLayerOutputHooks:
    """Install all-layer hooks plus sparse first-drift/router details."""

    def __init__(self, *, output_root: str, case_id: str) -> None:
        self.output_root = output_root
        self.case_id = case_id

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = prefix._find_model_root(model)
        previous = getattr(root, STATE_ATTRIBUTE, None)
        if isinstance(previous, dict):
            remove_hooks(root, previous)
        language = root.language_model
        if len(language.model.layers) != 40:
            raise RuntimeError("language layer count differs from the frozen model")
        diagnostic_layer = language.model.layers[ATTENTION_DIAGNOSTIC_LAYER]
        if getattr(diagnostic_layer, "layer_type", None) != "linear_attention":
            raise RuntimeError("language layer 1 is not linear attention")
        gdn_module = importlib.import_module(
            "vllm.model_executor.layers.mamba.gdn_linear_attn"
        )
        chunk_module = importlib.import_module(
            "vllm.model_executor.layers.fla.ops.chunk"
        )
        chunk_function_names = (
            "chunk_local_cumsum",
            "chunk_scaled_dot_kkt_fwd",
            "solve_tril",
            "recompute_w_u_fwd",
            "chunk_gated_delta_rule_fwd_h",
        )
        state: dict[str, Any] = {
            "output_root": self.output_root,
            "case_id": self.case_id,
            "captures": {},
            "handles": [],
            "router_bindings": [],
            "attention_bindings": [],
            "gdn_module": gdn_module,
            "chunk_module": chunk_module,
            "original_chunk_functions": {
                name: getattr(chunk_module, name) for name in chunk_function_names
            },
            "original_causal_conv1d_fn": gdn_module.causal_conv1d_fn,
            "original_fused_post_conv_prep": gdn_module.fused_post_conv_prep,
            "active_linear_diagnostic": False,
        }

        def capture(
            name: str, value: Any, *, allow_singleton_batch: bool = False
        ) -> None:
            tensor = prefix._first_tensor(value)
            if tensor is None or name in state["captures"]:
                return
            if tensor.ndim == 0 or (
                not allow_singleton_batch and tensor.shape[0] <= 1
            ):
                return
            state["captures"][name] = tensor.detach().contiguous().cpu()

        def capture_fla_internal(name: str, value: Any) -> None:
            if state["active_linear_diagnostic"]:
                capture(
                    component_name(1, name),
                    value,
                    allow_singleton_batch=True,
                )

        def instrumented_chunk_local_cumsum(*args: Any, **kwargs: Any) -> Any:
            output = state["original_chunk_functions"]["chunk_local_cumsum"](
                *args, **kwargs
            )
            capture_fla_internal("gdn_g_cumsum", output)
            return output

        def instrumented_chunk_scaled_dot_kkt_fwd(
            *args: Any, **kwargs: Any
        ) -> Any:
            output = state["original_chunk_functions"][
                "chunk_scaled_dot_kkt_fwd"
            ](*args, **kwargs)
            capture_fla_internal("gdn_chunk_matrix", output)
            return output

        def instrumented_solve_tril(*args: Any, **kwargs: Any) -> Any:
            output = state["original_chunk_functions"]["solve_tril"](
                *args, **kwargs
            )
            capture_fla_internal("gdn_chunk_matrix_inverse", output)
            return output

        def instrumented_recompute_w_u_fwd(*args: Any, **kwargs: Any) -> Any:
            output = state["original_chunk_functions"]["recompute_w_u_fwd"](
                *args, **kwargs
            )
            if state["active_linear_diagnostic"]:
                if not isinstance(output, (tuple, list)) or len(output) != 2:
                    raise RuntimeError("language layer 1 W/U geometry differs")
                capture_fla_internal("gdn_w", output[0])
                capture_fla_internal("gdn_u", output[1])
            return output

        def instrumented_chunk_gated_delta_rule_fwd_h(
            *args: Any, **kwargs: Any
        ) -> Any:
            output = state["original_chunk_functions"][
                "chunk_gated_delta_rule_fwd_h"
            ](*args, **kwargs)
            if state["active_linear_diagnostic"]:
                if not isinstance(output, (tuple, list)) or len(output) != 3:
                    raise RuntimeError("language layer 1 chunk-state geometry differs")
                capture_fla_internal("gdn_chunk_state", output[0])
                capture_fla_internal("gdn_v_new", output[1])
                capture_fla_internal("gdn_final_state", output[2])
            return output

        def next_input_norm_hook(next_layer: int):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                if not isinstance(output, (tuple, list)) or len(output) != 2:
                    raise RuntimeError(
                        f"language layer {next_layer} input norm differs"
                    )
                capture(component_name(next_layer - 1, "output"), output[1])
                if next_layer == ATTENTION_DIAGNOSTIC_LAYER:
                    capture(component_name(next_layer, "attention_input"), output[0])
                if next_layer == 3:
                    capture("layer_003_attention_input", output[0])

            return hook

        def attention_hook(
            _module: Any,
            args: Any,
            kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            output = kwargs.get("output")
            if output is None and len(args) > 1:
                output = args[1]
            capture(component_name(1, "attention_output"), output)
            state["active_linear_diagnostic"] = False

        def attention_pre_hook(
            _module: Any, _args: Any, _kwargs: dict[str, Any]
        ) -> None:
            state["active_linear_diagnostic"] = True

        def qkvz_projection_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            tensor = prefix._first_tensor(output)
            if tensor is None or tensor.ndim != 2 or tensor.shape[1] != 12288:
                raise RuntimeError("language layer 1 QKVZ geometry differs")
            capture("layer_001_qkvz_projection", tensor)

        def ba_projection_hook(_module: Any, _args: Any, output: Any) -> None:
            tensor = prefix._first_tensor(output)
            if tensor is None or tensor.ndim != 2 or tensor.shape[1] != 64:
                raise RuntimeError("language layer 1 BA geometry differs")
            projected_b, projected_a = tensor.chunk(2, dim=-1)
            capture("layer_001_a_projection", projected_a)
            capture("layer_001_b_projection", projected_b)

        def instrumented_causal_conv1d_fn(*args: Any, **kwargs: Any) -> Any:
            output = state["original_causal_conv1d_fn"](*args, **kwargs)
            if (
                state["active_linear_diagnostic"]
                and isinstance(output, torch.Tensor)
                and output.ndim == 2
            ):
                capture(component_name(1, "convolution"), output.transpose(0, 1))
            return output

        def instrumented_fused_post_conv_prep(
            *args: Any, **kwargs: Any
        ) -> Any:
            output = state["original_fused_post_conv_prep"](*args, **kwargs)
            if state["active_linear_diagnostic"]:
                if not isinstance(output, (tuple, list)) or len(output) != 5:
                    raise RuntimeError("language layer 1 post-conv geometry differs")
                for suffix, tensor in zip(
                    ("gdn_q", "gdn_k", "gdn_v", "gdn_g", "gdn_beta"),
                    output,
                    strict=True,
                ):
                    capture(component_name(1, suffix), tensor)
            return output

        def gated_norm_pre_hook(
            _module: Any, args: Any, kwargs: dict[str, Any]
        ) -> None:
            core = args[0] if args else kwargs.get("x")
            capture(component_name(1, "gdn_core"), core)

        def gated_norm_hook(_module: Any, _args: Any, output: Any) -> None:
            capture(component_name(1, "gdn_gated"), output)

        def post_attention_hook(_module: Any, _args: Any, output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("language layer 1 post-attention norm differs")
            capture(component_name(1, "post_attention_norm"), output[0])
            capture(component_name(1, "attention_residual"), output[1])

        def layer0_moe_hook(_module: Any, _args: Any, output: Any) -> None:
            capture("layer_000_combined_moe_output", output)

        def final_norm_hook(_module: Any, _args: Any, output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("language final norm residual contract differs")
            capture("language_final_norm", output[0])
            capture("layer_039_output", output[1])

        def instrument_router(layer_index: int, router: Any) -> None:
            original = router.select_experts

            def wrapped(*args: Any, **kwargs: Any) -> Any:
                output = original(*args, **kwargs)
                if not isinstance(output, (tuple, list)) or len(output) != 2:
                    raise RuntimeError(
                        f"language layer {layer_index} router output differs"
                    )
                weights, indices = output
                if not isinstance(indices, torch.Tensor):
                    raise RuntimeError(
                        f"language layer {layer_index} router indices unavailable"
                    )
                indices_i64 = indices.to(torch.int64)
                capture(component_name(layer_index, "router_indices"), indices_i64)
                if layer_index in DETAILED_ROUTER_LAYERS:
                    router_logits = kwargs.get("router_logits")
                    if router_logits is None and len(args) > 1:
                        router_logits = args[1]
                    if not isinstance(router_logits, torch.Tensor):
                        raise RuntimeError(
                            f"language layer {layer_index} router logits unavailable"
                        )
                    capture(component_name(layer_index, "router_logits"), router_logits)
                    capture(component_name(layer_index, "router_weights"), weights)
                    capture(
                        component_name(layer_index, "router_scores"),
                        torch.gather(router_logits.float(), 1, indices_i64),
                    )
                return output

            state["router_bindings"].append(
                {"router": router, "original": original}
            )
            router.select_experts = wrapped

        handles = state["handles"]
        handles.append(
            language.model.layers[0].mlp.register_forward_hook(layer0_moe_hook)
        )
        handles.append(
            diagnostic_layer.linear_attn.register_forward_pre_hook(
                attention_pre_hook, with_kwargs=True
            )
        )
        handles.append(
            diagnostic_layer.linear_attn.register_forward_hook(
                attention_hook, with_kwargs=True
            )
        )
        handles.append(
            diagnostic_layer.linear_attn.in_proj_qkvz.register_forward_hook(
                qkvz_projection_hook
            )
        )
        handles.append(
            diagnostic_layer.linear_attn.in_proj_ba.register_forward_hook(
                ba_projection_hook
            )
        )
        handles.append(
            diagnostic_layer.linear_attn.norm.register_forward_pre_hook(
                gated_norm_pre_hook, with_kwargs=True
            )
        )
        handles.append(
            diagnostic_layer.linear_attn.norm.register_forward_hook(
                gated_norm_hook
            )
        )
        handles.append(
            diagnostic_layer.post_attention_layernorm.register_forward_hook(
                post_attention_hook
            )
        )
        for next_layer in range(1, 40):
            handles.append(
                language.model.layers[next_layer].input_layernorm.register_forward_hook(
                    next_input_norm_hook(next_layer)
                )
            )
        handles.append(language.model.norm.register_forward_hook(final_norm_hook))
        for layer_index, layer in enumerate(language.model.layers):
            instrument_router(layer_index, layer.mlp.experts.router)
        gdn_module.causal_conv1d_fn = instrumented_causal_conv1d_fn
        gdn_module.fused_post_conv_prep = instrumented_fused_post_conv_prep
        chunk_module.chunk_local_cumsum = instrumented_chunk_local_cumsum
        chunk_module.chunk_scaled_dot_kkt_fwd = (
            instrumented_chunk_scaled_dot_kkt_fwd
        )
        chunk_module.solve_tril = instrumented_solve_tril
        chunk_module.recompute_w_u_fwd = instrumented_recompute_w_u_fwd
        chunk_module.chunk_gated_delta_rule_fwd_h = (
            instrumented_chunk_gated_delta_rule_fwd_h
        )
        setattr(root, STATE_ATTRIBUTE, state)
        return {
            "layer_count": len(language.model.layers),
            "attention_diagnostic_layer": ATTENTION_DIAGNOSTIC_LAYER,
            "detailed_router_layers": list(DETAILED_ROUTER_LAYERS),
        }


class FinalizeLanguageLayerOutputHooks:
    """Write the sparse attribution tensors and native-compatible ledger."""

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = prefix._find_model_root(model)
        state = getattr(root, STATE_ATTRIBUTE, None)
        if not isinstance(state, dict):
            raise RuntimeError("HTTP language attribution hooks were not installed")
        captures = state["captures"]
        projection_parts = (
            "layer_001_qkvz_projection",
            "layer_001_a_projection",
            "layer_001_b_projection",
        )
        if all(name in captures for name in projection_parts):
            captures[component_name(1, "fused_input_projection")] = torch.cat(
                tuple(captures[name] for name in projection_parts), dim=-1
            ).contiguous()
            captures[component_name(1, "a_projection")] = captures[
                "layer_001_a_projection"
            ]
            captures[component_name(1, "b_projection")] = captures[
                "layer_001_b_projection"
            ]
        missing = REQUIRED_COMPONENTS - set(captures)
        if missing:
            remove_hooks(root, state)
            raise RuntimeError(
                "missing HTTP language attribution components: "
                + ", ".join(sorted(missing))
            )
        output_root = Path(state["output_root"])
        case_id = state["case_id"]
        components = {
            name: prefix.write_raw_tensor(
                output_root, f"{case_id}/components/{name}", captures[name]
            )
            for name in sorted(REQUIRED_COMPONENTS)
        }
        labels = oracle_labels()
        case_root = output_root / case_id
        oracle_lines = [
            json.dumps(
                {
                    "event": "native_layer_oracle_tensor",
                    "label": label,
                    "file": Path(components[name]["path"])
                    .relative_to(case_id)
                    .as_posix(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for label, name in labels.items()
        ]
        (case_root / "oracle.jsonl").write_text(
            "\n".join(oracle_lines) + "\n", encoding="utf-8"
        )
        remove_hooks(root, state)
        return {
            "components": components,
            "oracle_labels": labels,
            "oracle_jsonl_sha256": prefix.sha256_file(
                case_root / "oracle.jsonl"
            ),
        }


class RemoveLanguageLayerOutputHooks:
    def __call__(self, model: Any) -> bool:
        root = prefix._find_model_root(model)
        state = getattr(root, STATE_ATTRIBUTE, None)
        if not isinstance(state, dict):
            return False
        remove_hooks(root, state)
        return True
