#!/usr/bin/env python3
"""Capture sparse HTTP VL language drift-attribution boundaries."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

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
    for layer in DETAILED_ROUTER_LAYERS:
        router_prefix = f"layer-{layer:03d}-return-layer_body-"
        for suffix in ("router_logits", "router_scores", "router_weights"):
            labels[router_prefix + suffix] = component_name(layer, suffix)
    return labels


def remove_hooks(root: Any, state: dict[str, Any]) -> None:
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
        state: dict[str, Any] = {
            "output_root": self.output_root,
            "case_id": self.case_id,
            "captures": {},
            "handles": [],
            "router_bindings": [],
            "attention_bindings": [],
        }

        def capture(name: str, value: Any) -> None:
            tensor = prefix._first_tensor(value)
            if tensor is None or name in state["captures"]:
                return
            if tensor.ndim == 0 or tensor.shape[0] <= 1:
                return
            state["captures"][name] = tensor.detach().contiguous().cpu()

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
            diagnostic_layer.linear_attn.register_forward_hook(
                attention_hook, with_kwargs=True
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
        setattr(root, STATE_ATTRIBUTE, state)
        return {
            "layer_count": len(language.model.layers),
            "attention_diagnostic_layer": ATTENTION_DIAGNOSTIC_LAYER,
            "detailed_router_layers": list(DETAILED_ROUTER_LAYERS),
        }


class FinalizeLanguageLayerOutputHooks:
    """Write the sparse attribution tensors and native-compatible ledger."""

    def __call__(self, model: Any) -> dict[str, Any]:
        root = prefix._find_model_root(model)
        state = getattr(root, STATE_ATTRIBUTE, None)
        if not isinstance(state, dict):
            raise RuntimeError("HTTP language attribution hooks were not installed")
        captures = state["captures"]
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
