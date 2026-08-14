#!/usr/bin/env python3
"""Capture all vLLM language-layer outputs and router choices for VL drift attribution."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREFIX_CAPTURE = ROOT / "scripts/capture-vllm-vl-language-prefix-diagnostics.py"
SPEC = importlib.util.spec_from_file_location(
    "aima_vl_language_layer_output_capture", PREFIX_CAPTURE
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the frozen language-prefix capture")
prefix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prefix
SPEC.loader.exec_module(prefix)


SCHEMA = "aima-amd395-qwen36/vl-language-layer-output-diagnostic-oracle/v1"
STATE_ATTRIBUTE = "_aima_vl_language_layer_output_diagnostic_state"
OUTPUT_COMPONENTS = {f"layer_{layer:03d}_output" for layer in range(40)}
ROUTER_COMPONENTS = {
    f"layer_{layer:03d}_router_indices" for layer in range(40)
}
REQUIRED_COMPONENTS = OUTPUT_COMPONENTS | ROUTER_COMPONENTS | {
    "language_final_norm",
    "layer_000_combined_moe_output",
    "layer_003_attention_input",
}


def component_name(layer: int, suffix: str) -> str:
    return f"layer_{layer:03d}_{suffix}"


def oracle_labels() -> dict[str, str]:
    labels: dict[str, str] = {"language-final-norm": "language_final_norm"}
    for layer in range(40):
        prefix_text = f"layer-{layer:03d}-"
        labels[prefix_text + "return-layer_body-output"] = component_name(
            layer, "output"
        )
        labels[prefix_text + "return-layer_body-router_indices"] = component_name(
            layer, "router_indices"
        )
    return labels


class InstallLanguageLayerOutputHooks:
    """Serializable worker callable that captures the first divergent layer."""

    def __init__(self, *, output_root: str, case_id: str) -> None:
        self.output_root = output_root
        self.case_id = case_id

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = prefix._find_model_root(model)
        previous = getattr(root, STATE_ATTRIBUTE, None)
        if isinstance(previous, dict):
            prefix._remove_hooks(root, previous)
        language = root.language_model
        if len(language.model.layers) != 40:
            raise RuntimeError("language layer count differs from the frozen model")
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
                if next_layer == 3:
                    capture("layer_003_attention_input", output[0])

            return hook

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
                _weights, indices = output
                if not isinstance(indices, torch.Tensor):
                    raise RuntimeError(
                        f"language layer {layer_index} router indices unavailable"
                    )
                capture(
                    component_name(layer_index, "router_indices"),
                    indices.to(torch.int64),
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
            "layers": {
                str(index): prefix._module_identity(layer)
                for index, layer in enumerate(language.model.layers)
            },
            "final_norm": prefix._module_identity(language.model.norm),
        }


class FinalizeLanguageLayerOutputHooks:
    """Write the compact all-layer diagnostic tensors and native labels."""

    def __call__(self, model: Any) -> dict[str, Any]:
        root = prefix._find_model_root(model)
        state = getattr(root, STATE_ATTRIBUTE, None)
        if not isinstance(state, dict):
            raise RuntimeError("language layer-output hooks were not installed")
        captures = state["captures"]
        missing = REQUIRED_COMPONENTS - set(captures)
        if missing:
            prefix._remove_hooks(root, state)
            raise RuntimeError(
                "missing language layer-output components: "
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
        prefix._remove_hooks(root, state)
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
        prefix._remove_hooks(root, state)
        return True


def main() -> int:
    import cloudpickle

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    prefix.SCHEMA = SCHEMA
    prefix.STATE_ATTRIBUTE = STATE_ATTRIBUTE
    prefix.REQUIRED_COMPONENTS = REQUIRED_COMPONENTS
    prefix.InstallLanguagePrefixDiagnosticHooks = InstallLanguageLayerOutputHooks
    prefix.FinalizeLanguagePrefixDiagnosticHooks = FinalizeLanguageLayerOutputHooks
    prefix.RemoveLanguagePrefixDiagnosticHooks = RemoveLanguageLayerOutputHooks
    prefix._oracle_labels = oracle_labels
    args = prefix.parse_args()
    result = prefix.capture(args)
    result.pop("integrity", None)
    result["schema"] = SCHEMA
    result["capture_script"] = prefix.file_component(
        Path(__file__), "scripts/capture-vllm-vl-language-layer-outputs.py"
    )
    result["diagnostic_scope"] = (
        "all 40 accumulated layer outputs, all router index rows, and final norm"
    )
    result = prefix.seal_manifest(result)
    manifest_path = args.output_root.resolve() / "manifest.json"
    prefix.atomic_json(manifest_path, result)
    print(
        json.dumps(
            {
                "output": str(manifest_path),
                "cases": len(result["cases"]),
                "sha256": prefix.sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"capture VL language layer outputs: {error}", file=sys.stderr)
        raise SystemExit(1)
