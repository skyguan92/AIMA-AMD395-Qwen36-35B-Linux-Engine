#!/usr/bin/env python3
"""Capture pinned vLLM language-layer-0 attribution boundaries for VL cases."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_oracle import (  # noqa: E402
    canonical_int_list_sha256,
    write_raw_tensor,
)
from aima_engine.vl_reference import (  # noqa: E402
    MODEL_REVISION,
    PINNED_PACKAGES,
    REFERENCE_ATTENTION_BACKEND,
    atomic_json,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
    verify_manifest_integrity,
)


SCHEMA = "aima-amd395-qwen36/vl-language-layer0-diagnostic-oracle/v1"
VL_ORACLE_SHA256 = "87dcdf76b7251f78da01a2a5f4312a9fb5c7d07a1ca2b2420566e77930f23d44"
CASE_IDS = (
    "image_local_png",
    "video_local_mp4",
    "multi_image",
    "multi_video",
    "mixed_image_video",
)
SOURCE_HASHES = {
    "vllm.model_executor.models.qwen3_5": (
        "6cbbe29a102a5e6207a1b1828976cbf442eca0fe9f5895b7e1ca74542bb5e8c0"
    ),
    "vllm.model_executor.models.qwen3_next": (
        "0b3a7f577757712b48a09d4ae849d091949be35975de3eb95da81f7ea5670934"
    ),
    "vllm.model_executor.layers.mamba.gdn_linear_attn": (
        "931affd80c786ffc224e01314e8f0a5482cbb878d406a0a5fac3fd76ea130dd2"
    ),
    "vllm.model_executor.layers.layernorm": (
        "4c4be7e915fa2977dee683a2304a9469719785449ed204dec197c30921fe4d1e"
    ),
    "vllm.model_executor.layers.linear": (
        "715ba882f6029d2cba21314b0e189a1a80947128b7cd4d505f4af9a86c3cc542"
    ),
    "vllm.model_executor.layers.fused_moe.shared_fused_moe": (
        "c944521d34792bf77bacb46e0de626d65dc61bd90f933a99e6a803e39bf0df1d"
    ),
}
REQUIRED_COMPONENTS = {
    "input_norm",
    "gdn_qkvz_projection",
    "gdn_qkv_projection",
    "gdn_z_projection",
    "gdn_ba_projection",
    "gdn_b_projection",
    "gdn_a_projection",
    "gdn_fused_input_projection",
    "gdn_core",
    "gdn_z",
    "gdn_gated_norm",
    "linear_attention_output",
    "attention_residual",
    "post_attention_norm",
    "router_logits",
    "shared_gate_logits",
    "shared_moe_output",
    "routed_moe_output",
    "combined_moe_output",
    "layer0_first_tensor",
    "layer_output",
}
ORACLE_LABELS = {
    "launch-000-out": "input_norm",
    "diagnostic-qkv": "gdn_qkv_projection",
    "diagnostic-z": "gdn_z_projection",
    "diagnostic-a": "gdn_a_projection",
    "diagnostic-b": "gdn_b_projection",
    "diagnostic-fused-input": "gdn_fused_input_projection",
    "launch-008-o": "gdn_core",
    "return-linear_attention-gated_out": "gdn_gated_norm",
    "return-linear_attention-output": "linear_attention_output",
    "launch-009-residual_out": "attention_residual",
    "launch-009-norm_out": "post_attention_norm",
    "diagnostic-h2": "post_attention_norm",
    "diagnostic-shared_out": "shared_moe_output",
    "diagnostic-routed_moe": "routed_moe_output",
    "diagnostic-moe_out": "combined_moe_output",
    "diagnostic-output": "layer_output",
}


def _load_base_capture() -> Any:
    path = ROOT / "scripts/capture-vllm-vl-oracles.py"
    spec = importlib.util.spec_from_file_location(
        "aima_vl_language_layer0_base_capture", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen VL capture helpers")
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


def _module_identity(module: Any) -> str:
    return f"{type(module).__module__}.{type(module).__name__}"


class InstallLanguageLayer0DiagnosticHooks:
    """Serializable worker callable for the smallest useful layer-0 ledger."""

    def __init__(self, *, output_root: str, case_id: str) -> None:
        self.output_root = output_root
        self.case_id = case_id

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = _find_model_root(model)
        previous = getattr(root, "_aima_vl_language_layer0_diagnostic_state", None)
        if isinstance(previous, dict):
            for handle in previous.get("handles", []):
                handle.remove()

        language = root.language_model
        if len(language.model.layers) < 2:
            raise RuntimeError("language model has no layer-1 residual boundary")
        layer0 = language.model.layers[0]
        layer1 = language.model.layers[1]
        if getattr(layer0, "layer_type", None) != "linear_attention":
            raise RuntimeError("language layer 0 is not linear attention")
        linear = layer0.linear_attn
        mlp = layer0.mlp
        state: dict[str, Any] = {
            "output_root": self.output_root,
            "case_id": self.case_id,
            "captures": {},
            "handles": [],
        }

        def capture(name: str, value: Any) -> None:
            tensor = _first_tensor(value)
            if tensor is None or name in state["captures"]:
                return
            # Every prompt case has more than one token. Decode calls are
            # deliberately excluded from this prefill-only attribution ledger.
            if tensor.ndim == 0 or tensor.shape[0] <= 1:
                return
            state["captures"][name] = tensor.detach().contiguous().cpu()

        def output_hook(name: str):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                capture(name, output)

            return hook

        def linear_hook(
            _module: Any,
            args: Any,
            kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            output = kwargs.get("output")
            if output is None and len(args) > 1:
                output = args[1]
            capture("linear_attention_output", output)

        def gated_norm_pre_hook(
            _module: Any, args: Any, kwargs: dict[str, Any]
        ) -> None:
            core = args[0] if args else kwargs.get("x")
            gate = args[1] if len(args) > 1 else kwargs.get("z")
            capture("gdn_core", core)
            capture("gdn_z", gate)

        def qkvz_projection_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            tensor = _first_tensor(output)
            if tensor is None or tensor.ndim != 2 or tensor.shape[1] != 12288:
                raise RuntimeError("GDN QKVZ projection geometry differs")
            capture("gdn_qkvz_projection", tensor)
            capture("gdn_qkv_projection", tensor[:, :8192])
            capture("gdn_z_projection", tensor[:, 8192:])

        def ba_projection_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            tensor = _first_tensor(output)
            if tensor is None or tensor.ndim != 2 or tensor.shape[1] != 64:
                raise RuntimeError("GDN BA projection geometry differs")
            capture("gdn_ba_projection", tensor)
            projected_b, projected_a = tensor.chunk(2, dim=-1)
            capture("gdn_b_projection", projected_b)
            capture("gdn_a_projection", projected_a)

        def post_attention_hook(_module: Any, _args: Any, output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("post-attention norm did not return two tensors")
            capture("post_attention_norm", output[0])
            capture("attention_residual", output[1])

        def experts_hook(_module: Any, _args: Any, output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("shared fused MoE did not return two tensors")
            # The pinned default runner returns (shared, routed).
            capture("shared_moe_output", output[0])
            capture("routed_moe_output", output[1])

        def layer0_hook(_module: Any, _args: Any, output: Any) -> None:
            capture("layer0_first_tensor", output)

        def layer1_norm_hook(_module: Any, _args: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)) and len(output) == 2:
                capture("layer_output", output[1])

        hooks = state["handles"]
        hooks.append(layer0.input_layernorm.register_forward_hook(output_hook("input_norm")))
        hooks.append(
            linear.in_proj_qkvz.register_forward_hook(
                qkvz_projection_hook
            )
        )
        hooks.append(
            linear.in_proj_ba.register_forward_hook(ba_projection_hook)
        )
        hooks.append(
            linear.norm.register_forward_pre_hook(
                gated_norm_pre_hook, with_kwargs=True
            )
        )
        hooks.append(linear.norm.register_forward_hook(output_hook("gdn_gated_norm")))
        hooks.append(linear.register_forward_hook(linear_hook, with_kwargs=True))
        hooks.append(layer0.post_attention_layernorm.register_forward_hook(post_attention_hook))
        hooks.append(mlp.gate.register_forward_hook(output_hook("router_logits")))
        hooks.append(
            mlp.shared_expert_gate.register_forward_hook(
                output_hook("shared_gate_logits")
            )
        )
        hooks.append(mlp.experts.register_forward_hook(experts_hook))
        hooks.append(mlp.register_forward_hook(output_hook("combined_moe_output")))
        hooks.append(layer0.register_forward_hook(layer0_hook))
        hooks.append(layer1.input_layernorm.register_forward_hook(layer1_norm_hook))
        root._aima_vl_language_layer0_diagnostic_state = state
        return {
            "layer0": _module_identity(layer0),
            "input_norm": _module_identity(layer0.input_layernorm),
            "linear_attention": _module_identity(linear),
            "gated_norm": _module_identity(linear.norm),
            "post_attention_norm": _module_identity(
                layer0.post_attention_layernorm
            ),
            "moe": _module_identity(mlp),
            "experts": _module_identity(mlp.experts),
            "layer1_input_norm": _module_identity(layer1.input_layernorm),
        }


class FinalizeLanguageLayer0DiagnosticHooks:
    """Write diagnostic tensors and a native-oracle compatibility ledger."""

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = _find_model_root(model)
        state = getattr(root, "_aima_vl_language_layer0_diagnostic_state", None)
        if not isinstance(state, dict):
            raise RuntimeError("language layer-0 diagnostic hooks were not installed")
        for handle in state.get("handles", []):
            handle.remove()
        captures = state["captures"]
        if {
            "gdn_qkvz_projection",
            "gdn_a_projection",
            "gdn_b_projection",
        }.issubset(captures):
            # The native derived layout is QKVZ/A/B, while vLLM's second
            # merged projection returns B/A. Reorder only the two 32-wide
            # views so this diagnostic compares arithmetic, not packing.
            captures["gdn_fused_input_projection"] = torch.cat(
                (
                    captures["gdn_qkvz_projection"],
                    captures["gdn_a_projection"],
                    captures["gdn_b_projection"],
                ),
                dim=-1,
            ).contiguous()
        missing = REQUIRED_COMPONENTS - set(captures)
        if missing:
            raise RuntimeError(
                "missing language layer-0 diagnostic components: "
                + ", ".join(sorted(missing))
            )
        output_root = Path(state["output_root"])
        case_id = state["case_id"]
        components = {
            name: write_raw_tensor(
                output_root, f"{case_id}/components/{name}", captures[name]
            )
            for name in sorted(REQUIRED_COMPONENTS)
        }
        case_root = output_root / case_id
        oracle_lines = []
        for label, component_name in ORACLE_LABELS.items():
            component_path = Path(components[component_name]["path"])
            relative_path = component_path.relative_to(case_id)
            oracle_lines.append(
                json.dumps(
                    {
                        "event": "native_layer_oracle_tensor",
                        "label": label,
                        "file": relative_path.as_posix(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        (case_root / "oracle.jsonl").write_text(
            "\n".join(oracle_lines) + "\n", encoding="utf-8"
        )
        delattr(root, "_aima_vl_language_layer0_diagnostic_state")
        return {
            "components": components,
            "oracle_labels": dict(ORACLE_LABELS),
            "oracle_jsonl_sha256": sha256_file(case_root / "oracle.jsonl"),
        }


class RemoveLanguageLayer0DiagnosticHooks:
    def __call__(self, model: Any) -> bool:
        root = _find_model_root(model)
        state = getattr(root, "_aima_vl_language_layer0_diagnostic_state", None)
        if not isinstance(state, dict):
            return False
        for handle in state.get("handles", []):
            handle.remove()
        delattr(root, "_aima_vl_language_layer0_diagnostic_state")
        return True


def _runtime_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PINNED_PACKAGES}


def _verify_serving_sources() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for module_name, expected_sha256 in SOURCE_HASHES.items():
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"frozen serving source differs: {module_name}")
        marker = "/site-packages/"
        path_text = path.as_posix()
        relative = path_text.split(marker, 1)[1] if marker in path_text else path.name
        result[module_name] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }
    return result


def _read_component(root: Path, record: Mapping[str, Any]) -> bytes:
    path = root / str(record["path"])
    payload = path.read_bytes()
    if len(payload) != int(record["bytes"]):
        raise RuntimeError(f"diagnostic tensor byte count differs: {path.name}")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise RuntimeError(f"diagnostic tensor hash differs: {path.name}")
    return payload


def _compare_component(
    *,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    actual_root: Path,
    expected_root: Path,
) -> dict[str, Any]:
    actual_payload = _read_component(actual_root, actual)
    expected_payload = _read_component(expected_root, expected)
    shape_exact = actual.get("shape") == expected.get("shape")
    dtype_exact = actual.get("dtype") == expected.get("dtype")
    payload_exact = actual_payload == expected_payload
    return {
        "shape_exact": shape_exact,
        "dtype_exact": dtype_exact,
        "payload_exact": payload_exact,
        "exact": shape_exact and dtype_exact and payload_exact,
        "actual_sha256": actual["sha256"],
        "expected_sha256": expected["sha256"],
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
        raise ValueError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required for the isolated "
            "offline apply_model diagnostic hooks"
        )
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
    import cloudpickle

    base = _load_base_capture()
    output_root = args.output_root.resolve()
    fixture_root = args.fixture_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"diagnostic output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source = git_identity(ROOT)
    if source["dirty"]:
        raise ValueError("language layer-0 diagnostic source must be a clean commit")

    reference, _launch, _processor = base._require_inputs(
        reference_path=args.reference_manifest,
        launch_path=args.launch_config,
        processor_path=args.processor_probe,
        fixture_root=fixture_root,
    )
    if socket.gethostname() != reference.get("host", {}).get("hostname"):
        raise ValueError("diagnostic host differs from the frozen reference")
    vl_manifest_path = args.vl_oracle_manifest.resolve()
    if sha256_file(vl_manifest_path) != VL_ORACLE_SHA256:
        raise ValueError("VL oracle manifest differs from the frozen contract")
    vl_manifest = load_json_object(vl_manifest_path)
    integrity_errors = verify_manifest_integrity(vl_manifest)
    if integrity_errors:
        raise ValueError(
            "VL oracle manifest integrity failed:\n- "
            + "\n- ".join(integrity_errors)
        )
    versions = _runtime_versions()
    for name, expected in PINNED_PACKAGES.items():
        actual = versions.get(name)
        if actual != expected and not actual.startswith(expected + "."):
            raise ValueError(f"runtime pin mismatch for {name}: {actual!r}")
    serving_sources = _verify_serving_sources()

    selected_case_ids = CASE_IDS if args.case_id == "all" else (args.case_id,)
    specs = {item["case_id"]: item for item in base.CASE_SPECS}
    vl_cases = {item["case_id"]: item for item in vl_manifest["cases"]}
    if set(CASE_IDS) != set(specs) or set(CASE_IDS) != set(vl_cases):
        raise ValueError("frozen VL case set differs from the diagnostic contract")

    # Worker subprocesses cannot import a module created from this hyphenated
    # filename. Serialize the hook definitions by value in the isolated RPC.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    llm_kwargs = base._llm_kwargs(args.model_dir.resolve(), fixture_root)
    llm_kwargs["skip_mm_profiling"] = True
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,
        seed=0,
    )
    cases: list[dict[str, Any]] = []
    try:
        for case_id in selected_case_ids:
            print(
                json.dumps({"event": "diagnostic_case_start", "case_id": case_id}),
                flush=True,
            )
            llm.reset_mm_cache()
            llm.llm_engine.reset_encoder_cache()
            messages = base._build_messages(specs[case_id], fixture_root)
            engine_input = llm._preprocess_chat_one(
                messages, chat_template_content_format="openai"
            )
            prompt_token_ids = [int(item) for item in engine_input["prompt_token_ids"]]
            prompt_sha256 = canonical_int_list_sha256(prompt_token_ids)
            frozen_prompt_sha256 = vl_cases[case_id]["processor"][
                "prompt_token_ids_sha256"
            ]
            if prompt_sha256 != frozen_prompt_sha256:
                raise RuntimeError(f"prompt tokens differ from frozen case: {case_id}")

            def install_callable(model: Any) -> dict[str, Any]:
                return InstallLanguageLayer0DiagnosticHooks(
                    output_root=str(output_root), case_id=case_id
                )(model)

            def finalize_callable(model: Any) -> dict[str, Any]:
                return FinalizeLanguageLayer0DiagnosticHooks()(model)

            def cleanup_callable(model: Any) -> bool:
                return RemoveLanguageLayer0DiagnosticHooks()(model)

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
                or len(installation) != 1
                or len(finalization) != 1
            ):
                raise RuntimeError("diagnostic capture requires one request and TP=1")
            record = finalization[0]
            expected = vl_cases[case_id]["boundaries"]["language_layer_0"]
            combined_comparison = _compare_component(
                actual=record["components"]["combined_moe_output"],
                expected=expected,
                actual_root=output_root,
                expected_root=args.vl_oracle_root.resolve(),
            )
            layer_comparison = _compare_component(
                actual=record["components"]["layer0_first_tensor"],
                expected=expected,
                actual_root=output_root,
                expected_root=args.vl_oracle_root.resolve(),
            )
            if not combined_comparison["exact"] or not layer_comparison["exact"]:
                raise RuntimeError(
                    f"diagnostic capture did not reproduce frozen layer 0: {case_id}"
                )
            cases.append(
                {
                    "case_id": case_id,
                    "prompt_tokens": len(prompt_token_ids),
                    "prompt_token_ids_sha256": prompt_sha256,
                    "model_modules": installation[0],
                    "components": record["components"],
                    "oracle_labels": record["oracle_labels"],
                    "oracle_jsonl_sha256": record["oracle_jsonl_sha256"],
                    "frozen_combined_moe_comparison": combined_comparison,
                    "frozen_layer0_hook_comparison": layer_comparison,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "diagnostic_case_complete",
                        "case_id": case_id,
                        "prompt_tokens": len(prompt_token_ids),
                    }
                ),
                flush=True,
            )
    finally:
        del llm

    return seal_manifest(
        {
            "schema": SCHEMA,
            "complete": True,
            "qualified_for_attribution_only": True,
            "acceptance_threshold_unchanged": True,
            "source": source,
            "model": {
                "revision": MODEL_REVISION,
                "reference_manifest": file_component(
                    args.reference_manifest,
                    "benchmarks/results/vl-reference-manifest.json",
                ),
                "vl_oracle_manifest_sha256": VL_ORACLE_SHA256,
            },
            "runtime": versions,
            "reference": {
                "attention_backend": REFERENCE_ATTENTION_BACKEND,
                "gdn_prefill_backend": "triton",
                "enforce_eager": True,
                "skip_mm_profiling": True,
                "vllm_allow_insecure_serialization": True,
                "serving_sources": serving_sources,
            },
            "case_selector": args.case_id,
            "required_components": sorted(REQUIRED_COMPONENTS),
            "oracle_labels": dict(ORACLE_LABELS),
            "cases": cases,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--processor-probe", type=Path, required=True)
    parser.add_argument("--vl-oracle-manifest", type=Path, required=True)
    parser.add_argument("--vl-oracle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-id", choices=("all", *CASE_IDS), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = capture(args)
    manifest_path = args.output_root.resolve() / "manifest.json"
    atomic_json(manifest_path, result)
    print(
        json.dumps(
            {
                "output": str(manifest_path),
                "cases": len(result["cases"]),
                "sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"capture VL language layer-0 diagnostics: {error}", file=sys.stderr)
        raise SystemExit(1)
