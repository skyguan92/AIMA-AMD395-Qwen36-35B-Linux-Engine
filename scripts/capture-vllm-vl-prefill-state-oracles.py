#!/usr/bin/env python3
"""Capture vLLM linear-attention state at the prefill-to-decode handoff."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import importlib
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

from aima_engine.vl_generation_oracle import (  # noqa: E402
    CASE_CONTRACTS,
    CASE_ORDER,
    validate_generation_oracle_manifest,
)
from aima_engine.vl_oracle import (  # noqa: E402
    canonical_int_list_sha256,
    write_raw_tensor,
)
from aima_engine.vl_prefill_state_oracle import (  # noqa: E402
    CONV_STATE_SHAPE,
    LINEAR_LAYER_INDICES,
    RECURRENT_STATE_SHAPE,
    STATE_COMPONENT_NAMES,
    VL_PREFILL_STATE_ORACLE_SCHEMA,
    validate_vl_prefill_state_oracle_manifest,
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
STATE_ATTRIBUTE = "_aima_vl_prefill_state_oracle_state"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load capture dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    attention = state.get("layer0_attention")
    original = state.get("original_forward_core")
    if attention is not None and original is not None:
        attention._forward_core = original
    if hasattr(root, STATE_ATTRIBUTE):
        delattr(root, STATE_ATTRIBUTE)


class InstallPrefillStateHooks:
    """Freeze every linear cache before layer 0 consumes decode token one."""

    def __init__(self, *, case_id: str) -> None:
        self.case_id = case_id

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = _find_model_root(model)
        previous = getattr(root, STATE_ATTRIBUTE, None)
        if isinstance(previous, dict):
            _remove_hooks(root, previous)
        language = root.language_model
        if len(language.model.layers) != 40:
            raise RuntimeError("language layer count differs from the frozen model")
        layer0_attention = language.model.layers[0].linear_attn
        gdn_module = importlib.import_module(
            "vllm.model_executor.layers.mamba.gdn_linear_attn"
        )
        original_forward_core = layer0_attention._forward_core
        state: dict[str, Any] = {
            "case_id": self.case_id,
            "captures": {},
            "capture_decode_call": 0,
            "cache_index": None,
            "conv_state_dim_first": None,
            "layer0_attention": layer0_attention,
            "original_forward_core": original_forward_core,
        }

        def capture_prefill_states() -> None:
            context = gdn_module.get_forward_context()
            metadata = context.attn_metadata
            if not isinstance(metadata, dict):
                return
            metadata = metadata[layer0_attention.prefix]
            if (
                metadata.num_prefills != 0
                or metadata.num_decodes <= 0
                or metadata.num_actual_tokens != 1
            ):
                return
            state["capture_decode_call"] += 1
            if state["captures"]:
                return
            indices = metadata.non_spec_state_indices_tensor.flatten()
            if indices.numel() < 1:
                raise RuntimeError("decode state index is unavailable")
            cache_index = int(indices[0].item())
            if cache_index < 0:
                raise RuntimeError("decode state index is invalid")
            dim_first = bool(gdn_module.is_conv_state_dim_first())
            state["cache_index"] = cache_index
            state["conv_state_dim_first"] = dim_first
            for layer_index in LINEAR_LAYER_INDICES:
                layer = language.model.layers[layer_index]
                if getattr(layer, "layer_type", None) != "linear_attention":
                    raise RuntimeError(
                        f"language layer {layer_index} is not linear attention"
                    )
                attention = layer.linear_attn
                conv_cache, recurrent_cache = attention.kv_cache
                conv_cache = (
                    conv_cache
                    if dim_first
                    else conv_cache.transpose(-1, -2)
                )
                conv = conv_cache[cache_index].detach().contiguous()
                recurrent = recurrent_cache[cache_index].detach().contiguous()
                if tuple(conv.shape) != CONV_STATE_SHAPE:
                    raise RuntimeError(
                        f"layer {layer_index} conv state geometry changed: "
                        f"{tuple(conv.shape)}"
                    )
                if tuple(recurrent.shape) != RECURRENT_STATE_SHAPE:
                    raise RuntimeError(
                        f"layer {layer_index} recurrent state geometry changed: "
                        f"{tuple(recurrent.shape)}"
                    )
                if conv.dtype != torch.bfloat16 or recurrent.dtype != torch.float32:
                    raise RuntimeError(
                        f"layer {layer_index} state dtype changed: "
                        f"{conv.dtype}/{recurrent.dtype}"
                    )
                state["captures"][
                    f"layer_{layer_index:03d}_conv_state"
                ] = conv.cpu()
                state["captures"][
                    f"layer_{layer_index:03d}_recurrent_state"
                ] = recurrent.cpu()

        def instrumented_forward_core(*args: Any, **kwargs: Any) -> Any:
            capture_prefill_states()
            return original_forward_core(*args, **kwargs)

        layer0_attention._forward_core = instrumented_forward_core
        setattr(root, STATE_ATTRIBUTE, state)
        return {"installed": True}


class FinalizePrefillStateHooks:
    def __init__(self, *, output_root: str) -> None:
        self.output_root = output_root

    def __call__(self, model: Any) -> dict[str, Any]:
        root = _find_model_root(model)
        state = getattr(root, STATE_ATTRIBUTE, None)
        if not isinstance(state, dict):
            raise RuntimeError("VL prefill state hooks were not installed")
        captures = state["captures"]
        missing = set(STATE_COMPONENT_NAMES) - set(captures)
        if missing:
            _remove_hooks(root, state)
            raise RuntimeError(
                "missing prefill state components: " + ", ".join(sorted(missing))
            )
        if state["capture_decode_call"] < 1:
            _remove_hooks(root, state)
            raise RuntimeError("prefill state capture never reached decode")

        output_root = Path(self.output_root)
        case_id = state["case_id"]
        components = {
            name: write_raw_tensor(
                output_root, f"{case_id}/components/{name}", captures[name]
            )
            for name in STATE_COMPONENT_NAMES
        }
        case_root = output_root / case_id
        ledger_lines = [
            json.dumps(
                {
                    "event": "native_layer_oracle_tensor",
                    "label": name,
                    "file": Path(components[name]["path"])
                    .relative_to(case_id)
                    .as_posix(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for name in STATE_COMPONENT_NAMES
        ]
        ledger = case_root / "oracle.jsonl"
        ledger.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")
        result = {
            "components": components,
            "oracle_jsonl": file_component(ledger, f"{case_id}/oracle.jsonl"),
            "capture_decode_call": 1,
            "observed_decode_calls": state["capture_decode_call"],
            "cache_index": state["cache_index"],
            "conv_state_dim_first": state["conv_state_dim_first"],
        }
        _remove_hooks(root, state)
        return result


class RemovePrefillStateHooks:
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
        raise ValueError(f"VL prefill state oracle root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source = git_identity(ROOT)
    if source["dirty"]:
        raise ValueError("VL prefill state capture source must be a clean commit")

    generation = load_module(
        GENERATION_CAPTURE, "aima_vl_prefill_state_generation_capture"
    )
    base = generation.load_module(
        generation.BASE_CAPTURE, "aima_vl_prefill_state_base_capture"
    )
    probe = generation.load_module(
        generation.CAPABILITY_PROBE, "aima_vl_prefill_state_capability_probe"
    )
    fixture_root = args.fixture_root.resolve()
    reference, _launch, _processor = base._require_inputs(
        reference_path=args.reference_manifest,
        launch_path=args.launch_config,
        processor_path=args.processor_probe,
        fixture_root=fixture_root,
    )
    if socket.gethostname() != reference.get("host", {}).get("hostname"):
        raise ValueError("VL prefill state host differs from the frozen reference")

    capability = load_json_object(args.capability_manifest)
    errors = generation.validate_capability_manifest(capability)
    if errors:
        raise ValueError("invalid capability manifest:\n- " + "\n- ".join(errors))
    render = load_json_object(args.api_render_manifest)
    errors = generation.validate_api_render_manifest(render)
    if errors:
        raise ValueError("invalid API render manifest:\n- " + "\n- ".join(errors))
    generation_path = args.generation_oracle_manifest.resolve()
    generation_root = args.generation_oracle_root.resolve()
    generation_oracle = load_json_object(generation_path)
    errors = validate_generation_oracle_manifest(
        generation_oracle, oracle_root=generation_root
    )
    if errors:
        raise ValueError("invalid generation oracle:\n- " + "\n- ".join(errors))
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
        raise RuntimeError("VL prefill state tool case set changed")

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
                raise RuntimeError(f"VL prefill state request drifted: {case_id}")

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
                raise RuntimeError(f"VL prefill state prompt differs: {case_id}")

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
            sampling = SamplingParams(
                temperature=0,
                max_tokens=2,
                seed=0,
                structured_outputs=structured,
            )

            def install_callable(model: Any) -> dict[str, Any]:
                return InstallPrefillStateHooks(case_id=case_id)(model)

            def finalize_callable(model: Any) -> dict[str, Any]:
                return FinalizePrefillStateHooks(
                    output_root=str(output_root)
                )(model)

            def cleanup_callable(model: Any) -> bool:
                return RemovePrefillStateHooks()(model)

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
                raise RuntimeError("VL prefill state capture requires TP=1")
            output_token_ids = [
                int(token) for token in outputs[0].outputs[0].token_ids
            ]
            expected_output_ids = generation_case["generation"][
                "output_token_ids"
            ][:2]
            if output_token_ids != expected_output_ids:
                raise RuntimeError(
                    f"VL prefill state output prefix changed: {case_id}"
                )
            cases.append(
                {
                    "case_id": case_id,
                    "passed": True,
                    "prompt_tokens": len(prompt_token_ids),
                    "prompt_token_ids_sha256": canonical_int_list_sha256(
                        prompt_token_ids
                    ),
                    "output_prefix_token_ids": output_token_ids,
                    "output_prefix_token_ids_sha256": canonical_int_list_sha256(
                        output_token_ids
                    ),
                    **finalization[0],
                }
            )
            print(
                json.dumps(
                    {
                        "event": "case_complete",
                        "case_id": case_id,
                        "states": len(finalization[0]["components"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = seal_manifest(
        {
            "schema": VL_PREFILL_STATE_ORACLE_SCHEMA,
            "captured_at": base.utc_now(),
            "complete": True,
            "qualified_for_state_attribution": True,
            "scope": "two-fixed-vllm-vl-prefill-to-first-decode-state-sets",
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in (
                        Path(__file__).resolve(),
                        ROOT / "aima_engine/vl_prefill_state_oracle.py",
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
                generation_path,
                "benchmarks/results/vl-generation-oracle-v0.1.0.json",
            ),
            "capture_control_plane": {
                "vllm_allow_insecure_serialization": True,
                "skip_mm_profiling": True,
                "maximum_tokens_per_case": 2,
                "capture_point": "before-layer-0-first-decode-update",
                "product_runtime_dependency": False,
            },
            "oracle_root": args.oracle_root_label,
            "cases": cases,
            "decision": {
                "two_prompt_prefixes_exact": len(cases) == 2,
                "two_prefill_state_sets_captured": len(cases) == 2,
                "g1_passed": False,
                "g2_passed": False,
                "g3_passed": False,
                "g4_passed": False,
                "g5_passed": False,
            },
        }
    )
    errors = validate_vl_prefill_state_oracle_manifest(
        manifest, oracle_root=output_root
    )
    if errors:
        raise RuntimeError(
            "VL prefill state oracle validation failed:\n- "
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
        "--oracle-root-label",
        default="benchmarks/oracles/vl-prefill-state-v0.1.0",
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
                "qualified": manifest["qualified_for_state_attribution"],
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
