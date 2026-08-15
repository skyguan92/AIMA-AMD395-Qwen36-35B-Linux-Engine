#!/usr/bin/env python3
"""Capture fixed-vLLM full-vocabulary logits at two VL generation drifts."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
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

from aima_engine.vl_capability import (  # noqa: E402
    validate_api_render_manifest,
    validate_capability_manifest,
)
from aima_engine.vl_generation_oracle import (  # noqa: E402
    CASE_CONTRACTS,
    CASE_ORDER,
    GENERATION_ORACLE_SCHEMA,
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
    canonical_json_sha256,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
)


BASE_CAPTURE = ROOT / "scripts/capture-vllm-vl-oracles.py"
CAPABILITY_PROBE = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"


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
    if isinstance(value, dict):
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


class InstallGenerationLogitsHook:
    """Serializable worker hook for one generated-token distribution."""

    def __init__(self, *, case_id: str, output_index: int) -> None:
        self.case_id = case_id
        self.output_index = output_index

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = _find_model_root(model)
        previous = getattr(root, "_aima_vl_generation_logits_state", None)
        if isinstance(previous, dict):
            for handle in previous.get("handles", []):
                handle.remove()
        state: dict[str, Any] = {
            "case_id": self.case_id,
            "target_output_index": self.output_index,
            "prefill_calls": 0,
            "decode_calls": 0,
            "call_shapes": [],
            "captured_output_index": None,
            "logits": None,
            "handles": [],
        }

        def logits_hook(_module: Any, _args: Any, output: Any) -> None:
            logits = _first_tensor(output)
            if logits is None or logits.ndim != 2:
                return
            if logits.shape[1] != MODEL_VOCABULARY_SIZE:
                raise RuntimeError(
                    f"generation vocabulary changed: {list(logits.shape)}"
                )
            state["call_shapes"].append(list(logits.shape))
            if logits.shape[0] > 1:
                state["prefill_calls"] += 1
                output_index = 0
            elif logits.shape[0] == 1:
                state["decode_calls"] += 1
                output_index = state["decode_calls"]
            else:
                return
            if (
                output_index == state["target_output_index"]
                and state["logits"] is None
            ):
                state["captured_output_index"] = output_index
                state["logits"] = logits[-1].detach().float().contiguous().cpu()

        handle = root.language_model.logits_processor.register_forward_hook(
            logits_hook
        )
        state["handles"].append(handle)
        root._aima_vl_generation_logits_state = state
        return {"installed": True}


class FinalizeGenerationLogitsHook:
    def __init__(self, *, output_root: str) -> None:
        self.output_root = output_root

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = _find_model_root(model)
        state = getattr(root, "_aima_vl_generation_logits_state", None)
        if not isinstance(state, dict):
            raise RuntimeError("generation logits hook was not installed")
        for handle in state.get("handles", []):
            handle.remove()
        logits = state.get("logits")
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "target generation logits were not captured: "
                f"target={state['target_output_index']} "
                f"shapes={state['call_shapes']}"
            )
        case_id = state["case_id"]
        component = write_raw_tensor(
            Path(self.output_root), f"{case_id}/divergence-logits", logits
        )
        values, indices = torch.topk(logits, k=20)
        target_id = CASE_CONTRACTS[case_id]["reference_token_id"]
        target_logit = float(logits[target_id].item())
        target_rank = int(torch.count_nonzero(logits > logits[target_id]).item()) + 1
        result = {
            "component": component,
            "captured_output_index": state["captured_output_index"],
            "prefill_calls": state["prefill_calls"],
            "decode_calls": state["decode_calls"],
            "call_shapes": state["call_shapes"],
            "raw_top_tokens": [
                {
                    "rank": rank + 1,
                    "token_id": int(token_id),
                    "logit": float(value),
                }
                for rank, (token_id, value) in enumerate(
                    zip(indices.tolist(), values.tolist(), strict=True)
                )
            ],
            "selected_token_id": target_id,
            "selected_token_logit": target_logit,
            "selected_token_raw_rank": target_rank,
        }
        delattr(root, "_aima_vl_generation_logits_state")
        return result


class RemoveGenerationLogitsHook:
    def __call__(self, model: Any) -> bool:
        root = _find_model_root(model)
        state = getattr(root, "_aima_vl_generation_logits_state", None)
        if not isinstance(state, dict):
            return False
        for handle in state.get("handles", []):
            handle.remove()
        delattr(root, "_aima_vl_generation_logits_state")
        return True


def canonical_round_trip(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    )


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
        raise ValueError(f"generation oracle root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source = git_identity(ROOT)
    if source["dirty"]:
        raise ValueError("generation oracle capture source must be a clean commit")
    base = load_module(BASE_CAPTURE, "aima_vl_generation_base_capture")
    probe = load_module(CAPABILITY_PROBE, "aima_vl_generation_capability_probe")
    fixture_root = args.fixture_root.resolve()
    reference, _launch, _processor = base._require_inputs(
        reference_path=args.reference_manifest,
        launch_path=args.launch_config,
        processor_path=args.processor_probe,
        fixture_root=fixture_root,
    )
    if socket.gethostname() != reference.get("host", {}).get("hostname"):
        raise ValueError("generation oracle host differs from the frozen reference")

    capability = load_json_object(args.capability_manifest)
    capability_errors = validate_capability_manifest(capability)
    if capability_errors:
        raise ValueError(
            "invalid capability manifest:\n- " + "\n- ".join(capability_errors)
        )
    render = load_json_object(args.api_render_manifest)
    render_errors = validate_api_render_manifest(render)
    if render_errors:
        raise ValueError(
            "invalid API render manifest:\n- " + "\n- ".join(render_errors)
        )
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
        raise RuntimeError("generation tool case set changed")

    for module in (sys.modules[__name__],):
        cloudpickle.register_pickle_by_value(module)
    llm_kwargs = base._llm_kwargs(args.model_dir.resolve(), fixture_root)
    # Qualification requests provide real media immediately, so the expensive
    # synthetic maximum-item profiling pass is unnecessary. Existing HTTP
    # language diagnostics use the same control and bind it in their manifest.
    llm_kwargs["skip_mm_profiling"] = True
    llm = LLM(**llm_kwargs)
    print(json.dumps({"event": "engine_ready"}, sort_keys=True), flush=True)
    cases: list[dict[str, Any]] = []
    try:
        for case_id in CASE_ORDER:
            contract = CASE_CONTRACTS[case_id]
            spec = specs[case_id]
            payload = canonical_round_trip(spec["payload"])
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
                raise RuntimeError(f"generation request drifted: {case_id}")

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
                raise RuntimeError(f"generation prompt differs from render: {case_id}")

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
                max_tokens=contract["max_tokens"],
                prompt_logprobs=1,
                seed=0,
                structured_outputs=structured,
            )

            target_index = contract["divergence_output_index"]

            def install_callable(model: Any) -> dict[str, Any]:
                return InstallGenerationLogitsHook(
                    case_id=case_id, output_index=target_index
                )(model)

            def finalize_callable(model: Any) -> dict[str, Any]:
                return FinalizeGenerationLogitsHook(
                    output_root=str(output_root)
                )(model)

            def cleanup_callable(model: Any) -> bool:
                return RemoveGenerationLogitsHook()(model)

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
                raise RuntimeError("generation capture requires one request and TP=1")

            completion = outputs[0].outputs[0]
            output_token_ids = [int(token) for token in completion.token_ids]
            output_digest = canonical_int_list_sha256(output_token_ids)
            if output_digest != contract["output_token_ids_sha256"]:
                raise RuntimeError(
                    f"frozen generation changed for {case_id}: {output_digest}"
                )
            if output_token_ids[target_index] != contract["reference_token_id"]:
                raise RuntimeError(f"generation target token changed: {case_id}")
            reference_logits = finalization[0]
            if reference_logits["captured_output_index"] != target_index:
                raise RuntimeError(f"generation logit step changed: {case_id}")
            cases.append(
                {
                    "case_id": case_id,
                    "passed": True,
                    "request": normalized_request,
                    "request_sha256": canonical_json_sha256(normalized_request),
                    "prompt_tokens": len(prompt_token_ids),
                    "prompt_token_ids_sha256": canonical_int_list_sha256(
                        prompt_token_ids
                    ),
                    "divergence_output_index": target_index,
                    "shared_reference_prefix_token_ids_sha256": (
                        canonical_int_list_sha256(output_token_ids[:target_index])
                    ),
                    "generation": {
                        "sampling": {
                            "temperature": 0,
                            "max_tokens": contract["max_tokens"],
                            "prompt_logprobs": 1,
                            "seed": 0,
                            "structured": contract["structured"],
                        },
                        "output_token_ids": output_token_ids,
                        "output_token_ids_sha256": output_digest,
                        "completion_tokens": len(output_token_ids),
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                    },
                    "reference_logits": reference_logits,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "case_complete",
                        "case_id": case_id,
                        "completion_tokens": len(output_token_ids),
                        "target_output_index": target_index,
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
            "schema": GENERATION_ORACLE_SCHEMA,
            "captured_at": base.utc_now(),
            "complete": True,
            "qualified_for_native_generation_comparison": True,
            "scope": "two-fixed-vllm-vl-tool-generations-and-first-divergence-logits",
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in (
                        Path(__file__).resolve(),
                        ROOT / "aima_engine/vl_generation_oracle.py",
                        BASE_CAPTURE,
                        CAPABILITY_PROBE,
                    )
                ],
            },
            "host": {"label": args.host_label, "hostname": socket.gethostname()},
            "model": {
                "repository": "Qwen/Qwen3.6-35B-A3B",
                "revision": MODEL_REVISION,
                "dtype": "bfloat16",
            },
            "runtime": {"packages": versions, "python_version": sys.version.split()[0]},
            "bindings": {
                "reference_manifest": file_component(
                    args.reference_manifest,
                    "benchmarks/results/vl-reference-manifest.json",
                ),
                "launch_config": file_component(
                    args.launch_config,
                    "benchmarks/results/vl-reference-launch.json",
                ),
                "processor_probe": file_component(
                    args.processor_probe,
                    "benchmarks/results/vl-processor-capability-v0.1.0.json",
                ),
                "capability_manifest": file_component(
                    args.capability_manifest,
                    "benchmarks/results/vl-capability-manifest.json",
                ),
                "api_render_manifest": file_component(
                    args.api_render_manifest,
                    "benchmarks/results/vl-api-render-manifest-v0.1.0.json",
                ),
            },
            "capture_control_plane": {
                "vllm_allow_insecure_serialization": True,
                "skip_mm_profiling": True,
                "scope": "isolated-offline-qualification-hook-rpc-only",
                "product_runtime_dependency": False,
            },
            "oracle_root": args.oracle_root_label,
            "cases": cases,
            "decision": {
                "two_tool_generations_exact": len(cases) == 2
                and all(case["passed"] for case in cases),
                "two_prompt_vectors_exact": len(cases) == 2,
                "two_divergence_logits_captured": len(cases) == 2,
                "g1_passed": False,
                "g2_passed": False,
                "g3_passed": False,
                "g4_passed": False,
                "g5_passed": False,
            },
        }
    )
    errors = validate_generation_oracle_manifest(
        manifest, oracle_root=output_root
    )
    if errors:
        raise RuntimeError(
            "generation oracle validation failed:\n- " + "\n- ".join(errors)
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--oracle-root-label",
        default="benchmarks/oracles/vl-generation-v0.1.0",
    )
    parser.add_argument("--host-label", default="amd395")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
        raise ValueError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required for isolated "
            "offline qualification hooks"
        )
    manifest = capture(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(manifest["cases"]),
                "qualified": manifest[
                    "qualified_for_native_generation_comparison"
                ],
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
