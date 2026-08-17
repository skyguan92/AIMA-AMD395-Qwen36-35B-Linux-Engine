#!/usr/bin/env python3
"""Capture fixed-vLLM logits at the two remaining task-quality drifts.

This is an attribution tool, not release evidence.  It verifies the frozen
prompt and generated prefix for each target before writing the full-vocabulary
row and a cases file consumable by ``vl-generation-logits-probe``.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_oracle import (  # noqa: E402
    canonical_int_list_sha256,
    write_raw_tensor,
)
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
)
from aima_engine.vl_task_quality import (  # noqa: E402
    build_cases,
    normalize_contract_request,
    validate_fixture_manifest,
    validate_reference_manifest,
)


BASE_CAPTURE = ROOT / "scripts/capture-vllm-vl-oracles.py"
CAPABILITY_PROBE = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
MODEL_ID = "aima-amd395-qwen36-35b"
MODEL_VOCABULARY_SIZE = 248_320
TARGETS = {
    "image_central_red_circle": 122,
    "video_blue_square_moves_down": 172,
}
SCHEMA = "aima-amd395-qwen36/vllm-vl-task-quality-divergence/v1"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load capture dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def first_tensor(value: Any) -> Any | None:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def find_model_root(model: Any) -> Any:
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


class InstallLogitsHook:
    """Serializable worker hook for one generated-token distribution."""

    def __init__(self, *, case_id: str, output_index: int) -> None:
        self.case_id = case_id
        self.output_index = output_index

    def __call__(self, model: Any) -> dict[str, Any]:
        root = find_model_root(model)
        previous = getattr(root, "_aima_task_quality_logits_state", None)
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

        def hook(_module: Any, _args: Any, output: Any) -> None:
            logits = first_tensor(output)
            if logits is None or logits.ndim != 2:
                return
            if logits.shape[1] != MODEL_VOCABULARY_SIZE:
                raise RuntimeError(
                    f"generation vocabulary changed: {list(logits.shape)}"
                )
            state["call_shapes"].append(list(logits.shape))
            if logits.shape[0] > 1:
                state["prefill_calls"] += 1
                # Prompt logprobs make the teacher-forced prompt matrix a
                # separate call. It is not generated output index zero.
                return
            if logits.shape[0] != 1:
                return
            state["decode_calls"] += 1
            output_index = state["decode_calls"] - 1
            if (
                output_index == state["target_output_index"]
                and state["logits"] is None
            ):
                state["captured_output_index"] = output_index
                state["logits"] = logits[-1].detach().float().contiguous().cpu()

        state["handles"].append(
            root.language_model.logits_processor.register_forward_hook(hook)
        )
        root._aima_task_quality_logits_state = state
        return {"installed": True}


class FinalizeLogitsHook:
    def __init__(
        self, *, output_root: str, expected_token_id: int
    ) -> None:
        self.output_root = output_root
        self.expected_token_id = expected_token_id

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = find_model_root(model)
        state = getattr(root, "_aima_task_quality_logits_state", None)
        if not isinstance(state, dict):
            raise RuntimeError("task-quality logits hook was not installed")
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
        target_logit = float(logits[self.expected_token_id].item())
        target_rank = (
            int(torch.count_nonzero(logits > logits[self.expected_token_id]).item())
            + 1
        )
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
            "expected_token_id": self.expected_token_id,
            "expected_token_logit": target_logit,
            "expected_token_raw_rank": target_rank,
        }
        delattr(root, "_aima_task_quality_logits_state")
        return result


class RemoveLogitsHook:
    def __call__(self, model: Any) -> bool:
        root = find_model_root(model)
        state = getattr(root, "_aima_task_quality_logits_state", None)
        if not isinstance(state, dict):
            return False
        for handle in state.get("handles", []):
            handle.remove()
        delattr(root, "_aima_task_quality_logits_state")
        return True


def canonical_round_trip(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def source_components() -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        BASE_CAPTURE,
        CAPABILITY_PROBE,
        ROOT / "aima_engine/vl_oracle.py",
        ROOT / "aima_engine/vl_reference.py",
        ROOT / "aima_engine/vl_task_quality.py",
    )
    return [
        file_component(path, path.relative_to(ROOT).as_posix())
        for path in paths
    ]


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import cloudpickle
    import torch
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.outputs import RequestOutput

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"diagnostic output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    fixture_root = args.fixture_root.resolve()
    fixture_manifest = load_json_object(fixture_root / "fixtures-manifest.json")
    fixture_errors = validate_fixture_manifest(fixture_manifest, fixture_root)
    if fixture_errors:
        raise ValueError("invalid task-quality fixtures:\n- " + "\n- ".join(fixture_errors))
    reference = load_json_object(args.reference)
    reference_errors = validate_reference_manifest(reference)
    if reference_errors:
        raise ValueError("invalid task-quality reference:\n- " + "\n- ".join(reference_errors))
    reference_cases = {case["case_id"]: case for case in reference["cases"]}

    base = load_module(BASE_CAPTURE, "aima_task_quality_divergence_base")
    probe = load_module(CAPABILITY_PROBE, "aima_task_quality_divergence_probe")
    versions = base._runtime_versions()
    for name, expected in base.PINNED_PACKAGES.items():
        actual = versions.get(name)
        if not isinstance(actual, str) or not (
            actual == expected or actual.startswith(expected + ".")
        ):
            raise RuntimeError(
                f"task-quality divergence runtime pin mismatch for {name}: "
                f"{actual!r}"
            )
    fixtures = probe.Fixtures(fixture_root, "http://127.0.0.1:9")
    specs = {
        spec["case_id"]: spec
        for spec in build_cases(fixtures, MODEL_ID)
        if spec["case_id"] in TARGETS
    }
    if set(specs) != set(TARGETS):
        raise RuntimeError("task-quality divergence case set changed")

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    llm_kwargs = base._llm_kwargs(args.model_dir.resolve(), fixture_root)
    llm_kwargs["skip_mm_profiling"] = True
    llm = LLM(**llm_kwargs)
    print(json.dumps({"event": "engine_ready"}, sort_keys=True), flush=True)
    cases: list[dict[str, Any]] = []
    native_cases: list[dict[str, Any]] = []
    try:
        for case_id, target_index in TARGETS.items():
            reference_case = reference_cases[case_id]
            reference_ids = [int(token) for token in reference_case["output_token_ids"]]
            expected_token = reference_ids[target_index]
            spec = specs[case_id]
            payload = canonical_round_trip(spec["payload"])
            normalized = normalize_contract_request(
                probe.recursive_replace(payload, spec["replacements"])
            )
            if normalized != reference_case["request"]:
                raise RuntimeError(f"task-quality request drifted: {case_id}")
            openai_request = ChatCompletionRequest.model_validate(payload)
            llm.reset_mm_cache()
            llm.llm_engine.reset_encoder_cache()
            engine_input = llm._preprocess_chat_one(
                openai_request.messages,
                chat_template_content_format="string",
                chat_template_kwargs=openai_request.chat_template_kwargs,
                tools=None,
            )
            prompt_ids = [int(token) for token in engine_input["prompt_token_ids"]]
            if prompt_ids != reference_case["render"]["prompt_token_ids"]:
                raise RuntimeError(f"task-quality prompt drifted: {case_id}")
            sampling = SamplingParams(
                temperature=0,
                max_tokens=target_index + 1,
                prompt_logprobs=1,
                seed=0,
            )

            def install_callable(model: Any) -> dict[str, Any]:
                return InstallLogitsHook(
                    case_id=case_id, output_index=target_index
                )(model)

            def finalize_callable(model: Any) -> dict[str, Any]:
                return FinalizeLogitsHook(
                    output_root=str(output_root),
                    expected_token_id=expected_token,
                )(model)

            def cleanup_callable(model: Any) -> bool:
                return RemoveLogitsHook()(model)

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
                raise RuntimeError("task-quality capture requires one request and TP=1")
            output_ids = [int(token) for token in outputs[0].outputs[0].token_ids]
            expected_prefix = reference_ids[: target_index + 1]
            if output_ids != expected_prefix:
                mismatch = next(
                    (
                        index
                        for index, (actual, expected) in enumerate(
                            zip(output_ids, expected_prefix, strict=False)
                        )
                        if actual != expected
                    ),
                    min(len(output_ids), len(expected_prefix)),
                )
                raise RuntimeError(
                    f"frozen task-quality prefix changed: {case_id} "
                    f"output_index={mismatch}"
                )
            logits = finalization[0]
            if logits["captured_output_index"] != target_index:
                raise RuntimeError(f"task-quality logit step changed: {case_id}")
            if logits["decode_calls"] != len(output_ids):
                raise RuntimeError(
                    f"task-quality logit call count changed: {case_id}"
                )
            if logits["raw_top_tokens"][0]["token_id"] != expected_token:
                raise RuntimeError(f"task-quality raw top-1 changed: {case_id}")
            component_path = output_root / logits["component"]["path"]
            cases.append(
                {
                    "case_id": case_id,
                    "prompt_tokens": len(prompt_ids),
                    "prompt_token_ids_sha256": canonical_int_list_sha256(prompt_ids),
                    "target_output_index": target_index,
                    "expected_prefix_token_ids_sha256": canonical_int_list_sha256(
                        reference_ids[:target_index]
                    ),
                    "expected_token_id": expected_token,
                    "captured_prefix_token_ids_sha256": canonical_int_list_sha256(
                        output_ids
                    ),
                    "reference_logits": logits,
                }
            )
            native_cases.append(
                {
                    "case_id": case_id,
                    "request": payload,
                    "expected_prefix_token_ids": reference_ids[:target_index],
                    "expected_reference_token_id": expected_token,
                    "expected_selected_token_id": expected_token,
                    "reference_logits": str(component_path),
                    "reference_logits_output_index": target_index,
                }
            )
            print(
                json.dumps(
                    {
                        "case_id": case_id,
                        "event": "task_quality_divergence_case_complete",
                        "target_output_index": target_index,
                        "expected_token_id": expected_token,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_json(args.native_cases, {"cases": native_cases})
    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "complete": len(cases) == len(TARGETS),
            "qualified_for_attribution": len(cases) == len(TARGETS),
            "scope": "two-task-quality-first-divergence-full-vocabulary-logits",
            "source": {
                **git_identity(ROOT),
                "files": source_components(),
            },
            "bindings": {
                "reference": file_component(
                    args.reference,
                    "benchmarks/results/vl-task-quality-reference-v0.1.0.json",
                ),
                "fixture_manifest": file_component(
                    fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-task-quality-v0.1.0/fixtures-manifest.json",
                ),
            },
            "runtime": {
                name: versions[name]
                for name in sorted(base.PINNED_PACKAGES)
            },
            "oracle_root": args.oracle_root_label,
            "cases": cases,
            "native_cases": file_component(args.native_cases, args.native_cases.name),
            "decision": {
                "two_prompts_exact": len(cases) == 2,
                "two_reference_prefixes_exact": len(cases) == 2,
                "two_reference_top1_rows_captured": len(cases) == 2,
                "product_runtime_dependency": False,
                "g1_passed": False,
                "g2_passed": False,
            },
        }
    )
    atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--oracle-root-label",
        default="diagnostics/vl-task-quality-divergence",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
        raise ValueError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required for isolated "
            "offline qualification hooks"
        )
    for path in (args.output, args.native_cases):
        if path.exists():
            raise ValueError(f"diagnostic output already exists: {path}")
    manifest = capture(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(manifest["cases"]),
                "qualified": manifest["qualified_for_attribution"],
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
