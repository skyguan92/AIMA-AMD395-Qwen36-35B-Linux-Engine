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
LAYER_CAPTURE = ROOT / "scripts/capture-vllm-vl-generation-layer-oracles.py"
MODEL_ID = "aima-amd395-qwen36-35b"
TARGETS = {
    "image_central_red_circle": 122,
    "video_blue_square_moves_down": 172,
}
SCHEMA = "aima-amd395-qwen36/vllm-vl-task-quality-divergence/v1"


def parse_case_int_overrides(
    values: list[str] | None,
    *,
    option: str,
    maximum: int | None = None,
) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values or []:
        case_id, separator, integer_text = value.partition("=")
        if separator != "=" or case_id not in TARGETS:
            raise ValueError(f"{option} must be CASE_ID=INTEGER")
        if case_id in overrides:
            raise ValueError(f"duplicate {option} case: {case_id}")
        try:
            integer = int(integer_text)
        except ValueError as error:
            raise ValueError(f"{option} must be CASE_ID=INTEGER") from error
        if integer < 0 or (maximum is not None and integer >= maximum):
            raise ValueError(f"{option} is out of range: {value}")
        overrides[case_id] = integer
    return overrides


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load capture dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_round_trip(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def source_components() -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        BASE_CAPTURE,
        CAPABILITY_PROBE,
        LAYER_CAPTURE,
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

    targets = dict(TARGETS)
    targets.update(
        parse_case_int_overrides(
            args.case_output_index,
            option="--case-output-index",
            maximum=1_024,
        )
    )
    extra_video_indices: list[int] = []
    if args.extra_video_output_indices:
        try:
            extra_video_indices = [
                int(value)
                for value in args.extra_video_output_indices.split(",")
            ]
        except ValueError as error:
            raise ValueError(
                "--extra-video-output-indices must be comma-separated integers"
            ) from error
        if (
            not extra_video_indices
            or len(set(extra_video_indices)) != len(extra_video_indices)
            or any(index <= 0 or index >= 1_024 for index in extra_video_indices)
        ):
            raise ValueError(
                "--extra-video-output-indices contains an invalid index"
            )
    capture_targets = [
        (case_id, case_id, target_index)
        for case_id, target_index in targets.items()
    ]
    capture_targets.extend(
        (
            f"video_blue_square_moves_down__output_{target_index:03d}",
            "video_blue_square_moves_down",
            target_index,
        )
        for target_index in extra_video_indices
        if target_index != targets["video_blue_square_moves_down"]
    )
    linear_layers = {case_id: 0 for case_id in targets}
    linear_layers.update(
        parse_case_int_overrides(
            args.case_linear_attention_layer,
            option="--case-linear-attention-layer",
            maximum=40,
        )
    )
    for case_id, layer_index in linear_layers.items():
        if layer_index % 4 == 3:
            raise ValueError(
                "--case-linear-attention-layer selected a full-attention "
                f"layer: {case_id}={layer_index}"
            )
    full_layers = {case_id: 3 for case_id in targets}
    full_layers.update(
        parse_case_int_overrides(
            args.case_full_attention_layer,
            option="--case-full-attention-layer",
            maximum=40,
        )
    )
    for case_id, layer_index in full_layers.items():
        if layer_index % 4 != 3:
            raise ValueError(
                "--case-full-attention-layer selected a linear-attention "
                f"layer: {case_id}={layer_index}"
            )
    full_projection_cases = set(args.case_full_attention_projection or [])

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
    layer = load_module(LAYER_CAPTURE, "aima_task_quality_divergence_layer")
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
        if spec["case_id"] in targets
    }
    if set(specs) != set(targets):
        raise RuntimeError("task-quality divergence case set changed")

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(layer)
    llm_kwargs = base._llm_kwargs(args.model_dir.resolve(), fixture_root)
    llm_kwargs["skip_mm_profiling"] = True
    llm = LLM(**llm_kwargs)
    print(json.dumps({"event": "engine_ready"}, sort_keys=True), flush=True)
    cases: list[dict[str, Any]] = []
    native_cases: list[dict[str, Any]] = []
    try:
        for artifact_case_id, case_id, target_index in capture_targets:
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
                return layer.InstallGenerationLayerHooks(
                    case_id=artifact_case_id,
                    output_index=target_index,
                    linear_attention_layer_index=linear_layers[case_id],
                    full_attention_layer_index=full_layers[case_id],
                    capture_full_attention_projection=(
                        case_id in full_projection_cases
                    ),
                )(model)

            def finalize_callable(model: Any) -> dict[str, Any]:
                return layer.FinalizeGenerationLayerHooks(
                    output_root=str(output_root)
                )(model)

            def cleanup_callable(model: Any) -> bool:
                return layer.RemoveGenerationLayerHooks()(model)

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
            layer_record = finalization[0]
            if layer_record["captured_logits_output_index"] != target_index:
                raise RuntimeError(f"task-quality logit step changed: {case_id}")
            if layer_record["logits_decode_calls"] != len(output_ids):
                raise RuntimeError(
                    f"task-quality logit call count changed: {case_id}"
                )
            if layer_record["target_logits_top1_token_id"] != expected_token:
                raise RuntimeError(f"task-quality raw top-1 changed: {case_id}")
            logits_component = layer_record["target_logits_component"]
            component_path = output_root / logits_component["path"]
            cases.append(
                {
                    "case_id": artifact_case_id,
                    "source_case_id": case_id,
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
                    "reference_logits": {
                        "component": logits_component,
                        "captured_output_index": layer_record[
                            "captured_logits_output_index"
                        ],
                        "prefill_calls": layer_record["logits_prefill_calls"],
                        "decode_calls": layer_record["logits_decode_calls"],
                        "top1_token_id": layer_record[
                            "target_logits_top1_token_id"
                        ],
                    },
                    "decode_layers": layer_record,
                }
            )
            native_cases.append(
                {
                    "case_id": artifact_case_id,
                    "request": payload,
                    "expected_prefix_token_ids": reference_ids[:target_index],
                    "expected_reference_token_id": expected_token,
                    "expected_selected_token_id": expected_token,
                    "reference_logits": str(component_path),
                    "reference_logits_output_index": target_index,
                    "reference_decode_output_index": target_index,
                    "reference_decode_linear_layer_index": linear_layers[case_id],
                    "reference_decode_full_attention_layer_index": full_layers[
                        case_id
                    ],
                    "reference_decode_boundary_dir": str(
                        (output_root / artifact_case_id).resolve()
                    ),
                    "reference_decode_linear_boundary_dir": str(
                        (output_root / artifact_case_id / "linear").resolve()
                    ),
                    "reference_decode_layer0_tail_boundary_dir": str(
                        (output_root / artifact_case_id / "layer0-tail").resolve()
                    ),
                    "reference_decode_full_attention_dir": str(
                        (
                            output_root
                            / artifact_case_id
                            / "full-attention"
                        ).resolve()
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "case_id": artifact_case_id,
                        "source_case_id": case_id,
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
            "complete": len(cases) == len(capture_targets),
            "qualified_for_attribution": len(cases) == len(capture_targets),
            "scope": (
                "two-task-quality-first-divergence-full-vocabulary-logits-"
                "plus-decode-layer-boundaries"
            ),
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
                "two_prompts_exact": len(cases) == len(targets) == 2,
                "two_reference_prefixes_exact": len(cases) == len(targets) == 2,
                "two_reference_top1_rows_captured": len(cases) == len(targets) == 2,
                "two_decode_layer_boundary_sets_captured": (
                    len(cases) == len(targets) == 2
                ),
                "two_selected_linear_boundary_sets_captured": (
                    len(cases) == len(targets) == 2
                ),
                "two_layer0_tail_boundary_sets_captured": (
                    len(cases) == len(targets) == 2
                ),
                "two_layer3_full_attention_sets_captured": (
                    len(cases) == len(targets) == 2
                ),
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
        "--case-output-index",
        action="append",
        help="override one target as CASE_ID=OUTPUT_INDEX",
    )
    parser.add_argument(
        "--case-linear-attention-layer",
        action="append",
        help="capture one selected linear layer as CASE_ID=LAYER_INDEX",
    )
    parser.add_argument(
        "--case-full-attention-layer",
        action="append",
        help="capture one selected full-attention layer as CASE_ID=LAYER_INDEX",
    )
    parser.add_argument(
        "--case-full-attention-projection",
        action="append",
        choices=tuple(TARGETS),
        help="capture layer-3 projection and tail boundaries for one case",
    )
    parser.add_argument(
        "--extra-video-output-indices",
        help="capture additional video targets as comma-separated indices",
    )
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
