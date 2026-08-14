#!/usr/bin/env python3
"""Capture all language-layer outputs for one real-HTTP-rendered VL case."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from aima_engine.vl_http_oracle import (  # noqa: E402
    validate_http_oracle_manifest,
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


SCHEMA = "aima-amd395-qwen36/vl-http-language-layer-diagnostic-oracle/v1"
BASE_CAPTURE = ROOT / "scripts/capture-vllm-vl-oracles.py"
LAYER_CAPTURE = (
    ROOT / "scripts/capture-vllm-vl-http-language-attribution.py"
)
PREFIX_CAPTURE = ROOT / "scripts/capture-vllm-vl-language-prefix-diagnostics.py"
HTTP_ORACLE_CONTRACT = ROOT / "aima_engine/vl_http_oracle.py"
CASE_IDS = (
    "image_local_png",
    "video_local_mp4",
    "multi_image",
    "multi_video",
    "mixed_image_video",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load diagnostic dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
        raise ValueError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required for the isolated "
            "offline diagnostic hooks"
        )
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"diagnostic output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source = git_identity(ROOT)
    if source["dirty"]:
        raise ValueError("HTTP language diagnostic source must be a clean commit")
    base = load_module(BASE_CAPTURE, "aima_vl_http_language_base_capture")
    layers = load_module(LAYER_CAPTURE, "aima_vl_http_language_layer_capture")

    fixture_root = args.fixture_root.resolve()
    reference, _launch, _processor = base._require_inputs(
        reference_path=args.reference_manifest,
        launch_path=args.launch_config,
        processor_path=args.processor_probe,
        fixture_root=fixture_root,
    )
    if socket.gethostname() != reference.get("host", {}).get("hostname"):
        raise ValueError("diagnostic host differs from the frozen reference")

    render_path = args.serving_render_manifest.resolve()
    http_oracle_path = args.http_oracle_manifest.resolve()
    render = load_json_object(render_path)
    http_oracle = load_json_object(http_oracle_path)
    errors = validate_http_oracle_manifest(
        http_oracle,
        render_manifest=render,
        render_manifest_sha256=sha256_file(render_path),
        oracle_root=args.http_oracle_root.resolve(),
    )
    if errors:
        raise ValueError(
            "invalid HTTP numerical oracle:\n- " + "\n- ".join(errors)
        )
    oracle_by_id = {case["case_id"]: case for case in http_oracle["cases"]}
    specs = {spec["case_id"]: spec for spec in base.CASE_SPECS}
    if set(oracle_by_id) != set(CASE_IDS) or set(specs) != set(CASE_IDS):
        raise ValueError("HTTP diagnostic case set changed")

    versions = base._runtime_versions()
    for name, expected in PINNED_PACKAGES.items():
        actual = versions.get(name)
        if not isinstance(actual, str) or not (
            actual == expected or actual.startswith(expected + ".")
        ):
            raise ValueError(f"runtime pin mismatch for {name}: {actual!r}")
    serving_sources = layers.prefix._verify_serving_sources()

    import cloudpickle
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput

    for module in (
        base,
        layers,
        layers.base,
        layers.prefix,
        sys.modules[__name__],
    ):
        cloudpickle.register_pickle_by_value(module)
    llm_kwargs = base._llm_kwargs(args.model_dir.resolve(), fixture_root)
    llm_kwargs["skip_mm_profiling"] = True
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,
        seed=0,
    )
    case_id = args.case_id
    try:
        llm.reset_mm_cache()
        llm.llm_engine.reset_encoder_cache()
        messages = base._build_messages(specs[case_id], fixture_root)
        engine_input = llm._preprocess_chat_one(
            messages, chat_template_content_format="string"
        )
        prompt_token_ids = [int(item) for item in engine_input["prompt_token_ids"]]
        expected_ids = oracle_by_id[case_id]["processor"]["prompt_token_ids"]
        if prompt_token_ids != expected_ids:
            raise RuntimeError(
                f"diagnostic prompt differs from the HTTP oracle: {case_id}"
            )

        def install_callable(model: Any) -> dict[str, Any]:
            return layers.InstallLanguageLayerOutputHooks(
                output_root=str(output_root), case_id=case_id
            )(model)

        def finalize_callable(model: Any) -> dict[str, Any]:
            return layers.FinalizeLanguageLayerOutputHooks()(model)

        def cleanup_callable(model: Any) -> bool:
            return layers.RemoveLanguageLayerOutputHooks()(model)

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
            raise RuntimeError("HTTP layer capture requires one request and TP=1")
        record = finalization[0]
        final_norm_comparison = layers.prefix._compare_component(
            actual=record["components"]["language_final_norm"],
            expected=oracle_by_id[case_id]["boundaries"]["language_final_norm"],
            actual_root=output_root,
            expected_root=args.http_oracle_root.resolve(),
        )
        if not final_norm_comparison["exact"]:
            raise RuntimeError("diagnostic final norm differs from HTTP oracle")
    finally:
        del llm

    return seal_manifest(
        {
            "schema": SCHEMA,
            "captured_at": utc_now(),
            "complete": True,
            "qualified_for_attribution_only": True,
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in (
                        Path(__file__).resolve(),
                        HTTP_ORACLE_CONTRACT,
                        BASE_CAPTURE,
                        LAYER_CAPTURE,
                        PREFIX_CAPTURE,
                    )
                ],
            },
            "model": {
                "revision": MODEL_REVISION,
                "http_oracle_manifest": file_component(
                    http_oracle_path,
                    "benchmarks/results/vl-http-oracle-manifest-v0.1.0.json",
                ),
                "serving_render_manifest": file_component(
                    render_path,
                    "benchmarks/results/vl-serving-render-manifest-v0.1.0.json",
                ),
            },
            "runtime": versions,
            "reference": {
                "chat_template_content_format": "string",
                "skip_mm_profiling": True,
                "vllm_allow_insecure_serialization": True,
                "serving_sources": serving_sources,
            },
            "case": {
                "case_id": case_id,
                "prompt_tokens": len(prompt_token_ids),
                "prompt_token_ids_sha256": oracle_by_id[case_id]["processor"][
                    "prompt_token_ids_sha256"
                ],
                "components": record["components"],
                "oracle_labels": record["oracle_labels"],
                "oracle_jsonl_sha256": record["oracle_jsonl_sha256"],
                "http_oracle_final_norm_comparison": final_norm_comparison,
            },
            "decision": {
                "diagnostic_only": True,
                "g1_passed": False,
                "g2_passed": False,
            },
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--processor-probe", type=Path, required=True)
    parser.add_argument("--serving-render-manifest", type=Path, required=True)
    parser.add_argument("--http-oracle-manifest", type=Path, required=True)
    parser.add_argument("--http-oracle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-id", choices=CASE_IDS, default="multi_video")
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
                "case_id": result["case"]["case_id"],
                "components": len(result["case"]["components"]),
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
        print(f"capture HTTP VL language layers: {error}", file=sys.stderr)
        raise SystemExit(1)
