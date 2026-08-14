#!/usr/bin/env python3
"""Capture numerical VL oracles from the five fixed real-HTTP prompt renders."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_http_oracle import (  # noqa: E402
    HTTP_ORACLE_ROOT,
    HTTP_ORACLE_SCOPE,
    HTTP_ORACLE_VARIANT,
    validate_http_oracle_manifest,
)
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    load_json_object,
    seal_manifest,
    sha256_file,
)
from aima_engine.vl_serving_render import (  # noqa: E402
    SERVING_RENDER_CASES,
    validate_serving_render_manifest,
)


BASE_CAPTURE = ROOT / "scripts/capture-vllm-vl-oracles.py"
DEFAULT_RENDER = (
    ROOT / "benchmarks/results/vl-serving-render-manifest-v0.1.0.json"
)


def load_base_capture() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aima_vl_http_oracle_base_capture", BASE_CAPTURE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen VL oracle capture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--processor-probe", type=Path, required=True)
    parser.add_argument(
        "--serving-render-manifest", type=Path, default=DEFAULT_RENDER
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-label", default="amd395")
    parser.add_argument("--max-tokens", type=int, default=8)
    return parser.parse_args()


def capture(args: argparse.Namespace) -> dict[str, object]:
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
        raise ValueError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required for the isolated "
            "offline apply_model oracle hooks"
        )
    if args.output.exists():
        raise ValueError("HTTP oracle output must not already exist")
    render_path = args.serving_render_manifest.resolve()
    render = load_json_object(render_path)
    render_errors = validate_serving_render_manifest(render)
    if render_errors:
        raise ValueError(
            "invalid serving render manifest:\n- " + "\n- ".join(render_errors)
        )
    if tuple(case["case_id"] for case in render["cases"]) != SERVING_RENDER_CASES:
        raise ValueError("serving render case order changed")

    base = load_base_capture()
    import cloudpickle

    cloudpickle.register_pickle_by_value(base)
    fixture_records = base._load_fixture_records(args.fixture_root.resolve())
    render_by_id = {case["case_id"]: case for case in render["cases"]}
    for spec in base.CASE_SPECS:
        case_id = spec["case_id"]
        request_sha256 = base.canonical_json_sha256(
            base._semantic_case(spec, fixture_records)
        )
        if request_sha256 != render_by_id[case_id]["oracle_request_sha256"]:
            raise ValueError(f"HTTP oracle request identity changed: {case_id}")
    args._chat_template_content_format = "string"
    args._oracle_root_label = HTTP_ORACLE_ROOT
    args._expected_prompt_vectors = {
        case_id: render_by_id[case_id]["prompt_token_ids"]
        for case_id in SERVING_RENDER_CASES
    }
    manifest = base.capture(args)
    for case in manifest["cases"]:
        render_case = render_by_id[case["case_id"]]
        case["http_render"] = {
            "prompt_tokens": render_case["prompt_tokens"],
            "prompt_token_ids_sha256": render_case[
                "prompt_token_ids_sha256"
            ],
            "render_transport_request_sha256": render_case[
                "render_transport_request_sha256"
            ],
            "private_prompt_token_ids_sha256": render_case[
                "private_prompt_token_ids_sha256"
            ],
            "private_prompt_matches_real_http": render_case[
                "private_prompt_matches_real_http"
            ],
        }
    manifest["bindings"]["serving_render_manifest"] = file_component(
        render_path,
        "benchmarks/results/vl-serving-render-manifest-v0.1.0.json",
    )
    manifest["scope"] = HTTP_ORACLE_SCOPE
    manifest["oracle_variant"] = dict(HTTP_ORACLE_VARIANT)
    manifest["capture_scripts"] = {
        "http_wrapper": file_component(
            Path(__file__).resolve(),
            "scripts/capture-vllm-vl-http-oracles.py",
        ),
        "base_oracle": file_component(
            BASE_CAPTURE, "scripts/capture-vllm-vl-oracles.py"
        ),
    }
    manifest["decision"] = {
        "five_real_http_numerical_oracles_exact": True,
        "g1_passed": False,
        "g2_passed": False,
    }
    manifest.pop("integrity", None)
    manifest = seal_manifest(manifest)
    errors = validate_http_oracle_manifest(
        manifest,
        render_manifest=render,
        render_manifest_sha256=sha256_file(render_path),
        oracle_root=args.output_root.resolve(),
    )
    if errors:
        raise RuntimeError(
            "HTTP oracle manifest validation failed:\n- " + "\n- ".join(errors)
        )
    atomic_json(args.output, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    manifest = capture(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(manifest["cases"]),
                "qualified": manifest["decision"][
                    "five_real_http_numerical_oracles_exact"
                ],
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
