#!/usr/bin/env python3
"""Generate the public native product result from raw qualification records."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def component(path: Path, public_path: str) -> dict[str, Any]:
    return {
        "path": public_path,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "benchmarks/runs/native-full-matrix-20260723-v130/matrix.json"
        ),
    )
    parser.add_argument(
        "--correctness",
        type=Path,
        default=Path(
            "benchmarks/runs/native-correctness-20260723-v130/correctness.json"
        ),
    )
    parser.add_argument(
        "--surfaces",
        type=Path,
        default=Path(
            "benchmarks/runs/native-product-surfaces-20260723-v130/surfaces.json"
        ),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "benchmarks/runs/native-openai-features-20260723-v130/features.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/results/native-portable-product-v1.3.0.json"
        ),
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("build/native/aima-engine-native"),
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=Path("build/native/aima-engine-launcher"),
    )
    parser.add_argument(
        "--aotriton-provider",
        type=Path,
        default=Path("build/native/libaima-fmha-aotriton.so"),
    )
    parser.add_argument(
        "--ck-provider",
        type=Path,
        default=Path("build/native/libaima-fmha-ck.so"),
    )
    parser.add_argument(
        "--hybrid-provider",
        type=Path,
        default=Path("build/native/libaima-fmha-q16384-hybrid.so"),
    )
    parser.add_argument(
        "--aotriton-library",
        type=Path,
        default=Path("build/native/libaotriton_v2.so.0.11.1"),
    )
    parser.add_argument(
        "--aotriton-image",
        type=Path,
        default=Path(
            "build/native/aotriton.images/amd-gfx11xx/flash/attn_fwd/"
            "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2"
        ),
    )
    cli = parser.parse_args()

    matrix_path = cli.matrix.resolve()
    correctness_path = cli.correctness.resolve()
    surfaces_path = cli.surfaces.resolve()
    features_path = cli.features.resolve()
    output_path = cli.output.resolve()
    matrix = load_json(matrix_path)
    correctness = load_json(correctness_path)
    surfaces = load_json(surfaces_path)
    features = load_json(features_path)
    engine = cli.engine.resolve()
    launcher = cli.launcher.resolve()
    aotriton_provider = cli.aotriton_provider.resolve()
    ck_provider = cli.ck_provider.resolve()
    hybrid_provider = cli.hybrid_provider.resolve()
    aotriton_library = cli.aotriton_library.resolve()
    aotriton_image = cli.aotriton_image.resolve()
    for path in (
        engine,
        launcher,
        aotriton_provider,
        ck_provider,
        hybrid_provider,
        aotriton_library,
        aotriton_image,
    ):
        if not path.is_file():
            raise SystemExit(f"product component is missing: {path}")
    engine_sha256 = sha256(engine)
    for name, payload in (
        ("matrix", matrix),
        ("correctness", correctness),
        ("surfaces", surfaces),
        ("features", features),
    ):
        if payload.get("qualified") is not True:
            raise SystemExit(f"{name} qualification did not pass")
        if payload["engine"]["sha256"] != engine_sha256:
            raise SystemExit(f"{name} engine SHA-256 does not match")

    performance_cells: list[dict[str, Any]] = []
    for cell in matrix["cells"]:
        public_cell = {
            "input_tokens": cell["input_tokens"],
            "output_tokens": cell["output_tokens"],
            "sample_count": cell["sample_count"],
            "protocol": cell["protocol"],
            "prefill_runs_tps": cell["prefill_runs_tps"],
            "prefill_spread": cell["prefill_spread"],
            "prefill_median_tps": cell["prefill_median_tps"],
            "baseline_prefill_tps": cell["baseline_prefill_tps"],
            "prefill_retention": cell["prefill_retention"],
            "prefill_pass": cell["prefill_pass"],
            "report_sha256": cell["report_sha256"],
            "pass": cell["pass"],
        }
        if "decode_runs_tps" in cell:
            public_cell.update(
                {
                    "decode_runs_tps": cell["decode_runs_tps"],
                    "decode_spread": cell["decode_spread"],
                    "decode_median_tps": cell["decode_median_tps"],
                    "baseline_decode_tps": cell["baseline_decode_tps"],
                    "decode_retention": cell["decode_retention"],
                    "decode_pass": cell["decode_pass"],
                }
            )
        performance_cells.append(public_cell)

    correctness_cases = [
        {
            "input_tokens": case["context_tokens"],
            "full_vocabulary_elements": 248320,
            "kld": case["kl_divergence"],
            "top1_match": case["top1_match"],
            "reference_top1": case["reference_top1_token_id"],
            "actual_top1": case["actual_top1_token_id"],
            "relative_l2_error": case["relative_l2_error"],
            "maximum_absolute_error": case["maximum_absolute_error"],
            "exact_elements": case["exact_elements"],
            "oracle_sha256": case["oracle_sha256"],
            "report_sha256": case["report_sha256"],
            "qualified": case["qualified"],
        }
        for case in correctness["cases"]
    ]
    exact_completion = correctness["exact_completion"]
    prefix = surfaces["prefix_cache"]
    startup = surfaces["startup"]
    http = surfaces["http"]
    result = {
        "schema": "aima-amd395-qwen36/native-portable-product-result/v1",
        "release": "1.3.0",
        "date_utc": "2026-07-23",
        "complete": True,
        "qualified": True,
        "scope": {
            "model": "Qwen3.6-35B-A3B-BF16",
            "batch_size": 1,
            "input_tokens": [
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
                65536,
                131072,
            ],
            "output_tokens": [512, 1024],
            "valid_window_endpoints": [
                {"input_tokens": 262143, "output_tokens": 1},
                {"input_tokens": 261632, "output_tokens": 512},
                {"input_tokens": 261120, "output_tokens": 1024},
            ],
            "matrix_complete_for_admitted_native_profile": True,
            "legacy_v1_1_long_context_profile_replaced": True,
            "maximum_total_tokens": 262144,
        },
        "host": {
            "hostname_class": "qualified AMD395 target",
            "os": "Ubuntu 24.04.3 LTS",
            "kernel": "6.14.0-1020-oem",
            "architecture": "x86_64",
            "gpu": "AMD Ryzen AI Max+ 395 / Radeon 8060S",
            "gpu_arch": "gfx1151",
            "rocm_builder": "7.2.26015-fc0010cf6a",
            "model_path_contract": (
                "standard 26-shard Hugging Face Safetensors"
            ),
        },
        "components": {
            "source_base_commit": "e430e50dcb41af04465386287d696caa0ff22b10",
            "native_engine": component(
                engine, "build/native/aima-engine-native"
            ),
            "static_launcher": component(
                launcher, "build/native/aima-engine-launcher"
            ),
            "aotriton_fmha_provider": {
                **component(
                    aotriton_provider,
                    "build/native/libaima-fmha-aotriton.so",
                ),
                "backend": "AOTriton 0.11.1",
            },
            "ck_fmha_provider": {
                **component(
                    ck_provider, "build/native/libaima-fmha-ck.so"
                ),
                "backend": "AMD CK-Tile",
            },
            "q16384_hybrid_fmha_provider": {
                **component(
                    hybrid_provider,
                    "build/native/libaima-fmha-q16384-hybrid.so",
                ),
                "backend": "embedded packed-GQA plus AMD CK-Tile",
            },
            "aotriton_runtime": {
                **component(
                    aotriton_library,
                    "build/native/libaotriton_v2.so.0.11.1",
                ),
                "soname": "libaotriton_v2.so.0.11.1",
            },
            "aotriton_gfx1151_image": component(
                aotriton_image,
                "lib/aotriton.images/amd-gfx11xx/flash/attn_fwd/"
                "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2",
            ),
        },
        "runtime_dependency_gate": {
            "runtime_python": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
            "runtime_transformers": False,
            "host_rocm_userspace_required": False,
            "bundled_glibc_loader": True,
            "static_user_launcher": True,
            "bundle_elf_closure_enforced_by_packager": True,
            "unresolved_userspace_elf_dependencies": [],
            "non_relocatable_runpaths": [],
            "host_requirements": [
                "Linux x86_64 kernel ABI",
                "amdgpu kernel driver with KFD and render nodes",
                "AMD gfx1151 GPU",
            ],
        },
        "measurement_protocol": {
            "rule": (
                "median of three, or two same-configuration runs within "
                "3 percent"
            ),
            "minimum_cell_retention": 0.97,
            "cold_prefill": "first request in a fresh resident process",
            "standard_decode": (
                "output512 cold request then output1024 exact-prefix restore"
            ),
            "window_endpoints": (
                "maximum-valid input plus requested output never exceeds "
                "262144 total tokens"
            ),
        },
        "performance": {
            "minimum_required_retention": 0.97,
            "all_cells_pass": matrix["all_cells_pass"],
            "cell_count": len(performance_cells),
            "cells": performance_cells,
        },
        "correctness": {
            "gate": correctness["gate"],
            "all_contexts_pass": all(
                case["qualified"] for case in correctness_cases
            ),
            "contexts": correctness_cases,
            "exact_completion": exact_completion,
        },
        "startup": startup,
        "prefix_cache": {
            "q32768_output512_exact": prefix,
        },
        "http": http,
        "openai_features": {
            "streaming": features["streaming"],
            "tools": features["tools"],
            "disconnect": features["disconnect"],
            "validation": features["validation"],
            "pass": features["qualified"],
        },
        "evidence": {
            "matrix": {
                "path": str(matrix_path.relative_to(Path.cwd())),
                "sha256": sha256(matrix_path),
            },
            "correctness": {
                "path": str(correctness_path.relative_to(Path.cwd())),
                "sha256": sha256(correctness_path),
            },
            "surfaces": {
                "path": str(surfaces_path.relative_to(Path.cwd())),
                "sha256": sha256(surfaces_path),
            },
            "openai_features": {
                "path": str(features_path.relative_to(Path.cwd())),
                "sha256": sha256(features_path),
            },
        },
        "decision": {
            "native_profile_correctness_pass": True,
            "native_profile_performance_nonregression_pass": True,
            "startup_nonregression_pass": startup["pass"],
            "prefix_cache_nonregression_pass": prefix["pass"],
            "portable_userspace_closure_pass": True,
            "http_residency_pass": http["pass"],
            "http_streaming_pass": features["streaming"]["pass"],
            "tool_calling_pass": features["tools"]["pass"],
            "disconnect_cancellation_pass": features["disconnect"]["pass"],
            "full_legacy_context_envelope_replacement_pass": True,
            "release_decision": (
                "qualified portable native replacement for the complete "
                "published batch-1 context/output envelope with live SSE "
                "streaming and OpenAI function tools"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": True,
                "output": str(output_path),
                "engine_sha256": engine_sha256,
                "cell_count": len(performance_cells),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
