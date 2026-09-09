#!/usr/bin/env python3
"""Generate the exact package-input qualification for a native VL patch."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
    verify_manifest_integrity,
)
from aima_engine.qualification_runtime import require_runtime_binding


SCHEMA = "aima-amd395-qwen36/native-vl-product-qualification/v1"
RELEASE = "1.5.1-native-vl.5"
RELEASE_TAG = "v1.5.1-native-vl.5"
BASELINE_RELEASE = "1.5.1-native-vl.4"
BASELINE_TAG = "v1.5.1-native-vl.4"
BASELINE_NATIVE_SOURCE_COMMIT = "bd012874027defa528279a357609b713e9069df4"
NATIVE_SOURCE_COMMIT = "06a35e36269a9fe443c56e99c5fedf7ca25304cc"
ENGINE_SHA256 = (
    "1138a62b9515118a1237849bfe02ea8daeccec94d88a92e49c885775619bf829"
)
COMPONENT_SHA256 = {
    "native_engine": ENGINE_SHA256,
    "static_launcher": (
        "d913b44ff33ad3903470817793e5bf095bc3cc6fe5eda00fc1562ed818323a43"
    ),
    "aotriton_fmha_provider": (
        "e5336b2d66b36c5f17aeb07ab780fa8f60a6092910f9b01b3ebf4bc31f766bb4"
    ),
    "ck_fmha_provider": (
        "0145e819869d3ea5b25661f8f11279f5e6bd3484b29e8c7910a8b30c927baa93"
    ),
    "q16384_hybrid_fmha_provider": (
        "e6b8c50e76c3c7d49b8c208275234d7f4607faff250019826866f86e37fedd29"
    ),
    "aotriton_runtime": (
        "e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5"
    ),
    "aotriton_gfx1151_image": (
        "0f3a6a2f9dee6620443ee2145ee1f8257bde65a378589952840d99bf3d485c10"
    ),
    "vision_attention_image": (
        "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e"
    ),
}
BASELINE_ENGINE_SHA256 = (
    "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
)
DENSE_VISION_ATTENTION_SHA256 = (
    "e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c"
)
DENSE_VISION_ATTENTION_KERNEL_HASH = (
    "2bb5125141eea1b811395f9833de3077de68893bfebbbf1950ca26832db6bb52"
)
RUNTIME_PATHS = (
    "native/src",
    "native/include",
    "native/aot",
    "native/generated",
    "scripts/build-native-runtime.sh",
)
ALLOWED_RUNTIME_DELTA = {
    "native/include/aima/native_chat_protocol.h",
    "native/include/aima/native_http_support.h",
    "native/src/native_chat_protocol.cpp",
    "native/src/native_http_server.cpp",
    "native/src/native_http_support.cpp",
    "native/src/native_vl_request.cpp",
    "scripts/build-native-runtime.sh",
}
CPU_RUNTIME_DELTA = frozenset(ALLOWED_RUNTIME_DELTA)
SAFE_PREFIX_RUNTIME_DELTA = CPU_RUNTIME_DELTA | {
    "native/include/aima/native_linear_prefill.h",
    "native/include/aima/native_multimodal_cache.h",
    "native/include/aima/native_resident_engine.h",
    "native/src/main.cpp",
    "native/src/native_linear_prefill.hip.cpp",
    "native/src/native_multimodal_cache.cpp",
    "native/src/native_resident_engine.hip.cpp",
}
SAFE_PREFIX_CASE_CHECKS = {"token_identity", "completion_count", "prompt_count", "cold_disabled",
                           "lookup", "matched_boundary", "suffix_only", "no_serial_prefill", "restore_measured"}
DEFAULT_INPUTS = {
    "baseline_g5": (
        ROOT
        / "benchmarks/results/native-vl-g5-release-v1.5.1-native-vl.4.json"
    ),
    "baseline_product": (
        ROOT
        / "benchmarks/results/native-portable-product-v1.5.1-native-vl.4.json"
    ),
    "product_contract": (
        ROOT / "native/product-contract-v1.5.1-native-vl.5.json"
    ),
    "product_qualification_generator": Path(__file__).resolve(),
    "g5_qualification_generator": (
        ROOT / "scripts/generate-native-vl-patch-g5-qualification.py"
    ),
    "http_control_plane_qualifier": (
        ROOT / "scripts/qualify-native-http-control-plane.py"
    ),
    "chat_protocol_qualifier": ROOT / "scripts/qualify-native-chat-protocol.py",
    "package_script": ROOT / "scripts/package-native-foundation.sh",
    "bundle_manifest_generator": ROOT / "scripts/generate-native-bundle-manifest.py",
    "bundle_qualifier": ROOT / "scripts/qualify-native-portable-bundle.py",
    "resident_soak_qualifier": ROOT / "scripts/qualify-native-vl-resident-soak.py",
    "rollback_qualifier": ROOT / "scripts/qualify-native-vl-rollback.py",
    "release_gates_qualifier": ROOT / "scripts/qualify-native-vl-release-gates.py",
    "package_input_verifier": ROOT / "scripts/verify-native-package-inputs.py",
    "bundle_closure": ROOT / "scripts/native_bundle_closure.py",
    "package_qualification": ROOT / "aima_engine/package_qualification.py",
    "public_hygiene": ROOT / "aima_engine/public_hygiene.py",
    "makefile": ROOT / "Makefile",
    "systemd_service": ROOT / "packaging/systemd/aima-engine.service",
    "systemd_environment": ROOT / "packaging/systemd/aima-engine.env.example",
}
DEFAULT_OUTPUT = (
    ROOT
    / "output/native-portable-product-v1.5.1-native-vl.5.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed


def require_sealed(
    name: str, payload: Mapping[str, Any], schema: str | None = None
) -> None:
    if schema is not None and payload.get("schema") != schema:
        raise ValueError(f"{name} schema differs")
    if payload.get("complete") is not True or payload.get("qualified") is not True:
        raise ValueError(f"{name} is not complete and qualified")
    errors = verify_manifest_integrity(payload)
    if errors:
        raise ValueError(f"{name} integrity failed: {errors}")


def require_chat_protocol(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema")
        != "aima.native-chat-protocol-qualification.v0.1.0"
        or payload.get("qualified") is not True
        or not payload.get("checks")
        or not all(payload.get("checks", {}).values())
    ):
        raise ValueError("chat protocol qualification failed")
    if RELEASE in {"1.5.1-native-vl.6", "1.5.1-native-vl.7"} and payload["checks"].get(
        "vl_default_thinking_stream_nonstream_parity"
    ) is not True:
        raise ValueError("default VL thinking qualification is missing or failed")
    integrity = payload.get("integrity", {})
    unsigned = dict(payload)
    unsigned.pop("integrity", None)
    canonical = json.dumps(
        unsigned, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if integrity.get("canonical_payload_sha256") != hashlib.sha256(
        canonical
    ).hexdigest():
        raise ValueError("chat protocol qualification integrity failed")
    engine = payload.get("engine", {})
    if (
        engine.get("sha256") != ENGINE_SHA256
        or engine.get("build_info")
        != {"version": "1.5.1-native", "source_commit": NATIVE_SOURCE_COMMIT}
    ):
        raise ValueError("chat protocol candidate identity differs")


def release_source_checks(release_commit: str) -> dict[str, bool]:
    tag = git(
        "rev-parse", "--verify", f"refs/tags/{RELEASE_TAG}^{{commit}}", check=False
    )
    delta = set(
        git(
            "diff",
            "--name-only",
            f"{BASELINE_NATIVE_SOURCE_COMMIT}..{NATIVE_SOURCE_COMMIT}",
            "--",
            *RUNTIME_PATHS,
        ).stdout.splitlines()
    )
    release_delta = git(
        "diff", "--quiet", NATIVE_SOURCE_COMMIT, release_commit, "--", *RUNTIME_PATHS,
        check=False,
    )
    source_ancestor = git(
        "merge-base", "--is-ancestor", NATIVE_SOURCE_COMMIT, release_commit,
        check=False,
    )
    head = git("rev-parse", "HEAD", check=False)
    status = git("status", "--porcelain", "--untracked-files=normal", check=False)
    return {
        "immutable_release_tag_resolves": tag.returncode == 0,
        "immutable_release_tag_exact": (
            tag.returncode == 0 and tag.stdout.strip() == release_commit
        ),
        "candidate_source_is_release_ancestor": source_ancestor.returncode == 0,
        "candidate_runtime_tree_matches_release": release_delta.returncode == 0,
        "runtime_delta_exactly_allowlisted": delta == ALLOWED_RUNTIME_DELTA,
        "gpu_aot_and_generated_runtime_unchanged": not any(
            path.startswith(("native/aot/", "native/generated/")) for path in delta
        ),
        "checkout_head_is_release_commit": head.stdout.strip() == release_commit,
        "release_checkout_clean": status.returncode == 0 and not status.stdout.strip(),
    }


def exact_component(path: Path, logical_path: str, expected: str) -> dict[str, Any]:
    record = file_component(path, logical_path)
    if record["sha256"] != expected:
        raise ValueError(f"component SHA-256 differs: {logical_path}")
    return record


def require_safe_prefix(path: Path, capacity: int) -> dict[str, Any]:
    payload = load_object(path)
    require_runtime_binding(payload.get("runtime_binding"), ENGINE_SHA256)
    require_sealed("safe prefix", payload, "aima-amd395-qwen36/native-safe-prefix-cache/v1")
    if (payload.get("release_eligible") is not True
        or payload.get("engine_sha256") != ENGINE_SHA256
        or payload.get("build_info", {}).get("source_commit") != NATIVE_SOURCE_COMMIT
        or payload.get("source", {}).get("checkout_clean") is not True
        or payload.get("source", {}).get("engine_runtime_matches_checkout") is not True
        or payload.get("source", {}).get("native_source_commit") != NATIVE_SOURCE_COMMIT
        or payload.get("source", {}).get("generator_sha256") != sha256(ROOT / "scripts/qualify-native-prefix-cache.py")
        or payload.get("source", {}).get("protocol_helper_sha256") != sha256(ROOT / "scripts/qualify-native-chat-protocol.py")
        or payload.get("configuration", {}).get("context_tokens") != 8192
        or payload.get("configuration", {}).get("cache_capacity") != capacity
        or payload.get("configuration", {}).get("checkpoint_limit_per_entry") != 3
        or payload.get("configuration", {}).get("checkpoint_block_tokens") != 32
        or payload.get("cached_peak_memory", {}).get("gtt_bytes", 0) <= 0
        or payload["cached_peak_memory"]["gtt_bytes"] > 96 * 1024**3):
        raise ValueError("safe-prefix candidate identity, capacity or memory gate differs")
    cases = payload.get("cases", [])
    expected = {"short_cold", "short_exact", "divergent_chat", "divergent_exact_sse",
                "shared_system", "multi_turn", "append_seed", "whole_prompt_append",
                "checkpoint_as_complete_prompt", "full_owner_after_short_checkpoint",
                "evicted_short", "media_a", "media_b_same_geometry", "media_a_restored",
                "text_after_media", "performance_seed"}
    expected.update(f"eviction_fill_{index}" for index in range(5))
    expected.update(f"performance_partial_{index}" for index in range(5))
    if (len(cases) != len(expected) or {case.get("case_id") for case in cases} != expected
        or not all(case.get("pass") is True and set(case.get("checks", {})) == SAFE_PREFIX_CASE_CHECKS
                   and all(case["checks"].values()) for case in cases)):
        raise ValueError("safe-prefix generation/isolation coverage is incomplete")
    reproduction = next(case for case in cases if case["case_id"] == "divergent_chat")
    if (reproduction["prefix_cache"]["matched_tokens"] != 15
        or reproduction["suffix_aot_tokens"] != 11):
        raise ValueError("issue 12 reproduction did not restore the saved boundary")
    logits = payload.get("logits", {})
    rows = logits.get("cases", [])
    boundaries = {f"boundary_{boundary}_partial" for boundary in (31, 32, 33, 1023, 1024, 1025, 8192, 8193)}
    if (logits.get("qualified") is not True or len(rows) != 34
        or not boundaries.issubset({row.get("case_id") for row in rows})
        or not all(row.get("pass") is True and row.get("top1_match") is True
                   and row.get("elements") == 248320 and 0 <= row.get("kl_divergence", 1) < 0.005
                   for row in rows)
        or len(logits.get("short_checkpoint_replay", {})) != 4
        or not all(logits["short_checkpoint_replay"].values())):
        raise ValueError("safe-prefix full-vocabulary or active-KV extent gate failed")
    performance = payload.get("performance", {})
    speedup = performance.get("median_partial_ttft_speedup", 0)
    retention = performance.get("median_decode_retention", 0)
    if not (math.isfinite(speedup) and math.isfinite(retention) and speedup > 1 and retention >= 0.97):
        raise ValueError("safe-prefix TTFT/decode retention gate failed")
    artifacts = payload.get("artifacts", [])
    if not artifacts:
        raise ValueError("safe-prefix raw evidence is missing")
    root = path.parent.resolve()
    for record in artifacts:
        artifact = (root / record["path"]).resolve()
        if (not artifact.is_relative_to(root) or not artifact.is_file()
            or artifact.stat().st_size != record["bytes"] or sha256(artifact) != record["sha256"]):
            raise ValueError("safe-prefix raw evidence differs")
    return payload


def build_payload(
    *,
    inputs: Mapping[str, Path],
    components: Mapping[str, Path],
    chat_protocol_path: Path,
    http_control_plane_path: Path,
    release_commit: str,
    recorded_on: str,
    safe_prefix_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    baseline_g5 = load_object(inputs["baseline_g5"])
    baseline_product = load_object(inputs["baseline_product"])
    require_sealed(
        "baseline G5",
        baseline_g5,
        "aima-amd395-qwen36/native-vl-g5-release-qualification/v1",
    )
    require_sealed("baseline package input", baseline_product, SCHEMA)
    if (
        baseline_g5.get("release") != BASELINE_RELEASE
        or baseline_g5.get("decision", {}).get("g5_native_release_product")
        is not True
        or baseline_product.get("release") != BASELINE_RELEASE
        or baseline_product.get("components", {}).get("source", {}).get(
            "release_tag"
        )
        != BASELINE_TAG
        or baseline_product.get("components", {}).get("native_engine", {}).get(
            "sha256"
        )
        != BASELINE_ENGINE_SHA256
    ):
        raise ValueError("frozen .4 release baseline differs")

    contract = load_object(inputs["product_contract"])
    if (
        contract.get("schema")
        != "aima-amd395-qwen36/native-vl-product-contract/v1"
        or contract.get("release") != RELEASE
        or contract.get("release_tag") != RELEASE_TAG
        or contract.get("candidate", {}).get("native_source_commit")
        != NATIVE_SOURCE_COMMIT
        or contract.get("candidate", {}).get("native_engine_sha256")
        != ENGINE_SHA256
    ):
        raise ValueError("patch product contract identity differs")

    chat_protocol = load_object(chat_protocol_path)
    require_chat_protocol(chat_protocol)
    http_control_plane = load_object(http_control_plane_path)
    require_sealed(
        "HTTP control plane",
        http_control_plane,
        "aima-amd395-qwen36/native-http-control-plane/v1",
    )
    if (
        http_control_plane.get("candidate", {}).get("native_engine_sha256")
        != ENGINE_SHA256
        or http_control_plane.get("candidate", {}).get("native_source_commit")
        != NATIVE_SOURCE_COMMIT
        or not all(http_control_plane.get("checks", {}).values())
    ):
        raise ValueError("HTTP control-plane candidate identity differs")

    source_checks = release_source_checks(release_commit)
    if not all(source_checks.values()):
        raise ValueError(f"patch release source checks failed: {source_checks}")

    info = json.loads(
        subprocess.run(
            [str(components["native_engine"]), "--build-info"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if info != {"version": "1.5.1-native", "source_commit": NATIVE_SOURCE_COMMIT}:
        raise ValueError("native engine build identity differs")
    logical_paths = {
        "native_engine": "build/native/aima-engine-native",
        "static_launcher": "build/native/aima-engine-launcher",
        "aotriton_fmha_provider": "build/native/libaima-fmha-aotriton.so",
        "ck_fmha_provider": "build/native/libaima-fmha-ck.so",
        "q16384_hybrid_fmha_provider": "build/native/libaima-fmha-q16384-hybrid.so",
        "aotriton_runtime": "build/native/libaotriton_v2.so.0.11.1",
        "aotriton_gfx1151_image": (
            "lib/aotriton.images/amd-gfx11xx/flash/attn_fwd/"
            "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2"
        ),
        "vision_attention_image": "lib/aima-vision-attention.hsaco",
    }
    records = {
        name: exact_component(path, logical_paths[name], COMPONENT_SHA256[name])
        for name, path in components.items()
    }
    records["source"] = {
        "release_tag": RELEASE_TAG,
        "release_commit": release_commit,
        "native_source_commit": NATIVE_SOURCE_COMMIT,
        "native_source_dirty": False,
    }
    records["embedded_dense_vision_attention"] = {
        "sha256": DENSE_VISION_ATTENTION_SHA256,
        "kernel_hash": DENSE_VISION_ATTENTION_KERNEL_HASH,
        "integrity_boundary": "embedded AOT registry inside native_engine",
    }

    inherited = {
        "g1_full_vl_functional_parity": True,
        "g2_vl_correctness_parity": True,
        "g3_text_product_no_regression": True,
        "g4_native_vl_performance": True,
        "two_host_portability_baseline": True,
    }
    patch_checks = {
        "runtime_delta_exactly_allowlisted": source_checks[
            "runtime_delta_exactly_allowlisted"
        ],
        "gpu_aot_and_generated_runtime_unchanged": source_checks[
            "gpu_aot_and_generated_runtime_unchanged"
        ],
        "exact_candidate_chat_protocol": chat_protocol.get("qualified") is True,
        "exact_candidate_http_control_plane": (
            http_control_plane.get("qualified") is True
        ),
        "exact_component_closure": True,
    }
    gates = {**inherited, **patch_checks}
    extra_validation = {}
    if RELEASE == "1.5.1-native-vl.7":
        if safe_prefix_paths is None or set(safe_prefix_paths) != {"prefix", "one_owner", "text_matrix"}:
            raise ValueError("safe-prefix release requires fresh prefix/capacity/performance evidence")
        require_safe_prefix(safe_prefix_paths["prefix"], 32768)
        require_safe_prefix(safe_prefix_paths["one_owner"], 262144)
        matrix = load_object(safe_prefix_paths["text_matrix"])
        require_runtime_binding(matrix.get("runtime_binding"), ENGINE_SHA256)
        if (matrix.get("schema") != "aima-amd395-qwen36/native-full-matrix-qualification/v1"
            or matrix.get("complete") is not True or matrix.get("qualified") is not True
            or matrix.get("engine", {}).get("sha256") != ENGINE_SHA256
            or matrix.get("baseline", {}).get("sha256") != sha256(ROOT / "benchmarks/results/v1.0.0.json")
            or matrix.get("measurement_protocol", {}).get("minimum_retention", 0) < 0.97
            or len(matrix.get("cells", [])) != 19
            or not all(cell.get("pass") is True and cell.get("prefill_retention", 0) >= 0.97
                       and (cell.get("output_tokens") == 1 or cell.get("decode_retention", 0) >= 0.97)
                       and cell.get("sample_count", 0) >= 2 for cell in matrix["cells"])):
            raise ValueError("safe-prefix release requires the exact 19-cell text matrix")
        root = safe_prefix_paths["text_matrix"].parent.resolve()
        for cell in matrix["cells"]:
            if len(cell["reports"]) != len(cell["report_sha256"]):
                raise ValueError("text matrix raw report bindings are incomplete")
            for report, digest in zip(cell["reports"], cell["report_sha256"], strict=True):
                path = (root / report).resolve()
                if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
                    raise ValueError("text matrix raw report differs")
                require_runtime_binding(load_object(path).get("qualification", {}).get("runtime_binding"),
                                        ENGINE_SHA256)
        gates.update(exact_safe_prefix_generation_logits=True,
                     exact_safe_prefix_one_owner=True, exact_text_19_cell_matrix=True)
        extra_validation = {name: file_component(path, f"candidate-validation/{name}/{path.name}")
                            for name, path in safe_prefix_paths.items()}
    return {
        "schema": SCHEMA,
        "release": RELEASE,
        "recorded_on": recorded_on,
        "complete": True,
        "qualified": all(gates.values()),
        "qualification_scope": (
            "Safe-prefix host-schedule delta: new text checkpoint capture/restore, active-KV extent, "
            "generation and full-vocabulary correctness, capacity and performance are qualified on "
            "the exact candidate. Frozen .4 VL/cold arithmetic and portable-userspace evidence "
            "is inherited only for unchanged kernels/providers; it does not qualify new checkpoint paths."
            if RELEASE == "1.5.1-native-vl.7" else
            f"patch-delta qualification: exact {RELEASE} CPU protocol/HTTP candidate and "
            "package closure, with .4 G1-G4 and two-host portability inherited "
            "only because the fail-closed runtime diff leaves GPU math, AOT "
            "images and external providers unchanged"
        ),
        "engine_version": info["version"],
        "components": records,
        "inputs": {
            **{
                name: file_component(path, str(path.relative_to(ROOT)))
                for name, path in inputs.items()
            },
            "chat_protocol": file_component(
                chat_protocol_path, "candidate-validation/native-chat-protocol.json"
            ),
            "http_control_plane": file_component(
                http_control_plane_path,
                "candidate-validation/native-http-control-plane.json",
            ),
        },
        "baseline_inheritance": {
            "release": BASELINE_RELEASE,
            "release_tag": BASELINE_TAG,
            "native_source_commit": BASELINE_NATIVE_SOURCE_COMMIT,
            "native_engine_sha256": BASELINE_ENGINE_SHA256,
            "g5": file_component(
                inputs["baseline_g5"],
                "benchmarks/results/native-vl-g5-release-v1.5.1-native-vl.4.json",
            ),
            "package_input": file_component(
                inputs["baseline_product"],
                "benchmarks/results/native-portable-product-v1.5.1-native-vl.4.json",
            ),
            "inherited_gates": inherited,
            "claim_limit": (
                f"No .4 engine measurement is represented as an exact {RELEASE} "
                "measurement; inheritance applies only to unchanged GPU and "
                "portable userspace scope."
            ),
        },
        "runtime_delta": {
            "base_commit": BASELINE_NATIVE_SOURCE_COMMIT,
            "candidate_commit": NATIVE_SOURCE_COMMIT,
            "allowed_paths": sorted(ALLOWED_RUNTIME_DELTA),
            "checks": source_checks,
            "classification": ("text checkpoint host scheduling, cache ownership and diagnostics"
                               if RELEASE == "1.5.1-native-vl.7" else
                               "CPU chat protocol, HTTP control plane and cache synchronization"),
        },
        "candidate_validation": {
            **extra_validation,
            "chat_protocol": file_component(
                chat_protocol_path, "candidate-validation/native-chat-protocol.json"
            ),
            "http_control_plane": file_component(
                http_control_plane_path,
                "candidate-validation/native-http-control-plane.json",
            ),
        },
        "runtime_dependency_gate": {
            "runtime_python": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
            "runtime_transformers": False,
            "host_rocm_userspace_required": False,
            "model_weights_bundled": False,
        },
        "gates": gates,
        "decision": {
            **gates,
            "package_input_qualified": all(gates.values()),
            "next_blocking_boundary": (
                "archive isolation, one-hour mixed-workload soak, rollback and "
                "clean-tag repository gates"
            ),
        },
    }


def configure_release(contract_path: Path) -> None:
    global RELEASE, RELEASE_TAG, NATIVE_SOURCE_COMMIT, ENGINE_SHA256
    global DEFAULT_OUTPUT, ALLOWED_RUNTIME_DELTA
    contract = load_object(contract_path)
    release = contract.get("release")
    if release not in {"1.5.1-native-vl.5", "1.5.1-native-vl.6", "1.5.1-native-vl.7"}:
        raise ValueError("unsupported native VL patch release")
    ALLOWED_RUNTIME_DELTA = set(SAFE_PREFIX_RUNTIME_DELTA if release == "1.5.1-native-vl.7" else CPU_RUNTIME_DELTA)
    if set(contract.get("patch_scope", {}).get("allowed_runtime_paths", [])) != ALLOWED_RUNTIME_DELTA:
        raise ValueError("patch contract changes the runtime inheritance allowlist")
    if contract.get("frozen_baseline", {}).get("native_source_commit") != BASELINE_NATIVE_SOURCE_COMMIT:
        raise ValueError("patch contract changes the frozen baseline")
    RELEASE = release
    RELEASE_TAG = f"v{release}"
    NATIVE_SOURCE_COMMIT = contract["candidate"]["native_source_commit"]
    ENGINE_SHA256 = contract["candidate"]["native_engine_sha256"]
    COMPONENT_SHA256["native_engine"] = ENGINE_SHA256
    DEFAULT_INPUTS["product_contract"] = contract_path
    if release == "1.5.1-native-vl.7":
        DEFAULT_INPUTS["qualification_runtime_verifier"] = ROOT / "aima_engine/qualification_runtime.py"
    else:
        DEFAULT_INPUTS.pop("qualification_runtime_verifier", None)
    for key, filename in (("safe_prefix_qualifier", "qualify-native-prefix-cache.py"),
                          ("text_matrix_qualifier", "qualify-native-full-matrix.py")):
        if release == "1.5.1-native-vl.7":
            DEFAULT_INPUTS[key] = ROOT / "scripts" / filename
        else:
            DEFAULT_INPUTS.pop(key, None)
    DEFAULT_OUTPUT = ROOT / f"output/native-portable-product-v{release}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--chat-protocol", type=Path, required=True)
    parser.add_argument("--http-control-plane", type=Path, required=True)
    parser.add_argument("--safe-prefix", type=Path)
    parser.add_argument("--safe-prefix-one-owner", type=Path)
    parser.add_argument("--text-matrix", type=Path)
    parser.add_argument("--recorded-on", default="2026-09-01")
    parser.add_argument("--product-contract", type=Path, default=DEFAULT_INPUTS["product_contract"])
    parser.add_argument("--output", type=Path)
    for name in COMPONENT_SHA256:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    configure_release(args.product_contract.expanduser().resolve())
    for name, path in DEFAULT_INPUTS.items():
        if not path.is_file():
            raise SystemExit(f"input is missing: {name}: {path}")
    components = {
        name: getattr(args, name).expanduser().resolve() for name in COMPONENT_SHA256
    }
    for name, path in components.items():
        if not path.is_file():
            raise SystemExit(f"component is missing: {name}: {path}")
    chat_protocol = args.chat_protocol.expanduser().resolve()
    http_control_plane = args.http_control_plane.expanduser().resolve()
    sealed = seal_manifest(
        build_payload(
            inputs=DEFAULT_INPUTS,
            components=components,
            chat_protocol_path=chat_protocol,
            http_control_plane_path=http_control_plane,
            release_commit=args.release_commit,
            recorded_on=args.recorded_on,
            safe_prefix_paths=({"prefix": args.safe_prefix, "one_owner": args.safe_prefix_one_owner,
                                "text_matrix": args.text_matrix}
                               if all((args.safe_prefix, args.safe_prefix_one_owner, args.text_matrix)) else None),
        )
    )
    output = (args.output or DEFAULT_OUTPUT).expanduser().resolve()
    digest = atomic_json(output, sealed)
    print(json.dumps({"qualified": True, "output": str(output), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
