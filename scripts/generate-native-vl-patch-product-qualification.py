#!/usr/bin/env python3
"""Generate the exact package-input qualification for native VL patch .5."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
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


def build_payload(
    *,
    inputs: Mapping[str, Path],
    components: Mapping[str, Path],
    chat_protocol_path: Path,
    http_control_plane_path: Path,
    release_commit: str,
    recorded_on: str,
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
    return {
        "schema": SCHEMA,
        "release": RELEASE,
        "recorded_on": recorded_on,
        "complete": True,
        "qualified": all(gates.values()),
        "qualification_scope": (
            "patch-delta qualification: exact .5 CPU protocol/HTTP candidate and "
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
                "No .4 engine measurement is represented as an exact .5 "
                "measurement; inheritance applies only to unchanged GPU and "
                "portable userspace scope."
            ),
        },
        "runtime_delta": {
            "base_commit": BASELINE_NATIVE_SOURCE_COMMIT,
            "candidate_commit": NATIVE_SOURCE_COMMIT,
            "allowed_paths": sorted(ALLOWED_RUNTIME_DELTA),
            "checks": source_checks,
            "classification": "CPU chat protocol, HTTP control plane and cache synchronization",
        },
        "candidate_validation": {
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--chat-protocol", type=Path, required=True)
    parser.add_argument("--http-control-plane", type=Path, required=True)
    parser.add_argument("--recorded-on", default="2026-09-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    for name in COMPONENT_SHA256:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
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
        )
    )
    output = args.output.expanduser().resolve()
    digest = atomic_json(output, sealed)
    print(json.dumps({"qualified": True, "output": str(output), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
