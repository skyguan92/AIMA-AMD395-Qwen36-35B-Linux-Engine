#!/usr/bin/env python3
"""Generate the exact-candidate package-input qualification for native VL."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
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
RELEASE = "1.5.1-native-vl.1"
RELEASE_TAG = "v1.5.1-native-vl.1"
NATIVE_SOURCE_COMMIT = "bd012874027defa528279a357609b713e9069df4"
ENGINE_SHA256 = (
    "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
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
DENSE_VISION_ATTENTION_SHA256 = (
    "e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c"
)
DENSE_VISION_ATTENTION_KERNEL_HASH = (
    "2bb5125141eea1b811395f9833de3077de68893bfebbbf1950ca26832db6bb52"
)
DEFAULT_INPUTS = {
    "g1": ROOT / "benchmarks/results/native-vl-g1-coverage-audit-v0.1.0.json",
    "g2": ROOT / "benchmarks/results/vl-correctness-v0.1.0.json",
    "g3": ROOT / "benchmarks/results/text-v151-nonregression-v0.1.0.json",
    "g4": ROOT / "benchmarks/results/vl-performance-v0.1.0.json",
    "envelope": ROOT / "benchmarks/results/native-vl-envelope-v0.1.0.json",
    "goal": ROOT / "docs/NATIVE_VL_GOAL.md",
    "media_test": ROOT / "tests/native_media_test.cpp",
    "cache_test": ROOT / "tests/native_multimodal_cache_test.cpp",
    "resident_source": ROOT / "native/src/native_resident_engine.hip.cpp",
    "vision_stack_source": (
        ROOT / "native/src/native_vision_aot_block_stack.hip.cpp"
    ),
    "media_source": ROOT / "native/src/native_media.cpp",
    "remote_media_source": ROOT / "native/src/native_remote_media.cpp",
    "product_contract": (
        ROOT / "native/product-contract-v1.5.1-native-vl.1.json"
    ),
    "product_qualification_generator": Path(__file__).resolve(),
    "package_script": ROOT / "scripts/package-native-foundation.sh",
    "bundle_manifest_generator": (
        ROOT / "scripts/generate-native-bundle-manifest.py"
    ),
    "bundle_qualifier": ROOT / "scripts/qualify-native-portable-bundle.py",
    "temperature_sampling_qualifier": (
        ROOT / "scripts/qualify-native-temperature-sampling.py"
    ),
    "resident_soak_qualifier": (
        ROOT / "scripts/qualify-native-vl-resident-soak.py"
    ),
    "rollback_qualifier": ROOT / "scripts/qualify-native-vl-rollback.py",
    "release_gates_qualifier": (
        ROOT / "scripts/qualify-native-vl-release-gates.py"
    ),
    "g5_qualification_generator": (
        ROOT / "scripts/generate-native-vl-g5-qualification.py"
    ),
    "release_provenance_generator": (
        ROOT / "scripts/generate-native-vl-release-provenance.py"
    ),
    "package_input_verifier": (
        ROOT / "scripts/verify-native-package-inputs.py"
    ),
    "bundle_closure": ROOT / "scripts/native_bundle_closure.py",
    "package_qualification": ROOT / "aima_engine/package_qualification.py",
    "public_hygiene": ROOT / "aima_engine/public_hygiene.py",
    "release_evidence": ROOT / "aima_engine/release_evidence.py",
    "makefile": ROOT / "Makefile",
    "systemd_service": ROOT / "packaging/systemd/aima-engine.service",
    "systemd_environment": (
        ROOT / "packaging/systemd/aima-engine.env.example"
    ),
}
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/native-portable-product-v1.5.1-native-vl.1.json"
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


def require_sealed_qualification(
    name: str,
    payload: Mapping[str, Any],
    schema: str,
) -> None:
    if payload.get("schema") != schema:
        raise ValueError(f"{name} schema differs from the frozen contract")
    if payload.get("complete") is not True or payload.get("qualified") is not True:
        raise ValueError(f"{name} is not complete and qualified")
    errors = verify_manifest_integrity(payload)
    if errors:
        raise ValueError(f"{name} integrity failed: {errors}")


def exact_component(path: Path, logical_path: str, expected: str) -> dict[str, Any]:
    record = file_component(path, logical_path)
    if record["sha256"] != expected:
        raise ValueError(f"component SHA-256 differs: {logical_path}")
    return record


def engine_build_info(engine: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(engine), "--build-info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"native --build-info failed: {completed.stderr.strip()}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("native --build-info returned a non-object")
    return value


def source_architecture_checks(inputs: Mapping[str, Path]) -> dict[str, bool]:
    resident = inputs["resident_source"].read_text(encoding="utf-8")
    vision = inputs["vision_stack_source"].read_text(encoding="utf-8")
    media = inputs["media_source"].read_text(encoding="utf-8")
    remote = inputs["remote_media_source"].read_text(encoding="utf-8")
    runtime_files = [
        path
        for root in (ROOT / "native/src", ROOT / "native/include")
        for path in root.rglob("*")
        if path.is_file()
    ]
    per_layer = re.compile(r"(?:layer|block)[_-]?\d+", re.IGNORECASE)
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--quiet",
            NATIVE_SOURCE_COMMIT,
            "--",
            "native/src",
            "native/include",
            "native/generated",
            "scripts/build-native-runtime.sh",
        ],
        check=False,
    )
    return {
        "runtime_source_matches_embedded_commit": diff.returncode == 0,
        "no_numbered_runtime_layer_or_block_sources": not any(
            per_layer.search(path.name) for path in runtime_files
        ),
        "single_shared_forty_layer_loop": (
            "for (std::size_t layer_index = 0; layer_index < 40;"
            in resident
        ),
        "single_parameterized_twenty_seven_block_loop": (
            "constexpr std::size_t kVisionBlockCount = 27;" in vision
            and "for (std::size_t block_index = 0;"
            in vision
        ),
        "descriptor_relative_local_media_open": (
            "openat" in media and "O_NOFOLLOW" in media
        ),
        "remote_media_has_explicit_allowlist": "allowlisted" in remote,
        "remote_media_disables_proxy_inheritance": (
            "CURLOPT_PROXY" in remote and "CURLOPT_NOPROXY" in remote
        ),
    }


def release_source_checks(release_commit: str) -> dict[str, bool]:
    tag = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-parse",
            "--verify",
            f"refs/tags/{RELEASE_TAG}^{{commit}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    native_ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            NATIVE_SOURCE_COMMIT,
            release_commit,
        ],
        check=False,
    )
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    tracked_diff = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", release_commit, "--"],
        check=False,
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "immutable_release_tag_resolves": tag.returncode == 0,
        "immutable_release_tag_exact": (
            tag.returncode == 0 and tag.stdout.strip() == release_commit
        ),
        "native_source_is_release_ancestor": native_ancestor.returncode == 0,
        "checkout_head_is_release_commit": (
            head.returncode == 0 and head.stdout.strip() == release_commit
        ),
        "tracked_tree_matches_release_commit": tracked_diff.returncode == 0,
        "release_checkout_clean": (
            status.returncode == 0 and status.stdout.strip() == ""
        ),
    }


def build_payload(
    *,
    inputs: Mapping[str, Path],
    components: Mapping[str, Path],
    release_commit: str,
    recorded_on: str,
) -> dict[str, Any]:
    payloads = {
        name: load_object(path)
        for name, path in inputs.items()
        if name
        in {"g1", "g2", "g3", "g4", "envelope"}
    }
    require_sealed_qualification(
        "G1",
        payloads["g1"],
        "aima-amd395-qwen36/native-vl-g1-coverage-audit/v1",
    )
    require_sealed_qualification(
        "G2",
        payloads["g2"],
        "aima-amd395-qwen36/native-vl-g2-qualification/v1",
    )
    require_sealed_qualification(
        "G3",
        payloads["g3"],
        "aima-amd395-qwen36/text-v151-nonregression/v1",
    )
    require_sealed_qualification(
        "G4", payloads["g4"], "aima-amd395-qwen36/vl-performance/v1"
    )
    require_sealed_qualification(
        "execution envelope",
        payloads["envelope"],
        "aima-amd395-qwen36/native-vl-envelope-qualification/v1",
    )

    candidate = {
        "source_commit": NATIVE_SOURCE_COMMIT,
        "binary_sha256": ENGINE_SHA256,
    }
    if payloads["g2"].get("candidate") != candidate:
        raise ValueError("G2 candidate identity differs")
    if payloads["g3"].get("candidate") != candidate:
        raise ValueError("G3 candidate identity differs")
    g4_candidate = payloads["g4"].get("artifact_identity", {}).get(
        "candidate", {}
    )
    if g4_candidate.get("source_commit") != NATIVE_SOURCE_COMMIT:
        raise ValueError("G4 candidate source identity differs")
    g4_files = {
        item.get("path"): item.get("sha256")
        for item in g4_candidate.get("files", [])
        if isinstance(item, Mapping)
    }
    expected_g4 = {
        "aima-engine-native": COMPONENT_SHA256["native_engine"],
        "libaima-fmha-aotriton.so": COMPONENT_SHA256[
            "aotriton_fmha_provider"
        ],
        "libaima-fmha-ck.so": COMPONENT_SHA256["ck_fmha_provider"],
        "libaima-fmha-q16384-hybrid.so": COMPONENT_SHA256[
            "q16384_hybrid_fmha_provider"
        ],
        "libaotriton_v2.so.0.11.1": COMPONENT_SHA256["aotriton_runtime"],
        "aima-vision-attention.hsaco": COMPONENT_SHA256[
            "vision_attention_image"
        ],
        (
            "aotriton.images/amd-gfx11xx/flash/attn_fwd/"
            "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2"
        ): COMPONENT_SHA256["aotriton_gfx1151_image"],
    }
    if g4_files != expected_g4:
        raise ValueError("G4 candidate runtime closure differs")

    decisions = {
        "g1_full_vl_functional_parity": (
            payloads["g1"].get("decision", {}).get("g1_passed") is True
        ),
        "g2_vl_correctness_parity": (
            payloads["g2"].get("decision", {}).get("g2_passed") is True
        ),
        "g3_text_product_no_regression": (
            payloads["g3"].get("decision", {}).get(
                "g3_text_product_no_regression"
            )
            is True
        ),
        "g4_native_vl_performance": payloads["g4"].get("qualified") is True,
    }
    if not all(decisions.values()):
        raise ValueError("not every G1-G4 gate is qualified")

    ready = payloads["envelope"].get("server", {}).get("ready", {})
    ready_checks = {
        "single_native_process": (
            payloads["envelope"].get("server", {}).get("checks", {}).get(
                "one_model_load"
            )
            is True
        ),
        "language_weights_ready": ready.get("language_model_tensor_count") == 693,
        "vision_weights_ready": ready.get("visual_model_tensor_count") == 333,
        "all_model_tensors_ready": ready.get("model_tensor_count") == 1026,
        "vision_warmup_before_ready": ready.get("vision_warmup_completed") is True,
        "general_vision_image_exact": (
            ready.get("vision_attention_image_sha256")
            == COMPONENT_SHA256["vision_attention_image"]
        ),
        "dense_vision_image_exact": (
            ready.get("vision_dense_image_attention_image_sha256")
            == DENSE_VISION_ATTENTION_SHA256
        ),
        "runtime_python_absent": ready.get("runtime_python") is False,
        "runtime_torch_absent": ready.get("runtime_torch") is False,
        "runtime_vllm_absent": ready.get("runtime_vllm") is False,
        "runtime_triton_absent": ready.get("runtime_triton") is False,
        "full_window_ready": ready.get("context_capacity") == 262144,
    }
    architecture_checks = source_architecture_checks(inputs)
    source_checks = release_source_checks(release_commit)
    if (
        not all(ready_checks.values())
        or not all(architecture_checks.values())
        or not all(source_checks.values())
    ):
        raise ValueError("native release-boundary static checks failed")

    info = engine_build_info(components["native_engine"])
    if info != {
        "version": "1.5.1-native",
        "source_commit": NATIVE_SOURCE_COMMIT,
    }:
        raise ValueError("native engine build identity differs")
    logical_component_paths = {
        "native_engine": "build/native/aima-engine-native",
        "static_launcher": "build/native/aima-engine-launcher",
        "aotriton_fmha_provider": "build/native/libaima-fmha-aotriton.so",
        "ck_fmha_provider": "build/native/libaima-fmha-ck.so",
        "q16384_hybrid_fmha_provider": (
            "build/native/libaima-fmha-q16384-hybrid.so"
        ),
        "aotriton_runtime": "build/native/libaotriton_v2.so.0.11.1",
        "aotriton_gfx1151_image": (
            "lib/aotriton.images/amd-gfx11xx/flash/attn_fwd/"
            "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2"
        ),
        "vision_attention_image": "lib/aima-vision-attention.hsaco",
    }
    component_records: dict[str, Any] = {
        name: exact_component(
            path, logical_component_paths[name], COMPONENT_SHA256[name]
        )
        for name, path in components.items()
    }
    component_records["source"] = {
        "release_tag": RELEASE_TAG,
        "release_commit": release_commit,
        "native_source_commit": NATIVE_SOURCE_COMMIT,
        "native_source_dirty": False,
    }
    component_records["embedded_dense_vision_attention"] = {
        "sha256": DENSE_VISION_ATTENTION_SHA256,
        "kernel_hash": DENSE_VISION_ATTENTION_KERNEL_HASH,
        "integrity_boundary": (
            "embedded AOT registry inside the exact native_engine payload"
        ),
    }

    return {
        "schema": SCHEMA,
        "release": RELEASE,
        "recorded_on": recorded_on,
        "complete": True,
        "qualified": True,
        "qualification_scope": (
            "exact G1-G4 candidate and complete package-input closure; final "
            "G5 promotion additionally requires the built archive, isolated "
            "bundle, second host, soak, rollback and release-evidence gates"
        ),
        "engine_version": info["version"],
        "components": component_records,
        "inputs": {
            name: file_component(path, str(path.relative_to(ROOT)))
            for name, path in inputs.items()
        },
        "model_contract": {
            "id": "Qwen/Qwen3.6-35B-A3B",
            "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
            "checkpoint_index_sha256": (
                "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
            ),
            "checkpoint_shards": 26,
            "weights_in_package": False,
            "input_contract": "external standard Hugging Face Safetensors",
        },
        "ready_boundary": {
            "checks": ready_checks,
            "command_to_ready_wall_ms": ready["command_to_ready_wall_ms"],
            "language_model_payload_bytes": ready[
                "language_model_payload_bytes"
            ],
            "visual_model_payload_bytes": ready["visual_model_payload_bytes"],
            "model_payload_bytes": ready["model_payload_bytes"],
            "vision_warmup_patches": ready["vision_warmup_patches"],
            "vision_image_count_warmup_patches": ready[
                "vision_image_count_warmup_patches"
            ],
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
        "source_architecture": {
            "checks": architecture_checks,
            "vision_stack": "one parameterized 27-block shared loop",
            "language_stack": "one parameterized 40-layer shared loop",
        },
        "release_source": {
            "tag": RELEASE_TAG,
            "commit": release_commit,
            "native_source_commit": NATIVE_SOURCE_COMMIT,
            "checks": source_checks,
        },
        "media_security_contract": {
            "local_file": (
                "descriptor-relative allowlisted open, no symlink traversal, "
                "bounded bytes"
            ),
            "remote_url": (
                "per-hop allowlist, bounded redirects/time/bytes, SSRF and "
                "credential rejection, proxy inheritance disabled"
            ),
            "live_gate_pending": "make check plus make security-scan in G5",
        },
        "gates": decisions,
        "decision": {
            **decisions,
            "package_input_qualified": True,
            "g5_native_release_product": False,
            "next_blocking_boundary": (
                "build and qualify the exact portable archive, then pass "
                "isolated-bundle, second-host, resident soak, rollback and "
                "release-evidence verification"
            ),
        },
    }


def verify_exact(path: Path, expected: Mapping[str, Any]) -> None:
    if load_object(path) != expected:
        raise SystemExit(f"native VL product qualification is stale: {path}")
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{digest}  {path.name}\n":
        raise SystemExit(f"native VL product sidecar is stale: {sidecar}")


def component_argument(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or name not in COMPONENT_SHA256 or not raw_path:
        raise argparse.ArgumentTypeError(
            "component must use a frozen NAME=PATH"
        )
    return name, Path(raw_path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--recorded-on", default="2026-08-24")
    parser.add_argument("--component", action="append", type=component_argument)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    for name in (
        "g1",
        "g2",
        "g3",
        "g4",
        "envelope",
    ):
        parser.add_argument(
            "--" + name,
            type=Path,
            default=DEFAULT_INPUTS[name],
        )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.release_commit):
        parser.error("release-commit must be a full lowercase Git commit")
    supplied = dict(args.component or [])
    if (
        len(supplied) != len(args.component or [])
        or set(supplied) != set(COMPONENT_SHA256)
    ):
        parser.error(
            "--component must provide exactly: "
            + ", ".join(COMPONENT_SHA256)
        )
    inputs = {
        **DEFAULT_INPUTS,
        **{
            name: getattr(args, name).expanduser().resolve()
            for name in (
                "g1",
                "g2",
                "g3",
                "g4",
                "envelope",
            )
        },
    }
    sealed = seal_manifest(
        build_payload(
            inputs=inputs,
            components=supplied,
            release_commit=args.release_commit,
            recorded_on=args.recorded_on,
        )
    )
    output = args.output.expanduser().resolve()
    if args.check:
        verify_exact(output, sealed)
        print(f"native VL product qualification: PASS ({output})")
        return 0
    digest = atomic_json(output, sealed)
    print(
        json.dumps(
            {
                "output": str(output),
                "qualified": sealed["qualified"],
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
