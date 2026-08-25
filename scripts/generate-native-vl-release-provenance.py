#!/usr/bin/env python3
"""Generate the immutable and tree-bound public evidence record for native VL."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.release_evidence import evidence_tree  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
    verify_manifest_integrity,
)


RELEASE = "1.5.1-native-vl.4"
RELEASE_TAG = "v1.5.1-native-vl.4"
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/"
    "native-release-provenance-v1.5.1-native-vl.4.json"
)
IMMUTABLE_PATHS = {
    "product_result": (
        ROOT
        / "benchmarks/results/native-vl-g5-release-v1.5.1-native-vl.4.json"
    ),
    "portable_bundle_result": (
        ROOT
        / "benchmarks/results/"
        "native-portable-bundle-v1.5.1-native-vl.4.json"
    ),
    "product_contract": (
        ROOT / "native/product-contract-v1.5.1-native-vl.4.json"
    ),
    "package_input_qualification": (
        ROOT
        / "benchmarks/results/"
        "native-portable-product-v1.5.1-native-vl.4.json"
    ),
    "g1": (
        ROOT / "benchmarks/results/native-vl-g1-coverage-audit-v0.1.0.json"
    ),
    "g2": ROOT / "benchmarks/results/vl-correctness-v0.1.0.json",
    "g3": (
        ROOT / "benchmarks/results/text-v151-nonregression-v0.1.0.json"
    ),
    "g4": ROOT / "benchmarks/results/vl-performance-v0.1.0.json",
    "temperature_sampling": (
        ROOT / "benchmarks/results/native-temperature-sampling-v0.1.0.json"
    ),
    "envelope": (
        ROOT / "benchmarks/results/native-vl-envelope-v0.1.0.json"
    ),
}
STANDALONE_EVIDENCE = {
    "g1_g2": (
        ROOT
        / "benchmarks/runs/native-vl-g1-g2-20260824-bd01287-final/"
        "g1/native-vl-capability-v0.1.0.json",
        ROOT / "benchmarks/runs/native-vl-g1-g2-20260824-bd01287-final",
    ),
    "g1_generation_raw": (
        ROOT
        / "benchmarks/results/native-vl-generation-current-head-v0.1.0-raw/"
        "probe.stdout.json",
        ROOT
        / "benchmarks/results/"
        "native-vl-generation-current-head-v0.1.0-raw",
    ),
    "g3_correctness": (
        ROOT
        / "benchmarks/runs/native-correctness-20260824-bd01287-final/"
        "correctness.json",
        ROOT / "benchmarks/runs/native-correctness-20260824-bd01287-final",
    ),
    "g3_doctor": (
        ROOT
        / "benchmarks/runs/native-doctor-20260824-bd01287-final/doctor.json",
        ROOT / "benchmarks/runs/native-doctor-20260824-bd01287-final",
    ),
    "g3_mmlu": (
        ROOT
        / "benchmarks/runs/native-mmlu256-eval-20260824-bd01287-final/"
        "mmlu256.json",
        ROOT / "benchmarks/runs/native-mmlu256-eval-20260824-bd01287-final",
    ),
    "g3_openai_features": (
        ROOT
        / "benchmarks/runs/native-openai-features-20260824-bd01287-final/"
        "features.json",
        ROOT / "benchmarks/runs/native-openai-features-20260824-bd01287-final",
    ),
    "g3_product_surfaces": (
        ROOT
        / "benchmarks/runs/native-product-surfaces-20260824-bd01287-final/"
        "surfaces.json",
        ROOT / "benchmarks/runs/native-product-surfaces-20260824-bd01287-final",
    ),
    "g3_text_matrix": (
        ROOT
        / "benchmarks/runs/"
        "native-paired-text-matrix-20260824-bd01287-final-balanced6/matrix.json",
        ROOT
        / "benchmarks/runs/"
        "native-paired-text-matrix-20260824-bd01287-final-balanced6",
    ),
    "g4_reference_availability_raw": (
        ROOT
        / "benchmarks/results/"
        "native-vl-performance-reference-availability-v0.1.0-raw/"
        "evidence-spec.json",
        ROOT
        / "benchmarks/results/"
        "native-vl-performance-reference-availability-v0.1.0-raw",
    ),
    "g4_vl_performance": (
        ROOT
        / "benchmarks/runs/native-vl-performance-20260824-bd01287-final/"
        "pair-01/summary.json",
        ROOT / "benchmarks/runs/native-vl-performance-20260824-bd01287-final",
    ),
}
PRODUCT_EVIDENCE_KEYS = (
    "primary_bundle",
    "second_bundle",
    "resident_soak",
    "rollback",
    "release_gates",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def public_record(summary: Path, tree: Path) -> dict[str, Any]:
    record = file_component(summary, str(summary.relative_to(ROOT)))
    record["tree_path"] = str(tree.relative_to(ROOT))
    return record


def build_payload(recorded_on: str) -> dict[str, Any]:
    for name, path in IMMUTABLE_PATHS.items():
        if not path.is_file():
            raise RuntimeError(f"immutable record is missing: {name}: {path}")
    for name, (summary, tree) in STANDALONE_EVIDENCE.items():
        if not summary.is_file() or not tree.is_dir():
            raise RuntimeError(f"public evidence is missing: {name}")
    final_result = load_object(IMMUTABLE_PATHS["product_result"])
    package_input = load_object(IMMUTABLE_PATHS["package_input_qualification"])
    bundle_result = load_object(IMMUTABLE_PATHS["portable_bundle_result"])
    temperature_sampling = load_object(
        IMMUTABLE_PATHS["temperature_sampling"]
    )
    if (
        final_result.get("schema")
        != "aima-amd395-qwen36/native-vl-g5-release-qualification/v1"
        or final_result.get("release") != RELEASE
        or final_result.get("qualified") is not True
        or final_result.get("decision", {}).get("g5_native_release_product")
        is not True
        or verify_manifest_integrity(final_result)
        or verify_manifest_integrity(package_input)
        or verify_manifest_integrity(bundle_result)
        or verify_manifest_integrity(temperature_sampling)
    ):
        raise RuntimeError("sealed native VL release records are invalid")
    source = final_result.get("source", {})
    if (
        source.get("release_tag") != RELEASE_TAG
        or source.get("release_commit") is None
        or source.get("native_source_commit") is None
    ):
        raise RuntimeError("final native VL source identity is incomplete")
    if (
        temperature_sampling.get("schema")
        != "aima-amd395-qwen36/native-temperature-sampling/v1"
        or temperature_sampling.get("complete") is not True
        or temperature_sampling.get("qualified") is not True
        or temperature_sampling.get("candidate", {}).get("source_commit")
        != source.get("native_source_commit")
        or temperature_sampling.get("candidate", {}).get("binary_sha256")
        != final_result.get("candidate", {}).get("native_engine_sha256")
        or temperature_sampling.get("post_g5_boundary", {}).get("g5_result")
        != file_component(
            IMMUTABLE_PATHS["product_result"],
            str(IMMUTABLE_PATHS["product_result"].relative_to(ROOT)),
        )
        or temperature_sampling.get("post_g5_boundary", {}).get(
            "g5_recorded_at"
        )
        != final_result.get("recorded_at")
        or temperature_sampling.get("decision", {}).get(
            "nonzero_temperature_supported"
        )
        is not True
    ):
        raise RuntimeError(
            "post-G5 temperature-sampling qualification is invalid"
        )
    package_inputs = package_input.get("inputs", {})
    for gate in ("g1", "g2", "g3", "g4", "envelope"):
        expected = file_component(
            IMMUTABLE_PATHS[gate],
            str(IMMUTABLE_PATHS[gate].relative_to(ROOT)),
        )
        if package_inputs.get(gate) != expected:
            raise RuntimeError(f"package input no longer binds {gate}")

    product_evidence: dict[str, dict[str, Any]] = {}
    for key in PRODUCT_EVIDENCE_KEYS:
        record = final_result.get("evidence", {}).get(key)
        if not isinstance(record, dict):
            raise RuntimeError(f"final result evidence is missing: {key}")
        path = ROOT / str(record.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256(path) != record.get("sha256")
        ):
            raise RuntimeError(f"final result evidence differs: {key}")
        product_evidence[key] = public_record(path, path.parent)
    public_evidence = {
        **product_evidence,
        **{
            name: public_record(summary, tree)
            for name, (summary, tree) in STANDALONE_EVIDENCE.items()
        },
    }
    public_trees = {}
    for name, record in public_evidence.items():
        tree = ROOT / record["tree_path"]
        value = evidence_tree(tree)
        value["path"] = record["tree_path"]
        public_trees[name] = value

    return {
        "schema": "aima-amd395-qwen36/native-release-provenance/v1",
        "release": RELEASE,
        "recorded_on": recorded_on,
        "complete": True,
        "release_tag": RELEASE_TAG,
        "release_commit": source["release_commit"],
        "native_source_commit": source["native_source_commit"],
        "clarification": (
            "The immutable tag binds the exact native runtime, G1-G4 evidence, "
            "release tooling and product contract. The portable archive, two-host "
            "qualification, one-hour mixed-workload soak, exact v1.5.1 rollback "
            "and post-G5 temperature-sampling qualification are additive "
            "hash-bound release records."
        ),
        "immutable_records": {
            name: file_component(path, str(path.relative_to(ROOT)))
            for name, path in IMMUTABLE_PATHS.items()
        },
        "public_evidence": public_evidence,
        "public_evidence_trees": public_trees,
        "claim_effect": (
            "G1-G5 passed for the exact native VL archive: one native resident "
            "process serves text, image, video and mixed media without framework "
            "or host ROCm userspace dependencies, preserves the v1.5.1 text "
            "product, reproduces on a second AMD395, survives the formal soak "
            "and rolls back to the exact v1.5.1 portable baseline. Seeded "
            "nonzero-temperature text/VL sampling is separately qualified only "
            "after that G5 boundary."
        ),
    }


def verify_exact(path: Path, expected: Mapping[str, Any]) -> None:
    if load_object(path) != expected:
        raise SystemExit(f"native VL release provenance is stale: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{sha256(path)}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise SystemExit(f"native VL release provenance sidecar is stale: {sidecar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-on", default="2026-08-24")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sealed = seal_manifest(build_payload(args.recorded_on))
    output = args.output.expanduser().resolve()
    if args.check:
        verify_exact(output, sealed)
        print(f"native VL release provenance: PASS ({output})")
        return 0
    digest = atomic_json(output, sealed)
    print(
        json.dumps(
            {
                "complete": True,
                "output": str(output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
