#!/usr/bin/env python3
"""Generate the additive, tree-bound v1.5.1-native-vl.5 evidence record."""

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


RELEASE = "1.5.1-native-vl.5"
RELEASE_TAG = "v1.5.1-native-vl.5"
RELEASE_COMMIT = "eb7d8ac30cea4401a068fd25f1f1379c72eaf448"
NATIVE_SOURCE_COMMIT = "06a35e36269a9fe443c56e99c5fedf7ca25304cc"
ENGINE_SHA256 = (
    "1138a62b9515118a1237849bfe02ea8daeccec94d88a92e49c885775619bf829"
)
ARCHIVE_NAME = "aima-engine-native-portable-194f2a673904.tar.zst"
ARCHIVE_SHA256 = (
    "59f30c4232b8459f3efcd7b8506cc71b957614c0aac1fa96a2eb4e15f52940a3"
)
PUBLIC_EVIDENCE_NAME = (
    "aima-engine-v1.5.1-native-vl.5-public-evidence.tar.zst"
)
RELEASE_URL = (
    "https://github.com/skyguan92/"
    "AIMA-AMD395-Qwen36-35B-Linux-Engine/releases/tag/"
    "v1.5.1-native-vl.5"
)
RESULTS = ROOT / "benchmarks/results"
DEFAULT_OUTPUT = RESULTS / "native-release-provenance-v1.5.1-native-vl.5.json"

IMMUTABLE_PATHS = {
    "product_result": (
        RESULTS / "native-vl-g5-release-v1.5.1-native-vl.5.json"
    ),
    "portable_bundle_result": (
        RESULTS / "native-portable-bundle-v1.5.1-native-vl.5.json"
    ),
    "product_contract": (
        ROOT / "native/product-contract-v1.5.1-native-vl.5.json"
    ),
    "package_input_qualification": (
        RESULTS / "native-portable-product-v1.5.1-native-vl.5.json"
    ),
    "chat_protocol": (
        RESULTS / "native-chat-protocol-v1.5.1-native-vl.5.json"
    ),
    "http_control_plane": (
        RESULTS / "native-http-control-plane-v1.5.1-native-vl.5.json"
    ),
    "archive_manifest": (
        RESULTS / "native-portable-manifest-v1.5.1-native-vl.5.json"
    ),
    "archive_checksum": RESULTS / f"{ARCHIVE_NAME}.sha256",
    "baseline_g5": (
        RESULTS / "native-vl-g5-release-v1.5.1-native-vl.4.json"
    ),
    "baseline_package_input": (
        RESULTS / "native-portable-product-v1.5.1-native-vl.4.json"
    ),
}

PUBLIC_EVIDENCE = {
    "chat_protocol": (
        ROOT
        / "benchmarks/runs/native-chat-protocol-20260901-vl5-final/"
        "chat-protocol.json",
        ROOT / "benchmarks/runs/native-chat-protocol-20260901-vl5-final",
    ),
    "http_control_plane": (
        ROOT
        / "benchmarks/runs/native-http-control-plane-20260901-vl5-final/"
        "http-control-plane.json",
        ROOT
        / "benchmarks/runs/native-http-control-plane-20260901-vl5-final",
    ),
    "bundle": (
        ROOT
        / "benchmarks/runs/native-portable-bundle-20260901-vl5-final/"
        "bundle.json",
        ROOT / "benchmarks/runs/native-portable-bundle-20260901-vl5-final",
    ),
    "resident_soak": (
        ROOT
        / "benchmarks/runs/native-vl-resident-soak-20260901-vl5-final/"
        "soak.json",
        ROOT
        / "benchmarks/runs/native-vl-resident-soak-20260901-vl5-final",
    ),
    "rollback": (
        ROOT
        / "benchmarks/runs/native-vl-rollback-20260901-vl5-final/"
        "rollback.json",
        ROOT / "benchmarks/runs/native-vl-rollback-20260901-vl5-final",
    ),
    "release_gates": (
        ROOT
        / "benchmarks/runs/native-vl-release-gates-20260901-vl5-final/"
        "release-gates.json",
        ROOT
        / "benchmarks/runs/native-vl-release-gates-20260901-vl5-final",
    ),
}


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


def require_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{sha256(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"sealed record sidecar differs: {path}")


def require_sealed(
    path: Path,
    *,
    schema: str,
    release: str | None = RELEASE,
    require_complete: bool = True,
) -> dict[str, Any]:
    payload = load_object(path)
    if (
        payload.get("schema") != schema
        or payload.get("qualified") is not True
        or (require_complete and payload.get("complete") is not True)
        or (release is not None and payload.get("release") != release)
        or verify_manifest_integrity(payload)
    ):
        raise RuntimeError(f"sealed release record is invalid: {path}")
    require_sidecar(path)
    return payload


def public_record(summary: Path, tree: Path) -> dict[str, Any]:
    record = file_component(summary, str(summary.relative_to(ROOT)))
    record["tree_path"] = str(tree.relative_to(ROOT))
    return record


def expected_final_component(path: Path) -> dict[str, Any]:
    return file_component(path, f"patch-release-evidence/{path.name}")


def require_checksum(path: Path, digest: str, name: str) -> None:
    if path.read_text(encoding="utf-8") != f"{digest}  {name}\n":
        raise RuntimeError(f"release asset checksum differs: {path}")


def build_payload(recorded_on: str) -> dict[str, Any]:
    for name, path in IMMUTABLE_PATHS.items():
        if not path.is_file():
            raise RuntimeError(f"immutable record is missing: {name}: {path}")
    for name, (summary, tree) in PUBLIC_EVIDENCE.items():
        if not summary.is_file() or not tree.is_dir():
            raise RuntimeError(f"public evidence is missing: {name}")

    final_result = require_sealed(
        IMMUTABLE_PATHS["product_result"],
        schema=(
            "aima-amd395-qwen36/"
            "native-vl-patch-g5-release-qualification/v1"
        ),
    )
    package_input = require_sealed(
        IMMUTABLE_PATHS["package_input_qualification"],
        schema="aima-amd395-qwen36/native-vl-product-qualification/v1",
    )
    bundle = require_sealed(
        IMMUTABLE_PATHS["portable_bundle_result"],
        schema="aima-amd395-qwen36/native-portable-bundle-qualification/v1",
    )
    chat = require_sealed(
        IMMUTABLE_PATHS["chat_protocol"],
        schema="aima.native-chat-protocol-qualification.v0.1.0",
        release=None,
        require_complete=False,
    )
    http = require_sealed(
        IMMUTABLE_PATHS["http_control_plane"],
        schema="aima-amd395-qwen36/native-http-control-plane/v1",
    )

    source = final_result.get("source", {})
    if (
        source.get("release_tag") != RELEASE_TAG
        or source.get("release_commit") != RELEASE_COMMIT
        or source.get("native_source_commit") != NATIVE_SOURCE_COMMIT
        or source.get("native_source_dirty") is not False
        or final_result.get("decision", {}).get("patch_release_promoted")
        is not True
        or final_result.get("archive", {}).get("sha256") != ARCHIVE_SHA256
        or final_result.get("candidate", {}).get("native_engine_sha256")
        != ENGINE_SHA256
    ):
        raise RuntimeError("final patch release identity differs")

    final_evidence_paths = {
        "package_input": IMMUTABLE_PATHS["package_input_qualification"],
        "bundle": PUBLIC_EVIDENCE["bundle"][0],
        "resident_soak": PUBLIC_EVIDENCE["resident_soak"][0],
        "rollback": PUBLIC_EVIDENCE["rollback"][0],
        "release_gates": PUBLIC_EVIDENCE["release_gates"][0],
        "baseline_g5": IMMUTABLE_PATHS["baseline_g5"],
        "product_contract": IMMUTABLE_PATHS["product_contract"],
    }
    for name, path in final_evidence_paths.items():
        if final_result.get("evidence", {}).get(name) != expected_final_component(
            path
        ):
            raise RuntimeError(f"final patch evidence differs: {name}")

    product_inputs = package_input.get("inputs", {})
    expected_inputs = {
        "chat_protocol": file_component(
            IMMUTABLE_PATHS["chat_protocol"],
            "candidate-validation/native-chat-protocol.json",
        ),
        "http_control_plane": file_component(
            IMMUTABLE_PATHS["http_control_plane"],
            "candidate-validation/native-http-control-plane.json",
        ),
        "baseline_g5": file_component(
            IMMUTABLE_PATHS["baseline_g5"],
            "benchmarks/results/native-vl-g5-release-v1.5.1-native-vl.4.json",
        ),
        "product_contract": file_component(
            IMMUTABLE_PATHS["product_contract"],
            "native/product-contract-v1.5.1-native-vl.5.json",
        ),
    }
    for name, expected in expected_inputs.items():
        if product_inputs.get(name) != expected:
            raise RuntimeError(f"package input no longer binds {name}")
    if (
        package_input.get("components", {}).get("native_engine", {}).get(
            "sha256"
        )
        != ENGINE_SHA256
        or package_input.get("components", {}).get("source") != source
        or chat.get("engine", {}).get("sha256") != ENGINE_SHA256
        or http.get("candidate", {}).get("native_engine_sha256")
        != ENGINE_SHA256
        or bundle.get("archive", {}).get("sha256") != ARCHIVE_SHA256
    ):
        raise RuntimeError("exact-candidate patch evidence differs")

    manifest_path = IMMUTABLE_PATHS["archive_manifest"]
    require_sidecar(manifest_path)
    manifest = load_object(manifest_path)
    file_records = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    if (
        manifest.get("complete") is not True
        or manifest.get("release") != RELEASE
        or manifest.get("source")
        != {
            "commit": RELEASE_COMMIT,
            "dirty": False,
            "native_commit": NATIVE_SOURCE_COMMIT,
            "release_tag": RELEASE_TAG,
        }
        or file_records.get("libexec/aima-engine.real", {}).get("sha256")
        != ENGINE_SHA256
        or manifest.get("native_vl", {}).get("enabled") is not True
    ):
        raise RuntimeError("portable archive manifest identity differs")
    require_checksum(
        IMMUTABLE_PATHS["archive_checksum"], ARCHIVE_SHA256, ARCHIVE_NAME
    )
    public_evidence = {
        name: public_record(summary, tree)
        for name, (summary, tree) in PUBLIC_EVIDENCE.items()
    }
    public_trees: dict[str, dict[str, Any]] = {}
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
        "release_commit": RELEASE_COMMIT,
        "native_source_commit": NATIVE_SOURCE_COMMIT,
        "release_url": RELEASE_URL,
        "release_assets": {
            "portable_archive": {
                "name": ARCHIVE_NAME,
                "bytes": final_result["archive"]["bytes"],
                "sha256": ARCHIVE_SHA256,
            },
            "public_evidence": {
                "name": PUBLIC_EVIDENCE_NAME,
                "checksum_sidecar_required": True,
                "contains_this_provenance_record": True,
            },
        },
        "clarification": (
            "The immutable .5 tag binds the exact patch runtime, tooling and "
            "product contract. The final archive, exact-candidate chat/HTTP "
            "qualifications, one-hour soak, rollback and release gates are "
            "additive hash-bound evidence and do not move that tag."
        ),
        "inheritance": final_result["inheritance"],
        "immutable_records": {
            name: file_component(path, str(path.relative_to(ROOT)))
            for name, path in IMMUTABLE_PATHS.items()
        },
        "public_evidence": public_evidence,
        "public_evidence_trees": public_trees,
        "claim_effect": (
            "The exact .5 archive passed isolated AMD395 bundle execution, "
            "3600 seconds and 360 requests of resident mixed traffic, exact "
            "v1.5.1 rollback and repository/security/evidence gates. The .4 "
            "G1-G4 and two-host results remain inherited baseline evidence, "
            "not exact .5 measurements or a second exact .5 host run."
        ),
    }


def verify_exact(path: Path, expected: Mapping[str, Any]) -> None:
    if load_object(path) != expected:
        raise SystemExit(f"native VL patch provenance is stale: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{sha256(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(
        encoding="utf-8"
    ) != expected_sidecar:
        raise SystemExit(f"native VL patch provenance sidecar is stale: {sidecar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-on", default="2026-09-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sealed = seal_manifest(build_payload(args.recorded_on))
    output = args.output.expanduser().resolve()
    if args.check:
        verify_exact(output, sealed)
        print(f"native VL patch release provenance: PASS ({output})")
        return 0
    digest = atomic_json(output, sealed)
    print(
        json.dumps(
            {"complete": True, "output": str(output), "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
