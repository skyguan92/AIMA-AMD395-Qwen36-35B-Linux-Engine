#!/usr/bin/env python3
"""Generate the final native VL .5 patch-release qualification."""

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


SCHEMA = "aima-amd395-qwen36/native-vl-patch-g5-release-qualification/v1"
RELEASE = "1.5.1-native-vl.5"
RELEASE_TAG = "v1.5.1-native-vl.5"
NATIVE_SOURCE_COMMIT = "06a35e36269a9fe443c56e99c5fedf7ca25304cc"
ENGINE_SHA256 = (
    "1138a62b9515118a1237849bfe02ea8daeccec94d88a92e49c885775619bf829"
)
BASELINE_RELEASE = "1.5.1-native-vl.4"
BASELINE_ENGINE_SHA256 = (
    "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
)
ROLLBACK_ARCHIVE_SHA256 = (
    "4e38f90fce3feb7bccf1965d87a3ec2bebddc439ce62e75fe1bc797c6ce1a5bc"
)
SHA256 = re.compile(r"[0-9a-f]{64}")
SCHEMAS = {
    "product_result": "aima-amd395-qwen36/native-vl-product-qualification/v1",
    "bundle": "aima-amd395-qwen36/native-portable-bundle-qualification/v1",
    "soak": "aima-amd395-qwen36/native-vl-resident-soak/v1",
    "rollback": "aima-amd395-qwen36/native-vl-rollback/v1",
    "release_gates": "aima-amd395-qwen36/native-vl-release-gates/v1",
}
BASELINE_G5 = (
    ROOT / "benchmarks/results/native-vl-g5-release-v1.5.1-native-vl.4.json"
)
PRODUCT_CONTRACT = ROOT / "native/product-contract-v1.5.1-native-vl.5.json"
DEFAULT_OUTPUT = ROOT / "output/native-vl-g5-release-v1.5.1-native-vl.5.json"


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


def require_sealed(name: str, path: Path, schema: str, release: str) -> dict[str, Any]:
    payload = load_object(path)
    if (
        payload.get("schema") != schema
        or payload.get("release") != release
        or payload.get("complete") is not True
        or payload.get("qualified") is not True
        or verify_manifest_integrity(payload)
    ):
        raise RuntimeError(f"{name} is incomplete, unsealed or has the wrong identity")
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{sha256(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"{name} checksum sidecar differs")
    return payload


def verify_raw_record(owner: Path, recorded_path: str, expected: str) -> None:
    if not recorded_path.startswith("raw/"):
        return
    root = owner.parent.resolve()
    path = (root / recorded_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"raw evidence escapes its root: {recorded_path}") from error
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"raw evidence differs: {path}")


def verify_raw_closure(owner: Path, value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            verify_raw_closure(owner, item)
        return
    if not isinstance(value, dict):
        return
    for path_key, digest_key in (
        ("path", "sha256"),
        ("report", "report_sha256"),
        ("response", "response_sha256"),
    ):
        path = value.get(path_key)
        digest = value.get(digest_key)
        if isinstance(path, str) and isinstance(digest, str):
            verify_raw_record(owner, path, digest)
    for item in value.values():
        verify_raw_closure(owner, item)


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def exact_bundle_checks(
    payload: Mapping[str, Any], *, archive_sha256: str
) -> dict[str, bool]:
    doctor = payload.get("doctor", {})
    host = doctor.get("host", {})
    vl_smoke = payload.get("vl_smoke", {})
    return {
        "primary_host_role": payload.get("host_role") == "primary_amd395",
        "host_fingerprint_present": SHA256.fullmatch(
            str(host.get("fingerprint_sha256", ""))
        )
        is not None,
        "host_is_amd395": (
            host.get("gpu", {}).get("architecture") == "gfx1151"
            and host.get("gtt_bytes", 0) >= 96 * 1024**3
        ),
        "archive_exact": payload.get("archive", {}).get("sha256") == archive_sha256,
        "source_exact": (
            payload.get("source", {}).get("native_source_commit")
            == NATIVE_SOURCE_COMMIT
        ),
        "manifest_native_vl": payload.get("manifest", {})
        .get("native_vl", {})
        .get("enabled")
        is True,
        "isolated_environment": payload.get("isolated_environment", {}).get("keys")
        == ["HOME", "LANG", "PATH"],
        "doctor_qualified": (
            doctor.get("qualified") is True
            and doctor.get("bundle_detected_complete") is True
        ),
        "provider_smokes_qualified": (
            len(payload.get("provider_smokes", [])) == 3
            and all(
                item.get("qualified") is True
                for item in payload.get("provider_smokes", [])
            )
        ),
        "vl_smoke_qualified": (
            vl_smoke.get("qualified") is True
            and vl_smoke.get("checks", {}).get("five_requests_qualified") is True
            and vl_smoke.get("checks", {}).get("single_model_load") is True
        ),
        "public_hygiene": (
            payload.get("public_hygiene", {}).get("passed") is True
            and payload.get("public_hygiene", {}).get("finding_count") == 0
        ),
    }


def build_payload(
    *, paths: Mapping[str, Path], archive: Path, recorded_at: str
) -> dict[str, Any]:
    payloads = {
        name: require_sealed(name, paths[name], schema, RELEASE)
        for name, schema in SCHEMAS.items()
    }
    for name in ("bundle", "soak", "rollback", "release_gates"):
        verify_raw_closure(paths[name], payloads[name])

    baseline = require_sealed(
        "baseline G5",
        BASELINE_G5,
        "aima-amd395-qwen36/native-vl-g5-release-qualification/v1",
        BASELINE_RELEASE,
    )
    baseline_gates = baseline.get("gates", {})
    baseline_cross_host = baseline.get("host_checks", {}).get("cross_host", {})
    inherited_checks = {
        "baseline_engine_exact": baseline.get("candidate", {}).get(
            "native_engine_sha256"
        )
        == BASELINE_ENGINE_SHA256,
        "baseline_g1_g4_all": all(
            baseline_gates.get(name) is True
            for name in (
                "g1_full_vl_functional_parity",
                "g2_vl_correctness_parity",
                "g3_text_product_no_regression",
                "g4_native_vl_performance",
            )
        ),
        "baseline_two_host_portability": (
            baseline_gates.get("second_amd395") is True
            and all(baseline_cross_host.values())
        ),
        "baseline_fully_promoted": baseline.get("decision", {}).get(
            "g5_native_release_product"
        )
        is True,
    }

    product = payloads["product_result"]
    source = product.get("components", {}).get("source", {})
    product_checks = {
        "release_tag_exact": source.get("release_tag") == RELEASE_TAG,
        "native_source_exact": source.get("native_source_commit")
        == NATIVE_SOURCE_COMMIT,
        "native_engine_exact": product.get("components", {})
        .get("native_engine", {})
        .get("sha256")
        == ENGINE_SHA256,
        "source_clean": source.get("native_source_dirty") is False,
        "all_patch_gates": all(product.get("gates", {}).values()),
        "package_input_qualified": product.get("decision", {}).get(
            "package_input_qualified"
        )
        is True,
    }
    if not all(product_checks.values()):
        raise RuntimeError(f"package-input qualification differs: {product_checks}")
    release_commit = source.get("release_commit")
    if (
        not isinstance(release_commit, str)
        or git_output("rev-parse", "HEAD") != release_commit
        or git_output("rev-parse", "--verify", f"refs/tags/{RELEASE_TAG}^{{commit}}")
        != release_commit
        or git_output("status", "--porcelain", "--untracked-files=normal")
    ):
        raise RuntimeError("final patch G5 generation requires the clean tagged checkout")

    contract = load_object(PRODUCT_CONTRACT)
    if (
        contract.get("release") != RELEASE
        or contract.get("candidate", {}).get("native_source_commit")
        != NATIVE_SOURCE_COMMIT
        or contract.get("candidate", {}).get("native_engine_sha256")
        != ENGINE_SHA256
    ):
        raise RuntimeError("patch product contract identity differs")

    archive_digest = sha256(archive)
    checksum = archive.with_name(archive.name + ".sha256")
    if (
        not checksum.is_file()
        or checksum.read_text(encoding="utf-8").split()[0] != archive_digest
    ):
        raise RuntimeError("candidate archive checksum differs")
    bundle_checks = exact_bundle_checks(payloads["bundle"], archive_sha256=archive_digest)

    soak = payloads["soak"]
    soak_checks = {
        "archive_exact": soak.get("archive", {}).get("sha256") == archive_digest,
        "primary_host": soak.get("host_role") == "primary_amd395",
        "one_hour": soak.get("measurement", {}).get("elapsed_seconds", 0) >= 3600,
        "minimum_requests": soak.get("measurement", {}).get("request_count", 0)
        >= 240,
        "all_checks_pass": all(soak.get("checks", {}).values()),
        "single_model_load": soak.get("decision", {}).get(
            "single_model_load_preserved"
        )
        is True,
    }
    rollback = payloads["rollback"]
    rollback_checks = {
        "candidate_archive_exact": rollback.get("candidate", {})
        .get("archive", {})
        .get("sha256")
        == archive_digest,
        "baseline_archive_exact": rollback.get("rollback_target", {})
        .get("archive", {})
        .get("sha256")
        == ROLLBACK_ARCHIVE_SHA256,
        "primary_host": rollback.get("host_role") == "primary_amd395",
        "all_checks_pass": all(rollback.get("checks", {}).values()),
        "rollback_decision": rollback.get("decision", {}).get(
            "rollback_to_exact_v1_5_1_passed"
        )
        is True,
    }
    release_gates = payloads["release_gates"]
    static_checks = {
        "source_exact": release_gates.get("source") == source,
        "all_checks_pass": all(release_gates.get("checks", {}).values()),
        "make_check": release_gates.get("decision", {}).get("make_check_passed")
        is True,
        "security_scan": release_gates.get("decision", {}).get(
            "make_security_scan_passed"
        )
        is True,
        "evidence_preflight": release_gates.get("decision", {}).get(
            "make_verify_evidence_passed"
        )
        is True,
    }
    decisions = {
        "inherited_frozen_g1_g4": all(inherited_checks.values()),
        "inherited_two_host_portable_userspace": inherited_checks[
            "baseline_two_host_portability"
        ],
        "exact_patch_package_input": all(product_checks.values()),
        "exact_patch_archive_on_amd395": all(bundle_checks.values()),
        "resident_mixed_workload_soak": all(soak_checks.values()),
        "rollback": all(rollback_checks.values()),
        "repository_security_evidence_gates": all(static_checks.values()),
    }
    if not all(decisions.values()):
        raise RuntimeError(f"not every patch release gate passed: {decisions}")

    evidence_paths = {
        "package_input": paths["product_result"],
        "bundle": paths["bundle"],
        "resident_soak": paths["soak"],
        "rollback": paths["rollback"],
        "release_gates": paths["release_gates"],
        "baseline_g5": BASELINE_G5,
        "product_contract": PRODUCT_CONTRACT,
    }
    return {
        "schema": SCHEMA,
        "release": RELEASE,
        "recorded_at": recorded_at,
        "complete": True,
        "qualified": True,
        "source": source,
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": archive_digest,
            "checksum_sidecar": checksum.name,
            "checksum_verified": True,
        },
        "candidate": {
            "native_source_commit": NATIVE_SOURCE_COMMIT,
            "native_engine_sha256": ENGINE_SHA256,
            "single_resident_process": True,
            "runtime_python": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
            "runtime_transformers": False,
            "host_rocm_userspace_required": False,
        },
        "inheritance": {
            "baseline_release": BASELINE_RELEASE,
            "checks": inherited_checks,
            "scope": "unchanged GPU math, providers, AOT images and portable userspace",
            "claim_limit": (
                "The .4 G1-G4 and two-host results are inherited baseline "
                "evidence, not exact .5 measurements or a second .5 host run."
            ),
        },
        "evidence": {
            name: file_component(path, f"patch-release-evidence/{path.name}")
            for name, path in evidence_paths.items()
        },
        "bundle_checks": bundle_checks,
        "soak_checks": soak_checks,
        "rollback_checks": rollback_checks,
        "static_release_checks": static_checks,
        "gates": decisions,
        "decision": {
            **decisions,
            "patch_release_promoted": True,
            "next_blocking_boundary": None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--product-result", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--soak", type=Path, required=True)
    parser.add_argument("--rollback", type=Path, required=True)
    parser.add_argument("--release-gates", type=Path, required=True)
    parser.add_argument("--recorded-at", default="2026-09-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = {
        name: getattr(args, name).expanduser().resolve()
        for name in ("product_result", "bundle", "soak", "rollback", "release_gates")
    }
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"input is missing: {name}: {path}")
    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"archive is missing: {archive}")
    sealed = seal_manifest(
        build_payload(paths=paths, archive=archive, recorded_at=args.recorded_at)
    )
    output = args.output.expanduser().resolve()
    digest = atomic_json(output, sealed)
    output.with_name(output.name + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"qualified": True, "output": str(output), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
