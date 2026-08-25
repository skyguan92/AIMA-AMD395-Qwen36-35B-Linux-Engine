#!/usr/bin/env python3
"""Generate the final G5 release qualification from all live release gates."""

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


SCHEMA = "aima-amd395-qwen36/native-vl-g5-release-qualification/v1"
RELEASE = "1.5.1-native-vl.2"
RELEASE_TAG = "v1.5.1-native-vl.2"
NATIVE_SOURCE_COMMIT = "bd012874027defa528279a357609b713e9069df4"
ENGINE_SHA256 = (
    "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
)
SHA256 = re.compile(r"[0-9a-f]{64}")
BASELINE_ARCHIVE_SHA256 = (
    "4e38f90fce3feb7bccf1965d87a3ec2bebddc439ce62e75fe1bc797c6ce1a5bc"
)
DEFAULTS = {
    "product_result": (
        ROOT
        / "benchmarks/results/native-portable-product-v1.5.1-native-vl.2.json"
    ),
    "primary_bundle": (
        ROOT
        / "benchmarks/runs/native-portable-bundle-20260824-bd01287-final/"
        "bundle.json"
    ),
    "second_bundle": (
        ROOT
        / "benchmarks/runs/native-portable-baiying-20260824-bd01287-final/"
        "bundle.json"
    ),
    "soak": (
        ROOT
        / "benchmarks/runs/native-vl-resident-soak-20260824-bd01287-final/"
        "soak.json"
    ),
    "rollback": (
        ROOT
        / "benchmarks/runs/native-vl-rollback-20260824-bd01287-final/"
        "rollback.json"
    ),
    "release_gates": (
        ROOT
        / "benchmarks/runs/native-vl-release-gates-20260824-bd01287-final/"
        "release-gates.json"
    ),
    "product_contract": (
        ROOT / "native/product-contract-v1.5.1-native-vl.2.json"
    ),
}
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/"
    "native-vl-g5-release-v1.5.1-native-vl.2.json"
)
SCHEMAS = {
    "product_result": (
        "aima-amd395-qwen36/native-vl-product-qualification/v1"
    ),
    "primary_bundle": (
        "aima-amd395-qwen36/native-portable-bundle-qualification/v1"
    ),
    "second_bundle": (
        "aima-amd395-qwen36/native-portable-bundle-qualification/v1"
    ),
    "soak": "aima-amd395-qwen36/native-vl-resident-soak/v1",
    "rollback": "aima-amd395-qwen36/native-vl-rollback/v1",
    "release_gates": "aima-amd395-qwen36/native-vl-release-gates/v1",
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


def require_sealed(
    name: str, path: Path, schema: str
) -> dict[str, Any]:
    payload = load_object(path)
    if (
        payload.get("schema") != schema
        or payload.get("release") != RELEASE
        or payload.get("complete") is not True
        or payload.get("qualified") is not True
    ):
        raise RuntimeError(f"{name} is incomplete or has the wrong identity")
    integrity_errors = verify_manifest_integrity(payload)
    if integrity_errors:
        raise RuntimeError(f"{name} integrity failed: {integrity_errors}")
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{sha256(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"{name} checksum sidecar differs")
    return payload


def verify_raw_record(
    owner: Path, recorded_path: str, expected_sha256: str
) -> None:
    if not recorded_path.startswith("raw/"):
        return
    root = owner.parent.resolve()
    path = (root / recorded_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"raw evidence escapes its qualification directory: {recorded_path}"
        ) from error
    if not path.is_file() or sha256(path) != expected_sha256:
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
    payload: Mapping[str, Any], *, host_role: str, archive_sha256: str
) -> dict[str, bool]:
    doctor = payload.get("doctor", {})
    vl_smoke = payload.get("vl_smoke", {})
    manifest = payload.get("manifest", {})
    return {
        "host_role_exact": payload.get("host_role") == host_role,
        "host_fingerprint_present": SHA256.fullmatch(
            str(doctor.get("host", {}).get("fingerprint_sha256", ""))
        )
        is not None,
        "archive_exact": payload.get("archive", {}).get("sha256")
        == archive_sha256,
        "source_exact": (
            payload.get("source", {}).get("native_source_commit")
            == NATIVE_SOURCE_COMMIT
        ),
        "manifest_native_vl": manifest.get("native_vl", {}).get("enabled")
        is True,
        "isolated_environment": payload.get("isolated_environment", {}).get(
            "keys"
        )
        == ["HOME", "LANG", "PATH"],
        "doctor_qualified": (
            doctor.get("qualified") is True
            and doctor.get("bundle_detected_complete") is True
            and doctor.get("host", {}).get("role") == host_role
            and doctor.get("host", {}).get("gpu", {}).get("architecture")
            == "gfx1151"
            and doctor.get("host", {}).get("gtt_bytes", 0)
            >= 96 * 1024 * 1024 * 1024
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
            and vl_smoke.get("checks", {}).get("five_requests_qualified")
            is True
            and vl_smoke.get("checks", {}).get("single_model_load") is True
        ),
        "public_hygiene": (
            payload.get("public_hygiene", {}).get("passed") is True
            and payload.get("public_hygiene", {}).get("finding_count") == 0
        ),
    }


def build_payload(
    *,
    paths: Mapping[str, Path],
    archive: Path,
    recorded_at: str,
) -> dict[str, Any]:
    payloads = {
        name: require_sealed(name, paths[name], schema)
        for name, schema in SCHEMAS.items()
    }
    for name in (
        "primary_bundle",
        "second_bundle",
        "soak",
        "rollback",
        "release_gates",
    ):
        verify_raw_closure(paths[name], payloads[name])

    product = payloads["product_result"]
    contract = load_object(paths["product_contract"])
    if (
        contract.get("schema")
        != "aima-amd395-qwen36/native-vl-product-contract/v1"
        or contract.get("release") != RELEASE
        or contract.get("release_tag") != RELEASE_TAG
        or contract.get("candidate", {}).get("native_source_commit")
        != NATIVE_SOURCE_COMMIT
        or contract.get("candidate", {}).get("native_engine_sha256")
        != ENGINE_SHA256
        or contract.get("target", {}).get("maximum_resident_memory_bytes")
        != 96 * 1024 * 1024 * 1024
    ):
        raise RuntimeError("native VL product contract identity differs")
    source = product.get("components", {}).get("source", {})
    if (
        source.get("release_tag") != RELEASE_TAG
        or source.get("native_source_commit") != NATIVE_SOURCE_COMMIT
        or source.get("native_source_dirty") is not False
        or product.get("components", {}).get("native_engine", {}).get(
            "sha256"
        )
        != ENGINE_SHA256
        or product.get("decision", {}).get("package_input_qualified")
        is not True
        or not all(product.get("gates", {}).values())
    ):
        raise RuntimeError("package-input product qualification identity differs")
    release_commit = source.get("release_commit")
    if (
        not isinstance(release_commit, str)
        or git_output(
            "rev-parse", "--verify", f"refs/tags/{RELEASE_TAG}^{{commit}}"
        )
        != release_commit
        or git_output("rev-parse", "HEAD") != release_commit
        or git_output("status", "--porcelain", "--untracked-files=normal")
    ):
        raise RuntimeError("final G5 generation requires the clean tagged checkout")

    archive_digest = sha256(archive)
    checksum = archive.with_name(archive.name + ".sha256")
    if (
        not checksum.is_file()
        or checksum.read_text(encoding="utf-8").split()[0] != archive_digest
    ):
        raise RuntimeError("candidate archive checksum differs")
    primary_checks = exact_bundle_checks(
        payloads["primary_bundle"],
        host_role="primary_amd395",
        archive_sha256=archive_digest,
    )
    second_checks = exact_bundle_checks(
        payloads["second_bundle"],
        host_role="second_amd395",
        archive_sha256=archive_digest,
    )
    primary_host = payloads["primary_bundle"]["doctor"]["host"]
    second_host = payloads["second_bundle"]["doctor"]["host"]
    host_checks = {
        "primary_bundle_all": all(primary_checks.values()),
        "second_bundle_all": all(second_checks.values()),
        "roles_distinct": (
            primary_host.get("role") == "primary_amd395"
            and second_host.get("role") == "second_amd395"
        ),
        "live_host_facts_distinct": (
            primary_host.get("fingerprint_sha256")
            != second_host.get("fingerprint_sha256")
        ),
    }
    soak = payloads["soak"]
    soak_checks = {
        "archive_exact": soak.get("archive", {}).get("sha256")
        == archive_digest,
        "primary_host": soak.get("host_role") == "primary_amd395",
        "one_hour": soak.get("measurement", {}).get("elapsed_seconds", 0)
        >= 3600,
        "minimum_requests": soak.get("measurement", {}).get(
            "request_count", 0
        )
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
        == BASELINE_ARCHIVE_SHA256,
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
        "make_check": release_gates.get("decision", {}).get(
            "make_check_passed"
        )
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
        "g1_full_vl_functional_parity": product["gates"][
            "g1_full_vl_functional_parity"
        ],
        "g2_vl_correctness_parity": product["gates"][
            "g2_vl_correctness_parity"
        ],
        "g3_text_product_no_regression": product["gates"][
            "g3_text_product_no_regression"
        ],
        "g4_native_vl_performance": product["gates"][
            "g4_native_vl_performance"
        ],
        "portable_archive": all(primary_checks.values()),
        "second_amd395": all(second_checks.values())
        and all(host_checks.values()),
        "resident_mixed_workload_soak": all(soak_checks.values()),
        "rollback": all(rollback_checks.values()),
        "repository_security_evidence_gates": all(static_checks.values()),
    }
    if not all(decisions.values()):
        raise RuntimeError(f"not every G1-G5 release gate passed: {decisions}")

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
            "ready_includes_language_and_vision": True,
            "runtime_python": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
            "runtime_transformers": False,
            "host_rocm_userspace_required": False,
        },
        "evidence": {
            name: file_component(path, logical)
            for name, path, logical in (
                (
                    "package_input",
                    paths["product_result"],
                    "benchmarks/results/"
                    "native-portable-product-v1.5.1-native-vl.2.json",
                ),
                (
                    "primary_bundle",
                    paths["primary_bundle"],
                    "benchmarks/runs/native-portable-bundle-"
                    "20260824-bd01287-final/bundle.json",
                ),
                (
                    "second_bundle",
                    paths["second_bundle"],
                    "benchmarks/runs/native-portable-baiying-"
                    "20260824-bd01287-final/bundle.json",
                ),
                (
                    "resident_soak",
                    paths["soak"],
                    "benchmarks/runs/native-vl-resident-soak-"
                    "20260824-bd01287-final/soak.json",
                ),
                (
                    "rollback",
                    paths["rollback"],
                    "benchmarks/runs/native-vl-rollback-"
                    "20260824-bd01287-final/rollback.json",
                ),
                (
                    "release_gates",
                    paths["release_gates"],
                    "benchmarks/runs/native-vl-release-gates-"
                    "20260824-bd01287-final/release-gates.json",
                ),
                (
                    "product_contract",
                    paths["product_contract"],
                    "native/product-contract-v1.5.1-native-vl.2.json",
                ),
            )
        },
        "host_checks": {
            "primary": primary_checks,
            "second": second_checks,
            "cross_host": host_checks,
            "primary_facts": primary_host,
            "second_facts": second_host,
        },
        "soak_checks": soak_checks,
        "rollback_checks": rollback_checks,
        "static_release_checks": static_checks,
        "gates": decisions,
        "decision": {
            **decisions,
            "g5_native_release_product": True,
            "release_promoted": True,
            "next_blocking_boundary": None,
        },
    }


def verify_exact(path: Path, expected: Mapping[str, Any]) -> None:
    if load_object(path) != expected:
        raise SystemExit(f"G5 release qualification is stale: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{sha256(path)}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise SystemExit(f"G5 release qualification sidecar is stale: {sidecar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        parser.add_argument(
            "--" + name.replace("_", "-"), type=Path, default=default
        )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--recorded-at", default="2026-08-24")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    paths = {
        name: getattr(args, name).expanduser().resolve() for name in DEFAULTS
    }
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"{name} is missing: {path}")
    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"archive is missing: {archive}")
    sealed = seal_manifest(
        build_payload(
            paths=paths,
            archive=archive,
            recorded_at=args.recorded_at,
        )
    )
    output = args.output.expanduser().resolve()
    if args.check:
        verify_exact(output, sealed)
        print(f"native VL G5 release qualification: PASS ({output})")
        return 0
    digest = atomic_json(output, sealed)
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": True,
                "output": str(output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
