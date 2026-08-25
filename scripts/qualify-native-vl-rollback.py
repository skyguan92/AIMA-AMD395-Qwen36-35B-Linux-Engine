#!/usr/bin/env python3
"""Prove rollback from the native VL candidate to the exact v1.5.1 archive."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.public_hygiene import scan_bytes  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
    verify_manifest_integrity,
)


SCHEMA = "aima-amd395-qwen36/native-vl-rollback/v1"
SOAK_SCHEMA = "aima-amd395-qwen36/native-vl-resident-soak/v1"
BASELINE_RELEASE = "1.5.1"
BASELINE_TAG = "v1.5.1"
BASELINE_RELEASE_COMMIT = "6f3e669ac897eaabfeceb7f193a5e02708a4d95e"
BASELINE_NATIVE_COMMIT = "65c198415709dad6d046c247acab3dc9df2a95a0"
BASELINE_ARCHIVE_SHA256 = (
    "4e38f90fce3feb7bccf1965d87a3ec2bebddc439ce62e75fe1bc797c6ce1a5bc"
)
BASELINE_ENGINE_SHA256 = (
    "a9f18771175757af080c8a1d8d7e3fb3906c9aa41b43a496686103b626f80262"
)
BASELINE_LAUNCHER_SHA256 = (
    "ac43fb95a8bad8f9fb4e0f4eac9cadc4fb92f22189f4f35ce21a81f1d56fcf98"
)
EXACT_OUTPUT_SHA256 = (
    "aa910692fd03ed4a8e89c04497751e3a28eee36c6148237f7e97c74a6dd68201"
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


def publicize(value: Any, replacements: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, str):
        result = value
        for private, logical in replacements:
            result = result.replace(private, logical)
        return result
    if isinstance(value, list):
        return [publicize(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: publicize(item, replacements) for key, item in value.items()
        }
    return value


def verify_recursive_manifest(bundle: Path) -> dict[str, Any]:
    manifest = load_object(bundle / "manifest.json")
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            raise RuntimeError("baseline manifest contains a non-object entry")
        path = bundle / str(item.get("path", ""))
        if item.get("type") == "symlink":
            if not path.is_symlink() or path.readlink().as_posix() != item.get(
                "target"
            ):
                raise RuntimeError(f"baseline manifest symlink differs: {path}")
        elif (
            item.get("type") != "file"
            or not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or sha256(path) != item.get("sha256")
        ):
            raise RuntimeError(f"baseline manifest file differs: {path}")
    if manifest.get("complete") is not True:
        raise RuntimeError("baseline manifest is incomplete")
    return manifest


def public_hygiene_passes(root: Path) -> bool:
    return not any(
        scan_bytes(path.relative_to(root).as_posix(), path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    )


def require_gpu_idle() -> None:
    occupied = subprocess.run(
        ["fuser", "/dev/kfd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if occupied.returncode != 0:
        return
    subprocess.run(["fuser", "-v", "/dev/kfd"], check=False)
    raise SystemExit(75)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-soak", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host-role", default="primary_amd395")
    args = parser.parse_args()

    candidate_soak_path = args.candidate_soak.expanduser().resolve()
    baseline_archive = args.baseline_archive.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise SystemExit(f"output directory must be empty: {output_dir}")
    if not candidate_soak_path.is_file():
        raise SystemExit(f"candidate soak is missing: {candidate_soak_path}")
    if not baseline_archive.is_file():
        raise SystemExit(f"baseline archive is missing: {baseline_archive}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")

    candidate_soak = load_object(candidate_soak_path)
    if (
        candidate_soak.get("schema") != SOAK_SCHEMA
        or candidate_soak.get("complete") is not True
        or candidate_soak.get("qualified") is not True
        or candidate_soak.get("host_role") != args.host_role
        or candidate_soak.get("decision", {}).get(
            "one_hour_resident_mixed_workload_passed"
        )
        is not True
        or candidate_soak.get("checks", {}).get("clean_shutdown") is not True
        or verify_manifest_integrity(candidate_soak)
    ):
        raise SystemExit("candidate soak is incomplete, unsealed or on another host")
    soak_sidecar = candidate_soak_path.with_name(
        candidate_soak_path.name + ".sha256"
    )
    if (
        not soak_sidecar.is_file()
        or soak_sidecar.read_text(encoding="utf-8")
        != f"{sha256(candidate_soak_path)}  {candidate_soak_path.name}\n"
    ):
        raise SystemExit("candidate soak checksum sidecar differs")

    baseline_digest = sha256(baseline_archive)
    baseline_checksum = baseline_archive.with_name(
        baseline_archive.name + ".sha256"
    )
    if (
        baseline_digest != BASELINE_ARCHIVE_SHA256
        or not baseline_checksum.is_file()
        or baseline_checksum.read_text(encoding="utf-8").split()[0]
        != baseline_digest
    ):
        raise SystemExit("baseline v1.5.1 archive identity differs")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    doctor_path = raw_dir / "doctor.json"
    probe_path = raw_dir / "q8192-exact128.json"
    load_path = raw_dir / "q8192-exact128.load.json"
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with tempfile.TemporaryDirectory(prefix="aima-native-vl-rollback-") as tmp:
        temporary_root = Path(tmp)
        extraction = temporary_root / "extract"
        extraction.mkdir()
        subprocess.run(
            [
                "tar",
                "--zstd",
                "-xf",
                str(baseline_archive),
                "-C",
                str(extraction),
            ],
            check=True,
        )
        roots = [path for path in extraction.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("baseline archive must contain one root directory")
        bundle = roots[0]
        manifest = verify_recursive_manifest(bundle)
        launcher = bundle / "bin/aima-engine"
        engine = bundle / "libexec/aima-engine.real"
        if (
            sha256(engine) != BASELINE_ENGINE_SHA256
            or sha256(launcher) != BASELINE_LAUNCHER_SHA256
            or manifest.get("release") != BASELINE_RELEASE
            or manifest.get("source", {}).get("release_tag") != BASELINE_TAG
            or manifest.get("source", {}).get("commit")
            != BASELINE_RELEASE_COMMIT
            or manifest.get("source", {}).get("native_commit")
            != BASELINE_NATIVE_COMMIT
            or manifest.get("source", {}).get("dirty") is not False
        ):
            raise RuntimeError("extracted v1.5.1 rollback identity differs")

        isolated_home = temporary_root / "home"
        isolated_home.mkdir()
        environment = {
            "HOME": str(isolated_home),
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
        }
        version = subprocess.run(
            [str(launcher), "--version"],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        ).stdout.strip()
        build_info = subprocess.run(
            [str(launcher), "--build-info"],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        build_payload = json.loads(build_info.stdout)
        doctor_command = [
            str(launcher),
            "doctor",
            "--model-dir",
            str(model_dir),
            "--json",
        ]
        doctor = subprocess.run(
            doctor_command,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        doctor_payload = json.loads(doctor.stdout)
        required_checks = [
            item
            for item in doctor_payload.get("checks", [])
            if isinstance(item, dict) and item.get("required") is True
        ]
        expected_tokens = ",".join(["1000"] * 128)
        probe_command = [
            str(launcher),
            "resident-session-probe",
            "--model-dir",
            str(model_dir),
            "--context-tokens",
            "8192",
            "--uniform-input-token-id",
            "1000",
            "--max-new-tokens",
            "128",
            "--requests",
            "1",
            "--disable-prefix-cache",
            "--expected-token-ids",
            expected_tokens,
            "--report",
            str(load_path),
        ]
        require_gpu_idle()
        probe = subprocess.run(
            probe_command,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        probe_payload = json.loads(probe.stdout)
        request = probe_payload.get("requests", [{}])[0]
        checks = {
            "candidate_soak_completed_before_rollback": str(
                candidate_soak.get("recorded_at", "")
            )
            <= started_at,
            "candidate_shutdown_clean": candidate_soak.get("checks", {}).get(
                "clean_shutdown"
            )
            is True,
            "baseline_archive_exact": True,
            "recursive_manifest_exact": True,
            "isolated_environment": sorted(environment)
            == ["HOME", "LANG", "PATH"],
            "baseline_version_exact": version
            == "aima-engine-native 1.5.1-native",
            "baseline_build_info_exact": build_payload
            == {
                "version": "1.5.1-native",
                "source_commit": BASELINE_NATIVE_COMMIT,
            },
            "doctor_qualified": (
                doctor.returncode == 0
                and doctor_payload.get("complete") is True
                and doctor_payload.get("qualified") is True
                and len(required_checks) >= 13
                and all(item.get("passed") is True for item in required_checks)
            ),
            "probe_qualified": (
                probe.returncode == 0
                and probe_payload.get("complete") is True
                and probe_payload.get("model_loads") == 1
                and probe_payload.get("expected_tokens_match") is True
                and probe_payload.get("repeat_tokens_identical") is True
                and probe_payload.get("runtime_python") is False
                and probe_payload.get("runtime_torch") is False
                and probe_payload.get("runtime_vllm") is False
                and probe_payload.get("runtime_triton") is False
            ),
            "q8192_exact128_restored": (
                request.get("prompt_tokens") == 8192
                and request.get("completion_tokens") == 128
                and request.get("output_token_ids_sha256")
                == EXACT_OUTPUT_SHA256
                and request.get("first_token_certified") is True
                and request.get("all_decode_tokens_certified") is True
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"v1.5.1 rollback failed: {checks}")

        replacements = (
            (str(bundle), "${AIMA_BASELINE_BUNDLE_ROOT}"),
            (str(model_dir), "${AIMA_MODEL_DIR}"),
            (str(output_dir), "${AIMA_OUTPUT_DIR}"),
            (str(temporary_root), "${AIMA_ISOLATED_ROOT}"),
        )
        doctor_payload = publicize(doctor_payload, replacements)
        probe_payload = publicize(probe_payload, replacements)
        load_payload = publicize(load_object(load_path), replacements)
        doctor_command = publicize(doctor_command, replacements)
        probe_command = publicize(probe_command, replacements)
        atomic_json(doctor_path, doctor_payload)
        atomic_json(probe_path, probe_payload)
        atomic_json(load_path, load_payload)

    checks["public_hygiene"] = public_hygiene_passes(output_dir)
    if checks["public_hygiene"] is not True:
        raise RuntimeError("rollback output failed public hygiene")

    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "release": candidate_soak["release"],
            "recorded_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "started_at": started_at,
            "complete": True,
            "qualified": True,
            "host_role": args.host_role,
            "candidate": {
                "release": candidate_soak["release"],
                "archive": candidate_soak["archive"],
                "soak": file_component(
                    candidate_soak_path,
                    "benchmarks/runs/native-vl-resident-soak/soak.json",
                ),
                "clean_shutdown": True,
            },
            "rollback_target": {
                "release": BASELINE_RELEASE,
                "release_tag": BASELINE_TAG,
                "release_commit": BASELINE_RELEASE_COMMIT,
                "native_source_commit": BASELINE_NATIVE_COMMIT,
                "archive": {
                    "name": baseline_archive.name,
                    "bytes": baseline_archive.stat().st_size,
                    "sha256": baseline_digest,
                    "checksum_sidecar": baseline_checksum.name,
                    "checksum_verified": True,
                },
                "native_engine_sha256": BASELINE_ENGINE_SHA256,
                "static_launcher_sha256": BASELINE_LAUNCHER_SHA256,
            },
            "commands": {
                "doctor": doctor_command,
                "exact_q8192_output128": probe_command,
            },
            "checks": checks,
            "raw_artifacts": {
                name: file_component(path, f"raw/{path.name}")
                for name, path in (
                    ("doctor", doctor_path),
                    ("exact_q8192_output128", probe_path),
                    ("load_report", load_path),
                )
            },
            "decision": {
                "rollback_to_exact_v1_5_1_passed": True,
                "baseline_portable_doctor_passed": True,
                "baseline_exact_output_restored": True,
            },
        }
    )
    output = output_dir / "rollback.json"
    digest = atomic_json(output, payload)
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
