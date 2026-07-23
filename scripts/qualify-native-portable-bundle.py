#!/usr/bin/env python3
"""Qualify an extracted release archive with no host userspace environment."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any

from native_bundle_closure import audit_bundle


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def publicize(value: Any, model_dir: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(ROOT), "${AIMA_REPO_ROOT}").replace(
            str(model_dir), "${AIMA_MODEL_DIR}"
        )
    if isinstance(value, list):
        return [publicize(item, model_dir) for item in value]
    if isinstance(value, dict):
        return {
            key: publicize(item, model_dir) for key, item in value.items()
        }
    return value


def verify_manifest(bundle: Path) -> dict[str, Any]:
    manifest = load_json(bundle / "manifest.json")
    if manifest.get("complete") is not True:
        raise RuntimeError("bundle manifest is not complete")
    checked_files = 0
    checked_symlinks = 0
    for entry in manifest["files"]:
        path = bundle / entry["path"]
        if entry["type"] == "symlink":
            if not path.is_symlink() or path.readlink().as_posix() != entry[
                "target"
            ]:
                raise RuntimeError(f"bundle symlink mismatch: {entry['path']}")
            checked_symlinks += 1
            continue
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["bytes"])
            or sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"bundle file mismatch: {entry['path']}")
        checked_files += 1
    return {
        "schema": manifest["schema"],
        "complete": True,
        "checked_files": checked_files,
        "checked_symlinks": checked_symlinks,
        "payload_bytes_excluding_manifest": manifest[
            "payload_bytes_excluding_manifest"
        ],
        "attention_providers": manifest["attention_providers"],
    }


def run_smoke(
    *,
    launcher: Path,
    model_dir: Path,
    output_dir: Path,
    context: int,
    expected_primary: str,
    expected_secondary: str,
    expected_secondary_layers: list[int],
    environment: dict[str, str],
) -> dict[str, Any]:
    report = output_dir / "raw" / f"q{context}-output1.json"
    load_report = report.with_name(report.stem + ".load.json")
    command = [
        str(launcher),
        "resident-session-probe",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        str(context),
        "--uniform-input-token-id",
        "1",
        "--max-new-tokens",
        "1",
        "--requests",
        "1",
        "--disable-prefix-cache",
        "--report",
        str(load_report),
    ]
    print(
        json.dumps(
            {"event": "bundle_smoke_start", "context_tokens": context},
            sort_keys=True,
        ),
        flush=True,
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        env=environment,
        check=False,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("bundle smoke emitted non-object JSON")
    load = payload["load"]
    request = payload["requests"][0]
    qualified = bool(
        completed.returncode == 0
        and payload["complete"] is True
        and payload["runtime_python"] is False
        and payload["runtime_torch"] is False
        and payload["runtime_vllm"] is False
        and payload["runtime_triton"] is False
        and payload["model_loads"] == 1
        and request["prompt_tokens"] == context
        and request["completion_tokens"] == 1
        and request["first_token_certified"] is True
        and request["all_decode_tokens_certified"] is True
        and load["fmha_provider_backend"] == expected_primary
        and load["secondary_fmha_provider_backend"] == expected_secondary
        and load["secondary_fmha_layers"] == expected_secondary_layers
    )
    payload["bundle_qualification"] = {
        "command": command,
        "environment_keys": sorted(environment),
        "load_report": str(load_report),
        "load_report_sha256": sha256(load_report),
        "qualified": qualified,
    }
    payload = publicize(payload, model_dir)
    atomic_json(report, payload)
    if not qualified:
        raise RuntimeError(f"bundle smoke failed: q{context}")
    print(
        json.dumps(
            {
                "event": "bundle_smoke_complete",
                "context_tokens": context,
                "command_to_ready_wall_ms": load[
                    "command_to_ready_wall_ms"
                ],
                "prefill_tokens_per_second": request[
                    "prefill_tokens_per_second"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "context_tokens": context,
        "output_tokens": 1,
        "primary_provider": load["fmha_provider_backend"],
        "secondary_provider": load["secondary_fmha_provider_backend"],
        "secondary_layers": load["secondary_fmha_layers"],
        "command_to_ready_wall_ms": load["command_to_ready_wall_ms"],
        "prefill_tokens_per_second": request["prefill_tokens_per_second"],
        "output_token_ids_sha256": request["output_token_ids_sha256"],
        "report": f"raw/{report.name}",
        "report_sha256": sha256(report),
        "qualified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--product-result",
        type=Path,
        default=Path(
            "benchmarks/results/native-portable-product-v1.3.0.json"
        ),
    )
    cli = parser.parse_args()

    archive = cli.archive.expanduser().resolve()
    model_dir = cli.model_dir.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    product_result = load_json(cli.product_result.resolve())
    if not archive.is_file():
        raise SystemExit(f"archive is missing: {archive}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")
    archive_sha256 = sha256(archive)
    checksum_path = archive.with_name(archive.name + ".sha256")
    checksum_verified = False
    if checksum_path.is_file():
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        checksum_verified = expected == archive_sha256
    if not checksum_verified:
        raise SystemExit("archive checksum sidecar is missing or mismatched")

    with tempfile.TemporaryDirectory(prefix="aima-bundle-qualification-") as tmp:
        extraction = Path(tmp) / "extract"
        extraction.mkdir()
        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(extraction)],
            check=True,
        )
        roots = [path for path in extraction.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("archive must contain exactly one root directory")
        bundle = roots[0]
        if stat.S_IMODE(bundle.stat().st_mode) != 0o755:
            raise RuntimeError("archive root directory must have mode 0755")
        launcher = bundle / "bin/aima-engine"
        engine = bundle / "libexec/aima-engine.real"
        if sha256(engine) != product_result["components"]["native_engine"][
            "sha256"
        ]:
            raise RuntimeError("archive engine does not match product result")
        if sha256(launcher) != product_result["components"]["static_launcher"][
            "sha256"
        ]:
            raise RuntimeError("archive launcher does not match product result")
        manifest = verify_manifest(bundle)
        closure = audit_bundle(bundle)
        isolated_home = Path(tmp) / "home"
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
        help_text = subprocess.run(
            [str(launcher), "--help"],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        ).stdout
        if (
            version != "aima-engine-native 1.3.0-native"
            or "131072" not in help_text
            or "input261120/output1024" not in help_text
        ):
            raise RuntimeError("portable launcher public CLI is incomplete")
        smokes = [
            run_smoke(
                launcher=launcher,
                model_dir=model_dir,
                output_dir=output_dir,
                context=context,
                expected_primary=primary,
                expected_secondary=secondary,
                expected_secondary_layers=layers,
                environment=environment,
            )
            for context, primary, secondary, layers in (
                (1024, "AOTriton 0.11.1", "", []),
                (
                    16384,
                    "packed-GQA/CK-Tile hybrid",
                    "CK-Tile",
                    [39],
                ),
                (65536, "CK-Tile", "AOTriton 0.11.1", [39]),
            )
        ]

    result = {
        "schema": "aima-amd395-qwen36/native-portable-bundle-qualification/v1",
        "release": "1.3.0",
        "complete": True,
        "qualified": True,
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
            "checksum_sidecar": checksum_path.name,
            "checksum_verified": checksum_verified,
            "root_mode": "0755",
        },
        "manifest": manifest,
        "elf_closure": {
            "complete": closure["complete"],
            "launcher_static": closure["launcher_static"],
            "x86_64_dynamic_object_count": closure[
                "x86_64_dynamic_object_count"
            ],
            "provided_soname_count": closure["provided_soname_count"],
            "unresolved_userspace_dependencies": closure[
                "unresolved_userspace_dependencies"
            ],
            "non_relocatable_runpaths": closure[
                "non_relocatable_runpaths"
            ],
            "host_userspace_dependencies": closure[
                "host_userspace_dependencies"
            ],
            "maximum_bundled_glibc_abi": closure[
                "maximum_bundled_glibc_abi"
            ],
        },
        "isolated_environment": {
            "keys": ["HOME", "LANG", "PATH"],
            "host_ld_library_path": False,
            "host_rocm_path": False,
            "host_python_path": False,
            "version": version,
        },
        "provider_smokes": smokes,
        "host_requirements": [
            "Linux x86_64 kernel ABI",
            "amdgpu kernel driver with KFD and render nodes",
            "AMD gfx1151 GPU",
        ],
    }
    atomic_json(output_dir / "bundle.json", result)
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": True,
                "archive_sha256": archive_sha256,
                "output": str(output_dir / "bundle.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
