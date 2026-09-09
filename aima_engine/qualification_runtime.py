"""Bind native qualification to the frozen portable userspace and loader."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from aima_engine.vl_reference import canonical_json_sha256


BASELINE = Path(__file__).resolve().parents[1] / (
    "benchmarks/results/native-portable-manifest-v1.5.1-native-vl.5.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_runtime_path(path: str) -> bool:
    return path.startswith(("lib/", "amdgcn/", "share/hip/", "share/certs/")) or path == "bin/aima-engine"


def runtime_inventory(manifest: dict) -> dict:
    files = manifest.get("files", [])
    paths = [item["path"] for item in files]
    if manifest.get("complete") is not True or len(paths) != len(set(paths)):
        raise ValueError("portable runtime manifest is incomplete or duplicated")
    return {item["path"]: item for item in files if is_runtime_path(item["path"])}


def expected_binding(engine_sha256: str, baseline: Path = BASELINE) -> dict:
    inventory = runtime_inventory(json.loads(baseline.read_text()))
    return {
        "schema": "aima-amd395-qwen36/pinned-qualification-runtime/v1",
        "engine_sha256": engine_sha256,
        "baseline_manifest_sha256": sha256(baseline),
        "runtime_inventory_sha256": canonical_json_sha256(inventory),
        "runtime_entry_count": len(inventory),
        "launcher_sha256": inventory["bin/aima-engine"]["sha256"],
        "execution": "static-launcher/bundled-loader/--inhibit-cache/--library-path",
        "verifier_sha256": sha256(Path(__file__)),
    }


def require_runtime_binding(binding: dict | None, engine_sha256: str,
                            baseline: Path = BASELINE) -> None:
    if binding != expected_binding(engine_sha256, baseline):
        raise ValueError("qualification does not bind the exact pinned portable runtime")


def bind_runtime(engine: Path, runtime_root: Path | None,
                 baseline: Path = BASELINE) -> tuple[Path, dict | None]:
    if runtime_root is None:
        return engine, None
    root = runtime_root.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    inventory = runtime_inventory(manifest)
    frozen = runtime_inventory(json.loads(baseline.read_text()))
    if not frozen or inventory != frozen:
        raise ValueError("qualification runtime differs from frozen portable userspace")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if (path.is_file() or path.is_symlink()) and is_runtime_path(path.relative_to(root).as_posix())}
    if actual != set(inventory):
        raise ValueError("qualification runtime has unlisted or missing dependencies")
    # Verify the entire capsule, not only the inventory's claimed checksums.
    for item in manifest["files"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("unsafe portable runtime member")
        path = root / relative
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("portable runtime member escapes capsule")
        if item["type"] == "symlink":
            valid = path.is_symlink() and path.readlink().as_posix() == item["target"]
        else:
            valid = (item["type"] == "file" and not path.is_symlink() and path.is_file()
                     and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"])
        if not valid:
            raise ValueError(f"portable runtime member differs: {relative}")
    payload = root / "libexec/aima-engine.real"
    if payload.is_symlink() or not payload.is_file() or sha256(payload) != sha256(engine):
        raise ValueError("portable runtime capsule contains a different native engine")
    launcher = root / "bin/aima-engine"
    if not os.access(launcher, os.X_OK):
        raise ValueError("portable runtime launcher is not executable")
    return launcher, expected_binding(sha256(engine), baseline)
