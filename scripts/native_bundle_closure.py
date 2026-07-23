#!/usr/bin/env python3
"""Audit the relocatable native bundle's complete x86-64 ELF closure."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any


def _readelf(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["readelf", *args, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _is_x86_64_elf(path: Path) -> bool:
    header = _readelf(path, "-h")
    return "Machine:" in header and "Advanced Micro Devices X86-64" in header


def _dynamic_fields(path: Path) -> tuple[list[str], str | None, list[str]]:
    dynamic = _readelf(path, "-d")
    needed = re.findall(r"Shared library: \[([^]]+)\]", dynamic)
    sonames = re.findall(r"Library soname: \[([^]]+)\]", dynamic)
    runpaths = re.findall(r"Library runpath: \[([^]]+)\]", dynamic)
    if len(sonames) > 1:
        raise RuntimeError(f"multiple SONAMEs in {path}: {sonames}")
    return needed, sonames[0] if sonames else None, runpaths


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def audit_bundle(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    launcher = bundle / "bin/aima-engine"
    engine = bundle / "libexec/aima-engine.real"
    library_dir = bundle / "lib"
    for required in (launcher, engine, library_dir):
        if not required.exists():
            raise RuntimeError(f"native bundle entry is missing: {required}")

    launcher_program_headers = _readelf(launcher, "-l")
    launcher_needed, _, _ = _dynamic_fields(launcher)
    if "INTERP" in launcher_program_headers or launcher_needed:
        raise RuntimeError("portable launcher must be a fully static ELF")

    objects = [engine]
    objects.extend(
        path
        for path in sorted(library_dir.iterdir())
        if path.is_file() and not path.is_symlink() and _is_x86_64_elf(path)
    )
    providers: dict[str, str] = {}
    object_contracts: list[dict[str, Any]] = []
    for path in objects:
        needed, soname, runpaths = _dynamic_fields(path)
        relative = path.relative_to(bundle).as_posix()
        providers[path.name] = relative
        if soname is not None:
            providers[soname] = relative
        object_contracts.append(
            {
                "path": relative,
                "soname": soname,
                "needed": needed,
                "runpaths": runpaths,
            }
        )

    unresolved: dict[str, list[str]] = {}
    non_relocatable_runpaths: dict[str, list[str]] = {}
    for item in object_contracts:
        missing = [name for name in item["needed"] if name not in providers]
        if missing:
            unresolved[item["path"]] = missing
        invalid = [
            entry
            for runpath in item["runpaths"]
            for entry in runpath.split(":")
            if entry and not entry.startswith("$ORIGIN")
        ]
        if invalid:
            non_relocatable_runpaths[item["path"]] = invalid
    if unresolved:
        raise RuntimeError(
            "native bundle has unresolved userspace ELF dependencies: "
            + json.dumps(unresolved, sort_keys=True)
        )
    if non_relocatable_runpaths:
        raise RuntimeError(
            "native bundle contains non-relocatable ELF RUNPATH entries: "
            + json.dumps(non_relocatable_runpaths, sort_keys=True)
        )

    glibc_versions: set[str] = set()
    for path in objects:
        glibc_versions.update(
            re.findall(r"GLIBC_([0-9]+(?:\.[0-9]+)+)", _readelf(path, "--version-info"))
        )
    maximum_glibc_abi = (
        max(glibc_versions, key=_version_key) if glibc_versions else None
    )

    return {
        "schema": "aima-amd395-qwen36/native-bundle-elf-closure/v1",
        "complete": True,
        "launcher_static": True,
        "bundled_dynamic_loader": "lib/ld-linux-x86-64.so.2",
        "engine": "libexec/aima-engine.real",
        "library_search": "bundled loader --inhibit-cache --library-path BUNDLE/lib",
        "x86_64_dynamic_object_count": len(object_contracts),
        "provided_soname_count": len(providers),
        "unresolved_userspace_dependencies": {},
        "non_relocatable_runpaths": {},
        "host_userspace_dependencies": [],
        "maximum_bundled_glibc_abi": maximum_glibc_abi,
        "objects": object_contracts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_bundle(args.bundle)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
