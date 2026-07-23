#!/usr/bin/env python3
"""Generate a deterministic inventory for a portable native product bundle."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from native_bundle_closure import audit_bundle


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dynamic_contract(binary: Path) -> tuple[list[str], str]:
    result = subprocess.run(
        ["readelf", "-d", str(binary)],
        capture_output=True,
        text=True,
        check=True,
    )
    needed = re.findall(r"Shared library: \[([^]]+)\]", result.stdout)
    runpaths = re.findall(r"Library runpath: \[([^]]+)\]", result.stdout)
    if runpaths != ["$ORIGIN/../lib"]:
        raise RuntimeError(f"unexpected native executable RUNPATH: {runpaths}")
    if "libamdhip64.so.7" not in needed:
        raise RuntimeError("native executable is not linked to the qualified HIP ABI")
    return needed, runpaths[0]


def validate_local_markdown_links(bundle: Path) -> int:
    link_pattern = re.compile(r"\]\(([^)]+)\)")
    checked = 0
    missing: list[str] = []
    for document in sorted(bundle.rglob("*.md")):
        for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            checked += 1
            resolved = (document.parent / target).resolve()
            try:
                resolved.relative_to(bundle)
            except ValueError:
                missing.append(f"{document.relative_to(bundle)} -> {target}")
                continue
            if not resolved.exists():
                missing.append(f"{document.relative_to(bundle)} -> {target}")
    if missing:
        raise RuntimeError(f"bundle contains broken local Markdown links: {missing}")
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    launcher = bundle / "bin/aima-engine"
    binary = bundle / "libexec/aima-engine.real"
    if not launcher.is_file() or not binary.is_file():
        raise RuntimeError("native bundle launcher or engine payload is missing")
    required_product_files = (
        bundle / "lib/libaima-fmha-aotriton.so",
        bundle / "lib/libaima-fmha-ck.so",
        bundle / "lib/libaima-fmha-q16384-hybrid.so",
        bundle / "lib/libaotriton_v2.so.0.11.1",
        bundle
        / "lib/aotriton.images/amd-gfx11xx/flash/attn_fwd/"
        "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2",
    )
    missing = [str(path.relative_to(bundle)) for path in required_product_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"native product payload is incomplete: {missing}")

    needed, runpath = dynamic_contract(binary)
    elf_closure = audit_bundle(bundle)
    markdown_link_count = validate_local_markdown_links(bundle)
    files = []
    for path in sorted(bundle.rglob("*")):
        if path.name == "manifest.json":
            continue
        if path.is_symlink():
            files.append(
                {
                    "path": path.relative_to(bundle).as_posix(),
                    "type": "symlink",
                    "target": path.readlink().as_posix(),
                }
            )
            continue
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(bundle).as_posix(),
                "type": "file",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema": "aima-amd395-qwen36/native-portable-bundle/v3",
        "complete": True,
        "release": "1.2.0",
        "status": "portable_native_inference_runtime",
        "target": "Linux x86_64, amdgpu/KFD host driver, AMD gfx1151",
        "runtime_python": False,
        "runtime_torch": False,
        "runtime_vllm": False,
        "runtime_triton": False,
        "runtime_transformers": False,
        "native_tokenizer": True,
        "native_inference_engine": True,
        "model_weights_included": False,
        "attention_providers": {
            "1024": "lib/libaima-fmha-aotriton.so",
            "2048": "lib/libaima-fmha-aotriton.so",
            "4096": "lib/libaima-fmha-aotriton.so",
            "8192": "lib/libaima-fmha-ck.so",
            "16384": "lib/libaima-fmha-q16384-hybrid.so",
            "32768": "lib/libaima-fmha-ck.so",
            "long_context_primary": "lib/libaima-fmha-ck.so",
            "long_context_layer39": "lib/libaima-fmha-aotriton.so",
            "aotriton_runtime": "lib/libaotriton_v2.so.0.11.1",
            "selection": "automatic by admitted static context",
        },
        "binary": {
            "path": "libexec/aima-engine.real",
            "runpath": runpath,
            "needed": needed,
        },
        "launcher": {
            "path": "bin/aima-engine",
            "static": True,
            "shared_library_dependencies": [],
        },
        "elf_closure": elf_closure,
        "bundled_rocm_userspace": [
            "lib/libamdhip64.so.7",
            "lib/libhsa-runtime64.so.1",
            "lib/librocprofiler-register.so.0",
            "lib/libamd_comgr.so.3",
            "lib/libhsa-amd-aqlprofile64.so.1",
            "amdgcn/bitcode",
            "share/hip/version",
        ],
        "bundled_attention_userspace": [
            "lib/libaima-fmha-aotriton.so",
            "lib/libaima-fmha-ck.so",
            "lib/libaima-fmha-q16384-hybrid.so",
            "lib/libaotriton_v2.so.0.11.1",
            "one shape-selected gfx1151 AOTriton code object",
        ],
        "statically_linked_components": ["ICU 74.2"],
        "bundled_system_userspace": [
            "glibc dynamic loader and libc/libm",
            "libstdc++ and libgcc runtime",
            "libelf",
            "libdrm and libdrm_amdgpu",
            "libnuma",
            "zlib and zstd",
            "liblzma",
        ],
        "host_contract": [
            "Linux x86_64 kernel",
            "amdgpu kernel driver with KFD and render nodes",
            "AMD gfx1151 GPU",
        ],
        "documentation": {
            "local_markdown_links_checked": markdown_link_count,
            "broken_local_markdown_links": [],
        },
        "files": files,
        "payload_bytes_excluding_manifest": sum(
            item.get("bytes", 0) for item in files
        ),
    }
    (bundle / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
