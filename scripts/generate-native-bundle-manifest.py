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
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import verify_manifest_integrity
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
    parser.add_argument("--release", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--native-source-commit", required=True)
    parser.add_argument("--source-dirty", action="store_true")
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
        bundle / "lib/aima-vision-attention.hsaco",
        bundle
        / "lib/aotriton.images/amd-gfx11xx/flash/attn_fwd/"
        "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2",
    )
    missing = [str(path.relative_to(bundle)) for path in required_product_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"native product payload is incomplete: {missing}")

    qualification_path = bundle / "share/aima/qualification.json"
    if not qualification_path.is_file():
        raise RuntimeError("native product qualification is missing")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if not isinstance(qualification, dict):
        raise RuntimeError("native product qualification root is not an object")
    native_vl = (
        qualification.get("schema")
        == "aima-amd395-qwen36/native-vl-product-qualification/v1"
    )
    native_vl_record: dict[str, object] = {"enabled": False}
    if native_vl:
        integrity_errors = verify_manifest_integrity(qualification)
        source = qualification.get("components", {}).get("source", {})
        if (
            integrity_errors
            or qualification.get("complete") is not True
            or qualification.get("qualified") is not True
            or qualification.get("release") != args.release
            or not isinstance(source, dict)
            or source.get("release_tag") != args.release_tag
            or source.get("release_commit") != args.source_commit
            or source.get("native_source_commit")
            != args.native_source_commit
            or source.get("native_source_dirty") is not False
            or args.source_dirty
        ):
            raise RuntimeError(
                "native VL qualification integrity or source identity differs"
            )
        components = qualification.get("components", {})
        external = components.get("vision_attention_image", {})
        dense = components.get("embedded_dense_vision_attention", {})
        external_image = bundle / "lib/aima-vision-attention.hsaco"
        component_paths = {
            "native_engine": bundle / "libexec/aima-engine.real",
            "static_launcher": bundle / "bin/aima-engine",
            "aotriton_fmha_provider": (
                bundle / "lib/libaima-fmha-aotriton.so"
            ),
            "ck_fmha_provider": bundle / "lib/libaima-fmha-ck.so",
            "q16384_hybrid_fmha_provider": (
                bundle / "lib/libaima-fmha-q16384-hybrid.so"
            ),
            "aotriton_runtime": bundle / "lib/libaotriton_v2.so.0.11.1",
            "aotriton_gfx1151_image": (
                bundle
                / "lib/aotriton.images/amd-gfx11xx/flash/attn_fwd/"
                "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2"
            ),
            "vision_attention_image": external_image,
        }
        component_closure_exact = all(
            isinstance(components.get(name), dict)
            and components[name].get("bytes") == path.stat().st_size
            and components[name].get("sha256") == sha256_file(path)
            for name, path in component_paths.items()
        )
        if (
            not component_closure_exact
            or not isinstance(external, dict)
            or external.get("sha256") != sha256_file(external_image)
            or not isinstance(dense, dict)
            or dense.get("sha256")
            != "e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c"
            or dense.get("kernel_hash")
            != "2bb5125141eea1b811395f9833de3077de68893bfebbbf1950ca26832db6bb52"
        ):
            raise RuntimeError("native VL vision image closure is incomplete")
        product_contract = json.loads(
            (bundle / "share/aima/product-contract.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            not isinstance(product_contract, dict)
            or product_contract.get("schema")
            != "aima-amd395-qwen36/native-vl-product-contract/v1"
            or product_contract.get("release") != args.release
            or product_contract.get("release_tag") != args.release_tag
        ):
            raise RuntimeError("native VL product contract differs")
        native_vl_record = {
            "enabled": True,
            "single_resident_process": True,
            "ready_includes_language_and_vision": True,
            "language_tensor_count": 693,
            "visual_tensor_count": 333,
            "model_tensor_count": 1026,
            "general_vision_attention": {
                "path": "lib/aima-vision-attention.hsaco",
                "sha256": external["sha256"],
            },
            "dense_vision_attention": {
                "embedded_in": "libexec/aima-engine.real",
                "sha256": dense["sha256"],
                "kernel_hash": dense["kernel_hash"],
            },
        }

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
        "release": args.release,
        "source": {
            "release_tag": args.release_tag,
            "commit": args.source_commit,
            "native_commit": args.native_source_commit,
            "dirty": args.source_dirty,
        },
        "status": "portable_native_inference_runtime",
        "target": "Linux x86_64, amdgpu/KFD host driver, AMD gfx1151",
        "runtime_python": False,
        "runtime_torch": False,
        "runtime_vllm": False,
        "runtime_triton": False,
        "runtime_transformers": False,
        "native_tokenizer": True,
        "native_inference_engine": True,
        "native_vl": native_vl_record,
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
            "hash-locked gfx1151 vision attention: external general image "
            "plus embedded dense-image variant",
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
            "libpng, libjpeg-turbo, libwebp and libsharpyuv",
            "minimal FFmpeg avformat/avcodec/avutil/swscale",
            "minimal curl with c-ares",
            "OpenSSL and the bundled CA certificate store",
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
