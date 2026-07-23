#!/usr/bin/env python3
"""Export a minimal, path-clean gfx1151 AOT kernel closure from trace evidence."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any


HASH_CHUNK_BYTES = 1024 * 1024
FORBIDDEN_BINARY_MARKERS = (b"/home/", b"/data/", b"site-packages")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid trace JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"trace record is not an object at {path}:{line_number}")
            records.append(value)
    return records


def normalized_source_path(value: str) -> str | None:
    shape_marker = "/benchmarks/shape-lab/"
    if shape_marker in value:
        return "benchmarks/shape-lab/" + value.split(shape_marker, 1)[1]
    package_marker = "/site-packages/"
    if package_marker in value:
        relative = value.split(package_marker, 1)[1]
        if relative.startswith(("vllm/", "triton/")):
            return relative
    return None


def source_provenance(source_path: Path | None) -> list[dict[str, str]]:
    if source_path is None:
        return []
    text = source_path.read_text(encoding="utf-8")
    paths = {
        normalized
        for match in re.finditer(r'loc\("([^"\n]+\.py)"', text)
        if (normalized := normalized_source_path(match.group(1))) is not None
    }
    result: list[dict[str, str]] = []
    for path in sorted(paths):
        if path.startswith("vllm/"):
            owner, license_id = "vLLM", "Apache-2.0"
        elif path.startswith("triton/"):
            owner, license_id = "Triton", "MIT"
        else:
            owner, license_id = "Approaching AI Authors", "Apache-2.0"
        result.append({"path": path, "owner": owner, "license": license_id})
    return result


def one_cache_file(records: list[dict[str, Any]], suffix: str) -> dict[str, Any] | None:
    by_sha: dict[str, dict[str, Any]] = {}
    for record in records:
        for item in record["cache_files"]:
            if str(item["name"]).endswith(suffix):
                by_sha[str(item["sha256"])] = item
    if not by_sha:
        return None
    if len(by_sha) != 1:
        raise RuntimeError(f"one kernel hash maps to multiple {suffix} files")
    item = next(iter(by_sha.values()))
    path = Path(str(item["path"]))
    if not path.is_file():
        raise RuntimeError(f"trace cache artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != item["sha256"]:
        raise RuntimeError(f"trace cache artifact hash changed: {path}")
    if path.stat().st_size != int(item["bytes"]):
        raise RuntimeError(f"trace cache artifact size changed: {path}")
    return {**item, "resolved_path": path}


def regular_argument_geometry(arguments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for argument in arguments:
        if argument.get("abi_type") == "constexpr":
            continue
        keep = {"name", "abi_type", "kind", "shape", "stride", "dtype", "element_size", "value", "is_null"}
        result.append({key: value for key, value in argument.items() if key in keep})
    return result


def build_manifest(
    trace_paths: list[Path],
    output_dir: Path,
    llvm_strip: Path,
    context_tokens: int,
    completion_tokens: int,
    source_commit: str,
    engine_sha256: str,
) -> dict[str, Any]:
    trace_records: list[dict[str, Any]] = []
    trace_inputs: list[dict[str, Any]] = []
    for path in trace_paths:
        resolved = path.resolve()
        records = read_trace(resolved)
        trace_records.extend(records)
        trace_inputs.append(
            {
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
                "records": len(records),
            }
        )

    hard_errors = [
        record
        for record in trace_records
        if record.get("event") in {"trace_launch_error", "trace_autotune_error"}
    ]
    if hard_errors:
        raise RuntimeError(f"trace contains {len(hard_errors)} launch/autotune errors")
    launches = [record for record in trace_records if record.get("event") == "triton_launch"]
    if not launches:
        raise RuntimeError("trace contains no Triton launches")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in launches:
        metadata = record.get("metadata", {})
        target = metadata.get("target", {})
        if target != {"backend": "hip", "arch": "gfx1151", "warp_size": 32}:
            raise RuntimeError(f"unqualified AOT target: {target}")
        if metadata.get("num_ctas") != 1 or metadata.get("launch_cooperative_grid") is not False:
            raise RuntimeError("clustered/cooperative Triton kernels are not yet supported")
        grouped[str(record["kernel_hash"])].append(record)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite AOT output directory: {output_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    kernels_dir = temporary / "kernels"
    kernels_dir.mkdir()
    kernels: list[dict[str, Any]] = []
    raw_hsaco_bytes = 0
    packaged_hsaco_bytes = 0
    try:
        for kernel_hash, records in sorted(grouped.items()):
            first = records[0]
            metadata = first["metadata"]
            signature = first["signature"]
            if any(record["metadata"] != metadata for record in records):
                raise RuntimeError(f"metadata drift for kernel hash {kernel_hash}")
            if any(record["signature"] != signature for record in records):
                raise RuntimeError(f"signature drift for kernel hash {kernel_hash}")

            arguments = first["arguments"]
            constants = {
                str(argument["name"]): argument.get("value")
                for argument in arguments
                if argument.get("abi_type") == "constexpr"
            }
            if any(
                {
                    str(argument["name"]): argument.get("value")
                    for argument in record["arguments"]
                    if argument.get("abi_type") == "constexpr"
                }
                != constants
                for record in records
            ):
                raise RuntimeError(f"compile-constant drift for kernel hash {kernel_hash}")

            hsaco = one_cache_file(records, ".hsaco")
            if hsaco is None:
                raise RuntimeError(f"kernel hash has no HSACO: {kernel_hash}")
            source = one_cache_file(records, ".source")
            name = str(metadata["name"])
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
            relative_image = Path("kernels") / f"{kernel_hash[:16]}-{safe_name}.hsaco"
            destination = temporary / relative_image
            shutil.copyfile(hsaco["resolved_path"], destination)
            subprocess.run(
                [str(llvm_strip), "--strip-debug", str(destination)],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = destination.read_bytes()
            markers = [marker.decode("ascii") for marker in FORBIDDEN_BINARY_MARKERS if marker in payload]
            if markers:
                raise RuntimeError(f"stripped HSACO still contains private paths {markers}: {name}")
            packaged_sha = sha256_file(destination)
            raw_hsaco_bytes += int(hsaco["bytes"])
            packaged_hsaco_bytes += destination.stat().st_size

            launch_variants_by_key: dict[str, dict[str, Any]] = {}
            for record in records:
                variant = {
                    "grid": [int(item) for item in record["grid"]],
                    "arguments": regular_argument_geometry(record["arguments"]),
                }
                launch_variants_by_key[canonical(variant)] = variant

            audit_files: dict[str, dict[str, Any]] = {}
            for record in records:
                for item in record["cache_files"]:
                    key = str(item["sha256"])
                    audit_files[key] = {
                        "name": str(item["name"]),
                        "bytes": int(item["bytes"]),
                        "sha256": key,
                    }
            kernels.append(
                {
                    "kernel_hash": kernel_hash,
                    "symbol": name,
                    "metadata": metadata,
                    "signature": signature,
                    "regular_abi": [
                        {"name": argument["name"], "type": argument["abi_type"]}
                        for argument in arguments
                        if argument.get("abi_type") != "constexpr"
                    ],
                    "compile_constants": constants,
                    "launch_variants": [
                        launch_variants_by_key[key] for key in sorted(launch_variants_by_key)
                    ],
                    "image": {
                        "path": str(relative_image),
                        "bytes": destination.stat().st_size,
                        "sha256": packaged_sha,
                        "raw_bytes": int(hsaco["bytes"]),
                        "raw_sha256": str(hsaco["sha256"]),
                        "transformation": "llvm-strip --strip-debug",
                    },
                    "source_provenance": source_provenance(
                        source["resolved_path"] if source else None
                    ),
                    "audit_artifacts": [audit_files[key] for key in sorted(audit_files)],
                }
            )

        manifest = {
            "schema": "aima-amd395-qwen36/native-aot-closure/v1",
            "status": (
                f"bounded_q{context_tokens}_output{completion_tokens}_"
                "executor_migration_evidence"
            ),
            "target": {"backend": "HIP", "arch": "gfx1151", "warp_size": 32},
            "source": {
                "commit": source_commit,
                "engine_sha256": engine_sha256,
                "compiler": {"name": "Triton", "version": "3.6.0", "license": "MIT"},
                "vllm_version": "0.19.1rc1.dev300+g29e5d1020.rocm721",
                "vllm_license": "Apache-2.0",
                "traces": trace_inputs,
            },
            "coverage": {
                "context_tokens": context_tokens,
                "completion_tokens": completion_tokens,
                "cache_state": "warm_compilation_cache_cold_prefix_request",
                "full_matrix_complete": False,
            },
            "abi": {
                "launch": "hipModuleLaunchKernel",
                "hidden_arguments": ["global_scratch", "profile_scratch"],
                "hidden_argument_values": [0, 0],
            },
            "kernel_count": len(kernels),
            "kernel_symbol_count": len({kernel["symbol"] for kernel in kernels}),
            "launch_variant_count": sum(len(kernel["launch_variants"]) for kernel in kernels),
            "raw_hsaco_bytes": raw_hsaco_bytes,
            "packaged_hsaco_bytes": packaged_hsaco_bytes,
            "kernels": kernels,
            "non_claims": [
                "not_yet_a_complete_native_model_executor",
                "not_yet_full_context_matrix_coverage",
                "not_yet_a_correctness_or_performance_promotion",
            ],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llvm-strip", type=Path, default=Path("/opt/rocm/llvm/bin/llvm-strip"))
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--completion-tokens", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--engine-sha256", required=True)
    args = parser.parse_args()
    if not args.llvm_strip.is_file():
        raise SystemExit(f"llvm-strip does not exist: {args.llvm_strip}")
    if args.context_tokens <= 0 or args.completion_tokens <= 0:
        raise SystemExit("context/completion token counts must be positive")
    manifest = build_manifest(
        trace_paths=args.trace,
        output_dir=args.output_dir.resolve(),
        llvm_strip=args.llvm_strip.resolve(),
        context_tokens=args.context_tokens,
        completion_tokens=args.completion_tokens,
        source_commit=args.source_commit,
        engine_sha256=args.engine_sha256,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "kernel_count": manifest["kernel_count"],
                "kernel_symbol_count": manifest["kernel_symbol_count"],
                "launch_variant_count": manifest["launch_variant_count"],
                "raw_hsaco_bytes": manifest["raw_hsaco_bytes"],
                "packaged_hsaco_bytes": manifest["packaged_hsaco_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
