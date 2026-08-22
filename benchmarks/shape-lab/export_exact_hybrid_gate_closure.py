#!/usr/bin/env python3
"""Export the pinned exact-hybrid routed gate/up Triton closure."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import torch


CORRECTION_BLOCK_N = 64
CORRECTION_GRID_SIZE = 8192 // CORRECTION_BLOCK_N


def load_probe(path: Path):
    spec = importlib.util.spec_from_file_location("routed_exact_scalar_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load probe module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_image(compiled, output: Path, llvm_strip: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(compiled.asm["hsaco"])
    subprocess.run(
        [str(llvm_strip), "--strip-debug", str(output)],
        check=True,
    )
    return {
        "path": f"kernels/{output.name}",
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coefficient", type=float, required=True)
    parser.add_argument(
        "--llvm-strip",
        type=Path,
        default=Path("/opt/rocm/llvm/bin/llvm-strip"),
    )
    args = parser.parse_args()
    probe = load_probe(args.probe.resolve())
    hidden = torch.empty(2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((1, 1024, 2048), device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.zeros(8, device="cuda", dtype=torch.int32)
    output = torch.empty((8, 1024), device="cuda", dtype=torch.bfloat16)
    flagged_indices = torch.empty(8192, device="cuda", dtype=torch.int32)
    flagged_count = torch.zeros(1, device="cuda", dtype=torch.int32)

    flag = probe.hybrid_scalar_block_flag_kernel.warmup(
        hidden,
        weight,
        expert_ids,
        output,
        flagged_indices,
        flagged_count,
        2048,
        1024,
        BLOCK_M=4,
        CHUNK_K=2048,
        ERROR_COEFFICIENT=args.coefficient,
        CORRECT_SUBNORMALS=True,
        num_warps=8,
        grid=(8, 256),
    )
    correction = probe.sparse_wmma_correction_kernel.warmup(
        hidden,
        weight,
        expert_ids,
        output,
        flagged_indices,
        flagged_count,
        2048,
        1024,
        BLOCK_K=256,
        BLOCK_N=CORRECTION_BLOCK_N,
        num_warps=4,
        num_stages=2,
        grid=(CORRECTION_GRID_SIZE,),
    )
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root must be empty: {output_root}")
    kernel_root = output_root / "kernels"
    flag_name = f"{flag.hash[:16]}-{flag.name}.hsaco"
    correction_name = f"{correction.hash[:16]}-{correction.name}.hsaco"
    flag_image = export_image(flag, kernel_root / flag_name, args.llvm_strip)
    correction_image = export_image(
        correction, kernel_root / correction_name, args.llvm_strip
    )
    manifest = {
        "schema": "aima-amd395-qwen36/native-aot-closure/v1",
        "status": "qualified_exact_hybrid_routed_gate_up",
        "target": {
            "arch": "gfx1151",
            "backend": "HIP",
            "warp_size": 32,
        },
        "source": {
            "exporter": (
                "benchmarks/shape-lab/"
                "export_exact_hybrid_gate_closure.py"
            ),
            "kernel_source": (
                "benchmarks/shape-lab/routed_exact_scalar_probe.py"
            ),
            "triton_version": str(probe.triton.__version__),
        },
        "kernel_count": 2,
        "kernel_symbol_count": 2,
        "launch_variant_count": 2,
        "packaged_hsaco_bytes": (
            flag_image["bytes"] + correction_image["bytes"]
        ),
        "kernels": [
            {
                "kernel_hash": flag.hash,
                "symbol": flag.name,
                "metadata": {
                    "num_warps": flag.metadata.num_warps,
                    "warp_size": flag.metadata.warp_size,
                    "shared": flag.metadata.shared,
                    "error_coefficient": args.coefficient,
                    "block_m": 4,
                    "correct_subnormals": True,
                },
                "image": flag_image,
            },
            {
                "kernel_hash": correction.hash,
                "symbol": correction.name,
                "metadata": {
                    "num_warps": correction.metadata.num_warps,
                    "warp_size": correction.metadata.warp_size,
                    "shared": correction.metadata.shared,
                    "block_n": CORRECTION_BLOCK_N,
                    "grid_size": CORRECTION_GRID_SIZE,
                    "maximum_flagged_values": 8192,
                },
                "image": correction_image,
            },
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
