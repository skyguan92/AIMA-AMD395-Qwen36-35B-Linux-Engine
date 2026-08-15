#!/usr/bin/env python3
"""Exercise current vLLM's production GDN gated norm for AOT tracing."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def raw_bytes(tensor: Any) -> bytes:
    import torch

    return (
        tensor.detach()
        .contiguous()
        .cpu()
        .view(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from vllm.model_executor.layers.layernorm import RMSNormGated

    torch.manual_seed(395)
    device = "cuda"
    dtype = torch.bfloat16
    core = torch.randn((1, 1, 32, 128), device=device, dtype=dtype)
    gate = torch.randn_like(core)
    core_before = core.clone()
    gate_before = gate.clone()
    norm = RMSNormGated(
        128,
        eps=1.0e-6,
        group_size=None,
        norm_before_gate=True,
        device=torch.device(device),
        dtype=dtype,
        activation="silu",
    )
    with torch.no_grad():
        norm.weight.copy_(
            torch.randn((128,), device=device, dtype=dtype)
        )
        output = norm(core, gate)
    torch.cuda.synchronize()

    input_unchanged = bool(torch.equal(core, core_before))
    gate_unchanged = bool(torch.equal(gate, gate_before))
    output_finite = bool(torch.isfinite(output.float()).all())
    if (
        output.shape != core.shape
        or output.dtype != dtype
        or not input_unchanged
        or not gate_unchanged
        or not output_finite
    ):
        raise RuntimeError("linear gated norm decode contract changed")

    result = {
        "schema": "aima-amd395-qwen36/linear-gated-norm-decode-aot-trace/v1",
        "complete": True,
        "qualified_for_native_decode_replacement": True,
        "source_api": "vllm.model_executor.layers.layernorm.RMSNormGated",
        "geometry": {
            "batch": 1,
            "tokens": 1,
            "linear_heads": 32,
            "head_dimension": 128,
            "rows": 32,
            "epsilon": 1.0e-6,
            "norm_before_gate": True,
            "activation": "silu",
            "dtype": "torch.bfloat16",
        },
        "checks": {
            "input_unchanged": input_unchanged,
            "gate_unchanged": gate_unchanged,
            "output_finite": output_finite,
            "output_sha256": hashlib.sha256(raw_bytes(output)).hexdigest(),
        },
    }
    output_path = args.output.resolve()
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite trace result: {output_path}")
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"trace linear gated norm decode AOT: {error}")
        raise SystemExit(1)
