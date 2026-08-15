#!/usr/bin/env python3
"""Exercise the current vLLM causal-conv decode kernel for AOT tracing."""

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


def digest(tensor: Any) -> str:
    return hashlib.sha256(raw_bytes(tensor)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-production-parity",
        action="store_true",
        help="also run the indexed production variant and require raw parity",
    )
    args = parser.parse_args()

    import torch
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    torch.manual_seed(395)
    device = "cuda"
    source_x = torch.randn((1, 8_192), device=device, dtype=torch.bfloat16)
    source_state = torch.randn(
        (1, 8_192, 3), device=device, dtype=torch.bfloat16
    )
    weight = torch.randn((8_192, 4), device=device, dtype=torch.bfloat16)

    direct_x = source_x.clone()
    direct_state = source_state.clone()
    direct_state_before = direct_state.clone()
    direct_index = torch.tensor([0], device=device, dtype=torch.int32)
    direct_output = causal_conv1d_update(
        direct_x,
        direct_state,
        weight,
        None,
        "silu",
        conv_state_indices=direct_index,
        null_block_id=None,
        validate_data=False,
    )
    torch.cuda.synchronize()

    direct_state_changed = not bool(
        torch.equal(direct_state, direct_state_before)
    )
    output_finite = bool(torch.isfinite(direct_output.float()).all())
    state_finite = bool(torch.isfinite(direct_state.float()).all())
    if not direct_state_changed or not output_finite or not state_finite:
        raise RuntimeError("causal-conv direct decode contract changed")

    production_output_equal = None
    production_state_equal = None
    production_output_sha256 = None
    production_state_sha256 = None
    if args.verify_production_parity:
        production_x = source_x.clone()
        production_state = torch.cat(
            (torch.zeros_like(source_state), source_state.clone()), dim=0
        )
        production_index = torch.tensor(
            [1], device=device, dtype=torch.int32
        )
        production_output = causal_conv1d_update(
            production_x,
            production_state,
            weight,
            None,
            "silu",
            conv_state_indices=production_index,
            validate_data=False,
        )
        torch.cuda.synchronize()
        production_output_equal = bool(
            torch.equal(production_output, direct_output)
        )
        production_state_equal = raw_bytes(production_state[1]) == raw_bytes(
            direct_state
        )
        production_output_sha256 = digest(production_output)
        production_state_sha256 = digest(production_state[1])
        if not production_output_equal or not production_state_equal:
            raise RuntimeError(
                "causal-conv direct specialization differs from production"
            )

    result = {
        "schema": "aima-amd395-qwen36/causal-conv-decode-aot-trace/v1",
        "complete": True,
        "qualified_for_native_decode_replacement": (
            args.verify_production_parity
        ),
        "source_api": (
            "vllm.model_executor.layers.mamba.ops.causal_conv1d."
            "causal_conv1d_update"
        ),
        "geometry": {
            "batch": 1,
            "channels": 8_192,
            "state_length": 3,
            "kernel_width": 4,
            "activation": "silu",
            "direct_state_rows": 1,
            "direct_state_index": 0,
            "null_block_id": None,
        },
        "checks": {
            "direct_state_changed": direct_state_changed,
            "output_finite": output_finite,
            "state_finite": state_finite,
            "direct_output_sha256": digest(direct_output),
            "direct_state_sha256": digest(direct_state),
            "production_parity_checked": args.verify_production_parity,
            "production_output_equal": production_output_equal,
            "production_state_equal": production_state_equal,
            "production_output_sha256": production_output_sha256,
            "production_state_sha256": production_state_sha256,
        },
    }
    output_path = args.output.resolve()
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite trace result: {output_path}")
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"trace causal-conv decode AOT: {error}")
        raise SystemExit(1)
