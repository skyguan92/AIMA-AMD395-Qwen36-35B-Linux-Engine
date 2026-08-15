#!/usr/bin/env python3
"""Exercise the current vLLM packed GDN decode kernel for AOT tracing."""

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
        tensor.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    torch.manual_seed(395)
    device = "cuda"
    mixed_qkv = torch.randn((1, 8192), device=device, dtype=torch.bfloat16)
    a = torch.randn((1, 32), device=device, dtype=torch.bfloat16)
    b = torch.randn((1, 32), device=device, dtype=torch.bfloat16)
    a_log = torch.randn((32,), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((32,), device=device, dtype=torch.bfloat16)
    state = torch.randn((2, 32, 128, 128), device=device, dtype=torch.float32)
    state_before = state.clone()
    output = torch.empty((1, 1, 32, 128), device=device, dtype=torch.bfloat16)
    state_indices = torch.tensor([1], device=device, dtype=torch.int32)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        128**-0.5,
        state,
        output,
        state_indices,
        True,
    )
    torch.cuda.synchronize()

    guard_unchanged = bool(torch.equal(state[0], state_before[0]))
    selected_state_changed = not bool(torch.equal(state[1], state_before[1]))
    output_finite = bool(torch.isfinite(output.float()).all())
    if not guard_unchanged or not selected_state_changed or not output_finite:
        raise RuntimeError("packed decode state-index or output contract changed")

    result = {
        "schema": "aima-amd395-qwen36/packed-linear-decode-aot-trace/v1",
        "complete": True,
        "qualified_for_native_decode_replacement": True,
        "source_api": (
            "vllm.model_executor.layers.mamba.gdn_linear_attn."
            "fused_recurrent_gated_delta_rule_packed_decode"
        ),
        "geometry": {
            "batch": 1,
            "mixed_qkv_elements": 8192,
            "linear_heads": 32,
            "key_dimension": 128,
            "value_dimension": 128,
            "state_rows": 2,
            "selected_state_index": 1,
            "scale": 128**-0.5,
        },
        "checks": {
            "guard_state_unchanged": guard_unchanged,
            "selected_state_changed": selected_state_changed,
            "output_finite": output_finite,
            "output_sha256": hashlib.sha256(raw_bytes(output)).hexdigest(),
            "selected_state_sha256": hashlib.sha256(raw_bytes(state[1])).hexdigest(),
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
        print(f"trace packed linear decode AOT: {error}")
        raise SystemExit(1)
