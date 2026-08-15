#!/usr/bin/env python3
"""Exercise current vLLM's singleton Qwen3.6 routed MoE for AOT tracing."""

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
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )

    torch.manual_seed(395)
    device = "cuda"
    dtype = torch.bfloat16
    hidden_size = 2_048
    intermediate_size = 512
    experts = 256
    top_k = 8

    hidden_states = torch.randn((1, hidden_size), device=device, dtype=dtype)
    router_logits = torch.randn((1, experts), device=device, dtype=dtype)
    gate_up = torch.randn(
        (experts, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
    )
    down = torch.randn(
        (experts, hidden_size, intermediate_size),
        device=device,
        dtype=dtype,
    )
    hidden_before = hidden_states.clone()
    gate_up_before = gate_up.clone()
    down_before = down.clone()

    with torch.no_grad():
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=True,
        )
        output = fused_experts(
            hidden_states=hidden_states,
            w1=gate_up,
            w2=down,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
    torch.cuda.synchronize()

    checks = {
        "hidden_states_unchanged": bool(torch.equal(hidden_states, hidden_before)),
        "gate_up_unchanged": bool(torch.equal(gate_up, gate_up_before)),
        "down_unchanged": bool(torch.equal(down, down_before)),
        "output_finite": bool(torch.isfinite(output.float()).all()),
        "topk_weights_finite": bool(torch.isfinite(topk_weights).all()),
        "topk_ids_in_range": bool(
            torch.logical_and(topk_ids >= 0, topk_ids < experts).all()
        ),
    }
    if (
        output.shape != (1, hidden_size)
        or output.dtype != dtype
        or topk_weights.shape != (1, top_k)
        or topk_weights.dtype != torch.float32
        or topk_ids.shape != (1, top_k)
        or topk_ids.dtype != torch.int32
        or token_expert_indices.shape != (1, top_k)
        or token_expert_indices.dtype != torch.int32
        or not all(checks.values())
    ):
        raise RuntimeError("routed MoE singleton decode contract changed")

    result = {
        "schema": "aima-amd395-qwen36/routed-moe-decode-aot-trace/v1",
        "complete": True,
        "qualified_for_aot_capture": True,
        "qualified_for_native_decode_replacement": False,
        "source_apis": [
            "vllm.model_executor.layers.fused_moe.router.fused_topk_router.fused_topk",
            "vllm.model_executor.layers.fused_moe.fused_moe.fused_experts",
        ],
        "geometry": {
            "tokens": 1,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "experts": experts,
            "top_k": top_k,
            "renormalize": True,
            "hidden_dtype": str(dtype),
            "router_logits_dtype": str(router_logits.dtype),
            "topk_weights_dtype": str(topk_weights.dtype),
            "topk_ids_dtype": str(topk_ids.dtype),
        },
        "checks": {
            **checks,
            "topk_weights_sha256": hashlib.sha256(
                raw_bytes(topk_weights)
            ).hexdigest(),
            "topk_ids_sha256": hashlib.sha256(raw_bytes(topk_ids)).hexdigest(),
            "output_sha256": hashlib.sha256(raw_bytes(output)).hexdigest(),
        },
        "non_claims": [
            "not_a_model_weight_correctness_oracle",
            "not_yet_integrated_into_native_decode",
            "not_a_promotion_result",
        ],
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
        print(f"trace routed MoE decode AOT: {error}")
        raise SystemExit(1)
