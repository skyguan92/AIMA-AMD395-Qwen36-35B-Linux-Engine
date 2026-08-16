#!/usr/bin/env python3
"""Exercise current vLLM singleton unified attention for AOT tracing."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    parser.add_argument("--sequence-length", type=int, default=350)
    args = parser.parse_args()
    if args.sequence_length <= 1:
        raise ValueError("sequence length must include a non-empty decode prefix")

    import torch
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention,
    )

    torch.manual_seed(395)
    device = "cuda"
    dtype = torch.bfloat16
    query_heads = 16
    kv_heads = 2
    head_size = 256
    # The frozen Qwen3.6-VL worker allocates paged KV blocks at the admitted
    # multimodal bucket size.  This compile-time value is part of the Triton
    # kernel identity and must match the production cache, even for q=1 decode.
    block_size = 1_056
    blocks = (args.sequence_length + block_size - 1) // block_size
    sequence_threshold_3d = 64
    softmax_segments = 16

    query = torch.randn(
        (1, query_heads, head_size), device=device, dtype=dtype
    )
    key_cache = torch.randn(
        (blocks, block_size, kv_heads, head_size),
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    output = torch.empty_like(query)
    query_starts = torch.tensor([0, 1], device=device, dtype=torch.int32)
    sequence_lengths = torch.tensor(
        [args.sequence_length], device=device, dtype=torch.int32
    )
    block_table = torch.arange(blocks, device=device, dtype=torch.int32).reshape(
        1, blocks
    )
    descale = torch.tensor(1.0, device=device).expand(1, kv_heads)
    segment_output = torch.empty(
        (
            sequence_threshold_3d,
            query_heads,
            softmax_segments,
            head_size,
        ),
        device=device,
        dtype=torch.float32,
    )
    segment_max = torch.empty(
        (sequence_threshold_3d, query_heads, softmax_segments),
        device=device,
        dtype=torch.float32,
    )
    segment_expsum = torch.empty_like(segment_max)

    with torch.no_grad():
        unified_attention(
            query,
            key_cache,
            value_cache,
            output,
            query_starts,
            1,
            sequence_lengths,
            args.sequence_length,
            1.0 / math.sqrt(head_size),
            True,
            (-1, -1),
            block_table,
            0.0,
            None,
            descale,
            descale,
            sequence_threshold_3d,
            softmax_segments,
            segment_output,
            segment_max,
            segment_expsum,
        )
    torch.cuda.synchronize()

    output_repeat = torch.empty_like(query)
    with torch.no_grad():
        unified_attention(
            query,
            key_cache,
            value_cache,
            output_repeat,
            query_starts,
            1,
            sequence_lengths,
            args.sequence_length,
            1.0 / math.sqrt(head_size),
            True,
            (-1, -1),
            block_table,
            0.0,
            None,
            descale,
            descale,
            sequence_threshold_3d,
            softmax_segments,
            segment_output,
            segment_max,
            segment_expsum,
        )
    torch.cuda.synchronize()

    checks = {
        "output_finite": bool(torch.isfinite(output.float()).all()),
        "repeat_exact": bool(torch.equal(output, output_repeat)),
    }
    if output.shape != query.shape or output.dtype != dtype or not all(
        checks.values()
    ):
        raise RuntimeError("unified-attention singleton decode contract changed")

    result = {
        "schema": "aima-amd395-qwen36/unified-attention-decode-aot-trace/v1",
        "complete": True,
        "qualified_for_aot_capture": True,
        "qualified_for_native_decode_replacement": False,
        "source_api": (
            "vllm.v1.attention.ops.triton_unified_attention.unified_attention"
        ),
        "geometry": {
            "sequences": 1,
            "query_tokens": 1,
            "sequence_length": args.sequence_length,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_size": head_size,
            "block_size": block_size,
            "physical_blocks": blocks,
            "sequence_threshold_3d": sequence_threshold_3d,
            "softmax_segments": softmax_segments,
            "attention_path": "segmented_3d_plus_reduce",
            "softmax_scale": 1.0 / math.sqrt(head_size),
            "causal": True,
            "sliding_window": [-1, -1],
            "dtype": str(dtype),
        },
        "checks": {
            **checks,
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
        print(f"trace unified attention decode AOT: {error}")
        raise SystemExit(1)
