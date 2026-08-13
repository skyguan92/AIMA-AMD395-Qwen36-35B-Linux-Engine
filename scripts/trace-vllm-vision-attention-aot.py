#!/usr/bin/env python3
"""Exercise the frozen Triton vision-attention kernel for AOT tracing."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CASES = (
    ("image_local_png", 256, (0, 256)),
    ("video_local_mp4", 128, (0, 64, 128)),
)
HEADS = 16
HEAD_DIMENSION = 72


def _read_bf16(path: Path, elements: int) -> Any:
    import torch

    payload = bytearray(path.read_bytes())
    if len(payload) != 2 * elements:
        raise RuntimeError(f"raw BF16 tensor has the wrong size: {path}")
    return torch.frombuffer(payload, dtype=torch.bfloat16).clone()


def _raw_bytes(tensor: Any) -> bytes:
    import torch

    return (
        tensor.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes()
    )


def trace_case(root: Path, case_id: str, patches: int,
               cu_seqlens: tuple[int, ...]) -> dict[str, Any]:
    import torch
    from vllm.v1.attention.ops.triton_prefill_attention import (
        context_attention_fwd,
    )

    case_root = root / case_id
    elements = patches * HEADS * HEAD_DIMENSION
    query = _read_bf16(case_root / "query_rotated.bin", elements).reshape(
        patches, HEADS, HEAD_DIMENSION
    ).to("cuda")
    key = _read_bf16(case_root / "key_rotated.bin", elements).reshape(
        patches, HEADS, HEAD_DIMENSION
    ).to("cuda")
    value = _read_bf16(case_root / "value.bin", elements).reshape(
        patches, HEADS, HEAD_DIMENSION
    ).to("cuda")
    starts = torch.tensor(cu_seqlens[:-1], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(
        [right - left for left, right in zip(cu_seqlens, cu_seqlens[1:])],
        dtype=torch.int32,
        device="cuda",
    )
    output = torch.empty_like(query)
    context_attention_fwd(
        query,
        key,
        value,
        output,
        starts,
        lengths,
        max(lengths.tolist()),
        is_causal=False,
    )
    torch.cuda.synchronize()
    actual = _raw_bytes(output)
    expected = (case_root / "attention.bin").read_bytes()
    if actual != expected:
        raise RuntimeError(f"standalone Triton attention drifted: {case_id}")
    return {
        "case_id": case_id,
        "patches": patches,
        "cu_seqlens": list(cu_seqlens),
        "elements": elements,
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
        "actual_sha256": hashlib.sha256(actual).hexdigest(),
        "exact": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.block_oracle_root.resolve()
    cases = [trace_case(root, *case) for case in CASES]
    result = {
        "schema": "aima-amd395-qwen36/vision-attention-aot-trace/v1",
        "complete": True,
        "all_exact": all(case["exact"] for case in cases),
        "cases": cases,
    }
    args.output.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"trace vision attention AOT: {error}")
        raise SystemExit(1)
