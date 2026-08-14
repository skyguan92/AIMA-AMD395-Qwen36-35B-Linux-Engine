#!/usr/bin/env python3
"""Trace and qualify the short-VL FLA recompute-W/U AOT variant."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "aima-amd395-qwen36/vl-http-language-layer-diagnostic-oracle/v1"
CASE_ID = "multi_video"
NATIVE_BUCKET_TOKENS = 1024


def raw_bytes(tensor: Any) -> bytes:
    import torch

    return (
        tensor.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes()
    )


def load_tensor(root: Path, record: dict[str, Any], dtype: Any) -> Any:
    import torch

    path = root / str(record["path"])
    payload = bytearray(path.read_bytes())
    if len(payload) != int(record["bytes"]):
        raise RuntimeError(f"raw tensor size changed: {path}")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise RuntimeError(f"raw tensor hash changed: {path}")
    tensor = torch.frombuffer(payload, dtype=dtype).clone()
    return tensor.reshape(record["shape"]).to("cuda")


def compare(actual: Any, expected: Any) -> dict[str, Any]:
    import torch

    actual_payload = raw_bytes(actual)
    expected_payload = raw_bytes(expected)
    mismatches = torch.nonzero((actual != expected).flatten()).flatten()
    return {
        "elements": actual.numel(),
        "exact_elements": actual.numel() - mismatches.numel(),
        "first_mismatch_index": (
            int(mismatches[0]) if mismatches.numel() else None
        ),
        "expected_sha256": hashlib.sha256(expected_payload).hexdigest(),
        "actual_sha256": hashlib.sha256(actual_payload).hexdigest(),
        "exact": actual_payload == expected_payload,
    }


def pad_tokens(tensor: Any, tokens: int) -> Any:
    if tensor.ndim < 2 or tensor.shape[0] != 1 or tensor.shape[1] >= tokens:
        raise RuntimeError("short-VL padded replay geometry differs")
    padded = tensor.new_zeros((1, tokens, *tensor.shape[2:]))
    padded[:, : tensor.shape[1]].copy_(tensor)
    return padded


def trace(manifest_path: Path, oracle_root: Path) -> dict[str, Any]:
    import torch
    from vllm.model_executor.layers.fla.ops.wy_fast import (
        recompute_w_u_fwd,
        recompute_w_u_fwd_kernel,
    )

    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    if (
        manifest.get("schema") != SCHEMA
        or not manifest.get("complete", False)
        or not manifest.get("qualified_for_attribution_only", False)
        or manifest.get("case", {}).get("case_id") != CASE_ID
    ):
        raise RuntimeError("HTTP language diagnostic manifest differs")
    components = manifest["case"]["components"]

    key = load_tensor(oracle_root, components["layer_001_gdn_k"], torch.bfloat16)
    value = load_tensor(
        oracle_root, components["layer_001_gdn_v"], torch.bfloat16
    )
    beta = load_tensor(
        oracle_root, components["layer_001_gdn_beta"], torch.float32
    )
    inverse = load_tensor(
        oracle_root,
        components["layer_001_gdn_chunk_matrix_inverse"],
        torch.bfloat16,
    )
    g_cumsum = load_tensor(
        oracle_root, components["layer_001_gdn_g_cumsum"], torch.float32
    )
    expected_w = load_tensor(
        oracle_root, components["layer_001_gdn_w"], torch.bfloat16
    )
    expected_u = load_tensor(
        oracle_root, components["layer_001_gdn_u"], torch.bfloat16
    )

    autotuner = recompute_w_u_fwd_kernel.fn
    autotuner.cache.clear()
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)
    beta = beta.unsqueeze(0)
    actual_w, actual_u = recompute_w_u_fwd(
        k=key,
        v=value,
        beta=beta,
        A=inverse,
        g_cumsum=g_cumsum,
        cu_seqlens=None,
    )
    torch.cuda.synchronize()
    padded_w, padded_u = recompute_w_u_fwd(
        k=pad_tokens(key, NATIVE_BUCKET_TOKENS),
        v=pad_tokens(value, NATIVE_BUCKET_TOKENS),
        beta=pad_tokens(beta, NATIVE_BUCKET_TOKENS),
        A=pad_tokens(inverse, NATIVE_BUCKET_TOKENS),
        g_cumsum=pad_tokens(g_cumsum, NATIVE_BUCKET_TOKENS),
        cu_seqlens=None,
    )
    torch.cuda.synchronize()
    if len(autotuner.cache) != 1:
        raise RuntimeError("recompute-W/U autotune cache differs")
    config = next(iter(autotuner.cache.values()))
    selection = {
        "num_warps": int(config.num_warps),
        "num_stages": int(config.num_stages),
        "num_ctas": int(config.num_ctas),
    }
    comparisons = {
        "direct_w": compare(actual_w, expected_w),
        "direct_u": compare(actual_u, expected_u),
        "native_bucket_w": compare(
            padded_w[:, : key.shape[1]], expected_w
        ),
        "native_bucket_u": compare(
            padded_u[:, : key.shape[1]], expected_u
        ),
    }
    if selection != {"num_warps": 4, "num_stages": 2, "num_ctas": 1}:
        raise RuntimeError(f"short-VL recompute-W/U selection differs: {selection}")
    if not all(item["exact"] for item in comparisons.values()):
        raise RuntimeError("short-VL recompute-W/U output differs")
    return {
        "schema": "aima-amd395-qwen36/vl-recompute-w-u-aot-trace/v1",
        "complete": True,
        "qualified_for_short_vl_prefill": True,
        "source_http_language_diagnostic_sha256": hashlib.sha256(
            manifest_payload
        ).hexdigest(),
        "case_id": CASE_ID,
        "prompt_tokens": manifest["case"]["prompt_tokens"],
        "native_bucket_tokens": NATIVE_BUCKET_TOKENS,
        "autotune_key": {
            "H": 32,
            "K": 128,
            "V": 128,
            "BT": 64,
            "BK": 64,
            "BV": 64,
            "IS_VARLEN": False,
        },
        "selection": selection,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = trace(
        args.diagnostic_manifest.resolve(), args.diagnostic_root.resolve()
    )
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite trace result: {output}")
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"trace VL recompute W/U AOT: {error}")
        raise SystemExit(1)
