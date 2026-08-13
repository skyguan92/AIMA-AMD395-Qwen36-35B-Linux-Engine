#!/usr/bin/env python3
"""Generate a pointer-free native decode launch schedule from validation traces."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LINEAR_SEQUENCE = [
    "triton_rmsnorm_kernel",
    "triton_fused_input_proj_conv_kernel",
    "fused_sigmoid_gating_delta_rule_update_kernel",
    "triton_linear_gated_norm_kernel",
    "triton_rmsnorm_kernel",
    "triton_matvec_kernel",
    "triton_router_topk_stage1_kernel",
    "triton_router_topk_stage2_softmax_kernel",
    "raw_row_gate_up_activation_kernel",
    "raw_row_down_sum_kernel",
]
FULL_SEQUENCE = [
    "triton_rmsnorm_kernel",
    "triton_matvec_kernel",
    "triton_head_norm_rope_kernel",
    "triton_head_norm_rope_kernel",
    "triton_rmsnorm_kernel",
    "triton_matvec_kernel",
    "triton_router_topk_stage1_kernel",
    "triton_router_topk_stage2_softmax_kernel",
    "raw_row_gate_up_activation_kernel",
    "raw_row_down_sum_kernel",
]
FINAL_SEQUENCE = [
    "triton_rmsnorm_kernel",
    "triton_lm_head_rowwise_int8_gemv_kernel",
]
PREFILL_LINEAR_SEQUENCE = [
    "triton_rmsnorm_kernel",
    "_causal_conv1d_fwd_kernel",
    "_fused_post_conv_kernel",
    "chunk_local_cumsum_scalar_kernel",
    "chunk_scaled_dot_kkt_fwd_kernel",
    "merge_16x16_to_32x32_inverse_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
    "triton_linear_gated_norm_from_invstd_kernel",
    "triton_prefill_fused_add_rmsnorm_kernel",
    "fused_moe_kernel",
    "fused_moe_kernel",
]
PREFILL_LINEAR_SEQUENCE_Q32768 = [
    "triton_rmsnorm_kernel",
    "triton_prefill_direct_conv_kernel",
    "_fused_post_conv_kernel",
    "chunk_local_cumsum_scalar_kernel",
    "chunk_scaled_dot_kkt_fwd_kernel",
    "merge_16x16_to_32x32_inverse_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
    "triton_prefill_fused_add_rmsnorm_kernel",
    "fused_moe_kernel",
    "fused_moe_kernel",
]
PREFILL_LINEAR_SEQUENCE_Q1024 = [
    "triton_rmsnorm_kernel",
    "triton_prefill_direct_conv_kernel",
    "_fused_post_conv_kernel",
    "chunk_local_cumsum_scalar_kernel",
    "chunk_scaled_dot_kkt_fwd_kernel",
    "merge_16x16_to_64x64_inverse_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
    "triton_prefill_fused_add_rmsnorm_kernel",
    "fused_moe_kernel",
    "fused_moe_kernel",
]
PREFILL_LINEAR_SEQUENCE_SPLIT_TAIL = [
    "triton_rmsnorm_kernel",
    "_causal_conv1d_fwd_kernel",
    "_fused_post_conv_kernel",
    "chunk_local_cumsum_scalar_kernel",
    "chunk_scaled_dot_kkt_fwd_kernel",
    "merge_16x16_to_32x32_inverse_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
    "triton_prefill_fused_add_rmsnorm_kernel",
    "fused_moe_kernel",
    "fused_moe_kernel",
]
PREFILL_FULL_SEQUENCE = [
    "triton_rmsnorm_kernel",
    "triton_prefill_fused_add_rmsnorm_kernel",
    "fused_moe_kernel",
    "fused_moe_kernel",
]
PREFILL_FINAL_SEQUENCE = ["triton_lm_head_rowwise_int8_gemv_kernel"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"JSONL record is not an object at {path}:{line_number}")
            records.append(value)
    return records


def caller_scalar(record: dict[str, Any], name: str) -> Any:
    for frame in record.get("caller_context", []):
        scalars = frame.get("scalars", {})
        if name in scalars:
            return scalars[name]
    return None


def normalized_name(name: str, layer_index: int | None) -> str:
    if name.startswith("tensors.") and layer_index is not None:
        return f"layer_weights.{layer_index}." + name
    return name


def binding_kind(name: str) -> str:
    if name.startswith(("layer_weights.", "global_tensors.")):
        return "model_or_derived_weight"
    if name.startswith("transient."):
        return "transient_workspace"
    return "resident_state_or_workspace"


def prefill_linear_sequence(context_tokens: int) -> list[str]:
    if context_tokens == 1024:
        return PREFILL_LINEAR_SEQUENCE_Q1024
    if context_tokens == 8192:
        return PREFILL_LINEAR_SEQUENCE
    if context_tokens in {7168, 7680, 8191}:
        return PREFILL_LINEAR_SEQUENCE_SPLIT_TAIL
    if context_tokens > 0:
        return PREFILL_LINEAR_SEQUENCE_Q32768
    raise RuntimeError(
        "no qualified prefill linear-attention sequence is registered for "
        f"context {context_tokens}"
    )


def regular_argument_geometry(arguments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "name",
        "abi_type",
        "kind",
        "shape",
        "stride",
        "dtype",
        "element_size",
        "value",
        "is_null",
    }
    return [
        {key: value for key, value in argument.items() if key in keep}
        for argument in arguments
        if argument.get("abi_type") != "constexpr"
    ]


def embedded_kernel_for_record(
    record: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    kernel_hash = str(record["kernel_hash"])
    by_hash = {
        str(item["kernel_hash"]): item for item in manifest.get("kernels", [])
    }
    if kernel_hash in by_hash:
        return by_hash[kernel_hash], False
    geometry = regular_argument_geometry(record["arguments"])
    grid = [int(item) for item in record["grid"]]
    symbol = str(record["metadata"]["name"])
    candidates = [
        item
        for item in manifest.get("kernels", [])
        if str(item.get("symbol")) == symbol
        and any(
            variant.get("grid") == grid
            and variant.get("arguments") == geometry
            for variant in item.get("launch_variants", [])
        )
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"trace kernel {kernel_hash} ({symbol}) has {len(candidates)} "
            "ABI-compatible embedded variants"
        )
    return candidates[0], True


def candidate_score(
    candidate: dict[str, Any],
    argument: dict[str, Any],
    layer_index: int | None,
) -> tuple[int, int, int, int, int, str]:
    name = normalized_name(str(candidate["name"]), layer_index)
    layer_prefix = f"layer_weights.{layer_index}." if layer_index is not None else ""
    return (
        1 if layer_prefix and name.startswith(layer_prefix) else 0,
        1 if name.startswith(("layer_weights.", "global_tensors.")) else 0,
        1 if candidate.get("shape") == argument.get("shape") else 0,
        1 if candidate.get("stride") == argument.get("stride") else 0,
        1 if candidate.get("dtype") == argument.get("dtype") else 0,
        name,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tensor-registry", type=Path, required=True)
    parser.add_argument("--aot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=8192,
        help="Qualified request context represented by the trace (default: 8192).",
    )
    args = parser.parse_args()
    if args.context_tokens <= 0:
        parser.error("--context-tokens must be positive")

    trace_path = args.trace.resolve()
    registry_path = args.tensor_registry.resolve()
    manifest_path = args.aot_manifest.resolve()
    registry = load_json(registry_path)
    manifest = load_json(manifest_path)
    records = load_jsonl(trace_path)
    qualified_prefill_linear = (
        prefill_linear_sequence(args.context_tokens)
        if args.phase == "prefill"
        else PREFILL_LINEAR_SEQUENCE
    )
    if args.phase == "decode":
        launches = [
            record
            for record in records
            if record.get("event") == "triton_launch"
            and caller_scalar(record, "mode") == "decode"
            and caller_scalar(record, "position_start") == args.context_tokens
        ]
        expected_launches = 402
    else:
        launches = [
            record
            for record in records
            if record.get("event") == "triton_launch"
            and caller_scalar(record, "mode") == "prefill"
            and caller_scalar(record, "tokens") == args.context_tokens
            and caller_scalar(record, "position_start") == 0
        ]
        expected_launches = (
            30 * len(qualified_prefill_linear)
            + 10 * len(PREFILL_FULL_SEQUENCE)
            + len(PREFILL_FINAL_SEQUENCE)
        )
    if len(launches) != expected_launches:
        raise RuntimeError(
            f"expected {expected_launches} qualified {args.phase} launches, "
            f"got {len(launches)}"
        )
    embedded_kernels = [embedded_kernel_for_record(record, manifest) for record in launches]

    tensors = list(registry["tensors"])
    by_data_pointer: dict[int, list[dict[str, Any]]] = {}
    for tensor in tensors:
        by_data_pointer.setdefault(int(tensor["data_ptr"]), []).append(tensor)

    transient_ids: dict[tuple[int, int], str] = {}
    mapped_arguments = 0
    transient_arguments = 0
    schedule: list[dict[str, Any]] = []
    by_layer: dict[int, list[str]] = {index: [] for index in range(40)}
    substituted_launches = 0
    substituted_hashes: dict[str, str] = {}

    for phase_sequence, (record, embedded_selection) in enumerate(
        zip(launches, embedded_kernels)
    ):
        embedded_kernel, substituted = embedded_selection
        embedded_metadata = embedded_kernel["metadata"]
        if substituted:
            substituted_launches += 1
            substituted_hashes[str(record["kernel_hash"])] = str(
                embedded_kernel["kernel_hash"]
            )
        symbol = str(record["metadata"]["name"])
        is_final_logits = any(
            frame.get("function") == "final_logits"
            for frame in record.get("caller_context", [])
        ) or (
            args.phase == "prefill"
            and symbol == "triton_lm_head_rowwise_int8_gemv_kernel"
        )
        layer_value = None if is_final_logits else caller_scalar(record, "layer_index")
        layer_index = int(layer_value) if layer_value is not None else None
        if layer_index is not None:
            by_layer[layer_index].append(symbol)
        regular_arguments: list[dict[str, Any]] = []
        for argument in record["arguments"]:
            if argument.get("abi_type") == "constexpr":
                continue
            if argument.get("kind") != "tensor":
                regular_arguments.append(
                    {
                        "name": argument["name"],
                        "abi_type": argument.get("abi_type"),
                        "kind": argument.get("kind"),
                        "value": argument.get("value"),
                    }
                )
                continue
            pointer = int(argument["data_ptr"])
            candidates = list(by_data_pointer.get(pointer, []))
            if not candidates:
                candidates = [
                    tensor
                    for tensor in tensors
                    if int(tensor["storage_data_ptr"])
                    <= pointer
                    < int(tensor["storage_data_ptr"]) + int(tensor["storage_nbytes"])
                ]
            if candidates:
                candidate = max(
                    candidates,
                    key=lambda item: candidate_score(item, argument, layer_index),
                )
                binding = normalized_name(str(candidate["name"]), layer_index)
                storage_pointer = int(candidate["storage_data_ptr"])
                storage_nbytes = int(candidate["storage_nbytes"])
                mapped_arguments += 1
            else:
                storage_pointer = int(argument["storage_data_ptr"])
                storage_nbytes = int(argument["storage_nbytes"])
                key = (storage_pointer, storage_nbytes)
                if key not in transient_ids:
                    transient_ids[key] = f"transient.{len(transient_ids)}"
                binding = transient_ids[key]
                transient_arguments += 1
            regular_arguments.append(
                {
                    "name": argument["name"],
                    "abi_type": argument.get("abi_type"),
                    "kind": "tensor",
                    "binding": binding,
                    "binding_kind": binding_kind(binding),
                    "storage_nbytes": storage_nbytes,
                    "byte_offset": pointer - storage_pointer,
                    "shape": argument.get("shape"),
                    "stride": argument.get("stride"),
                    "dtype": argument.get("dtype"),
                }
            )
        schedule.append(
            {
                f"{args.phase}_sequence": phase_sequence,
                "source_sequence": int(record["sequence"]),
                "layer_index": layer_index,
                "layer_type": (
                    "full_attention"
                    if layer_index is not None and layer_index % 4 == 3
                    else "linear_attention"
                    if layer_index is not None
                    else "global"
                ),
                "kernel_hash": embedded_kernel["kernel_hash"],
                "source_kernel_hash": record["kernel_hash"],
                "embedded_variant_substituted": substituted,
                "symbol": symbol,
                "grid": [int(item) for item in record["grid"]],
                "num_warps": int(embedded_metadata["num_warps"]),
                "warp_size": int(embedded_metadata["warp_size"]),
                "shared_memory_bytes": int(embedded_metadata["shared"]),
                "arguments": regular_arguments,
                "call_path": [
                    {
                        "function": frame["function"],
                        "source": frame["source"],
                        "line": int(frame["line"]),
                    }
                    for frame in record.get("caller_context", [])
                ],
            }
        )

    for layer_index in range(40):
        if args.phase == "decode":
            expected = FULL_SEQUENCE if layer_index % 4 == 3 else LINEAR_SEQUENCE
        else:
            expected = (
                PREFILL_FULL_SEQUENCE
                if layer_index % 4 == 3
                else qualified_prefill_linear
            )
        if by_layer[layer_index] != expected:
            raise RuntimeError(
                f"qualified {args.phase} sequence drift at layer {layer_index}: "
                f"{by_layer[layer_index]}"
            )
    expected_final = FINAL_SEQUENCE if args.phase == "decode" else PREFILL_FINAL_SEQUENCE
    if [item["symbol"] for item in schedule[-len(expected_final):]] != expected_final:
        raise RuntimeError(f"qualified {args.phase} final-logit AOT sequence drift")

    total_tensor_arguments = mapped_arguments + transient_arguments
    payload = {
        "schema": f"aima-amd395-qwen36/native-{args.phase}-aot-schedule/v1",
        "status": "qualified_schedule_contract_native_executor_wiring_pending",
        "source": {
            "engine_sha256": args.engine_sha256,
            "trace_sha256": sha256_file(trace_path),
            "tensor_registry_sha256": sha256_file(registry_path),
            "aot_manifest_sha256": sha256_file(manifest_path),
            "trace_records": len(records),
        },
        "request": {
            "context_tokens": args.context_tokens,
            "phase_tokens": 1 if args.phase == "decode" else args.context_tokens,
            "position_start": args.context_tokens if args.phase == "decode" else 0,
            "batch_size": 1,
            "gpu_arch": "gfx1151",
        },
        "closure": {
            "launch_count": len(schedule),
            "layer_launch_count": sum(len(value) for value in by_layer.values()),
            "final_logit_launch_count": len(expected_final),
            "linear_layer_count": 30,
            "full_attention_layer_count": 10,
            "linear_launches_per_layer": len(
                LINEAR_SEQUENCE if args.phase == "decode" else qualified_prefill_linear
            ),
            "full_attention_launches_per_layer": len(
                FULL_SEQUENCE if args.phase == "decode" else PREFILL_FULL_SEQUENCE
            ),
            "tensor_argument_count": total_tensor_arguments,
            "registry_mapped_tensor_arguments": mapped_arguments,
            "transient_tensor_arguments": transient_arguments,
            "registry_mapping_fraction": mapped_arguments / total_tensor_arguments,
            "transient_storage_slots": len(transient_ids),
            "raw_device_pointers_retained": False,
            "embedded_variant_substitution_launches": substituted_launches,
            "embedded_variant_substitutions": substituted_hashes,
        },
        "layer_templates": {
            "linear_attention": (
                LINEAR_SEQUENCE if args.phase == "decode" else qualified_prefill_linear
            ),
            "full_attention": (
                FULL_SEQUENCE if args.phase == "decode" else PREFILL_FULL_SEQUENCE
            ),
            "final_logits": expected_final,
        },
        "non_aot_boundaries": ([
            {
                "owner": "linear_attention_output_projection",
                "count_per_token": 30,
                "native_provider": "qualified native BF16 wvSplitK",
                "state": "provider_qualified_executor_wiring_pending",
            },
            {
                "owner": "shared_expert_activation_and_down_projection",
                "count_per_token": 40,
                "native_provider": "pointwise HIP plus qualified native BF16 wvSplitK",
                "state": "down_provider_qualified_pointwise_and_executor_wiring_pending",
            },
            {
                "owner": "full_attention_core_gate_and_output_projection",
                "count_per_token": 10,
                "native_provider": "pending native grouped attention plus projection wiring",
                "state": "pending",
            },
            {
                "owner": "residual_add_state_promotion_and_sampling",
                "count_per_token": 1,
                "native_provider": "pending small HIP kernels and host control",
                "state": "pending",
            },
            {
                "owner": "certified_lm_head_shortlist_and_exact_selection",
                "count_per_token": 1,
                "native_provider": "embedded int8 AOT first stage; certificate tail pending",
                "state": "pending",
            },
        ] if args.phase == "decode" else [
            {
                "owner": "embedding_lookup_and_rotary_tables",
                "native_provider": "native HIP pointwise kernels",
                "state": "pending_executor_wiring",
            },
            {
                "owner": "dense_prefill_projections",
                "native_provider": "native hipBLASLt BF16 GEMM",
                "state": "dominant_shape_provider_qualified_executor_wiring_pending",
            },
            {
                "owner": "full_attention_prefill_core",
                "count_per_request": 10,
                "native_provider": "native CK-compatible or hipBLASLt attention path",
                "state": "pending",
            },
            {
                "owner": "router_topk_and_selected_expert_metadata",
                "count_per_request": 40,
                "native_provider": "native HIP reductions plus embedded fused MoE code objects",
                "state": "pending",
            },
            {
                "owner": "prefill_state_promotion_prefix_snapshot_and_exact_lm_head_selection",
                "count_per_request": 1,
                "native_provider": "resident HIP state plus native certificate",
                "state": "pending",
            },
        ]),
        "schedule": schedule,
        "non_claims": [
            f"not_a_complete_native_{args.phase}_executor",
            (
                "not_a_prefill_schedule_contract"
                if args.phase == "decode"
                else "not_a_complete_native_prefill_executor"
            ),
            "not_full_context_matrix_coverage",
            "trace_overhead_is_not_performance_evidence",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "launch_count": len(schedule),
                "mapped": mapped_arguments,
                "transient": transient_arguments,
                "transient_slots": len(transient_ids),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
