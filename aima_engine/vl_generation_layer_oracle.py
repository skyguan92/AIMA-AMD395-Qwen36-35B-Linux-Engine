"""Contracts for fixed-vLLM VL decode-boundary attribution oracles."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aima_engine.vl_generation_oracle import (
    CASE_CONTRACTS,
    CASE_ORDER,
    MODEL_VOCABULARY_SIZE,
)
from aima_engine.vl_oracle import verify_raw_tensor
from aima_engine.vl_reference import sha256_file, verify_manifest_integrity


GENERATION_LAYER_ORACLE_SCHEMA = (
    "aima-amd395-qwen36/vl-generation-layer-oracle/v4"
)
HIDDEN_SIZE = 2_048
FIRST_DECODE_LINEAR_OUTPUT_INDEX = 1
FULL_ATTENTION_LAYER = 3
BOUNDARY_NAMES = tuple(
    f"layer_{layer:03d}_output" for layer in range(40)
) + ("language_final_norm",)
LINEAR_ATTENTION_BOUNDARY_SPECS = {
    "input_norm": ([HIDDEN_SIZE], "torch.bfloat16", 2),
    "qkv_projection": ([8_192], "torch.bfloat16", 2),
    "z_projection": ([4_096], "torch.bfloat16", 2),
    "a_projection": ([32], "torch.bfloat16", 2),
    "b_projection": ([32], "torch.bfloat16", 2),
    "post_conv_mixed_qkv": ([8_192], "torch.bfloat16", 2),
    "conv_state_before": ([8_192, 3], "torch.bfloat16", 2),
    "conv_state_after": ([8_192, 3], "torch.bfloat16", 2),
    "recurrent_state_before": ([32, 128, 128], "torch.float32", 4),
    "recurrent_output": ([4_096], "torch.bfloat16", 2),
    "recurrent_state_after": ([32, 128, 128], "torch.float32", 4),
    "gated_norm": ([4_096], "torch.bfloat16", 2),
    "attention_output": ([HIDDEN_SIZE], "torch.bfloat16", 2),
}
LAYER0_TAIL_BOUNDARY_SPECS = {
    "attention_residual": ([HIDDEN_SIZE], "torch.bfloat16", 2),
    "post_attention_norm": ([HIDDEN_SIZE], "torch.bfloat16", 2),
    "shared_gate_logits": ([1], "torch.bfloat16", 2),
    "shared_gate_up_projection": ([1_024], "torch.bfloat16", 2),
    "shared_activation": ([512], "torch.bfloat16", 2),
    "shared_down_projection": ([HIDDEN_SIZE], "torch.bfloat16", 2),
    "shared_moe_output": ([HIDDEN_SIZE], "torch.bfloat16", 2),
    "router_logits": ([256], "torch.bfloat16", 2),
    "router_weights": ([8], "torch.float32", 4),
    "router_indices": ([8], "torch.int32", 4),
    "routed_gate_up_projection": ([8, 1_024], "torch.bfloat16", 2),
    "routed_activation": ([8, 512], "torch.bfloat16", 2),
    "routed_weighted_expert_outputs": (
        [8, HIDDEN_SIZE],
        "torch.bfloat16",
        2,
    ),
    "routed_moe_output": ([HIDDEN_SIZE], "torch.bfloat16", 2),
    "combined_moe_output": ([HIDDEN_SIZE], "torch.bfloat16", 2),
}
NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES = (
    "input_norm",
    "conv_state_before",
    "qkv_projection",
    "z_projection",
    "a_projection",
    "b_projection",
    "post_conv_mixed_qkv",
    "conv_state_after",
    "recurrent_state_before",
    "recurrent_output",
    "recurrent_state_after",
    "gated_norm",
    "attention_output",
)
FULL_ATTENTION_DECODE_COMPONENT_NAMES = (
    "query",
    "key_cache",
    "value_cache",
    "block_table",
    "sequence_lengths",
    "query_starts",
    "k_descale",
    "v_descale",
    "output",
)
FULL_ATTENTION_PROJECTION_COMPONENT_NAMES = (
    "qkv_projection",
    *FULL_ATTENTION_DECODE_COMPONENT_NAMES,
    "gated_attention",
    "projected_attention",
    "attention_residual",
    "post_attention_norm",
    *(
        name
        for name in LAYER0_TAIL_BOUNDARY_SPECS
        if name not in {"attention_residual", "post_attention_norm"}
    ),
)


def _validate_boundary_set(
    value: Any,
    *,
    case_id: str,
    label: str,
    expected_decode_call: int,
    expected_layer_index: int,
    specs: Mapping[str, tuple[list[int], str, int]],
    oracle_root: Path | None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"generation layer {label} boundaries are missing: {case_id}"]
    if value.get("target_decode_call") != expected_decode_call:
        errors.append(f"generation layer {label} target changed: {case_id}")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("layer_index") != (
        expected_layer_index
    ):
        errors.append(f"generation layer {label} index changed: {case_id}")
    components = value.get("components")
    if not isinstance(components, dict):
        return errors + [
            f"generation layer {label} components are missing: {case_id}"
        ]
    if set(components) != set(specs):
        errors.append(f"generation layer {label} component set changed: {case_id}")
    for name, (shape, dtype, element_size) in specs.items():
        component = components.get(name)
        if not isinstance(component, dict):
            continue
        elements = 1
        for dimension in shape:
            elements *= dimension
        if component.get("shape") != shape:
            errors.append(
                f"generation layer {label} shape changed: {case_id}/{name}"
            )
        if component.get("dtype") != dtype:
            errors.append(
                f"generation layer {label} dtype changed: {case_id}/{name}"
            )
        if component.get("bytes") != elements * element_size:
            errors.append(
                f"generation layer {label} byte count changed: {case_id}/{name}"
            )
        if oracle_root is not None:
            errors.extend(verify_raw_tensor(component, oracle_root))

    ledger = value.get("oracle_jsonl")
    if not isinstance(ledger, dict):
        errors.append(f"generation layer {label} ledger is missing: {case_id}")
    elif oracle_root is not None:
        relative = ledger.get("path")
        if not isinstance(relative, str) or not relative:
            errors.append(
                f"generation layer {label} ledger path is missing: {case_id}"
            )
        else:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                errors.append(
                    f"generation layer {label} ledger path is unsafe: {case_id}"
                )
            else:
                resolved = oracle_root / path
                if not resolved.is_file():
                    errors.append(
                        f"generation layer {label} ledger is absent: {case_id}"
                    )
                else:
                    if ledger.get("bytes") != resolved.stat().st_size:
                        errors.append(
                            f"generation layer {label} ledger size changed: {case_id}"
                        )
                    if ledger.get("sha256") != sha256_file(resolved):
                        errors.append(
                            f"generation layer {label} ledger digest changed: {case_id}"
                        )
    return errors


def _validate_full_attention_set(
    value: Any,
    *,
    case_id: str,
    label: str,
    expected_decode_call: int,
    oracle_root: Path | None,
) -> list[str]:
    """Validate the fixed singleton paged-attention capture geometry."""

    if not isinstance(value, dict):
        return [f"generation layer {label} boundaries are missing: {case_id}"]
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        return [f"generation layer {label} metadata is missing: {case_id}"]

    integer_fields = (
        "sequence_length",
        "block_size",
        "logical_blocks",
        "query_heads",
        "kv_heads",
        "head_size",
    )
    if any(
        not isinstance(metadata.get(name), int)
        or isinstance(metadata.get(name), bool)
        for name in integer_fields
    ):
        return [f"generation layer {label} metadata changed: {case_id}"]
    sequence_length = metadata["sequence_length"]
    block_size = metadata["block_size"]
    logical_blocks = metadata["logical_blocks"]
    if (
        sequence_length <= 0
        or block_size != 1_056
        or logical_blocks != (sequence_length + block_size - 1) // block_size
        or logical_blocks != 1
        or metadata["query_heads"] != 16
        or metadata["kv_heads"] != 2
        or metadata["head_size"] != 256
    ):
        return [f"generation layer {label} metadata changed: {case_id}"]

    specs = {
        "query": ([1, 16, 256], "torch.bfloat16", 2),
        "key_cache": (
            [logical_blocks, block_size, 2, 256],
            "torch.bfloat16",
            2,
        ),
        "value_cache": (
            [logical_blocks, block_size, 2, 256],
            "torch.bfloat16",
            2,
        ),
        "block_table": ([1, logical_blocks], "torch.int32", 4),
        "sequence_lengths": ([1], "torch.int32", 4),
        "query_starts": ([2], "torch.int32", 4),
        "k_descale": ([1, 2], "torch.float32", 4),
        "v_descale": ([1, 2], "torch.float32", 4),
        "output": ([1, 16, 256], "torch.bfloat16", 2),
    }
    return _validate_boundary_set(
        value,
        case_id=case_id,
        label=label,
        expected_decode_call=expected_decode_call,
        expected_layer_index=FULL_ATTENTION_LAYER,
        specs=specs,
        oracle_root=oracle_root,
    )


def validate_generation_layer_oracle_manifest(
    payload: Mapping[str, Any], *, oracle_root: Path | None = None
) -> list[str]:
    """Return every violation that makes decode boundaries non-comparable."""

    errors: list[str] = []
    if payload.get("schema") != GENERATION_LAYER_ORACLE_SCHEMA:
        errors.append(
            f"generation layer oracle schema must be {GENERATION_LAYER_ORACLE_SCHEMA}"
        )
    if payload.get("complete") is not True:
        errors.append("generation layer oracle is not complete")
    if payload.get("qualified_for_decode_attribution") is not True:
        errors.append("generation layer oracle is not attribution-qualified")
    errors.extend(verify_manifest_integrity(payload))

    binding = payload.get("generation_oracle")
    if not isinstance(binding, dict) or not isinstance(
        binding.get("sha256"), str
    ):
        errors.append("generation layer oracle has no generation binding")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        return errors + ["generation layer oracle cases must be an array"]
    case_ids = [
        case.get("case_id") if isinstance(case, dict) else None
        for case in cases
    ]
    if tuple(case_ids) != CASE_ORDER:
        errors.append("generation layer oracle case order changed")

    for case in cases:
        if not isinstance(case, dict) or case.get("case_id") not in CASE_CONTRACTS:
            errors.append("generation layer oracle contains a malformed case")
            continue
        case_id = case["case_id"]
        contract = CASE_CONTRACTS[case_id]
        if case.get("passed") is not True:
            errors.append(f"generation layer oracle case did not pass: {case_id}")
        if case.get("target_output_index") != contract["divergence_output_index"]:
            errors.append(f"generation layer target changed: {case_id}")
        if case.get("target_token_id") != contract["reference_token_id"]:
            errors.append(f"generation layer target token changed: {case_id}")
        if case.get("captured_logits_output_index") != contract[
            "divergence_output_index"
        ]:
            errors.append(f"generation layer logits step changed: {case_id}")
        if case.get("target_layer_decode_call") != contract[
            "divergence_output_index"
        ]:
            errors.append(f"generation layer decode call changed: {case_id}")
        if case.get("first_decode_logits_output_index") != (
            FIRST_DECODE_LINEAR_OUTPUT_INDEX
        ):
            errors.append(
                f"generation layer first-decode logits step changed: {case_id}"
            )
        for field, label in (
            ("target_logits_component", "target logits"),
            ("first_decode_logits_component", "first-decode logits"),
        ):
            logits_component = case.get(field)
            if not isinstance(logits_component, dict):
                errors.append(f"generation layer {label} are missing: {case_id}")
                continue
            if logits_component.get("shape") != [MODEL_VOCABULARY_SIZE]:
                errors.append(
                    f"generation layer {label} shape changed: {case_id}"
                )
            if logits_component.get("dtype") != "torch.float32":
                errors.append(
                    f"generation layer {label} dtype changed: {case_id}"
                )
            if logits_component.get("bytes") != MODEL_VOCABULARY_SIZE * 4:
                errors.append(
                    f"generation layer {label} byte count changed: {case_id}"
                )
            if oracle_root is not None:
                errors.extend(verify_raw_tensor(logits_component, oracle_root))
        target_logits_component = case.get("target_logits_component")
        if (
            isinstance(target_logits_component, dict)
            and target_logits_component.get("sha256")
            != case.get("target_logits_sha256")
        ):
            errors.append(f"generation layer target logits hash changed: {case_id}")

        components = case.get("components")
        if not isinstance(components, dict):
            errors.append(f"generation layer components are missing: {case_id}")
            continue
        if set(components) != set(BOUNDARY_NAMES):
            errors.append(f"generation layer component set changed: {case_id}")
        for name in BOUNDARY_NAMES:
            component = components.get(name)
            if not isinstance(component, dict):
                continue
            if component.get("shape") != [HIDDEN_SIZE]:
                errors.append(f"generation layer shape changed: {case_id}/{name}")
            if component.get("dtype") != "torch.bfloat16":
                errors.append(f"generation layer dtype changed: {case_id}/{name}")
            if component.get("bytes") != HIDDEN_SIZE * 2:
                errors.append(f"generation layer byte count changed: {case_id}/{name}")
            if oracle_root is not None:
                errors.extend(verify_raw_tensor(component, oracle_root))

        oracle_jsonl = case.get("oracle_jsonl")
        if not isinstance(oracle_jsonl, dict):
            errors.append(f"generation layer oracle ledger is missing: {case_id}")
        elif oracle_root is not None:
            relative = oracle_jsonl.get("path")
            if not isinstance(relative, str) or not relative:
                errors.append(f"generation layer oracle ledger path is missing: {case_id}")
            else:
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts:
                    errors.append(
                        f"generation layer oracle ledger path is unsafe: {case_id}"
                    )
                else:
                    resolved = oracle_root / path
                    if not resolved.is_file():
                        errors.append(
                            f"generation layer oracle ledger is absent: {case_id}"
                        )
                    else:
                        if oracle_jsonl.get("bytes") != resolved.stat().st_size:
                            errors.append(
                                f"generation layer oracle ledger size changed: {case_id}"
                            )
                        if oracle_jsonl.get("sha256") != sha256_file(resolved):
                            errors.append(
                                f"generation layer oracle ledger digest changed: {case_id}"
                            )

        errors.extend(
            _validate_boundary_set(
                case.get("linear_attention"),
                case_id=case_id,
                label="linear-attention",
                expected_decode_call=contract["divergence_output_index"],
                expected_layer_index=0,
                specs=LINEAR_ATTENTION_BOUNDARY_SPECS,
                oracle_root=oracle_root,
            )
        )
        errors.extend(
            _validate_full_attention_set(
                case.get("full_attention"),
                case_id=case_id,
                label="full-attention",
                expected_decode_call=contract["divergence_output_index"],
                oracle_root=oracle_root,
            )
        )
        errors.extend(
            _validate_full_attention_set(
                case.get("first_decode_full_attention"),
                case_id=case_id,
                label="first-decode full-attention",
                expected_decode_call=FIRST_DECODE_LINEAR_OUTPUT_INDEX,
                oracle_root=oracle_root,
            )
        )
        errors.extend(
            _validate_boundary_set(
                case.get("first_decode_linear_attention"),
                case_id=case_id,
                label="first-decode linear-attention",
                expected_decode_call=FIRST_DECODE_LINEAR_OUTPUT_INDEX,
                expected_layer_index=0,
                specs=LINEAR_ATTENTION_BOUNDARY_SPECS,
                oracle_root=oracle_root,
            )
        )
        errors.extend(
            _validate_boundary_set(
                case.get("layer0_tail"),
                case_id=case_id,
                label="layer-0 tail",
                expected_decode_call=contract["divergence_output_index"],
                expected_layer_index=0,
                specs=LAYER0_TAIL_BOUNDARY_SPECS,
                oracle_root=oracle_root,
            )
        )
        errors.extend(
            _validate_boundary_set(
                case.get("first_decode_layer0_tail"),
                case_id=case_id,
                label="first-decode layer-0 tail",
                expected_decode_call=FIRST_DECODE_LINEAR_OUTPUT_INDEX,
                expected_layer_index=0,
                specs=LAYER0_TAIL_BOUNDARY_SPECS,
                oracle_root=oracle_root,
            )
        )

    decision = payload.get("decision")
    if not isinstance(decision, dict):
        errors.append("generation layer oracle decision is missing")
    else:
        for name in (
            "two_target_prefixes_exact",
            "two_target_logits_bound",
            "two_decode_boundary_sets_captured",
            "two_layer0_linear_attention_boundary_sets_captured",
            "two_first_decode_layer0_linear_attention_boundary_sets_captured",
            "two_layer0_tail_boundary_sets_captured",
            "two_first_decode_layer0_tail_boundary_sets_captured",
            "two_routed_moe_stage_sets_captured",
            "two_layer3_unified_attention_sets_captured",
            "two_first_decode_layer3_unified_attention_sets_captured",
        ):
            if decision.get(name) is not True:
                errors.append(f"generation layer oracle decision failed: {name}")
        for gate in (
            "g1_passed",
            "g2_passed",
            "g3_passed",
            "g4_passed",
            "g5_passed",
        ):
            if decision.get(gate) is not False:
                errors.append(f"generation layer oracle must not close {gate}")
    return errors
