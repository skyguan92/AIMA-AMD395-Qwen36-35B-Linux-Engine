"""Contracts for fixed-vLLM VL decode-boundary attribution oracles."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aima_engine.vl_generation_oracle import CASE_CONTRACTS, CASE_ORDER
from aima_engine.vl_oracle import verify_raw_tensor
from aima_engine.vl_reference import sha256_file, verify_manifest_integrity


GENERATION_LAYER_ORACLE_SCHEMA = (
    "aima-amd395-qwen36/vl-generation-layer-oracle/v3"
)
HIDDEN_SIZE = 2_048
FIRST_DECODE_LINEAR_OUTPUT_INDEX = 1
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


def _validate_linear_attention_boundary_set(
    value: Any,
    *,
    case_id: str,
    label: str,
    expected_decode_call: int,
    oracle_root: Path | None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"generation layer {label} boundaries are missing: {case_id}"]
    if value.get("target_decode_call") != expected_decode_call:
        errors.append(f"generation layer {label} target changed: {case_id}")
    components = value.get("components")
    if not isinstance(components, dict):
        return errors + [
            f"generation layer {label} components are missing: {case_id}"
        ]
    if set(components) != set(LINEAR_ATTENTION_BOUNDARY_SPECS):
        errors.append(f"generation layer {label} component set changed: {case_id}")
    for name, (shape, dtype, element_size) in (
        LINEAR_ATTENTION_BOUNDARY_SPECS.items()
    ):
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
            _validate_linear_attention_boundary_set(
                case.get("linear_attention"),
                case_id=case_id,
                label="linear-attention",
                expected_decode_call=contract["divergence_output_index"],
                oracle_root=oracle_root,
            )
        )
        errors.extend(
            _validate_linear_attention_boundary_set(
                case.get("first_decode_linear_attention"),
                case_id=case_id,
                label="first-decode linear-attention",
                expected_decode_call=FIRST_DECODE_LINEAR_OUTPUT_INDEX,
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
