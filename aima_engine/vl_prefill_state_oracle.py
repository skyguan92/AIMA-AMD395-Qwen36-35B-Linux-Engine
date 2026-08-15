"""Contracts for fixed-vLLM VL prefill-to-decode state oracles."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aima_engine.vl_generation_oracle import CASE_ORDER
from aima_engine.vl_oracle import verify_raw_tensor
from aima_engine.vl_reference import sha256_file, verify_manifest_integrity


VL_PREFILL_STATE_ORACLE_SCHEMA = (
    "aima-amd395-qwen36/vl-prefill-state-oracle/v1"
)
LINEAR_LAYER_INDICES = tuple(layer for layer in range(40) if layer % 4 != 3)
CONV_STATE_SHAPE = (8_192, 3)
RECURRENT_STATE_SHAPE = (32, 128, 128)
STATE_COMPONENT_NAMES = tuple(
    name
    for layer in LINEAR_LAYER_INDICES
    for name in (
        f"layer_{layer:03d}_conv_state",
        f"layer_{layer:03d}_recurrent_state",
    )
)


def _validate_component(
    errors: list[str], case_id: str, name: str, component: Any,
    oracle_root: Path | None,
) -> None:
    if not isinstance(component, dict):
        errors.append(f"VL prefill state component is missing: {case_id}/{name}")
        return
    recurrent = name.endswith("_recurrent_state")
    expected_shape = list(
        RECURRENT_STATE_SHAPE if recurrent else CONV_STATE_SHAPE
    )
    expected_dtype = "torch.float32" if recurrent else "torch.bfloat16"
    element_size = 4 if recurrent else 2
    expected_bytes = element_size
    for value in expected_shape:
        expected_bytes *= value
    if component.get("shape") != expected_shape:
        errors.append(f"VL prefill state shape changed: {case_id}/{name}")
    if component.get("dtype") != expected_dtype:
        errors.append(f"VL prefill state dtype changed: {case_id}/{name}")
    if component.get("bytes") != expected_bytes:
        errors.append(f"VL prefill state byte count changed: {case_id}/{name}")
    if oracle_root is not None:
        errors.extend(verify_raw_tensor(component, oracle_root))


def validate_vl_prefill_state_oracle_manifest(
    payload: Mapping[str, Any], *, oracle_root: Path | None = None
) -> list[str]:
    """Return every violation that makes prefill state attribution unsafe."""

    errors: list[str] = []
    if payload.get("schema") != VL_PREFILL_STATE_ORACLE_SCHEMA:
        errors.append(
            f"VL prefill state oracle schema must be "
            f"{VL_PREFILL_STATE_ORACLE_SCHEMA}"
        )
    if payload.get("complete") is not True:
        errors.append("VL prefill state oracle is not complete")
    if payload.get("qualified_for_state_attribution") is not True:
        errors.append("VL prefill state oracle is not attribution-qualified")
    errors.extend(verify_manifest_integrity(payload))

    binding = payload.get("generation_oracle")
    if not isinstance(binding, dict) or not isinstance(
        binding.get("sha256"), str
    ):
        errors.append("VL prefill state oracle has no generation binding")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        return errors + ["VL prefill state oracle cases must be an array"]
    case_ids = [
        case.get("case_id") if isinstance(case, dict) else None
        for case in cases
    ]
    if tuple(case_ids) != CASE_ORDER:
        errors.append("VL prefill state oracle case order changed")
    for case in cases:
        if not isinstance(case, dict) or case.get("case_id") not in CASE_ORDER:
            errors.append("VL prefill state oracle contains a malformed case")
            continue
        case_id = case["case_id"]
        if case.get("passed") is not True:
            errors.append(f"VL prefill state oracle case did not pass: {case_id}")
        if case.get("capture_decode_call") != 1:
            errors.append(f"VL prefill state capture point changed: {case_id}")
        components = case.get("components")
        if not isinstance(components, dict):
            errors.append(f"VL prefill state components are missing: {case_id}")
            continue
        if set(components) != set(STATE_COMPONENT_NAMES):
            errors.append(f"VL prefill state component set changed: {case_id}")
        for name in STATE_COMPONENT_NAMES:
            _validate_component(
                errors, case_id, name, components.get(name), oracle_root
            )

        ledger = case.get("oracle_jsonl")
        if not isinstance(ledger, dict):
            errors.append(f"VL prefill state ledger is missing: {case_id}")
        elif oracle_root is not None:
            relative = ledger.get("path")
            if not isinstance(relative, str) or not relative:
                errors.append(f"VL prefill state ledger path is missing: {case_id}")
            else:
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts:
                    errors.append(f"VL prefill state ledger path is unsafe: {case_id}")
                else:
                    resolved = oracle_root / path
                    if not resolved.is_file():
                        errors.append(f"VL prefill state ledger is absent: {case_id}")
                    else:
                        if ledger.get("bytes") != resolved.stat().st_size:
                            errors.append(
                                f"VL prefill state ledger size changed: {case_id}"
                            )
                        if ledger.get("sha256") != sha256_file(resolved):
                            errors.append(
                                f"VL prefill state ledger digest changed: {case_id}"
                            )

    decision = payload.get("decision")
    if not isinstance(decision, dict):
        errors.append("VL prefill state oracle decision is missing")
    else:
        for name in (
            "two_prompt_prefixes_exact",
            "two_prefill_state_sets_captured",
        ):
            if decision.get(name) is not True:
                errors.append(f"VL prefill state oracle decision failed: {name}")
        for gate in (
            "g1_passed",
            "g2_passed",
            "g3_passed",
            "g4_passed",
            "g5_passed",
        ):
            if decision.get(gate) is not False:
                errors.append(f"VL prefill state oracle must not close {gate}")
    return errors
