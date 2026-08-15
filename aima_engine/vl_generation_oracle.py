"""Contracts for frozen VL generation-divergence logits oracles."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aima_engine.vl_oracle import (
    canonical_int_list_sha256,
    verify_raw_tensor,
)
from aima_engine.vl_reference import verify_manifest_integrity


GENERATION_ORACLE_SCHEMA = (
    "aima-amd395-qwen36/vl-generation-divergence-oracle/v1"
)
MODEL_VOCABULARY_SIZE = 248_320
CASE_ORDER = ("tool_forced_image", "tool_auto_image")
CASE_CONTRACTS: dict[str, dict[str, Any]] = {
    "tool_forced_image": {
        "max_tokens": 64,
        "divergence_output_index": 14,
        "reference_token_id": 1040,
        "previous_native_token_id": 1149,
        "completion_tokens": 28,
        "output_token_ids_sha256": (
            "4a66d37fbf7490d4c2f05c51bbed47c18444116955ef6f13f77d135765552490"
        ),
        "structured": True,
    },
    "tool_auto_image": {
        "max_tokens": 192,
        "divergence_output_index": 93,
        "reference_token_id": 52850,
        "previous_native_token_id": 4965,
        "completion_tokens": 139,
        "output_token_ids_sha256": (
            "83e5dbb3ffb5d026b21991407bd07e06d92fecc840061ae4e17abc9e84447b41"
        ),
        "structured": False,
    },
}


def validate_generation_oracle_manifest(
    payload: Mapping[str, Any], *, oracle_root: Path | None = None
) -> list[str]:
    """Return every violation that makes generation logits non-comparable."""

    errors: list[str] = []
    if payload.get("schema") != GENERATION_ORACLE_SCHEMA:
        errors.append(
            f"generation oracle schema must be {GENERATION_ORACLE_SCHEMA}"
        )
    if payload.get("complete") is not True:
        errors.append("generation oracle is not complete")
    if payload.get("qualified_for_native_generation_comparison") is not True:
        errors.append("generation oracle is not qualified for native comparison")
    errors.extend(verify_manifest_integrity(payload))

    cases = payload.get("cases")
    if not isinstance(cases, list):
        return errors + ["generation oracle cases must be an array"]
    case_ids = [
        case.get("case_id") if isinstance(case, dict) else None for case in cases
    ]
    if tuple(case_ids) != CASE_ORDER:
        errors.append("generation oracle case order changed")

    for case in cases:
        if not isinstance(case, dict) or case.get("case_id") not in CASE_CONTRACTS:
            errors.append("generation oracle contains a malformed case")
            continue
        case_id = case["case_id"]
        contract = CASE_CONTRACTS[case_id]
        if case.get("passed") is not True:
            errors.append(f"generation oracle case did not pass: {case_id}")
        if case.get("divergence_output_index") != contract[
            "divergence_output_index"
        ]:
            errors.append(f"generation divergence index changed: {case_id}")

        generation = case.get("generation")
        if not isinstance(generation, dict):
            errors.append(f"generation record is missing: {case_id}")
        else:
            token_ids = generation.get("output_token_ids")
            if not isinstance(token_ids, list) or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in token_ids
            ):
                errors.append(f"generation token IDs are malformed: {case_id}")
            else:
                digest = canonical_int_list_sha256(token_ids)
                if digest != generation.get("output_token_ids_sha256"):
                    errors.append(f"generation token digest mismatch: {case_id}")
                if digest != contract["output_token_ids_sha256"]:
                    errors.append(f"frozen generation token vector changed: {case_id}")
                if len(token_ids) != contract["completion_tokens"]:
                    errors.append(f"generation token count changed: {case_id}")
                target = contract["divergence_output_index"]
                if len(token_ids) > target and token_ids[target] != contract[
                    "reference_token_id"
                ]:
                    errors.append(f"generation divergence token changed: {case_id}")

        logits = case.get("reference_logits")
        if not isinstance(logits, dict):
            errors.append(f"reference logits are missing: {case_id}")
            continue
        component = logits.get("component")
        if not isinstance(component, dict):
            errors.append(f"reference logits component is missing: {case_id}")
        else:
            if component.get("shape") != [MODEL_VOCABULARY_SIZE]:
                errors.append(f"reference logits vocabulary changed: {case_id}")
            if component.get("dtype") != "torch.float32":
                errors.append(f"reference logits must be FP32: {case_id}")
            if oracle_root is not None:
                errors.extend(verify_raw_tensor(component, oracle_root))
        if logits.get("selected_token_id") != contract["reference_token_id"]:
            errors.append(f"selected reference token changed: {case_id}")
        if logits.get("captured_output_index") != contract[
            "divergence_output_index"
        ]:
            errors.append(f"captured reference logit step changed: {case_id}")
        top = logits.get("raw_top_tokens")
        if not isinstance(top, list) or not top:
            errors.append(f"reference top logits are missing: {case_id}")
        elif top[0].get("rank") != 1:
            errors.append(f"reference top logits are not ranked: {case_id}")
        elif (
            not contract["structured"]
            and top[0].get("token_id") != contract["reference_token_id"]
        ):
            errors.append(f"unconstrained reference top1 changed: {case_id}")

    decisions = payload.get("decision")
    if not isinstance(decisions, dict):
        errors.append("generation oracle decision is missing")
    else:
        for name in (
            "two_tool_generations_exact",
            "two_prompt_vectors_exact",
            "two_divergence_logits_captured",
        ):
            if decisions.get(name) is not True:
                errors.append(f"generation oracle decision failed: {name}")
        for gate in ("g1_passed", "g2_passed", "g3_passed", "g4_passed", "g5_passed"):
            if decisions.get(gate) is not False:
                errors.append(f"generation oracle must not close {gate}")
    return errors
