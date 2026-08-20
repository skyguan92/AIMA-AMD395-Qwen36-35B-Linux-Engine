#!/usr/bin/env python3
"""Validate one fresh-process paired run of the frozen G4 matrix."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA = "aima-amd395-qwen36/vl-performance-matrix-pair/v1"
PAIR_ID = re.compile(r"^(.+)\.pair-([1-9][0-9]*)$")
VISION_ATTENTION_IMAGE_SHA256 = (
    "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e"
)
DENSE_IMAGE_VISION_ATTENTION_IMAGE_SHA256 = (
    "e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c"
)
IMAGE_OPTIMIZED_VISION_ATTENTION_MIN_VISUAL_TOKENS = 256
IMAGE_OPTIMIZED_VISION_ATTENTION_MAX_VISUAL_TOKENS = 1024


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def expected_vision_attention_image_sha256s(
    cell: Mapping[str, Any], native_vl: Mapping[str, Any]
) -> list[str] | None:
    cache_hit = native_vl.get("vision_embedding_cache_hit")
    batch_count = native_vl.get("vision_batch_count")
    if cache_hit is True:
        return []
    if (
        cache_hit is not False
        or isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or batch_count <= 0
    ):
        return None
    visual_tokens = cell.get("aggregate_visual_tokens")
    media = cell.get("media")
    image_optimized_batch = (
        not isinstance(visual_tokens, bool)
        and isinstance(visual_tokens, int)
        and IMAGE_OPTIMIZED_VISION_ATTENTION_MIN_VISUAL_TOKENS
        <= visual_tokens
        <= IMAGE_OPTIMIZED_VISION_ATTENTION_MAX_VISUAL_TOKENS
        and isinstance(media, list)
        and bool(media)
        and all(
            isinstance(item, Mapping) and item.get("modality") == "image"
            for item in media
        )
    )
    image_sha256 = (
        DENSE_IMAGE_VISION_ATTENTION_IMAGE_SHA256
        if image_optimized_batch
        else VISION_ATTENTION_IMAGE_SHA256
    )
    return [image_sha256] * batch_count


def diagnostic_module():
    path = Path(__file__).with_name("summarize-vl-performance-diagnostic.py")
    spec = importlib.util.spec_from_file_location("vl_pair_diagnostic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load VL diagnostic summarizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matrix_cells(matrix: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cells = matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("matrix cells must be a non-empty array")
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("matrix contains a malformed cell")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen:
            raise ValueError("matrix cell IDs must be unique strings")
        seen.add(cell_id)
        result.append(cell)
    return result


def group_record(
    root: Path, matrix: Mapping[str, Any], process_group: str, pair_index: int
) -> dict[str, Any]:
    group = next(
        (
            item
            for item in matrix.get("process_groups", [])
            if isinstance(item, Mapping)
            and item.get("process_group") == process_group
        ),
        None,
    )
    if not isinstance(group, Mapping):
        raise ValueError(f"matrix process group is missing: {process_group}")
    order_index = 0 if pair_index % 2 else 1
    balanced_orders = group.get("balanced_orders")
    if (
        not isinstance(balanced_orders, list)
        or len(balanced_orders) != 2
        or not isinstance(balanced_orders[order_index], list)
    ):
        raise ValueError(f"matrix orders are malformed: {process_group}")
    expected_order = list(balanced_orders[order_index])
    reference_order = (
        root / process_group / "reference/request-order.txt"
    ).read_text(encoding="utf-8").splitlines()
    candidate_order = (
        root / process_group / "candidate/request-order.txt"
    ).read_text(encoding="utf-8").splitlines()
    health = load_object(root / process_group / "candidate/health.json")
    return {
        "process_group": process_group,
        "media_cache": group.get("media_cache"),
        "prefix_cache": group.get("prefix_cache"),
        "expected_request_order": expected_order,
        "reference_request_order": reference_order,
        "candidate_request_order": candidate_order,
        "request_order_exact": (
            reference_order == candidate_order == expected_order
        ),
        "candidate_health": {
            "command_to_ready_wall_ms": health.get(
                "command_to_ready_wall_ms"
            ),
            "model_loads": health.get("model_loads"),
            "prefix_cache_entries": health.get("prefix_cache_entries"),
            "media_cache_capacity_bytes": health.get(
                "media_cache_capacity_bytes"
            ),
            "media_cache_capacity_entries": health.get(
                "media_cache_capacity_entries"
            ),
            "vision_warmup_completed": health.get(
                "vision_warmup_completed"
            ),
            "vision_warmup_patches": health.get("vision_warmup_patches"),
            "vision_warmup_visual_tokens": health.get(
                "vision_warmup_visual_tokens"
            ),
            "vision_image_count_warmup_patches": health.get(
                "vision_image_count_warmup_patches"
            ),
            "vision_image_count_warmup_visual_tokens": health.get(
                "vision_image_count_warmup_visual_tokens"
            ),
            "vision_plan_cache_entries_at_ready": health.get(
                "vision_plan_cache_entries_at_ready"
            ),
            "vision_attention_image_sha256": health.get(
                "vision_attention_image_sha256"
            ),
            "vision_dense_image_attention_image_sha256": health.get(
                "vision_dense_image_attention_image_sha256"
            ),
            "vision_warmup_plan_build_wall_ms": health.get(
                "vision_warmup_plan_build_wall_ms"
            ),
            "vision_warmup_encode_wall_ms": health.get(
                "vision_warmup_encode_wall_ms"
            ),
            "vision_image_count_warmup_plan_build_wall_ms": health.get(
                "vision_image_count_warmup_plan_build_wall_ms"
            ),
            "vision_image_count_warmup_encode_wall_ms": health.get(
                "vision_image_count_warmup_encode_wall_ms"
            ),
        },
    }


def cache_a_b_a_contract_exact(
    cells: list[Mapping[str, Any]], role: str
) -> bool:
    if role not in {"reference", "candidate"}:
        return False
    by_sequence: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        sequence = cell.get("cache_sequence")
        if isinstance(sequence, str):
            if sequence in by_sequence:
                return False
            by_sequence[sequence] = cell
    if set(by_sequence) != {"A1", "A2", "B", "A3"}:
        return False
    a_records = [by_sequence[name] for name in ("A1", "A2", "A3")]
    contracts = [record.get("contract") for record in a_records]
    audits = [record.get("response_audit") for record in a_records]
    if not all(isinstance(value, Mapping) for value in contracts + audits):
        return False
    templates = [value.get("template_sha256") for value in contracts]
    nonces = [value.get("prompt_nonce_sha256") for value in contracts]
    responses = [value.get(role) for value in audits]
    if not all(isinstance(value, Mapping) for value in responses):
        return False
    digest = responses[0].get("content_sha256")
    canonical_sha256 = re.compile(r"^[0-9a-f]{64}$")
    return (
        isinstance(templates[0], str)
        and canonical_sha256.fullmatch(templates[0]) is not None
        and templates.count(templates[0]) == len(templates)
        and isinstance(nonces[0], str)
        and canonical_sha256.fullmatch(nonces[0]) is not None
        and nonces.count(nonces[0]) == len(nonces)
        and isinstance(digest, str)
        and canonical_sha256.fullmatch(digest) is not None
        and all(response == responses[0] for response in responses)
    )


def cell_record(
    root: Path,
    cell: Mapping[str, Any],
    pair_index: int,
    diagnostic: Any,
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    process_group = str(cell["cache_process"])
    group_root = root / process_group
    relative = Path("requests") / f"{cell_id}.json"
    expectations = {
        "output_tokens": cell.get("output_tokens"),
        "prefix_cache_lookup": cell.get("prefix_cache_expectation"),
        "media_cache_mode": cell.get("media_cache_expectation"),
    }
    summary = diagnostic.build_summary(
        group_root,
        request_relative_path=relative,
        expectations=expectations,
    )
    reference = load_object(group_root / "reference" / relative)
    candidate = load_object(group_root / "candidate" / relative)
    benchmark_id = reference.get("benchmark_id")
    match = PAIR_ID.fullmatch(benchmark_id) if isinstance(benchmark_id, str) else None
    native = candidate.get("native_metrics")
    native_vl = native.get("vl") if isinstance(native, Mapping) else None
    prompt_tokens = summary.get("measurements", {}).get("reference", {}).get(
        "prompt_tokens"
    )
    prompt_range = cell.get("expected_prompt_tokens_range")
    prompt_in_range = (
        isinstance(prompt_range, list)
        and len(prompt_range) == 2
        and all(isinstance(value, int) for value in prompt_range)
        and isinstance(prompt_tokens, int)
        and prompt_range[0] <= prompt_tokens <= prompt_range[1]
    )
    logical_request = cell.get("request")
    captured_template = reference.get("request", {}).get("template_sha256")
    template_bound = (
        isinstance(logical_request, Mapping)
        and captured_template == logical_request.get("sha256")
        and candidate.get("request", {}).get("template_sha256")
        == captured_template
    )
    prompt_nonce = cell.get("prompt_nonce")
    expected_prompt_nonce_sha256 = None
    if isinstance(prompt_nonce, str) and prompt_nonce:
        try:
            expected_prompt_nonce_sha256 = hashlib.sha256(
                prompt_nonce.encode("ascii")
            ).hexdigest()
        except UnicodeEncodeError:
            pass
    visual_tokens = (
        native_vl.get("visual_tokens") if isinstance(native_vl, Mapping) else None
    )
    candidate_attention_images = (
        native_vl.get("vision_attention_image_sha256s")
        if isinstance(native_vl, Mapping)
        else None
    )
    expected_attention_images = (
        expected_vision_attention_image_sha256s(cell, native_vl)
        if isinstance(native_vl, Mapping)
        else None
    )
    checks = {
        "diagnostic_complete": summary.get("complete") is True,
        "benchmark_cell_and_pair_exact": (
            match is not None
            and match.group(1) == cell_id
            and int(match.group(2)) == pair_index
            and candidate.get("benchmark_id") == benchmark_id
        ),
        "request_template_bound": template_bound,
        "prompt_context_bucket_exact": prompt_in_range,
        "visual_token_count_exact": visual_tokens
        == cell.get("aggregate_visual_tokens"),
        "candidate_vision_attention_dispatch_exact": (
            candidate_attention_images == expected_attention_images
            and expected_attention_images is not None
        ),
        "expected_output_completed": (
            reference.get("response", {})
            .get("usage", {})
            .get("completion_tokens")
            == cell.get("output_tokens")
            and candidate.get("response", {})
            .get("usage", {})
            .get("completion_tokens")
            == cell.get("output_tokens")
        ),
        "prompt_nonce_exact": (
            reference.get("request", {}).get("prompt_nonce_sha256")
            == candidate.get("request", {}).get("prompt_nonce_sha256")
            == expected_prompt_nonce_sha256
            and expected_prompt_nonce_sha256 is not None
        ),
    }
    return {
        "cell_id": cell_id,
        "pair_index": pair_index,
        "process_group": process_group,
        "cache_sequence": cell.get("cache_sequence"),
        "complete": all(checks.values()),
        "pair_qualified": summary.get("qualified") is True,
        "checks": checks,
        "diagnostic_checks": summary.get("checks"),
        "diagnostic_thresholds": summary.get("diagnostic_thresholds"),
        "comparisons": summary.get("comparisons"),
        "measurements": summary.get("measurements"),
        "contract": {
            "template_sha256": captured_template,
            "request_summary": reference.get("request", {}).get("summary"),
            "media": reference.get("request", {}).get("media"),
            "text_padding": reference.get("request", {}).get("text_padding"),
            "prompt_nonce_sha256": reference.get("request", {}).get(
                "prompt_nonce_sha256"
            ),
            "response": {
                "finish_reason": reference.get("response", {}).get(
                    "finish_reason"
                ),
                "usage": reference.get("response", {}).get("usage"),
            },
            "hostname": reference.get("host", {}).get("hostname"),
            "candidate_runtime": (
                native.get("runtime") if isinstance(native, Mapping) else None
            ),
        },
        "response_audit": summary.get("response_audit"),
        "candidate_cache": {
            "prefix_lookup": (
                native.get("prefix_cache", {}).get("lookup")
                if isinstance(native, Mapping)
                else None
            ),
            "media_hits": (
                native_vl.get("media_cache_hits")
                if isinstance(native_vl, Mapping)
                else None
            ),
            "media_misses": (
                native_vl.get("media_cache_misses")
                if isinstance(native_vl, Mapping)
                else None
            ),
            "media_entries": (
                native_vl.get("media_cache_entries")
                if isinstance(native_vl, Mapping)
                else None
            ),
            "vision_plan_cache_hit": (
                native_vl.get("vision_plan_cache_hit")
                if isinstance(native_vl, Mapping)
                else None
            ),
            "vision_attention_image_sha256s": candidate_attention_images,
            "vision_plan_build_wall_ms": (
                native_vl.get("vision_plan_build_wall_ms")
                if isinstance(native_vl, Mapping)
                else None
            ),
            "vision_embedding_cache_hit": (
                native_vl.get("vision_embedding_cache_hit")
                if isinstance(native_vl, Mapping)
                else None
            ),
            "vision_embedding_cache_entries": (
                native_vl.get("vision_embedding_cache_entries")
                if isinstance(native_vl, Mapping)
                else None
            ),
            "vision_embedding_cache_resident_bytes": (
                native_vl.get("vision_embedding_cache_resident_bytes")
                if isinstance(native_vl, Mapping)
                else None
            ),
            "vision_embedding_cache_capacity_bytes": (
                native_vl.get("vision_embedding_cache_capacity_bytes")
                if isinstance(native_vl, Mapping)
                else None
            ),
        },
    }


def build_summary(root: Path, matrix_path: Path) -> dict[str, Any]:
    matrix = load_object(matrix_path)
    copied_matrix = root / "matrix.json"
    if sha256_file(matrix_path) != sha256_file(copied_matrix):
        raise ValueError("pair matrix copy differs from the selected manifest")
    execution_order = (root / "execution-order.txt").read_text(
        encoding="utf-8"
    ).strip()
    benchmark_ids = []
    for process_group in ("disabled", "enabled"):
        request_dir = root / process_group / "reference/requests"
        benchmark_ids.extend(
            load_object(path).get("benchmark_id")
            for path in sorted(request_dir.glob("*.json"))
        )
    pair_indices = {
        int(match.group(2))
        for value in benchmark_ids
        if isinstance(value, str) and (match := PAIR_ID.fullmatch(value))
    }
    if len(pair_indices) != 1:
        raise ValueError("matrix pair contains inconsistent benchmark indices")
    pair_index = next(iter(pair_indices))
    groups = [
        group_record(root, matrix, process_group, pair_index)
        for process_group in ("disabled", "enabled")
    ]
    diagnostic = diagnostic_module()
    cells = [
        cell_record(root, cell, pair_index, diagnostic)
        for cell in matrix_cells(matrix)
    ]
    expected_order = (
        "reference candidate" if pair_index % 2 else "candidate reference"
    )
    checks = {
        "execution_order_alternates": execution_order == expected_order,
        "all_group_request_orders_exact": all(
            group["request_order_exact"] for group in groups
        ),
        "all_cells_complete": all(cell["complete"] for cell in cells),
        "candidate_runtime_native": all(
            str(cell["contract"]["candidate_runtime"]).startswith(
                "native-resident-q"
            )
            for cell in cells
        ),
        "candidate_prefix_cache_absent": all(
            group["candidate_health"]["prefix_cache_entries"] == 0
            for group in groups
        ),
        "candidate_vision_warmup_disclosed": all(
            group["candidate_health"]["vision_warmup_completed"] is True
            and group["candidate_health"]["vision_warmup_patches"] == 1024
            and group["candidate_health"]["vision_warmup_visual_tokens"] == 256
            and group["candidate_health"][
                "vision_image_count_warmup_patches"
            ]
            == 4096
            and group["candidate_health"][
                "vision_image_count_warmup_visual_tokens"
            ]
            == 1024
            and finite_number(
                group["candidate_health"][
                    "vision_image_count_warmup_plan_build_wall_ms"
                ]
            )
            and group["candidate_health"][
                "vision_image_count_warmup_plan_build_wall_ms"
            ]
            > 0.0
            and finite_number(
                group["candidate_health"][
                    "vision_image_count_warmup_encode_wall_ms"
                ]
            )
            and group["candidate_health"][
                "vision_image_count_warmup_encode_wall_ms"
            ]
            > 0.0
            and group["candidate_health"][
                "vision_plan_cache_entries_at_ready"
            ]
            == 2
            for group in groups
        ),
        "candidate_vision_attention_variants_bound": all(
            group["candidate_health"]["vision_attention_image_sha256"]
            == VISION_ATTENTION_IMAGE_SHA256
            and (
                group["candidate_health"][
                    "vision_dense_image_attention_image_sha256"
                ]
                == DENSE_IMAGE_VISION_ATTENTION_IMAGE_SHA256
            )
            for group in groups
        ),
        "candidate_startup_observed": all(
            finite_number(
                group["candidate_health"]["command_to_ready_wall_ms"]
            )
            is not None
            for group in groups
        ),
        "reference_cache_a_b_a_contract_exact": (
            cache_a_b_a_contract_exact(cells, "reference")
        ),
        "candidate_cache_a_b_a_contract_exact": (
            cache_a_b_a_contract_exact(cells, "candidate")
        ),
    }
    return {
        "schema": SCHEMA,
        "complete": all(checks.values()),
        "scope": "one alternating pair of every frozen G4 matrix cell",
        "pair_index": pair_index,
        "execution_order": execution_order,
        "matrix": {
            "path": str(matrix_path),
            "sha256": sha256_file(matrix_path),
            "integrity": matrix.get("integrity"),
            "cell_count": len(cells),
        },
        "checks": checks,
        "process_groups": groups,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.pair_dir.resolve()
    matrix = (args.matrix or (root / "matrix.json")).resolve()
    output = (args.output or (root / "summary.json")).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite matrix-pair summary: {output}")
    result = build_summary(root, matrix)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "complete": result["complete"],
                "pair_index": result["pair_index"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
