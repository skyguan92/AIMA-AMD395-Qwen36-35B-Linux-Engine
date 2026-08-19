#!/usr/bin/env python3
"""Tests for complete G4 paired-matrix aggregation."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize-vl-performance-matrix.py"
SPEC = importlib.util.spec_from_file_location("vl_performance_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def cell_spec(cell_id: str, output: int) -> dict:
    return {
        "cell_id": cell_id,
        "source_execution_cells": ["source"],
        "source_capability_cases": [],
        "coverage": {"output": [str(output)]},
        "context_bucket": "short",
        "output_tokens": output,
        "cache_process": "disabled",
        "media_cache_expectation": "disabled",
    }


def measured_cell(cell_id: str, index: int, output: int) -> dict:
    ratios = {
        "ttft_candidate_over_reference": 0.94,
        "total_candidate_over_reference": 0.95,
        "prefill_tps_candidate_over_reference": 1.03,
        "prefill_path_tps_candidate_over_reference": 1.02,
        "vision_tps_candidate_over_reference": 1.01,
        "cold_vision_path_tps_candidate_over_reference": 1.00,
    }
    if output > 1:
        ratios["decode_tps_candidate_over_reference"] = 1.04
    return {
        "cell_id": cell_id,
        "pair_index": index,
        "process_group": "disabled",
        "complete": True,
        "checks": {"cell": True},
        "diagnostic_checks": {"diagnostic": True},
        "comparisons": ratios,
        "contract": {
            "template_sha256": "a" * 64,
            "response": {"content_sha256": "b" * 64},
        },
    }


def pair(index: int) -> dict:
    cells = [
        measured_cell("short-output1", index, 1),
        measured_cell("long-output512", index, 512),
    ]
    return {
        "pair_dir": f"pair-{index}",
        "summary_sha256": "c" * 64,
        "pair_index": index,
        "execution_order": (
            "reference candidate" if index % 2 else "candidate reference"
        ),
        "complete": True,
        "matrix": {"sha256": "d" * 64},
        "process_groups": [
            {
                "process_group": name,
                "candidate_health": {"command_to_ready_wall_ms": 43_000 + index},
            }
            for name in ("disabled", "enabled")
        ],
        "cells": cells,
    }


class VlPerformanceMatrixSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = {
            "complete": True,
            "derivation": {"cartesian_product": False},
            "required_coverage": {"output": ["1", "512"]},
            "observed_coverage": {"output": ["1", "512"]},
            "cells": [
                cell_spec("short-output1", 1),
                cell_spec("long-output512", 512),
            ],
        }
        self.identity = {"matrix": {"sha256": "d" * 64}}

    def test_every_cell_median_and_decode_gate_qualify(self) -> None:
        result = summary.aggregate(
            [pair(index) for index in range(1, 6)],
            self.matrix,
            self.identity,
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertEqual(result["pair_count"], 5)
        long_cell = next(
            cell for cell in result["cells"] if cell["output_tokens"] == 512
        )
        self.assertTrue(
            long_cell["gates"]["decode_paired_median_gte_reference"]
        )

    def test_one_regressed_cell_blocks_the_complete_matrix(self) -> None:
        pairs = [pair(index) for index in range(1, 6)]
        for current in pairs[1:]:
            current["cells"][0]["comparisons"][
                "vision_tps_candidate_over_reference"
            ] = 0.98
        result = summary.aggregate(pairs, self.matrix, self.identity)
        self.assertTrue(result["complete"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["gates"]["every_cell_paired_median_qualified"])

    def test_reference_unavailable_cell_is_explicit_and_never_a_pass(self) -> None:
        comparable = dict(self.matrix)
        comparable["schema"] = summary.COMPARABLE_MATRIX_SCHEMA
        comparable["derivation"] = {
            "full_cell_count": 3,
            "comparable_cell_count": 2,
            "reference_unavailable_cell_count": 1,
        }
        comparable["checks"] = {
            "partition_covers_every_frozen_cell": True,
        }
        comparable["full_cell_status"] = [
            {"cell_id": "short-output1", "status": "comparable"},
            {"cell_id": "long-output512", "status": "comparable"},
            {"cell_id": "near-window", "status": "reference_unavailable"},
        ]
        comparable["reference_unavailable"] = [
            {
                "cell_id": "near-window",
                "status": "reference_unavailable",
                "performance_decision": "not_comparable_not_candidate_pass",
            }
        ]
        result = summary.aggregate(
            [pair(index) for index in range(1, 6)],
            comparable,
            self.identity,
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(
            result["decision"], "qualified_on_all_reference_available_cells"
        )
        self.assertFalse(result["all_frozen_cells_performance_compared"])
        self.assertEqual(result["reference_unavailable_cell_count"], 1)
        self.assertEqual(
            result["reference_unavailable"][0]["performance_decision"],
            "not_comparable_not_candidate_pass",
        )
        self.assertTrue(
            result["gates"][
                "every_comparable_cell_paired_median_qualified"
            ]
        )


if __name__ == "__main__":
    unittest.main()
