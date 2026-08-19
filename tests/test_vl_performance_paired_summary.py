#!/usr/bin/env python3
"""Tests for paired G4 VL cell qualification."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize-vl-performance-pairs.py"
SPEC = importlib.util.spec_from_file_location("vl_pairs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
vl_pairs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vl_pairs)


def record(index: int, vision_ratio: float = 1.01) -> dict:
    return {
        "pair_index": index,
        "benchmark_base": "image-typical-output1",
        "execution_order": (
            "reference candidate" if index % 2 else "candidate reference"
        ),
        "complete": True,
        "all_pair_checks": True,
        "template_sha256": "a" * 64,
        "request_summary": {"max_tokens": 1},
        "media": [{"sha256": "b" * 64}],
        "response_contract": {"content_sha256": "c" * 64},
        "hostname": "amd395",
        "candidate_runtime": "native-resident-q272",
        "candidate_startup_ms": 43_000 + index,
        "candidate_vision_warmup": {
            "completed": True,
            "patches": 1024,
            "visual_tokens": 256,
            "plan_cache_entries_at_ready": 1,
        },
        "candidate_request_plan_cache_hit": True,
        "candidate_request_plan_build_wall_ms": 0.0,
        "ratios": {
            "ttft_candidate_over_reference": 0.93,
            "total_candidate_over_reference": 0.94,
            "prefill_tps_candidate_over_reference": 1.03,
            "vision_tps_candidate_over_reference": vision_ratio,
        },
    }


class VlPerformancePairedSummaryTest(unittest.TestCase):
    def test_five_alternating_pairs_own_the_cell_decision(self) -> None:
        result = vl_pairs.aggregate_records(
            [record(index) for index in range(1, 6)], {"bound": True}
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertEqual(result["pair_count"], 5)
        self.assertAlmostEqual(
            result["paired_medians"][
                "vision_tps_candidate_over_reference"
            ],
            1.01,
        )
        self.assertTrue(
            result["gates"][
                "candidate_startup_median_lte_44_9_seconds"
            ]
        )

    def test_one_fast_pair_cannot_hide_a_median_regression(self) -> None:
        ratios = [1.20, 0.99, 0.98, 0.97, 0.96]
        result = vl_pairs.aggregate_records(
            [record(index, ratios[index - 1]) for index in range(1, 6)],
            {"bound": True},
        )
        self.assertTrue(result["complete"])
        self.assertFalse(result["qualified"])
        self.assertFalse(
            result["gates"]["vision_paired_median_gte_reference"]
        )

    def test_non_alternating_order_fails_closed(self) -> None:
        records = [record(index) for index in range(1, 6)]
        records[1]["execution_order"] = "reference candidate"
        result = vl_pairs.aggregate_records(records, {"bound": True})
        self.assertFalse(result["complete"])
        self.assertFalse(result["qualified"])


if __name__ == "__main__":
    unittest.main()
