#!/usr/bin/env python3
"""Tests for explicit G4 fixed-reference availability partitioning."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from aima_engine.vl_reference import seal_manifest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify-vl-performance-reference-availability.py"
SPEC = importlib.util.spec_from_file_location("vl_reference_availability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
availability_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(availability_module)


def matrix() -> dict:
    cells = [
        {
            "cell_id": "short",
            "cache_process": "disabled",
            "coverage": {"context": ["short"]},
        },
        {
            "cell_id": "long",
            "cache_process": "disabled",
            "coverage": {"context": ["128k"]},
        },
        {
            "cell_id": "cache-a",
            "cache_process": "enabled",
            "coverage": {"cache": ["cold_media"]},
        },
    ]
    return seal_manifest(
        {
            "complete": True,
            "required_coverage": {
                "context": ["short", "128k"],
                "cache": ["cold_media"],
            },
            "observed_coverage": {
                "context": ["128k", "short"],
                "cache": ["cold_media"],
            },
            "process_groups": [
                {
                    "process_group": "disabled",
                    "balanced_orders": [
                        ["short", "long"],
                        ["long", "short"],
                    ],
                },
                {
                    "process_group": "enabled",
                    "balanced_orders": [["cache-a"], ["cache-a"]],
                    "ordered_cache_sequence": ["A1"],
                },
            ],
            "cells": cells,
        }
    )


def unavailable(cell_id: str = "long", *, decision: str | None = None) -> dict:
    return seal_manifest(
        {
            "complete": True,
            "cells": [
                {
                    "cell_id": cell_id,
                    "status": "reference_unavailable",
                    "performance_decision": decision
                    or "not_comparable_not_candidate_pass",
                    "complete": True,
                    "coverage": {"context": ["128k"]},
                }
            ],
        }
    )


class VlPerformanceReferenceAvailabilityTest(unittest.TestCase):
    def test_unavailable_cell_remains_in_full_status_ledger(self) -> None:
        result = availability_module.derive_comparable_matrix(
            matrix(), unavailable(), {"bound": True}, "2026-08-19T00:00:00Z"
        )
        self.assertTrue(result["complete"])
        self.assertEqual([cell["cell_id"] for cell in result["cells"]], [
            "short",
            "cache-a",
        ])
        self.assertEqual(result["derivation"]["full_cell_count"], 3)
        self.assertEqual(result["derivation"]["comparable_cell_count"], 2)
        self.assertEqual(
            result["reference_unavailable"][0]["performance_decision"],
            "not_comparable_not_candidate_pass",
        )
        self.assertEqual(
            result["process_groups"][0]["balanced_orders"],
            [["short"], ["short"]],
        )

    def test_unavailable_cache_sequence_cell_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "cannot truncate cache sequences"
        ):
            availability_module.derive_comparable_matrix(
                matrix(), unavailable("cache-a"), {}, "2026-08-19T00:00:00Z"
            )

    def test_unavailable_record_cannot_claim_candidate_pass(self) -> None:
        with self.assertRaisesRegex(ValueError, "partition is invalid"):
            availability_module.derive_comparable_matrix(
                matrix(),
                unavailable("long", decision="candidate_pass"),
                {},
                "2026-08-19T00:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
