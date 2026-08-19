#!/usr/bin/env python3
"""Tests for explicit G4 fixed-reference availability partitioning."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from aima_engine.vl_reference import seal_manifest
from aima_engine.vl_reference import verify_manifest_integrity


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify-vl-performance-reference-availability.py"
PUBLISHED_AVAILABILITY = (
    ROOT
    / "benchmarks/results/native-vl-performance-reference-availability-v0.1.0.json"
)
PUBLISHED_COMPARABLE_MATRIX = (
    ROOT / "benchmarks/fixtures/vl-performance-v0.1.0/comparable-matrix.json"
)
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
    def test_published_partition_is_integral_and_path_safe(self) -> None:
        availability = availability_module.load_json_object(
            PUBLISHED_AVAILABILITY
        )
        comparable = availability_module.load_json_object(
            PUBLISHED_COMPARABLE_MATRIX
        )
        self.assertEqual(verify_manifest_integrity(availability), [])
        self.assertEqual(verify_manifest_integrity(comparable), [])
        self.assertTrue(availability["complete"])
        self.assertTrue(comparable["complete"])
        self.assertEqual(availability["cell_count"], 3)
        self.assertEqual(comparable["derivation"]["comparable_cell_count"], 20)
        for path in (PUBLISHED_AVAILABILITY, PUBLISHED_COMPARABLE_MATRIX):
            payload = path.read_text(encoding="utf-8")
            for prefix in ("/home/", "/Users/", "/data/", "/tmp/aima-native"):
                self.assertNotIn(prefix, payload)

    def test_candidate_source_identity_requires_clean_full_commit(self) -> None:
        self.assertTrue(
            availability_module.clean_source_commit(
                "d9788179e7716b92ecc115907c50b19483b147fc"
            )
        )
        self.assertFalse(
            availability_module.clean_source_commit(
                "d9788179e7716b92ecc115907c50b19483b147fc-dirty"
            )
        )
        self.assertFalse(availability_module.clean_source_commit("d978817"))

    def test_candidate_closure_requires_exact_unique_files(self) -> None:
        files = [
            {"path": path, "bytes": 1, "sha256": "a" * 64}
            for path in sorted(availability_module.CANDIDATE_CLOSURE_PATHS)
        ]
        self.assertTrue(availability_module.exact_candidate_closure(files))
        files[-1] = dict(files[0])
        self.assertFalse(availability_module.exact_candidate_closure(files))

    def test_candidate_closure_rejects_invalid_hash_and_size(self) -> None:
        files = [
            {"path": path, "bytes": 1, "sha256": "a" * 64}
            for path in sorted(availability_module.CANDIDATE_CLOSURE_PATHS)
        ]
        files[0]["sha256"] = "not-a-sha256"
        self.assertFalse(availability_module.exact_candidate_closure(files))
        files[0]["sha256"] = "a" * 64
        files[0]["bytes"] = 0
        self.assertFalse(availability_module.exact_candidate_closure(files))

    def test_redacted_reference_log_metadata_is_required_and_safe(self) -> None:
        published = (
            'AIMA_REDACTED_LOG_METADATA {"schema":"'
            + availability_module.REDACTED_LOG_SCHEMA
            + '","source_bytes":42,"source_sha256":"'
            + "a" * 64
            + '","replacement_counts":{"private_home":1,'
            '"private_ipv4":1,"benchmark_root":1,"model_root":1}}\n'
            "HIP error: unspecified launch failure\nError code 719\n"
        )
        self.assertEqual(
            availability_module.redacted_log_metadata(published)["source_bytes"],
            42,
        )
        self.assertTrue(availability_module.public_log_safe(published))
        self.assertFalse(
            availability_module.public_log_safe("/ho" + "me/private/run")
        )
        self.assertFalse(
            availability_module.public_log_safe("192." + "168.10.20")
        )

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
