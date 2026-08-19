#!/usr/bin/env python3
"""Tests for the frozen pairwise G4 matrix and its generated requests."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from aima_engine.vl_reference import verify_manifest_integrity


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = (
    ROOT / "benchmarks/fixtures/vl-performance-v0.1.0/matrix.json"
)
GENERATOR_PATH = ROOT / "scripts/generate-vl-performance-matrix.py"
SPEC = importlib.util.spec_from_file_location("vl_performance_matrix", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VlPerformanceMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = load(MATRIX_PATH)

    def test_matrix_has_complete_required_surface_coverage(self) -> None:
        self.assertEqual(self.matrix["schema"], generator.SCHEMA)
        self.assertTrue(self.matrix["complete"])
        self.assertEqual(verify_manifest_integrity(self.matrix), [])
        self.assertEqual(len(self.matrix["cells"]), 23)
        self.assertEqual(
            {
                key: sorted(values)
                for key, values in self.matrix["required_coverage"].items()
            },
            self.matrix["observed_coverage"],
        )
        groups = {
            group["process_group"]: group
            for group in self.matrix["process_groups"]
        }
        self.assertEqual(len(groups["disabled"]["balanced_orders"][0]), 19)
        self.assertEqual(len(groups["enabled"]["balanced_orders"][0]), 4)
        self.assertEqual(
            groups["enabled"]["ordered_cache_sequence"],
            ["A1", "A2", "B", "A3"],
        )
        self.assertEqual(
            groups["disabled"]["balanced_orders"][1],
            list(reversed(groups["disabled"]["balanced_orders"][0])),
        )

    def test_every_cell_binds_one_exact_request_and_frozen_media(self) -> None:
        capability = load(ROOT / "benchmarks/results/vl-capability-manifest.json")
        envelope = load(
            ROOT / "benchmarks/results/vl-capability-envelope-v0.1.0.json"
        )
        capability_ids = {case["case_id"] for case in capability["cases"]}
        execution_ids = {cell["cell_id"] for cell in envelope["execution_cells"]}
        for cell in self.matrix["cells"]:
            request_path = ROOT / cell["request"]["path"]
            payload = load(request_path)
            self.assertEqual(digest(request_path), cell["request"]["sha256"])
            self.assertEqual(request_path.stat().st_size, cell["request"]["bytes"])
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["max_tokens"], cell["output_tokens"])
            self.assertEqual(
                generator.json.dumps(payload).count(generator.PROMPT_NONCE), 1
            )
            serialized = json.dumps(payload)
            self.assertEqual(
                serialized.count("file://${AIMA_VL_MEDIA_ROOT}/"),
                len(cell["media"]),
            )
            self.assertTrue(
                set(cell["source_execution_cells"]).issubset(execution_ids)
            )
            self.assertTrue(
                set(cell["source_capability_cases"]).issubset(capability_ids)
            )
            self.assertEqual(cell["prefix_cache_expectation"], "disabled")
            self.assertIsInstance(cell["prompt_nonce"], str)
            self.assertTrue(cell["prompt_nonce"])
            low, high = cell["expected_prompt_tokens_range"]
            self.assertLessEqual(low, high)
            self.assertLessEqual(cell["aggregate_visual_tokens"], high)

    def test_checked_in_matrix_reproduces_from_bound_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated = generator.build_matrix(
                load(ROOT / "benchmarks/results/vl-capability-manifest.json"),
                load(
                    ROOT
                    / "benchmarks/results/vl-capability-envelope-v0.1.0.json"
                ),
                load(
                    ROOT
                    / "benchmarks/fixtures/vl-envelope-v0.1.0/fixtures-manifest.json"
                ),
                requests_dir=Path(temporary),
                logical_requests_dir=(
                    "benchmarks/fixtures/vl-performance-v0.1.0/requests"
                ),
                bindings=self.matrix["bindings"],
                generated_at=self.matrix["generated_at"],
            )
        self.assertEqual(generated, self.matrix)

    def test_discrete_count_and_sampling_cells_keep_the_intended_boundary(self) -> None:
        by_id = {cell["cell_id"]: cell for cell in self.matrix["cells"]}
        self.assertEqual(
            by_id["image_count_max_q8k_output1"]["text_padding_tokens"],
            6_976,
        )
        self.assertEqual(
            by_id["video_count_max_q1k_output1"]["text_padding_tokens"],
            567,
        )
        for cell_id, padding in (
            ("video_sampling_max_q32k_output1", 19_404),
            ("video_sampling_clamp_q32k_output1", 19_348),
        ):
            cell = by_id[cell_id]
            self.assertEqual(cell["aggregate_visual_tokens"], 9_600)
            self.assertEqual(cell["text_padding_tokens"], padding)
            payload = load(ROOT / cell["request"]["path"])
            self.assertEqual(
                payload["media_io_kwargs"],
                {"video": {"num_frames": 768, "video_backend": "opencv"}},
            )

        self.assertEqual(
            {
                by_id[cell_id]["prompt_nonce"]
                for cell_id in (
                    "cache_a_cold_output1",
                    "cache_a_exact_output1",
                    "cache_a_restored_output1",
                )
            },
            {"cache-a"},
        )
        self.assertEqual(
            by_id["cache_b_cold_output1"]["prompt_nonce"], "cache-b"
        )


if __name__ == "__main__":
    unittest.main()
