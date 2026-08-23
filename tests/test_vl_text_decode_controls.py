#!/usr/bin/env python3
"""Tests for same-boundary G4 text decode controls."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize-vl-text-decode-controls.py"
SPEC = importlib.util.spec_from_file_location("vl_text_controls", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
controls_summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls_summary)

CAPTURE_SCRIPT = ROOT / "scripts" / "capture-vl-text-decode-control.py"
CAPTURE_SPEC = importlib.util.spec_from_file_location(
    "vl_text_control_capture", CAPTURE_SCRIPT
)
assert CAPTURE_SPEC is not None and CAPTURE_SPEC.loader is not None
control_capture = importlib.util.module_from_spec(CAPTURE_SPEC)
CAPTURE_SPEC.loader.exec_module(control_capture)

VL_CAPTURE_SCRIPT = ROOT / "scripts" / "capture-vl-performance-request.py"
VL_CAPTURE_SPEC = importlib.util.spec_from_file_location(
    "vl_performance_capture", VL_CAPTURE_SCRIPT
)
assert VL_CAPTURE_SPEC is not None and VL_CAPTURE_SPEC.loader is not None
vl_capture = importlib.util.module_from_spec(VL_CAPTURE_SPEC)
VL_CAPTURE_SPEC.loader.exec_module(vl_capture)


CONTROL_SPECS = {
    "text_q1024_output512": {
        "control_id": "text_q1024_output512",
        "g4_cell_id": "image_portrait_q1k_output512",
        "expected_prompt_tokens": 1024,
        "expected_completion_tokens": 512,
    },
    "text_q1039_output1024": {
        "control_id": "text_q1039_output1024",
        "g4_cell_id": "video_typical_q1k_output1024",
        "expected_prompt_tokens": 1039,
        "expected_completion_tokens": 1024,
    },
    "text_q8236_output512": {
        "control_id": "text_q8236_output512",
        "g4_cell_id": "mixed_multi_turn_q8k_output512",
        "expected_prompt_tokens": 8236,
        "expected_completion_tokens": 512,
    },
}


def paired_run(
    index: int, text_decode_tps: float = 100.0, vl_decode_tps: float = 101.0
) -> dict:
    request_order = (
        list(CONTROL_SPECS)
        if index % 2
        else list(reversed(CONTROL_SPECS))
    )
    return {
        "run_index": index,
        "directory": f"paired-control-{index:02d}",
        "manifest_sha256": "a" * 64,
        "matrix_sha256": "9" * 64,
        "health_sha256": "b" * 64,
        "health_contract": {"context_capacity": 262144},
        "candidate_binary_sha256": "c" * 64,
        "server_pid": 1000 + index,
        "host": "test-host",
        "request_order": request_order,
        "pair_order": ["text", "vl"] if index % 2 else ["vl", "text"],
        "samples": {
            control_id: {
                "text": {
                    "client_decode_tokens_per_second": text_decode_tps,
                    "engine_decode_tokens_per_second": text_decode_tps + 0.1,
                    "content_sha256": "d" * 64,
                    "output_token_ids_sha256": "e" * 64,
                    "raw_sha256": "f" * 64,
                    "request_index": 2,
                },
                "vl": {
                    "client_decode_tokens_per_second": vl_decode_tps,
                    "engine_decode_tokens_per_second": vl_decode_tps + 0.1,
                    "content_sha256": "3" * 64,
                    "output_token_ids_sha256": "4" * 64,
                    "raw_sha256": "5" * 64,
                    "request_index": 3,
                },
            }
            for control_id in CONTROL_SPECS
        },
    }


class VlTextDecodeControlsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = {
            "source_commit": "6" * 40,
            "files": [
                {"path": "aima-engine-native", "sha256": "c" * 64}
            ],
        }

    def test_manifest_and_capture_use_the_exact_same_timing_boundary(self) -> None:
        manifest_path = (
            ROOT
            / "benchmarks/fixtures/vl-text-decode-control-v0.1.0/manifest.json"
        )
        manifest = controls_summary.load_object(manifest_path)
        indexed = controls_summary.manifest_controls(manifest, manifest_path)
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.assertEqual(
            manifest_path.with_name("manifest.json.sha256").read_text(
                encoding="utf-8"
            ),
            f"{digest}  manifest.json\n",
        )
        self.assertEqual(set(indexed), set(CONTROL_SPECS))
        self.assertEqual(
            manifest["protocol"]["timing_boundary"],
            control_capture.TIMING_BOUNDARY,
        )
        self.assertEqual(
            controls_summary.TIMING_BOUNDARY,
            control_capture.TIMING_BOUNDARY,
        )
        self.assertEqual(
            controls_summary.TIMING_BOUNDARY,
            vl_capture.TIMING_BOUNDARY,
        )
        self.assertIn("adjacent", manifest["protocol"]["pairing"])
        self.assertIn("odd", manifest["protocol"]["pair_order"])

    def test_five_exact_shape_controls_qualify_every_decode_cell(self) -> None:
        result = controls_summary.aggregate(
            [paired_run(index) for index in range(1, 6)],
            CONTROL_SPECS,
            self.candidate,
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(len(result["cells"]), 3)
        for cell in result["cells"]:
            self.assertEqual(cell["pair_count"], 5)
            self.assertAlmostEqual(
                cell["paired_median_vl_over_text_client_decode"], 1.01
            )
            self.assertTrue(cell["qualified"])

    def test_one_decode_cell_below_text_blocks_qualification(self) -> None:
        pairs = [paired_run(index) for index in range(1, 6)]
        for pair in pairs:
            pair["samples"]["text_q1039_output1024"]["vl"][
                "client_decode_tokens_per_second"
            ] = 99.0
        result = controls_summary.aggregate(
            pairs,
            CONTROL_SPECS,
            self.candidate,
        )
        self.assertTrue(result["complete"])
        self.assertFalse(result["qualified"])
        failed = [cell for cell in result["cells"] if not cell["qualified"]]
        self.assertEqual(
            [cell["control_id"] for cell in failed],
            ["text_q1039_output1024"],
        )

    def test_service_contract_drift_marks_result_incomplete(self) -> None:
        pairs = [paired_run(index) for index in range(1, 6)]
        pairs[-1]["health_contract"] = {"context_capacity": 8192}
        result = controls_summary.aggregate(
            pairs,
            CONTROL_SPECS,
            self.candidate,
        )
        self.assertFalse(result["complete"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["checks"]["same_native_service_contract"])


if __name__ == "__main__":
    unittest.main()
