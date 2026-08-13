#!/usr/bin/env python3
"""Contracts for native Qwen3-VL M-RoPE positions and evidence."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/native-vl-mrope-v0.1.0.json"


class NativeMropeTest(unittest.TestCase):
    def test_native_mrope_is_wired_into_the_product_build(self) -> None:
        header = (ROOT / "native/include/aima/native_mrope.h").read_text(
            encoding="utf-8"
        )
        source = (ROOT / "native/src/native_mrope.cpp").read_text(
            encoding="utf-8"
        )
        runtime = (ROOT / "scripts/build-native-runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("class NativeMropePlan", header)
        self.assertIn("native_mrope_decode_position", header)
        self.assertIn("append_grid_positions", source)
        self.assertIn("video pad tokens are not contiguous", source)
        self.assertIn("grid size overflows", source)
        self.assertIn("native_mrope.cpp", runtime)

    def test_mrope_qualification_is_integer_exact_and_hash_bound(self) -> None:
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "ad21b570a67bcc8a443d004bd227232495adc8e3",
        )
        for record in result["source"]["files"]:
            self.assertEqual(
                hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest(),
                record["sha256"],
            )
        for record in (
            result["reference"]["reference_manifest"],
            result["reference"]["full_model_oracle"],
        ):
            self.assertEqual(
                hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest(),
                record["sha256"],
            )
        self.assertEqual(len(result["qualification_run"]["cases"]), 5)
        for case in result["qualification_run"]["cases"]:
            self.assertEqual(case["elements"], case["exact_elements"])
            self.assertEqual(
                case["expected_position_delta"], case["actual_position_delta"]
            )
            self.assertEqual(case["expected_sha256"], case["actual_sha256"])
        decision = result["decision"]
        self.assertEqual(
            decision["total_elements"], decision["total_exact_elements"]
        )
        self.assertTrue(decision["prompt_mrope_boundary_qualified"])
        self.assertTrue(decision["decode_position_formula_implemented"])
        self.assertFalse(decision["language_rotary_consumption_qualified"])
        self.assertFalse(decision["language_layer_0_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])


if __name__ == "__main__":
    unittest.main()
