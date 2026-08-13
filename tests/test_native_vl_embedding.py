#!/usr/bin/env python3
"""Contracts for native visual-row injection into language embeddings."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE_MANIFEST = ROOT / "benchmarks/results/vl-oracle-manifest.json"
ORACLE_ROOT = ROOT / "benchmarks/oracles/vl-v0.1.0"
QUALIFICATION_RESULT = ROOT / "benchmarks/results/native-vl-embedding-v0.1.0.json"
IMAGE_PAD_TOKEN = 248056
VIDEO_PAD_TOKEN = 248057
HIDDEN_ROW_BYTES = 2048 * 2


class NativeVlEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(ORACLE_MANIFEST.read_text(encoding="utf-8"))

    def test_native_plan_and_gpu_scatter_are_product_sources(self) -> None:
        header = (
            ROOT / "native/include/aima/native_vl_embedding.h"
        ).read_text(encoding="utf-8")
        plan = (ROOT / "native/src/native_vl_embedding.cpp").read_text(
            encoding="utf-8"
        )
        gpu = (ROOT / "native/src/native_vl_embedding.hip.cpp").read_text(
            encoding="utf-8"
        )
        runtime = (ROOT / "scripts/build-native-runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("class NativeVlEmbeddingPlan", header)
        self.assertIn("visual_embedding_offset", header)
        self.assertIn("orphan placeholder token", plan)
        self.assertIn("visual rows are not covered exactly once", plan)
        self.assertIn("native_vl_embedding_scatter_kernel", gpu)
        self.assertIn("launch_prompt_embeddings", gpu)
        self.assertIn("native_vl_embedding.cpp", runtime)
        self.assertIn("native_vl_embedding.hip.cpp", runtime)

    def test_frozen_masks_are_exactly_derivable_from_prompt_tokens(self) -> None:
        for case in self.manifest["cases"]:
            token_ids = case["processor"]["prompt_token_ids"]
            for modality, spans in case["processor"]["placeholders"].items():
                pad_token = (
                    IMAGE_PAD_TOKEN if modality == "image" else VIDEO_PAD_TOKEN
                )
                for span in spans:
                    derived = [
                        token_ids[span["offset"] + index] == pad_token
                        for index in range(span["length"])
                    ]
                    self.assertEqual(sum(derived), span["num_embeds"])
                    if "is_embed" in span:
                        self.assertEqual(derived, span["is_embed"])
                    else:
                        self.assertTrue(all(derived))

    def test_frozen_visual_rows_match_injected_prompt_rows_byte_exactly(self) -> None:
        for case in self.manifest["cases"]:
            injected_record = case["boundaries"]["injected_embeddings"]
            merger_record = case["boundaries"]["vision_merger"]
            injected = (ORACLE_ROOT / injected_record["path"]).read_bytes()
            merger = (ORACLE_ROOT / merger_record["path"]).read_bytes()
            token_ids = case["processor"]["prompt_token_ids"]
            positions: list[int] = []
            ordered_spans = sorted(
                (
                    (span["offset"], modality, span)
                    for modality, spans in case["processor"]["placeholders"].items()
                    for span in spans
                ),
                key=lambda item: item[0],
            )
            for _, modality, span in ordered_spans:
                pad_token = (
                    IMAGE_PAD_TOKEN if modality == "image" else VIDEO_PAD_TOKEN
                )
                positions.extend(
                    span["offset"] + index
                    for index in range(span["length"])
                    if token_ids[span["offset"] + index] == pad_token
                )
            selected = b"".join(
                injected[
                    position * HIDDEN_ROW_BYTES : (position + 1) * HIDDEN_ROW_BYTES
                ]
                for position in positions
            )
            self.assertEqual(len(selected), merger_record["bytes"])
            self.assertEqual(selected, merger, case["case_id"])

    def test_gpu_qualification_is_hash_bound_and_bit_exact(self) -> None:
        result = json.loads(QUALIFICATION_RESULT.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "ca92be35e16508ff3a469d9bd60c684c990d3502",
        )
        for record in result["source"]["files"]:
            self.assertEqual(
                hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest(),
                record["sha256"],
            )
        for dependency in result["dependencies"].values():
            self.assertEqual(
                hashlib.sha256((ROOT / dependency["path"]).read_bytes()).hexdigest(),
                dependency["sha256"],
            )
        run = result["qualification_run"]
        self.assertTrue(run["single_resident_weight_load"])
        self.assertEqual(len(run["cases"]), 5)
        self.assertEqual(
            {case["case_id"] for case in run["cases"]},
            {
                "image_local_png",
                "video_local_mp4",
                "multi_image",
                "multi_video",
                "mixed_image_video",
            },
        )
        for case in run["cases"]:
            self.assertEqual(case["elements"], case["exact_elements"])
            self.assertEqual(case["expected_sha256"], case["actual_sha256"])
            self.assertTrue(case["repeat_deterministic"])
            self.assertEqual(case["relative_l2_error"], 0.0)
            self.assertEqual(case["cosine_similarity"], 1.0)
        decision = result["decision"]
        self.assertEqual(
            decision["total_elements"], decision["total_exact_elements"]
        )
        self.assertTrue(decision["injected_embedding_boundary_qualified"])
        self.assertFalse(decision["mrope_positions_qualified"])
        self.assertFalse(decision["language_boundaries_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])


if __name__ == "__main__":
    unittest.main()
