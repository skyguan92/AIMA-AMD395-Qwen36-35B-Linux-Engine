from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate-native-vl-g3-qualification.py"
RESULT = ROOT / "benchmarks/results/text-v151-nonregression-v0.1.0.json"
RESULT_SIDECAR = RESULT.with_name(RESULT.name + ".sha256")
PAIRED_MATRIX_ROOT = (
    ROOT
    / "benchmarks/runs/native-paired-text-matrix-20260821-50289f1-balanced6"
)
QUALIFIED_COMMIT = "50289f1cbae150997ca82bbc054635932a2721c3"
QUALIFIED_BINARY_SHA256 = (
    "4bf377135bafe4dd0d449dc2c8563fa727ed47414eb4c7c7221ecb7e631711d0"
)
BASELINE_COMMIT = "65c198415709dad6d046c247acab3dc9df2a95a0"
BASELINE_BINARY_SHA256 = (
    "a9f18771175757af080c8a1d8d7e3fb3906c9aa41b43a496686103b626f80262"
)


def load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "aima_native_vl_g3_qualification", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the G3 qualification generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NativeVlG3QualificationLogicTest(unittest.TestCase):
    def test_candidate_identity_is_runtime_configurable_and_validated(self) -> None:
        generator = load_generator_module()
        generator.configure_candidate_identity("c" * 40, "d" * 64)
        self.assertEqual(generator.EXPECTED_SOURCE_COMMIT, "c" * 40)
        self.assertEqual(generator.EXPECTED_BINARY_SHA256, "d" * 64)
        with self.assertRaises(ValueError):
            generator.configure_candidate_identity("not-a-commit", "d" * 64)
        with self.assertRaises(ValueError):
            generator.configure_candidate_identity("c" * 40, "not-a-hash")

    def test_empty_matrix_cannot_claim_all_cells_strict(self) -> None:
        generator = load_generator_module()
        checks, summary = generator.check_paired_matrix(
            {
                "complete": True,
                "qualified": True,
                "all_cells_pass": True,
                "text_request_path_idle": True,
                "engines": {
                    "candidate": {
                        "sha256": QUALIFIED_BINARY_SHA256,
                        "build_info": {"source_commit": QUALIFIED_COMMIT},
                    },
                    "baseline": {
                        "sha256": BASELINE_BINARY_SHA256,
                        "build_info": {"source_commit": BASELINE_COMMIT},
                    },
                },
                "cells": [],
                "q8192_startup": {
                    "complete": True,
                    "qualified": True,
                    "candidate_median_ms": 1.0,
                    "checks": {"candidate_at_most_44_90_seconds": True},
                },
            }
        )
        self.assertFalse(checks["exactly_nineteen_frozen_cells"])
        self.assertFalse(
            checks["all_cells_six_pair_order_balanced_strict_no_regression"]
        )
        self.assertEqual(summary["cell_count"], 0)


class NativeVlG3QualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = RESULT.read_bytes()
        cls.result = json.loads(cls.payload)

    def test_record_is_current_and_reproducible(self) -> None:
        completed = subprocess.run(
            ["python3", str(SCRIPT), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("native VL G3 qualification: PASS", completed.stdout)

    def test_sidecar_and_canonical_seal_are_exact(self) -> None:
        digest = hashlib.sha256(self.payload).hexdigest()
        self.assertEqual(
            RESULT_SIDECAR.read_text(encoding="utf-8"),
            f"{digest}  {RESULT.name}\n",
        )
        canonical = {
            key: value
            for key, value in self.result.items()
            if key != "integrity"
        }
        canonical_bytes = json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(self.result["integrity"]["algorithm"], "sha256")
        self.assertEqual(
            self.result["integrity"]["canonical_payload_sha256"],
            hashlib.sha256(canonical_bytes).hexdigest(),
        )

    def test_all_inputs_are_content_bound(self) -> None:
        for component in self.result["inputs"].values():
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                component["sha256"],
            )

    def test_exact_candidate_and_release_identities_are_bound(self) -> None:
        self.assertEqual(
            self.result["schema"],
            "aima-amd395-qwen36/text-v151-nonregression/v1",
        )
        self.assertTrue(self.result["complete"])
        self.assertTrue(self.result["qualified"])
        self.assertEqual(
            self.result["candidate"],
            {
                "source_commit": QUALIFIED_COMMIT,
                "binary_sha256": QUALIFIED_BINARY_SHA256,
            },
        )
        self.assertEqual(
            self.result["baseline"],
            {
                "release": "v1.5.1",
                "source_commit": BASELINE_COMMIT,
                "binary_sha256": BASELINE_BINARY_SHA256,
            },
        )
        self.assertTrue(all(self.result["cross_evidence_checks"].values()))

    def test_nine_context_correctness_and_exact_completion_pass(self) -> None:
        checks = self.result["checks"]["correctness"]
        self.assertTrue(all(checks.values()))
        summary = self.result["summaries"]["correctness"]
        self.assertEqual(
            summary["context_tokens"],
            [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 261632],
        )
        self.assertEqual(summary["case_count"], 9)
        self.assertEqual(summary["top1_matches"], 9)
        self.assertLess(summary["maximum_kl_divergence"], 0.005)
        self.assertEqual(summary["exact_completion_tokens"], 128)

    def test_mmlu_api_and_idle_path_gates_pass(self) -> None:
        self.assertTrue(all(self.result["checks"]["mmlu256"].values()))
        mmlu = self.result["summaries"]["mmlu256"]
        self.assertGreaterEqual(mmlu["correct"], 218)
        self.assertEqual(mmlu["invalid_answers"], 0)
        self.assertEqual(mmlu["prompt_token_hash_matches"], 256)

        self.assertTrue(all(self.result["checks"]["openai_features"].values()))
        features = self.result["summaries"]["openai_features"]
        self.assertEqual(features["served"], 14)
        self.assertEqual(features["model_loads"], 1)
        self.assertEqual(features["text_path_idle_requests"], 14)

    def test_prefix_startup_memory_and_nineteen_cells_pass(self) -> None:
        self.assertTrue(all(self.result["checks"]["product_surfaces"].values()))
        surfaces = self.result["summaries"]["product_surfaces"]
        self.assertGreaterEqual(surfaces["prefix_pair_count"], 5)
        self.assertGreaterEqual(
            surfaces["prefix_ttft_speedup_candidate_over_baseline"], 1.0
        )
        self.assertGreaterEqual(
            surfaces["prefix_decode_retention_candidate_over_baseline"], 1.0
        )
        self.assertLessEqual(
            surfaces["startup_command_to_ready_median_ms"], 44_900.0
        )

        self.assertTrue(all(self.result["checks"]["paired_text_matrix"].values()))
        matrix = self.result["summaries"]["paired_text_matrix"]
        self.assertEqual(matrix["cell_count"], 19)
        self.assertEqual(matrix["minimum_pair_count"], 6)
        self.assertGreaterEqual(
            matrix["minimum_prefill_tps_candidate_over_baseline"], 1.0
        )
        self.assertGreaterEqual(
            matrix["minimum_decode_tps_candidate_over_baseline"], 1.0
        )
        self.assertLessEqual(
            matrix["maximum_total_wall_candidate_over_baseline"], 1.0
        )

        self.assertTrue(all(self.result["checks"]["doctor"].values()))
        doctor = self.result["summaries"]["doctor"]
        self.assertEqual(doctor["vram_bytes"], 512 * 1024 * 1024)
        self.assertGreaterEqual(doctor["gtt_bytes"], 96 * 1024**3)
        self.assertEqual(doctor["model_shards"], 26)

    def test_only_g3_is_promoted_by_this_record(self) -> None:
        decision = self.result["decision"]
        self.assertTrue(decision["g3_text_product_no_regression"])
        for gate in (
            "g1_full_vl_functional_parity",
            "g2_vl_correctness_parity",
            "g4_native_vl_performance",
            "g5_native_release_product",
        ):
            self.assertFalse(decision[gate])

    def test_public_record_contains_no_private_machine_paths(self) -> None:
        serialized = self.payload.decode("utf-8")
        for prefix in ("/home/", "/Users/", "/data/", "/tmp/"):
            self.assertNotIn(f'"{prefix}', serialized)

    def test_paired_raw_tree_contains_no_private_machine_paths(self) -> None:
        self.assertTrue(PAIRED_MATRIX_ROOT.is_dir())
        files = [
            path for path in PAIRED_MATRIX_ROOT.rglob("*") if path.is_file()
        ]
        self.assertGreater(len(files), 0)
        for path in files:
            payload = path.read_bytes()
            with self.subTest(path=path.relative_to(ROOT)):
                for prefix in (b"/home/", b"/Users/", b"/data/", b"/tmp/"):
                    self.assertNotIn(prefix, payload)


if __name__ == "__main__":
    unittest.main()
