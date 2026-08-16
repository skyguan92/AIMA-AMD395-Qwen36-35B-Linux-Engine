from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_task_quality import (
    CASE_ORDER,
    EOS_TOKEN_ID,
    MIN_REFERENCE_CASE_SCORE_MILLIONTHS,
    MIN_REFERENCE_MODALITY_SCORE_MILLIONTHS,
    validate_fixture_manifest,
    validate_reference_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-task-quality-v0.1.0"
FIXTURE_MANIFEST = FIXTURE_ROOT / "fixtures-manifest.json"
REFERENCE = ROOT / "benchmarks/results/vl-task-quality-reference-v0.1.0.json"
QUALIFIED_COMMIT = "c3ab45ad4d6215c7cfa5c6772c90e75e9d1db9cd"
REFERENCE_SHA256 = (
    "51b3d95e3ce420584d765350bfe6b73f76d5786a8d9d629cf7c3e69ac11b8bce"
)


def assert_sidecar(test: unittest.TestCase, path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    test.assertEqual(
        path.with_name(path.name + ".sha256").read_text(encoding="utf-8"),
        f"{digest}  {path.name}\n",
    )


def assert_component_current(test: unittest.TestCase, component: dict) -> None:
    path = ROOT / component["path"]
    test.assertTrue(path.is_file(), component["path"])
    test.assertEqual(path.stat().st_size, component["bytes"])
    test.assertEqual(
        hashlib.sha256(path.read_bytes()).hexdigest(), component["sha256"]
    )


class VlTaskQualityEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_payload = json.loads(FIXTURE_MANIFEST.read_bytes())
        cls.reference_bytes = REFERENCE.read_bytes()
        cls.reference = json.loads(cls.reference_bytes)

    def test_fixture_corpus_is_hash_bound_and_replayable(self) -> None:
        self.assertEqual(
            validate_fixture_manifest(self.fixture_payload, FIXTURE_ROOT), []
        )
        assert_sidecar(self, FIXTURE_MANIFEST)
        self.assertEqual(self.fixture_payload["encoder"]["backend"], "opencv")
        self.assertEqual(self.fixture_payload["encoder"]["version"], "4.13.0")
        self.assertEqual(len(self.fixture_payload["fixtures"]), len(CASE_ORDER))

    def test_reference_is_complete_sealed_and_source_bound(self) -> None:
        self.assertEqual(validate_reference_manifest(self.reference), [])
        assert_sidecar(self, REFERENCE)
        self.assertEqual(
            hashlib.sha256(self.reference_bytes).hexdigest(), REFERENCE_SHA256
        )
        self.assertTrue(self.reference["complete"])
        self.assertTrue(self.reference["qualified_for_native_replay"])
        self.assertFalse(self.reference["source"]["dirty"])
        self.assertEqual(self.reference["source"]["commit"], QUALIFIED_COMMIT)
        for component in self.reference["source"]["files"]:
            assert_component_current(self, component)
        for component in self.reference["bindings"].values():
            assert_component_current(self, component)

    def test_long_greedy_vectors_and_quality_floors_are_frozen(self) -> None:
        cases = self.reference["cases"]
        self.assertEqual(tuple(case["case_id"] for case in cases), CASE_ORDER)
        self.assertEqual(
            [case["render"]["prompt_tokens"] for case in cases],
            [225, 225, 220, 222, 227, 222, 497, 497, 496, 497, 496, 495],
        )
        self.assertEqual(
            [case["response"]["usage"]["completion_tokens"] for case in cases],
            [190, 192, 192, 192, 192, 169, 192, 192, 192, 192, 192, 192],
        )
        self.assertTrue(all(case["qualified"] for case in cases))
        self.assertTrue(
            all(all(case["qualification_checks"].values()) for case in cases)
        )
        self.assertTrue(
            all(
                case["score"]["score_millionths"]
                >= MIN_REFERENCE_CASE_SCORE_MILLIONTHS
                for case in cases
            )
        )
        self.assertEqual(
            sum(
                case["output_token_reconstruction"]["eos_appended"]
                for case in cases
            ),
            2,
        )
        for case in cases:
            if case["output_token_reconstruction"]["eos_appended"]:
                self.assertEqual(case["output_token_ids"][-1], EOS_TOKEN_ID)
        self.assertEqual(
            self.reference["aggregate"]["image"]["score_millionths"],
            1_000_000,
        )
        self.assertEqual(
            self.reference["aggregate"]["video"]["score_millionths"],
            947_368,
        )
        self.assertTrue(
            all(
                value["score_millionths"]
                >= MIN_REFERENCE_MODALITY_SCORE_MILLIONTHS
                for value in self.reference["aggregate"].values()
            )
        )

    def test_reference_is_public_and_does_not_promote_product_gates(self) -> None:
        serialized = self.reference_bytes.decode("utf-8")
        for prefix in ("/home/", "/Users/", "/data/", "/tmp/aima-native"):
            self.assertNotIn(prefix, serialized)
        for gate in (
            "g1_passed",
            "g2_passed",
            "g3_passed",
            "g4_passed",
            "g5_passed",
        ):
            self.assertFalse(self.reference["decision"][gate])


if __name__ == "__main__":
    unittest.main()
