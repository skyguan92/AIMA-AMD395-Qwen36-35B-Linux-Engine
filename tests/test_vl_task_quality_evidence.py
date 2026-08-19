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
    NATIVE_SCHEMA,
    validate_fixture_manifest,
    validate_reference_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-task-quality-v0.1.0"
FIXTURE_MANIFEST = FIXTURE_ROOT / "fixtures-manifest.json"
REFERENCE = ROOT / "benchmarks/results/vl-task-quality-reference-v0.1.0.json"
NATIVE = ROOT / "benchmarks/results/native-vl-task-quality-v0.1.0.json"
QUALIFIED_COMMIT = "c3ab45ad4d6215c7cfa5c6772c90e75e9d1db9cd"
REFERENCE_SHA256 = (
    "51b3d95e3ce420584d765350bfe6b73f76d5786a8d9d629cf7c3e69ac11b8bce"
)
NATIVE_QUALIFIED_COMMIT = "1c5f6387898d0ae37d06234c5930221fe0ec5404"
NATIVE_BINARY_SHA256 = (
    "8524beee2e98bb9d261bb00d6f1febefc980953d5d02c8e0b005f56c5ee98339"
)
NATIVE_SHA256 = (
    "4333e176ff9df0e3bb9c64b62d78a6559b4ca2a6ada8a500f2826288b5b9b249"
)


def assert_sidecar(test: unittest.TestCase, path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    test.assertEqual(
        path.with_name(path.name + ".sha256").read_text(encoding="utf-8"),
        f"{digest}  {path.name}\n",
    )


def assert_canonical_seal(
    test: unittest.TestCase, result: dict[str, object]
) -> None:
    canonical = {
        key: value for key, value in result.items() if key != "integrity"
    }
    canonical_bytes = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    integrity = result["integrity"]
    test.assertEqual(integrity["algorithm"], "sha256")
    test.assertEqual(
        integrity["canonical_payload_sha256"],
        hashlib.sha256(canonical_bytes).hexdigest(),
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
        cls.native_bytes = NATIVE.read_bytes()
        cls.native = json.loads(cls.native_bytes)

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

    def test_native_is_sealed_and_exactly_source_bound(self) -> None:
        result = self.native
        assert_sidecar(self, NATIVE)
        assert_canonical_seal(self, result)
        self.assertEqual(
            hashlib.sha256(self.native_bytes).hexdigest(), NATIVE_SHA256
        )
        self.assertEqual(result["schema"], NATIVE_SCHEMA)
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(
            result["source"]["commit"], NATIVE_QUALIFIED_COMMIT
        )
        self.assertEqual(
            result["build_info"]["source_commit"], NATIVE_QUALIFIED_COMMIT
        )
        self.assertEqual(result["binary"]["sha256"], NATIVE_BINARY_SHA256)
        for component in result["source"]["files"]:
            assert_component_current(self, component)
        for name in ("reference", "fixture_manifest"):
            assert_component_current(self, result["dependencies"][name])
        self.assertEqual(
            {
                name: result["dependencies"][name]["sha256"]
                for name in (
                    "fmha_provider",
                    "aotriton_runtime",
                    "aotriton_image",
                    "vision_attention_image",
                )
            },
            {
                "fmha_provider": (
                    "98e6c47c017837ab796e3ca2e8256740d1e9cb6ec2f460af45ee586cd5fb7bd1"
                ),
                "aotriton_runtime": (
                    "e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5"
                ),
                "aotriton_image": (
                    "0f3a6a2f9dee6620443ee2145ee1f8257bde65a378589952840d99bf3d485c10"
                ),
                "vision_attention_image": (
                    "e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c"
                ),
            },
        )

    def test_native_quality_and_generation_parity_pass_exactly(
        self,
    ) -> None:
        result = self.native
        matrix = result["matrix"]
        cases = matrix["cases"]
        self.assertEqual(tuple(case["case_id"] for case in cases), CASE_ORDER)
        self.assertEqual(matrix["required_cases"], 12)
        self.assertTrue(all(case["qualified"] for case in cases))
        self.assertTrue(
            all(all(case["qualification_checks"].values()) for case in cases)
        )
        self.assertEqual(matrix["exact_render_prompt_vectors"], "12/12")
        self.assertEqual(matrix["exact_generated_content"], "12/12")
        self.assertEqual(matrix["exact_output_token_vectors"], "12/12")
        self.assertEqual(matrix["exact_reference_usage"], "12/12")
        self.assertEqual(matrix["exact_reference_finish_reason"], "12/12")
        self.assertEqual(
            matrix["native_aggregate"], matrix["reference_aggregate"]
        )
        self.assertEqual(
            matrix["native_aggregate"]["image"]["score_millionths"],
            1_000_000,
        )
        self.assertEqual(
            matrix["native_aggregate"]["video"]["score_millionths"],
            947_368,
        )
        self.assertTrue(
            all(
                all(case["parity_diagnostics"].values())
                for case in cases
            )
        )
        self.assertTrue(all(result["launch"]["checks"].values()))
        self.assertEqual(
            result["launch"]["stopped"],
            {"event": "stopped", "model_loads": 1, "served": 12},
        )
        self.assertEqual(result["raw"]["stderr"]["bytes"], 0)
        decision = result["decision"]
        self.assertTrue(decision["twelve_task_quality_cases_qualified"])
        self.assertTrue(decision["twelve_render_prompt_vectors_exact"])
        self.assertTrue(decision["image_task_quality_not_below_reference"])
        self.assertTrue(decision["video_task_quality_not_below_reference"])
        self.assertTrue(decision["single_resident_model_load"])
        self.assertTrue(decision["twelve_long_greedy_cases_reference_exact"])
        self.assertTrue(decision["twelve_output_token_vectors_exact"])
        for runtime in ("python", "torch", "triton", "vllm"):
            self.assertFalse(decision[f"runtime_{runtime}"])

    def test_evidence_is_public_and_does_not_promote_product_gates(self) -> None:
        for payload, result in (
            (self.reference_bytes, self.reference),
            (self.native_bytes, self.native),
        ):
            serialized = payload.decode("utf-8")
            for prefix in ("/home/", "/Users/", "/data/", "/tmp/"):
                self.assertNotIn(prefix, serialized)
            for gate in (
                "g1_passed",
                "g2_passed",
                "g3_passed",
                "g4_passed",
                "g5_passed",
            ):
                self.assertFalse(result["decision"][gate])


if __name__ == "__main__":
    unittest.main()
