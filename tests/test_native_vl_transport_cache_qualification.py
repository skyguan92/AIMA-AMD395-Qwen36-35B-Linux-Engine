from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_transport_cache import (
    DISABLED_REPLAY,
    ENABLED_REPLAY,
    REFERENCE_CASE_ORDER,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
REFERENCE = RESULTS / "vl-transport-cache-reference-v0.1.0.json"
NATIVE = RESULTS / "native-vl-transport-cache-v0.1.0.json"
QUALIFIED_COMMIT = "82fc48f7d4a0af1f1b30e9abfd26d78f73780715"
QUALIFIED_BINARY_SHA256 = (
    "246f2f9126e4bc905d3a49617f51311206ee9285f90311501a120ecdfbfbcf7c"
)


def assert_sealed(test: unittest.TestCase, path: Path, result: dict) -> None:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    test.assertEqual(
        path.with_name(path.name + ".sha256").read_text(encoding="utf-8"),
        f"{digest}  {path.name}\n",
    )
    canonical = {key: value for key, value in result.items() if key != "integrity"}
    canonical_bytes = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    test.assertEqual(result["integrity"]["algorithm"], "sha256")
    test.assertEqual(
        result["integrity"]["canonical_payload_sha256"],
        hashlib.sha256(canonical_bytes).hexdigest(),
    )


def assert_components_current(
    test: unittest.TestCase, components: list[dict]
) -> None:
    for component in components:
        path = ROOT / component["path"]
        test.assertEqual(path.stat().st_size, component["bytes"])
        test.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), component["sha256"]
        )


class NativeVlTransportCacheQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference_payload = REFERENCE.read_bytes()
        cls.native_payload = NATIVE.read_bytes()
        cls.reference = json.loads(cls.reference_payload)
        cls.native = json.loads(cls.native_payload)

    def test_reference_is_sealed_complete_and_exactly_bound(self) -> None:
        result = self.reference
        assert_sealed(self, REFERENCE, result)
        self.assertEqual(
            result["schema"],
            "aima-amd395-qwen36/vl-transport-cache-reference/v1",
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(result["source"]["commit"], QUALIFIED_COMMIT)
        assert_components_current(self, result["source"]["files"])
        self.assertEqual(
            tuple(case["case_id"] for case in result["cases"]),
            REFERENCE_CASE_ORDER,
        )
        self.assertTrue(all(case["qualified"] for case in result["cases"]))
        self.assertTrue(
            all(
                all(case["qualification_checks"].values())
                for case in result["cases"]
            )
        )
        self.assertTrue(all(result["qualification_checks"].values()))
        for decision in (
            "ten_reference_cases_frozen",
            "verified_https_frozen",
            "video_sampling_overrides_frozen",
            "same_url_content_mutation_frozen",
            "mixed_order_and_mutation_frozen",
        ):
            self.assertTrue(result["decision"][decision], decision)

    def test_native_replay_is_reference_exact_and_cache_invariant(self) -> None:
        result = self.native
        assert_sealed(self, NATIVE, result)
        self.assertEqual(
            result["schema"],
            "aima-amd395-qwen36/native-vl-transport-cache/v1",
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(result["source"]["commit"], QUALIFIED_COMMIT)
        self.assertEqual(result["build_info"]["source_commit"], QUALIFIED_COMMIT)
        self.assertEqual(result["binary"]["sha256"], QUALIFIED_BINARY_SHA256)
        assert_components_current(self, result["source"]["files"])
        assert_components_current(
            self,
            [
                result["dependencies"]["reference"],
                result["dependencies"]["fixture_manifest"],
            ],
        )
        self.assertEqual(
            result["dependencies"]["fmha_provider"]["sha256"],
            "98e6c47c017837ab796e3ca2e8256740d1e9cb6ec2f460af45ee586cd5fb7bd1",
        )
        self.assertEqual(
            result["dependencies"]["vision_attention_image"]["sha256"],
            "b709a058a77d61e14db73c1ff7d7f4c20859d997bec811cad7339d3e59223d00",
        )
        self.assertFalse(result["dependencies"]["test_ca"]["private_key_recorded"])

        reference_cases = {
            case["case_id"]: case for case in self.reference["cases"]
        }
        expected = {
            "cache_enabled": tuple(item[0] for item in ENABLED_REPLAY),
            "cache_disabled": tuple(item[0] for item in DISABLED_REPLAY),
        }
        for name, observation_ids in expected.items():
            run = result["runs"][name]
            cases = run["cases"]
            self.assertEqual(
                tuple(case["observation_id"] for case in cases), observation_ids
            )
            self.assertTrue(all(case["qualified"] for case in cases))
            self.assertTrue(
                all(
                    all(case["qualification_checks"].values())
                    for case in cases
                )
            )
            for case in cases:
                reference = reference_cases[case["reference_case_id"]]
                self.assertEqual(case["request"], reference["request"])
                self.assertEqual(case["status_code"], reference["status_code"])
            self.assertTrue(all(run["checks"].values()))
            self.assertEqual(run["stopped"]["model_loads"], 1)
            self.assertEqual(run["raw"]["stderr"]["bytes"], 0)

        self.assertEqual(result["runs"]["cache_enabled"]["stopped"]["served"], 16)
        self.assertEqual(result["runs"]["cache_disabled"]["stopped"]["served"], 10)
        self.assertTrue(
            all(result["qualification_checks"]["cache"].values())
        )
        self.assertTrue(
            all(result["qualification_checks"]["servers"].values())
        )
        for decision in (
            "all_observations_reference_exact",
            "verified_https_qualified",
            "video_sampling_cache_identity_qualified",
            "video_content_a_b_a_qualified",
            "mixed_order_mutation_qualified",
            "long_generation_usage_qualified",
            "cache_disabled_and_error_invariance_qualified",
            "two_resident_model_loads",
        ):
            self.assertTrue(result["decision"][decision], decision)

    def test_evidence_is_public_and_does_not_promote_product_gates(self) -> None:
        for payload, result in (
            (self.reference_payload, self.reference),
            (self.native_payload, self.native),
        ):
            serialized = payload.decode("utf-8")
            for prefix in ("/home/", "/Users/", "/data/", "/tmp/"):
                self.assertNotIn(f'"{prefix}', serialized)
            self.assertNotIn("PRIVATE KEY", serialized)
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
