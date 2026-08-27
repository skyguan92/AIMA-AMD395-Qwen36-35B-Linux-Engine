from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_g1_extension import (
    CASE_ORDER,
    response_content,
    usage_signature,
)
from tests.evidence_test_utils import assert_components_at_commit


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
REFERENCE = RESULTS / "vl-g1-mixed-conversation-reference-v0.1.0.json"
NATIVE = RESULTS / "native-vl-g1-extension-v0.1.0.json"
QUALIFIED_COMMIT = "1842c8f6d281d6c8e91563205cda3fb66908d8a1"
NATIVE_QUALIFIED_COMMIT = "bd012874027defa528279a357609b713e9069df4"
QUALIFIED_BINARY_SHA256 = (
    "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
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
            hashlib.sha256(path.read_bytes()).hexdigest(),
            component["sha256"],
        )


class NativeVlG1ExtensionQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference_payload = REFERENCE.read_bytes()
        cls.native_payload = NATIVE.read_bytes()
        cls.reference = json.loads(cls.reference_payload)
        cls.native = json.loads(cls.native_payload)

    def test_reference_is_sealed_complete_and_exactly_bound(self) -> None:
        result = self.reference
        assert_sealed(self, REFERENCE, result)
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(result["source"]["commit"], QUALIFIED_COMMIT)
        assert_components_at_commit(
            self, result["source"]["files"], QUALIFIED_COMMIT
        )
        assert_components_current(self, list(result["bindings"].values()))
        self.assertEqual(
            tuple(case["case_id"] for case in result["cases"]), CASE_ORDER
        )
        self.assertEqual(
            [case["render"]["prompt_tokens"] for case in result["cases"]],
            [240, 214, 160, 153, 155],
        )
        self.assertTrue(all(case["qualified"] for case in result["cases"]))
        self.assertTrue(
            all(
                all(case["qualification_checks"].values())
                for case in result["cases"]
            )
        )
        self.assertTrue(result["decision"]["five_reference_cases_accepted"])
        self.assertTrue(result["decision"]["mixed_multi_item_orders_frozen"])
        self.assertTrue(result["decision"]["video_and_mixed_history_frozen"])
        self.assertTrue(result["decision"]["mixed_sse_frozen"])

    def test_native_replay_is_reference_exact_and_single_resident(self) -> None:
        result = self.native
        assert_sealed(self, NATIVE, result)
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(result["source"]["commit"], NATIVE_QUALIFIED_COMMIT)
        self.assertEqual(
            result["build_info"]["source_commit"], NATIVE_QUALIFIED_COMMIT
        )
        self.assertEqual(result["binary"]["sha256"], QUALIFIED_BINARY_SHA256)
        assert_components_at_commit(
            self, result["source"]["files"], NATIVE_QUALIFIED_COMMIT
        )
        for name in ("reference", "fixture_manifest"):
            assert_components_current(self, [result["dependencies"][name]])
        self.assertEqual(
            result["dependencies"]["fmha_provider"]["sha256"],
            "e5336b2d66b36c5f17aeb07ab780fa8f60a6092910f9b01b3ebf4bc31f766bb4",
        )
        self.assertEqual(
            tuple(case["case_id"] for case in result["cases"]), CASE_ORDER
        )
        references = {
            case["case_id"]: case for case in self.reference["cases"]
        }
        for case in result["cases"]:
            reference = references[case["case_id"]]
            self.assertTrue(case["qualified"], case["case_id"])
            self.assertTrue(
                all(case["qualification_checks"].values()), case["case_id"]
            )
            self.assertEqual(case["request"], reference["request"])
            self.assertEqual(
                response_content(case["response"]),
                response_content(reference["response"]),
            )
            self.assertEqual(
                usage_signature(case["response"]),
                usage_signature(reference["response"]),
            )
        self.assertEqual(
            result["launch"]["stopped"],
            {"event": "stopped", "model_loads": 1, "served": 5},
        )
        self.assertTrue(all(result["launch"]["checks"].values()))
        self.assertEqual(result["raw"]["stderr"]["bytes"], 0)
        for decision in (
            "five_cases_reference_exact",
            "mixed_multi_item_orders_qualified",
            "video_and_mixed_history_qualified",
            "mixed_sse_qualified",
            "single_resident_model_load",
        ):
            self.assertTrue(result["decision"][decision], decision)

    def test_extension_evidence_is_public_and_does_not_promote_gates(self) -> None:
        for payload, result in (
            (self.reference_payload, self.reference),
            (self.native_payload, self.native),
        ):
            serialized = payload.decode("utf-8")
            for prefix in ("/home/", "/Users/", "/data/", "/tmp/"):
                self.assertNotIn(f'"{prefix}', serialized)
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
