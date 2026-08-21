from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_error_limits import (
    NATIVE_COMPATIBLE_ERROR,
    NATIVE_REPLAY,
    REFERENCE_CASE_ORDER,
    REFERENCE_ERROR_CONTRACT,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
MEDIA_IO = RESULTS / "vl-media-io-reference-v0.1.0.json"
REFERENCE = RESULTS / "vl-error-limits-reference-v0.1.0.json"
NATIVE = RESULTS / "native-vl-error-limits-v0.1.0.json"
QUALIFIED_COMMIT = "7642995e772fbdc8ae763bcffb90f2da852987f0"
NATIVE_QUALIFIED_COMMIT = "50289f1cbae150997ca82bbc054635932a2721c3"
QUALIFIED_BINARY_SHA256 = (
    "4bf377135bafe4dd0d449dc2c8563fa727ed47414eb4c7c7221ecb7e631711d0"
)
MEDIA_IO_CASES = {
    "default_white": (
        "c779b79d2b3dc97c964b1f931bb9056602fba3b40eee297131e247680e36104e"
    ),
    "red": "debb77f47c8594a633976b272b192ac42db5f396de52c4ee8789a57854f176ef",
}


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
        test.assertEqual(path.stat().st_size, component["bytes"], component["path"])
        test.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            component["sha256"],
            component["path"],
        )


class NativeVlErrorLimitsQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payloads = {
            path: path.read_bytes() for path in (MEDIA_IO, REFERENCE, NATIVE)
        }
        cls.media_io = json.loads(cls.payloads[MEDIA_IO])
        cls.reference = json.loads(cls.payloads[REFERENCE])
        cls.native = json.loads(cls.payloads[NATIVE])

    def test_media_io_reference_is_exact_and_sealed(self) -> None:
        result = self.media_io
        assert_sealed(self, MEDIA_IO, result)
        self.assertEqual(
            result["schema"], "aima-amd395-qwen36/vl-media-io-reference/v1"
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(result["source"]["commit"], QUALIFIED_COMMIT)
        assert_components_current(self, result["source"]["files"])
        assert_components_current(self, [result["fixture"]])
        cases = {case["case_id"]: case for case in result["cases"]}
        self.assertEqual(set(cases), set(MEDIA_IO_CASES))
        for case_id, expected_sha256 in MEDIA_IO_CASES.items():
            case = cases[case_id]
            self.assertTrue(case["qualified"])
            self.assertEqual(case["rgb_sha256"], expected_sha256)
            self.assertEqual(case["expected_rgb_sha256"], expected_sha256)
        self.assertTrue(result["decision"]["default_white_exact"])
        self.assertTrue(result["decision"]["request_red_exact"])
        self.assertTrue(result["decision"]["background_changes_rgb_bytes"])

    def test_fixed_vllm_reference_freezes_all_ten_cases(self) -> None:
        result = self.reference
        assert_sealed(self, REFERENCE, result)
        self.assertEqual(
            result["schema"],
            "aima-amd395-qwen36/vl-error-limits-reference/v1",
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(result["source"]["commit"], QUALIFIED_COMMIT)
        assert_components_current(self, result["source"]["files"])
        assert_components_current(self, list(result["bindings"].values()))
        cases = result["cases"]
        self.assertEqual(tuple(case["case_id"] for case in cases), REFERENCE_CASE_ORDER)
        self.assertTrue(all(case["passed"] for case in cases))
        self.assertTrue(all(case["qualified"] for case in cases))
        self.assertTrue(
            all(all(case["qualification_checks"].values()) for case in cases)
        )
        self.assertTrue(all(result["qualification_checks"].values()))

        by_id = {case["case_id"]: case for case in cases}
        for case_id, contract in REFERENCE_ERROR_CONTRACT.items():
            status, error_type, error_code, _category = contract
            case = by_id[case_id]
            self.assertFalse(case["accepted"])
            self.assertEqual(case["status_code"], status)
            self.assertEqual(case["response"]["error"]["type"], error_type)
            self.assertEqual(case["response"]["error"]["code"], error_code)
        self.assertTrue(result["decision"]["ten_reference_cases_frozen"])
        self.assertTrue(result["decision"]["error_limit_categories_frozen"])

    def test_native_replay_is_compatible_and_cache_safe(self) -> None:
        result = self.native
        assert_sealed(self, NATIVE, result)
        self.assertEqual(
            result["schema"], "aima-amd395-qwen36/native-vl-error-limits/v1"
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(result["source"]["commit"], NATIVE_QUALIFIED_COMMIT)
        self.assertEqual(
            result["build_info"]["source_commit"], NATIVE_QUALIFIED_COMMIT
        )
        self.assertEqual(result["binary"]["sha256"], QUALIFIED_BINARY_SHA256)
        assert_components_current(self, result["source"]["files"])
        assert_components_current(
            self,
            [
                result["dependencies"]["error_fixture_manifest"],
                result["dependencies"]["fixture_manifest"],
                result["dependencies"]["media_io_oracle"],
                result["dependencies"]["reference"],
            ],
        )
        self.assertEqual(
            result["dependencies"]["fmha_provider"]["sha256"],
            "e5336b2d66b36c5f17aeb07ab780fa8f60a6092910f9b01b3ebf4bc31f766bb4",
        )
        self.assertEqual(
            result["dependencies"]["vision_attention_image"]["sha256"],
            "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e",
        )

        cases = result["run"]["cases"]
        self.assertEqual(
            tuple(
                (case["observation_id"], case["reference_case_id"])
                for case in cases
            ),
            NATIVE_REPLAY,
        )
        self.assertTrue(all(case["passed"] for case in cases))
        self.assertTrue(all(case["qualified"] for case in cases))
        self.assertTrue(
            all(all(case["qualification_checks"].values()) for case in cases)
        )

        reference = {case["case_id"]: case for case in self.reference["cases"]}
        native_status, native_type, native_code = NATIVE_COMPATIBLE_ERROR
        for case in cases:
            frozen = reference[case["reference_case_id"]]
            self.assertEqual(case["request"], frozen["request"])
            if case["reference_case_id"] not in REFERENCE_ERROR_CONTRACT:
                self.assertEqual(case["status_code"], frozen["status_code"])
                continue
            self.assertFalse(case["accepted"])
            self.assertEqual(case["status_code"], native_status)
            self.assertEqual(case["response"]["error"]["type"], native_type)
            self.assertEqual(case["response"]["error"]["code"], native_code)
            self.assertTrue(case["response"]["error"]["message"])

        self.assertTrue(all(result["run"]["checks"].values()))
        self.assertEqual(result["run"]["stopped"], {
            "event": "stopped",
            "model_loads": 1,
            "served": 8,
        })
        self.assertEqual(result["run"]["raw"]["stderr"]["bytes"], 0)
        self.assertTrue(
            all(result["qualification_checks"]["cache"].values())
        )
        self.assertTrue(
            all(result["qualification_checks"]["server"].values())
        )
        self.assertTrue(result["decision"]["thirteen_observations_reference_exact"])
        self.assertTrue(result["decision"]["error_limit_categories_qualified"])
        self.assertTrue(result["decision"]["rgba_background_cache_identity_qualified"])
        self.assertTrue(result["decision"]["one_resident_model_load"])
        for runtime in ("python", "torch", "triton", "vllm"):
            self.assertFalse(result["decision"][f"runtime_{runtime}"])

    def test_public_evidence_does_not_promote_product_gates(self) -> None:
        for path, result in (
            (MEDIA_IO, self.media_io),
            (REFERENCE, self.reference),
            (NATIVE, self.native),
        ):
            serialized = self.payloads[path].decode("utf-8")
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
