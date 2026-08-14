from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_serving_render import (
    SERVING_RENDER_CASES,
    validate_serving_render_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/vl-serving-render-manifest-v0.1.0.json"
SIDECAR = RESULT.with_name(RESULT.name + ".sha256")
SOURCE_COMMIT = "2af339a31e6a9d982a90bd521f2558fc0f18ad5e"
EXPECTED_PROMPTS = {
    "image_local_png": (
        82,
        "6462d85f92fb077999f1b028006631937910dc5cadf4ea6fdda8805982863701",
    ),
    "video_local_mp4": (
        64,
        "6e632c4f208a83c4c0fdb66fe16a8104e4578b2c89e6542247f720084e78c446",
    ),
    "multi_image": (
        186,
        "3859b4d1942df0d6fa9d4e67c42a833bae351096c0cbc107c636cdcb434124cd",
    ),
    "multi_video": (
        131,
        "f11eb4c121339971b871b52c1d535803627336e201f6fc759062b5635dbe19b6",
    ),
    "mixed_image_video": (
        134,
        "d2d4f5cace39ca45fa3f57dd73745e351a629240e91a5a58fe6ff79f2094e76f",
    ),
}


class VlServingRenderEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = RESULT.read_bytes()
        cls.result = json.loads(cls.payload)

    def test_sidecar_seal_and_validator(self) -> None:
        digest = hashlib.sha256(self.payload).hexdigest()
        self.assertEqual(
            SIDECAR.read_text(encoding="utf-8"),
            f"{digest}  {RESULT.name}\n",
        )
        self.assertEqual(validate_serving_render_manifest(self.result), [])

    def test_clean_source_and_inputs_are_hash_bound(self) -> None:
        self.assertEqual(self.result["source"]["commit"], SOURCE_COMMIT)
        self.assertFalse(self.result["source"]["dirty"])
        for component in self.result["source"]["files"]:
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                component["sha256"],
            )
        for component in self.result["bindings"].values():
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                component["sha256"],
            )

    def test_five_real_http_prompt_vectors_are_exact(self) -> None:
        cases = self.result["cases"]
        self.assertEqual(
            tuple(case["case_id"] for case in cases), SERVING_RENDER_CASES
        )
        for case in cases:
            expected_tokens, expected_hash = EXPECTED_PROMPTS[case["case_id"]]
            self.assertEqual(case["prompt_tokens"], expected_tokens)
            self.assertEqual(case["prompt_token_ids_sha256"], expected_hash)
            self.assertFalse(case["private_prompt_matches_real_http"])
        self.assertTrue(
            self.result["decision"]["five_serving_render_cases_5_of_5"]
        )
        self.assertTrue(
            self.result["decision"][
                "private_preprocessor_boundary_distinguished_5_of_5"
            ]
        )
        self.assertFalse(self.result["decision"]["g1_passed"])
        self.assertFalse(self.result["decision"]["g2_passed"])

    def test_evidence_contains_no_private_machine_paths(self) -> None:
        serialized = self.payload.decode("utf-8")
        for private_prefix in ("/home/", "/Users/", "/data/", "/tmp/aima-native"):
            self.assertNotIn(private_prefix, serialized)


if __name__ == "__main__":
    unittest.main()
