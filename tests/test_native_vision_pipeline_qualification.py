from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate-native-vision-pipeline-qualification.py"


def load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "aima_native_vision_pipeline_qualification", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the vision-pipeline generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def exact_comparison(elements: int, digest: str) -> dict[str, object]:
    return {
        "passed": True,
        "bit_exact": True,
        "elements": elements,
        "exact_elements": elements,
        "finite_elements": elements,
        "expected_sha256": digest,
        "actual_sha256": digest,
        "relative_l2_error": 0.0,
        "cosine_similarity": 1.0,
    }


class NativeVisionPipelineQualificationLogicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = load_generator_module()
        self.attention_sha256 = "a" * 64
        self.payload = {
            "schema": "aima-amd395-qwen36/native-vision-pipeline-oracle/v2",
            "complete": True,
            "patches": 256,
            "merged_tokens": 64,
            "group_count": 1,
            "groups": [{"patches": 256, "merged_tokens": 64}],
            "temporary_bytes": 1,
            "metadata_resident_bytes": 1,
            "library_workspace_bytes": 1,
            "median_ms": 1.0,
            "attention_image_sha256": self.attention_sha256,
            "comparisons": {
                "vision_block_0": exact_comparison(294_912, "b" * 64),
                "vision_block_13": exact_comparison(294_912, "c" * 64),
                "vision_block_26": exact_comparison(294_912, "d" * 64),
                "vision_merger": exact_comparison(131_072, "e" * 64),
            },
            "repeat_actual_sha256": "e" * 64,
            "repeat_deterministic": True,
        }

    def test_exact_case_passes(self) -> None:
        record, checks = self.generator.summarize_case(
            "image_local_png", self.payload, self.attention_sha256
        )
        self.assertTrue(all(checks.values()))
        self.assertTrue(record["passed"])
        self.assertEqual(
            [item["name"] for item in record["boundaries"]],
            list(self.generator.BOUNDARY_ORDER),
        )

    def test_one_nonexact_boundary_fails_closed(self) -> None:
        self.payload["comparisons"]["vision_block_13"]["exact_elements"] -= 1
        record, checks = self.generator.summarize_case(
            "image_local_png", self.payload, self.attention_sha256
        )
        self.assertFalse(checks["all_boundaries_bit_exact"])
        self.assertFalse(record["passed"])

    def test_candidate_and_probe_identities_are_validated(self) -> None:
        self.generator.validate_identity(
            "1" * 40, "2" * 64, "3" * 64, "4" * 64, "5" * 40
        )
        with self.assertRaises(ValueError):
            self.generator.validate_identity(
                "not-a-commit", "2" * 64, "3" * 64, "4" * 64, "5" * 40
            )
        with self.assertRaises(ValueError):
            self.generator.validate_identity(
                "1" * 40, "not-a-hash", "3" * 64, "4" * 64, "5" * 40
            )


if __name__ == "__main__":
    unittest.main()
