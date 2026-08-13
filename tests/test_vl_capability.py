from __future__ import annotations

import unittest

from aima_engine.vl_capability import (
    EXPECTED_MAX_ITEMS_PER_PROMPT,
    EXPECTED_MAX_MODEL_LEN,
    EXPECTED_MAX_TOKENS_PER_ITEM,
    PROCESSOR_PROBE_SCHEMA,
    REQUIRED_API_CASES,
    REQUIRED_API_SURFACES,
    REQUIRED_IMAGE_CASES,
    REQUIRED_VIDEO_RESIZE_CASES,
    REQUIRED_VIDEO_SAMPLING_CASES,
    validate_capability_manifest,
    validate_processor_probe,
)
from aima_engine.vl_reference import CAPABILITY_SCHEMA, MODEL_REVISION, PINNED_PACKAGES


def valid_probe() -> dict:
    def cases(names: set[str]) -> list[dict]:
        return [
            {"case_id": name, "outcome": "accepted", "result": {}}
            for name in sorted(names)
        ]

    return {
        "schema": PROCESSOR_PROBE_SCHEMA,
        "complete": True,
        "qualified": True,
        "model": {"revision": MODEL_REVISION},
        "runtime": {"packages": dict(PINNED_PACKAGES)},
        "vllm_budget": {
            "max_model_len": EXPECTED_MAX_MODEL_LEN,
            "supported_limits": {"image": None, "video": None},
            "max_tokens_per_item": dict(EXPECTED_MAX_TOKENS_PER_ITEM),
            "max_items_per_prompt": dict(EXPECTED_MAX_ITEMS_PER_PROMPT),
            "max_items_per_batch": dict(EXPECTED_MAX_ITEMS_PER_PROMPT),
            "encoder_budget_tokens": EXPECTED_MAX_MODEL_LEN,
        },
        "image_resize_cases": cases(REQUIRED_IMAGE_CASES),
        "video_resize_cases": cases(REQUIRED_VIDEO_RESIZE_CASES),
        "video_sampling_cases": cases(REQUIRED_VIDEO_SAMPLING_CASES),
        "deterministic_processor_fixtures": [
            {
                "modality": modality,
                "outputs": {"input_ids": {"sha256": "a" * 64}},
            }
            for modality in ("image", "video")
        ],
    }


def valid_capability_manifest() -> dict:
    surface_cycle = sorted(REQUIRED_API_SURFACES)
    cases = []
    for index, (case_id, expected) in enumerate(REQUIRED_API_CASES.items()):
        cases.append(
            {
                "case_id": case_id,
                "surfaces": [surface_cycle[index % len(surface_cycle)]],
                "expected_accept": expected,
                "passed": True,
                "status_code": 200 if expected else 400,
                "request_sha256": "a" * 64,
                "response_sha256": "b" * 64,
            }
        )
    return {
        "schema": CAPABILITY_SCHEMA,
        "complete": True,
        "qualified": True,
        "bindings": {
            "processor_probe": {"sha256": "c" * 64},
            "fixture_manifest": {"sha256": "d" * 64},
        },
        "cases": cases,
    }


class VlCapabilityTest(unittest.TestCase):
    def test_complete_processor_probe_is_accepted(self) -> None:
        self.assertEqual(validate_processor_probe(valid_probe()), [])

    def test_budget_drift_is_rejected(self) -> None:
        probe = valid_probe()
        probe["vllm_budget"]["max_items_per_prompt"]["image"] = 15
        self.assertIn(
            "vLLM derived media-count boundary drifted",
            validate_processor_probe(probe),
        )

    def test_missing_discrete_boundary_is_rejected(self) -> None:
        probe = valid_probe()
        probe["image_resize_cases"] = [
            item
            for item in probe["image_resize_cases"]
            if item["case_id"] != "factor_plus_one"
        ]
        errors = validate_processor_probe(probe)
        self.assertIn("missing image resize cases: factor_plus_one", errors)

    def test_missing_video_fixture_is_rejected(self) -> None:
        probe = valid_probe()
        probe["deterministic_processor_fixtures"] = [
            probe["deterministic_processor_fixtures"][0]
        ]
        self.assertIn(
            "deterministic processor fixtures must cover image and video",
            validate_processor_probe(probe),
        )

    def test_complete_api_capability_manifest_is_accepted(self) -> None:
        self.assertEqual(validate_capability_manifest(valid_capability_manifest()), [])

    def test_missing_or_false_api_case_is_rejected(self) -> None:
        manifest = valid_capability_manifest()
        manifest["cases"] = manifest["cases"][1:]
        manifest["cases"][0]["passed"] = False
        errors = validate_capability_manifest(manifest)
        self.assertTrue(any("missing API capability cases" in error for error in errors))
        self.assertTrue(any("did not pass" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
