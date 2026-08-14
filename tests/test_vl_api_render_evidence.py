from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_capability import (
    API_RENDER_MEDIA_COUNTS,
    API_RENDER_TOOL_CASES,
    API_RENDER_USAGELESS_CASES,
    EXPECTED_FORCED_TOOL_STRUCTURED_OUTPUTS,
    REQUIRED_API_RENDER_CASES,
    validate_api_render_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/vl-api-render-manifest-v0.1.0.json"
RESULT_SIDECAR = RESULT.with_name(RESULT.name + ".sha256")
CAPABILITY = ROOT / "benchmarks/results/vl-capability-manifest.json"
QUALIFIED_COMMIT = "6e309d9e85c0fe79545dd0597255a514af5bc015"
EXPECTED_PROMPT_TOKENS = {
    "residency_text_before": 15,
    "image_local_png": 82,
    "image_data_jpeg": 112,
    "image_http_webp": 112,
    "image_transparent_png": 88,
    "multi_image_interleaved": 184,
    "video_local_mp4": 63,
    "video_data_mp4": 62,
    "video_http_avi": 78,
    "multi_video": 128,
    "mixed_image_then_video": 130,
    "mixed_video_then_image": 162,
    "conversation_prior_image": 109,
    "conversation_media_replace": 174,
    "tool_history_with_image": 395,
    "tool_forced_image": 349,
    "tool_auto_image": 352,
    "stream_image": 80,
    "stream_video": 62,
    "residency_text_after": 18,
}


class VlApiRenderEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = RESULT.read_bytes()
        cls.result = json.loads(cls.payload)
        cls.capability = json.loads(CAPABILITY.read_bytes())

    def test_sidecar_seal_and_fail_closed_validator(self) -> None:
        digest = hashlib.sha256(self.payload).hexdigest()
        self.assertEqual(
            RESULT_SIDECAR.read_text(encoding="utf-8"),
            f"{digest}  {RESULT.name}\n",
        )
        self.assertEqual(validate_api_render_manifest(self.result), [])

    def test_clean_source_and_all_inputs_are_hash_bound(self) -> None:
        source = self.result["source"]
        self.assertEqual(source["commit"], QUALIFIED_COMMIT)
        self.assertFalse(source["dirty"])
        for component in source["files"]:
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

    def test_fixture_normalized_requests_match_the_full_server_probe(self) -> None:
        references = {
            case["case_id"]: case for case in self.capability["cases"]
        }
        for case in self.result["cases"]:
            reference = references[case["case_id"]]
            self.assertEqual(case["request"], reference["request"])
            self.assertEqual(case["surfaces"], reference["surfaces"])
            self.assertEqual(
                case["reference_transport_request_sha256"],
                reference["request_sha256"],
            )

    def test_all_success_prompts_and_usage_semantics_are_frozen(self) -> None:
        cases = self.result["cases"]
        self.assertEqual(
            tuple(case["case_id"] for case in cases),
            REQUIRED_API_RENDER_CASES,
        )
        self.assertEqual(
            {case["case_id"]: case["prompt_tokens"] for case in cases},
            EXPECTED_PROMPT_TOKENS,
        )
        for case in cases:
            case_id = case["case_id"]
            if case_id in API_RENDER_USAGELESS_CASES:
                self.assertIsNone(case["reference_usage_prompt_tokens"])
                self.assertIsNone(case["reference_usage_delta"])
            else:
                self.assertEqual(case["reference_usage_delta"], 0)

    def test_multimodal_spans_and_forced_tool_decoder_are_exact(self) -> None:
        cases = {case["case_id"]: case for case in self.result["cases"]}
        for case_id, expected_counts in API_RENDER_MEDIA_COUNTS.items():
            observed = {
                modality: len(spans)
                for modality, spans in cases[case_id]["mm_placeholders"].items()
            }
            self.assertEqual(observed, expected_counts)
        video_span = cases["video_local_mp4"]["mm_placeholders"]["video"][0]
        self.assertEqual(video_span["length"], 48)
        self.assertEqual(video_span["pad_token_count"], 32)
        self.assertEqual(
            cases["tool_forced_image"]["structured_outputs"],
            EXPECTED_FORCED_TOOL_STRUCTURED_OUTPUTS,
        )

    def test_decision_does_not_claim_g1_or_g2(self) -> None:
        decision = self.result["decision"]
        for name in (
            "success_render_cases_20_of_20",
            "non_tool_non_stream_render_matches_full_usage",
            "tool_render_matches_full_usage",
            "named_tool_json_schema_bound",
        ):
            self.assertTrue(decision[name])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])

    def test_evidence_contains_no_private_machine_paths(self) -> None:
        serialized = self.payload.decode("utf-8")
        for private_prefix in (
            "/home/",
            "/Users/",
            "/data/",
            "/tmp/aima-native",
            "file:///tmp/",
        ):
            self.assertNotIn(private_prefix, serialized)


if __name__ == "__main__":
    unittest.main()
