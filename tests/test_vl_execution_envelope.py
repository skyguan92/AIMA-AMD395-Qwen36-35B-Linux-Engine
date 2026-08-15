from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from aima_engine.vl_execution import (
    NON_HTTP_CELL_MODES,
    build_http_probe_specs,
    execution_cell_coverage,
    validate_fixture_manifest,
    validate_http_observation,
    validate_processor_probe_observation,
    validate_vision_probe_observation,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-envelope-v0.1.0"
FIXTURE_MANIFEST = FIXTURE_ROOT / "fixtures-manifest.json"
ENVELOPE_PATH = ROOT / "benchmarks/results/vl-capability-envelope-v0.1.0.json"


class VlExecutionEnvelopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.envelope = json.loads(ENVELOPE_PATH.read_text(encoding="utf-8"))
        cls.fixtures = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
        cls.probes = build_http_probe_specs(
            cls.envelope, cls.fixtures, FIXTURE_ROOT
        )

    def test_fixture_corpus_is_hash_bound_and_complete(self) -> None:
        self.assertEqual(validate_fixture_manifest(self.fixtures, FIXTURE_ROOT), [])
        self.assertEqual(len(self.fixtures["fixtures"]), 16)
        records = {
            item["fixture_id"]: item for item in self.fixtures["fixtures"]
        }
        maximum = records[
            "video-sampling-above-maximum-18432f-24fps-256x256.mp4"
        ]
        self.assertEqual(maximum["frame_count"], 18_432)
        self.assertEqual(maximum["duration_seconds"], 768.0)
        self.assertEqual(
            records["video-rejected-spatial-2f-2fps-32x31.avi"]["height"],
            31,
        )

    def test_every_envelope_cell_has_an_explicit_execution_mode(self) -> None:
        coverage = execution_cell_coverage(self.envelope, self.probes)
        self.assertEqual(len(coverage), 23)
        self.assertNotIn("missing", coverage.values())
        self.assertEqual(
            {mode for mode in coverage.values()},
            {"http", "native-processor-probe", "native-vision-probe"},
        )
        self.assertEqual(
            set(NON_HTTP_CELL_MODES),
            {"video_sampling_option_conflict", "image_full_encoder_budget"},
        )
        source = (
            ROOT / "native/tools/vl_envelope_vision_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vl-envelope-vision-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("image_full_encoder_budget", source)
        self.assertIn("kMediaItems = 16", source)
        self.assertIn("native_qwen36_vision_batches(all_grids)", source)
        self.assertIn("pipeline.launch(", source)
        self.assertIn("repeat_deterministic", source)
        self.assertIn("vl_envelope_vision_probe.hip.cpp", build)
        qualifier = (
            ROOT / "scripts/qualify-native-vl-envelope.py"
        ).read_text(encoding="utf-8")
        self.assertIn("native VL execution-envelope cell", qualifier)
        self.assertIn("validate_http_observation", qualifier)
        self.assertIn('"262143"', qualifier)
        self.assertIn('"262144"', qualifier)
        self.assertIn("--client-timeout-seconds", qualifier)
        self.assertIn(
            '"--client-timeout-seconds", type=float, default=7200.0',
            qualifier,
        )
        self.assertIn("timeout=args.client_timeout_seconds", qualifier)
        self.assertIn(
            'fmha_provider.with_name(\n        "libaima-fmha-ck.so"',
            qualifier,
        )
        self.assertNotIn(
            '"--fmha-provider",\n        str(fmha_provider),',
            qualifier,
        )
        self.assertIn("automatic_long_context_fmha_policy", qualifier)
        self.assertIn("native_execution_qualification_complete", qualifier)

    def test_http_plan_has_exact_success_error_and_boundary_counts(self) -> None:
        self.assertEqual(len(self.probes), 23)
        self.assertEqual(sum(probe["expected_accept"] for probe in self.probes), 17)
        self.assertEqual(
            sum(not probe["expected_accept"] for probe in self.probes), 6
        )
        by_id = {probe["probe_id"]: probe for probe in self.probes}
        self.assertEqual(
            by_id["mixed_cross_batch_boundary"]["expected"],
            {
                "cell_id": "mixed_cross_batch_boundary",
                "outcome": "accepted",
                "status_code": 200,
                "media_counts": {"image": 1, "video": 1},
                "visual_tokens": 16_388,
                "vision_patches": 65_552,
                "vision_batch_count": 2,
                "vision_max_batch_tokens": 16_384,
                "vision_max_batch_patches": 65_536,
            },
        )
        self.assertEqual(
            by_id["video_sampling_maximum"]["expected"]["visual_tokens"],
            9_600,
        )
        self.assertEqual(
            by_id["image_near_window_maximum"]["expected"][
                "vision_batch_count"
            ],
            15,
        )
        self.assertEqual(
            by_id["video_full_item_budget"]["expected"][
                "vision_batch_count"
            ],
            21,
        )

    def test_native_decoder_clamps_the_above_maximum_sampling_boundary(self) -> None:
        decoder = (ROOT / "native/src/native_video_decoder.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "std::min({count, total_frames, maximum_sampled_frames})",
            decoder,
        )
        self.assertNotIn("sampled video frame count exceeds the limit", decoder)

    def test_http_observation_validator_fails_closed(self) -> None:
        probe = next(
            item for item in self.probes if item["probe_id"] == "image_minimum"
        )
        expected = probe["expected"]
        observation = {
            "status_code": 200,
            "accepted": True,
            "response": {
                "usage": {
                    "prompt_tokens": 82,
                    "completion_tokens": 1,
                    "total_tokens": 83,
                },
                "choices": [{"finish_reason": "length", "message": {}}],
                "aima_amd395": {
                    "runtime": "native-resident-q82",
                    "oracle_tensor_reads": 0,
                    "aot_prefill_segments": 1,
                    "padded_prefill_tokens": 942,
                    "mrope": {
                        "enabled": True,
                        "full_attention_launches": 10,
                        "fmha_launches": 0,
                        "unified_attention_launches": 10,
                    },
                    "vl": {
                        "enabled": True,
                        "media_count": 1,
                        "image_count": 1,
                        "video_count": 0,
                        "visual_tokens": 64,
                        "vision_patches": 256,
                        "vision_batch_count": 1,
                        "vision_max_batch_tokens": 64,
                        "vision_max_batch_patches": 256,
                        "vision_encode_wall_ms": 1.0,
                    },
                },
            },
        }
        self.assertTrue(all(validate_http_observation(observation, expected).values()))
        tampered = copy.deepcopy(observation)
        tampered["response"]["aima_amd395"]["vl"]["visual_tokens"] = 63
        checks = validate_http_observation(tampered, expected)
        self.assertFalse(checks["visual_tokens_exact"])
        self.assertFalse(all(checks.values()))
        tampered = copy.deepcopy(observation)
        tampered["response"]["aima_amd395"]["mrope"][
            "unified_attention_launches"
        ] = 0
        checks = validate_http_observation(tampered, expected)
        self.assertFalse(checks["mrope_dispatch_accounted"])
        self.assertFalse(checks["mrope_initial_padding_only"])

    def test_non_http_probe_validators_fail_closed(self) -> None:
        processor = validate_processor_probe_observation(
            0, "native_vl_processor_test: PASS\n", ""
        )
        self.assertTrue(all(processor.values()))
        self.assertFalse(
            validate_processor_probe_observation(0, "PASS\n", "")[
                "exact_pass_marker"
            ]
        )
        cell = next(
            item
            for item in self.envelope["execution_cells"]
            if item["cell_id"] == "image_full_encoder_budget"
        )
        vision = {
            "schema": (
                "aima-amd395-qwen36/"
                "native-vl-envelope-vision-probe/v1"
            ),
            "complete": True,
            "cell_id": "image_full_encoder_budget",
            "media_items": 16,
            "visual_tokens": 262_144,
            "vision_patches": 1_048_576,
            "vision_batch_count": 16,
            "vision_max_batch_tokens": 16_384,
            "vision_max_batch_patches": 65_536,
            "executed_batches": 16,
            "output_elements_per_batch": 33_554_432,
            "finite_output_elements": 536_870_912,
            "expected_finite_output_elements": 536_870_912,
            "repeat_output_sha256": "a" * 64,
            "repeat_deterministic": True,
            "weight_payload_bytes": 893_142_496,
            "attention_image_sha256": "b" * 64,
            "total_vision_wall_ms": 1.0,
        }
        self.assertTrue(all(validate_vision_probe_observation(vision, cell).values()))
        vision["finite_output_elements"] -= 1
        self.assertFalse(
            validate_vision_probe_observation(vision, cell)[
                "all_outputs_finite"
            ]
        )


if __name__ == "__main__":
    unittest.main()
