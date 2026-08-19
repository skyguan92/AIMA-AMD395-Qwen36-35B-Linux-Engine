from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize-vl-performance-diagnostic.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_vl_performance_diagnostic", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class VlPerformanceDiagnosticSummaryLogicTest(unittest.TestCase):
    def test_ratio_is_fail_closed_for_missing_or_nonpositive_values(self) -> None:
        self.assertEqual(summary.ratio(2, 4), 0.5)
        self.assertIsNone(summary.ratio(None, 4))
        self.assertIsNone(summary.ratio(2, 0))
        self.assertIsNone(summary.ratio(float("nan"), 1))

    def test_millisecond_conversion_accepts_zero_and_rejects_invalid(self) -> None:
        self.assertEqual(summary.seconds_from_milliseconds(250), 0.25)
        self.assertEqual(summary.seconds_from_milliseconds(0), 0.0)
        self.assertIsNone(summary.seconds_from_milliseconds(None))
        self.assertIsNone(summary.seconds_from_milliseconds(-1))

    def test_stage_arithmetic_is_fail_closed(self) -> None:
        self.assertAlmostEqual(summary.add_nonnegative(0.1, 0.2), 0.3)
        self.assertIsNone(summary.add_nonnegative(0.1, None))
        self.assertIsNone(summary.add_nonnegative(0.1, -0.2))
        self.assertAlmostEqual(summary.subtract_stage(0.7, 0.2), 0.5)
        self.assertIsNone(summary.subtract_stage(0.2, 0.2))
        self.assertIsNone(summary.subtract_stage(0.2, 0.3))

    def test_exactly_one_processor_and_encoder_record_are_required(
        self,
    ) -> None:
        self.assertIsNone(summary.one_mm_record({}))
        self.assertIsNone(
            summary.one_mm_record(
                {"multimodal": {"merged": {"a": {}, "b": {}}}}
            )
        )
        self.assertEqual(
            summary.one_mm_record(
                {
                    "multimodal": {
                        "merged": {
                            "a": {
                                "preprocessor_total_secs": 0.1,
                                "encoder_forward_secs": 0.2,
                            }
                        }
                    }
                }
            ),
            {
                "preprocessor_total_secs": 0.1,
                "encoder_forward_secs": 0.2,
                "processor_record_count": 1,
                "encoder_record_count": 1,
            },
        )
        self.assertEqual(
            summary.one_mm_record(
                {
                    "multimodal": {
                        "merged": {
                            "processor": {"preprocessor_total_secs": 0.1},
                            "worker": {"encoder_forward_secs": 0.2},
                        }
                    }
                }
            ),
            {
                "preprocessor_total_secs": 0.1,
                "encoder_forward_secs": 0.2,
                "processor_record_count": 1,
                "encoder_record_count": 1,
            },
        )
        self.assertEqual(
            summary.one_mm_record(
                {
                    "multimodal": {
                        "merged": {
                            "processor": {"preprocessor_total_secs": 0.001},
                        }
                    }
                }
            ),
            {
                "preprocessor_total_secs": 0.001,
                "encoder_forward_secs": 0.0,
                "processor_record_count": 1,
                "encoder_record_count": 0,
            },
        )
        self.assertIsNone(
            summary.one_mm_record(
                {
                    "multimodal": {
                        "merged": {
                            "processor-a": {"preprocessor_total_secs": 0.1},
                            "processor-b": {"preprocessor_total_secs": 0.1},
                            "worker": {"encoder_forward_secs": 0.2},
                        }
                    }
                }
            )
        )

    def test_warmup_contract_normalizes_model_specific_response_fields(
        self,
    ) -> None:
        payload = {
            "id": "request-specific",
            "model": "engine-specific",
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 1,
                "total_tokens": 18,
                "prompt_tokens_details": None,
            },
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "Here"},
                }
            ],
        }
        contract = summary.warmup_contract(payload)
        self.assertEqual(contract["prompt_tokens"], 17)
        self.assertEqual(contract["completion_tokens"], 1)
        self.assertEqual(contract["finish_reason"], "length")
        self.assertEqual(len(contract["content_sha256"]), 64)
        self.assertNotIn("model", contract)

    def test_build_summary_uses_symmetric_stage_subtraction(self) -> None:
        benchmark_id = "diagnostic.pair-1"
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 1,
            "total_tokens": 101,
        }
        request = {
            "complete": True,
            "benchmark_id": benchmark_id,
            "request": {
                "template_sha256": "a" * 64,
                "summary": {},
                "media": [
                    {
                        "index": 0,
                        "modality": "image",
                        "path": "${AIMA_VL_MEDIA_ROOT}/image.png",
                        "bytes": 1,
                        "sha256": "c" * 64,
                    }
                ],
                "text_padding": {
                    "tokens": 0,
                    "unit_sha256": "d" * 64,
                    "frozen_single_token_id": 830,
                },
            },
            "response": {
                "content_sha256": "b" * 64,
                "finish_reason": "length",
                "usage": usage,
            },
        }
        warmup = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "One"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for role in ("reference", "candidate"):
                role_dir = root / role
                role_dir.mkdir()
                payload = {
                    **request,
                    "engine_role": role,
                    "timings": {"ttft_seconds": 0.9, "total_seconds": 1.0},
                }
                if role == "reference":
                    payload["prometheus"] = {
                        "delta": {
                            "vllm:request_prefill_time_seconds_count": 1,
                            "vllm:request_prefill_time_seconds_sum": 0.7,
                            "vllm:time_to_first_token_seconds_count": 1,
                            "vllm:request_decode_time_seconds_sum": 0,
                        }
                    }
                else:
                    payload["response"] = {
                        **request["response"],
                        "content_sha256": "e" * 64,
                    }
                    payload["native_metrics"] = {
                        "prompt_tokens": 100,
                        "ttft_ms": 600,
                        "prefix_cache": {"lookup": "miss"},
                        "vl": {
                            "visual_tokens": 50,
                            "vision_plan_build_wall_ms": 100,
                            "vision_encode_wall_ms": 200,
                            "media_load_decode_wall_ms": 10,
                            "processor_wall_ms": 20,
                            "logical_projection_tokens": 100,
                            "logical_projection_plan_count": 7,
                            "logical_projection_workspace_bytes": 1024,
                            "logical_projection_plan_build_wall_ms": 50,
                            "logical_projection_plan_reused": False,
                        },
                    }
                (role_dir / "request.json").write_text(json.dumps(payload))
                (role_dir / "text-warmup.json").write_text(json.dumps(warmup))

            stage = {
                "benchmark_id": benchmark_id,
                "status_code": 200,
                "request_error": None,
                "stats_error": None,
                "media": {"media_load_decode_secs": 0.01},
                "multimodal": {
                    "merged": {
                        "request": {
                            "preprocessor_total_secs": 0.02,
                            "encoder_forward_secs": 0.2,
                        }
                    }
                },
            }
            (root / "reference/vllm-vl-stages.jsonl").write_text(
                json.dumps(stage) + "\n"
            )
            result = summary.build_summary(root)

        self.assertTrue(result["complete"])
        self.assertTrue(result["checks"]["response_shape_and_length_exact"])
        self.assertNotEqual(
            result["response_audit"]["reference"]["content_sha256"],
            result["response_audit"]["candidate"]["content_sha256"],
        )
        reference = result["measurements"]["reference"]
        candidate = result["measurements"]["candidate"]
        self.assertAlmostEqual(reference["llm_prefill_seconds"], 0.5)
        self.assertAlmostEqual(candidate["cold_vision_seconds"], 0.3)
        self.assertEqual(candidate["logical_projection_plan_count"], 7)
        self.assertAlmostEqual(
            candidate["logical_projection_plan_build_seconds"], 0.05
        )
        self.assertAlmostEqual(candidate["llm_prefill_seconds"], 0.3)
        self.assertAlmostEqual(
            result["comparisons"]["prefill_tps_candidate_over_reference"],
            5 / 3,
        )
        self.assertAlmostEqual(
            result["comparisons"]["vision_tps_candidate_over_reference"],
            1.0,
        )
        self.assertAlmostEqual(
            result["comparisons"][
                "cold_vision_path_tps_candidate_over_reference"
            ],
            2 / 3,
        )

    def test_exact_media_cache_hit_requires_candidate_encoder_skip(self) -> None:
        benchmark_id = "cache-hit.pair-1"
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 1,
            "total_tokens": 101,
        }
        media = [
            {
                "index": 0,
                "modality": "image",
                "path": "${AIMA_VL_MEDIA_ROOT}/image.png",
                "bytes": 1,
                "sha256": "c" * 64,
            }
        ]
        warmup = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
            "choices": [
                {"finish_reason": "length", "message": {"content": "One"}}
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for role in ("reference", "candidate"):
                role_dir = root / role
                role_dir.mkdir()
                payload = {
                    "complete": True,
                    "benchmark_id": benchmark_id,
                    "engine_role": role,
                    "request": {
                        "template_sha256": "a" * 64,
                        "summary": {},
                        "media": media,
                        "text_padding": {
                            "tokens": 0,
                            "unit_sha256": "d" * 64,
                            "frozen_single_token_id": 830,
                        },
                    },
                    "response": {
                        "content_sha256": ("b" if role == "reference" else "e")
                        * 64,
                        "finish_reason": "length",
                        "usage": usage,
                    },
                    "timings": {"ttft_seconds": 0.5, "total_seconds": 0.6},
                }
                if role == "reference":
                    payload["prometheus"] = {
                        "delta": {
                            "vllm:request_prefill_time_seconds_count": 1,
                            "vllm:request_prefill_time_seconds_sum": 0.5,
                            "vllm:time_to_first_token_seconds_count": 1,
                            "vllm:request_decode_time_seconds_sum": 0,
                        }
                    }
                else:
                    payload["native_metrics"] = {
                        "prompt_tokens": 100,
                        "ttft_ms": 500,
                        "prefix_cache": {"lookup": "disabled"},
                        "vl": {
                            "visual_tokens": 50,
                            "vision_plan_build_wall_ms": 0,
                            "vision_encode_wall_ms": 20,
                            "media_load_decode_wall_ms": 1,
                            "processor_wall_ms": 0,
                            "media_cache_hits": 1,
                            "media_cache_misses": 0,
                            "media_cache_entries": 1,
                            "vision_embedding_cache_hit": True,
                            "vision_embedding_cache_entries": 1,
                            "vision_embedding_cache_resident_bytes": 256000,
                            "vision_embedding_cache_capacity_bytes": 536870912,
                        },
                    }
                (role_dir / "request.json").write_text(json.dumps(payload))
                (role_dir / "text-warmup.json").write_text(json.dumps(warmup))

            stage = {
                "benchmark_id": benchmark_id,
                "status_code": 200,
                "request_error": None,
                "stats_error": None,
                "media": {"media_load_decode_secs": 0.001},
                "multimodal": {
                    "merged": {
                        "processor": {"preprocessor_total_secs": 0.001}
                    }
                },
            }
            (root / "reference/vllm-vl-stages.jsonl").write_text(
                json.dumps(stage) + "\n"
            )
            expectations = {
                "output_tokens": 1,
                "prefix_cache_lookup": "disabled",
                "media_cache_mode": "exact",
            }
            executed = summary.build_summary(root, expectations=expectations)
            self.assertTrue(executed["complete"])
            self.assertFalse(executed["qualified"])
            self.assertEqual(
                executed["comparisons"][
                    "vision_cache_hit_candidate_seconds"
                ],
                0.02,
            )

            candidate_path = root / "candidate/request.json"
            candidate = json.loads(candidate_path.read_text())
            candidate["native_metrics"]["vl"]["vision_encode_wall_ms"] = 0
            candidate_path.write_text(json.dumps(candidate))
            skipped = summary.build_summary(root, expectations=expectations)

            candidate["native_metrics"]["vl"][
                "vision_embedding_cache_hit"
            ] = False
            candidate_path.write_text(json.dumps(candidate))
            unbound = summary.build_summary(root, expectations=expectations)

        self.assertTrue(skipped["complete"])
        self.assertTrue(skipped["qualified"])
        self.assertFalse(
            skipped["metric_applicability"]["vision_throughput"]["applicable"]
        )
        self.assertEqual(
            skipped["comparisons"]["vision_cache_hit_candidate_seconds"],
            0.0,
        )
        self.assertTrue(
            skipped["checks"]["candidate_vision_embedding_cache_expected"]
        )
        self.assertTrue(
            skipped["measurements"]["candidate"][
                "vision_embedding_cache_hit"
            ]
        )
        self.assertFalse(unbound["complete"])
        self.assertFalse(
            unbound["checks"]["candidate_vision_embedding_cache_expected"]
        )


if __name__ == "__main__":
    unittest.main()
