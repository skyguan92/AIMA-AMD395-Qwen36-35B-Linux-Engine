from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "qualify-native-surfaces.py"
SPEC = importlib.util.spec_from_file_location("native_surfaces", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
surfaces = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(surfaces)


class NativeSurfacesQualificationTest(unittest.TestCase):
    def test_prefix_cpu_list_is_strict_and_canonical(self) -> None:
        self.assertEqual(surfaces.parse_cpu_list("8-15"), "8-15")
        self.assertEqual(surfaces.parse_cpu_list("2,4-7"), "2,4-7")
        for invalid in ("", "15-8", "1,", "1 2", "0-3:2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    surfaces.parse_cpu_list(invalid)

    def test_defaults_match_native_vl_goal(self) -> None:
        self.assertEqual(surfaces.STARTUP_CEILING_MS, 44_900.0)
        self.assertEqual(surfaces.MINIMUM_PREFIX_PAIRS, 5)
        self.assertEqual(
            surfaces.FROZEN_V151_PREFIX_OBSERVATION["ttft_speedup"],
            2636.9250000546567,
        )
        self.assertEqual(
            surfaces.FROZEN_V151_PREFIX_OBSERVATION["decode_retention"],
            1.0002958825348782,
        )
        self.assertEqual(
            surfaces.MINIMUM_PREFIX_TTFT_SPEEDUP,
            110.11994260509346,
        )
        self.assertEqual(
            surfaces.MINIMUM_PREFIX_DECODE_RETENTION,
            0.999653457424567,
        )

    def test_public_paths_bind_external_candidate_and_output(self) -> None:
        engine = Path("/tmp/candidate/build/aima-engine-native")
        model = Path("/srv/private/model")
        output = Path("/tmp/evidence/surfaces")
        baseline = Path("/tmp/release/libexec/aima-engine.real")
        value = {
            "command": [
                str(baseline),
                str(engine),
                "--model-dir",
                str(model),
            ],
            "candidate_provider": str(
                engine.parent / "libaima-fmha-ck.so"
            ),
            "baseline_provider": str(
                baseline.parent / "libaima-fmha-aotriton.so"
            ),
            "baseline_bundle_provider": str(
                baseline.parent.parent
                / "lib"
                / "libaima-fmha-aotriton.so"
            ),
            "report": str(output / "raw" / "run.json"),
        }
        self.assertEqual(
            surfaces.publicize(
                value,
                engine=engine,
                model_dir=model,
                output_dir=output,
                baseline_engine=baseline,
            ),
            {
                "command": [
                    "${AIMA_BASELINE_ENGINE}",
                    "${AIMA_ENGINE}",
                    "--model-dir",
                    "${AIMA_MODEL_DIR}",
                ],
                "candidate_provider": (
                    "${AIMA_ENGINE_DIR}/libaima-fmha-ck.so"
                ),
                "baseline_provider": (
                    "${AIMA_BASELINE_ENGINE_DIR}/libaima-fmha-aotriton.so"
                ),
                "baseline_bundle_provider": (
                    "${AIMA_BASELINE_BUNDLE_ROOT}/lib/"
                    "libaima-fmha-aotriton.so"
                ),
                "report": "${AIMA_OUTPUT_DIR}/raw/run.json",
            },
        )

    def test_valid_prefix_evidence_can_fail_performance_gate(self) -> None:
        common = {
            "output_token_ids_sha256": "a" * 64,
            "first_token_certified": True,
            "all_decode_tokens_certified": True,
            "mrope_enabled": False,
            "mrope_position_upload_bytes": 0,
            "mrope_full_attention_launches": 0,
            "mrope_decode_steps": 0,
            "prefill_vl_unified_attention_launches": 0,
            "vl_logical_projection_tokens": 0,
            "vl_logical_projection_plan_count": 0,
            "vl_logical_projection_workspace_bytes": 0,
            "vl_logical_projections_enabled": False,
            "vl_logical_projection_plan_build_wall_ms": 0.0,
            "prefix_cache_restore_wall_ms": 0.0,
            "prefix_cache_active_kv_reused": False,
        }
        payload = {
            "schema": "aima-amd395-qwen36/native-resident-session-probe/v1",
            "complete": True,
            "model_loads": 1,
            "request_count": 2,
            "repeat_tokens_identical": True,
            "runtime_python": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
            "requests": [
                {
                    **common,
                    "prefix_cache_lookup": "miss",
                    "prefill_wall_ms": 26_460.0,
                    "decode_tokens_per_second": 31.0,
                },
                {
                    **common,
                    "prefix_cache_lookup": "exact",
                    "prefix_cache_matched_tokens": 32768,
                    "prefix_cache_suffix_tokens": 0,
                    "prefix_cache_suffix_aot_launches": 0,
                    "prefix_cache_suffix_native_launches": 0,
                    "prefix_cache_restore_wall_ms": 8.0,
                    "prefix_cache_active_kv_reused": True,
                    "prefill_wall_ms": 10.0,
                    "decode_tokens_per_second": 31.0031,
                },
            ],
            "qualification": {"engine_sha256": "b" * 64},
        }
        self.assertTrue(
            surfaces.prefix_cache_report_valid(
                payload, engine_sha256="b" * 64
            )
        )
        self.assertFalse(
            surfaces.prefix_cache_report_qualified(
                payload,
                engine_sha256="b" * 64,
                minimum_ttft_speedup=2637.0,
                minimum_decode_retention=1.0003,
            )
        )
        payload["requests"][1]["prefix_cache_active_kv_reused"] = False
        self.assertFalse(
            surfaces.prefix_cache_report_valid(
                payload, engine_sha256="b" * 64
            )
        )

    def test_paired_prefix_medians_own_no_regression_decision(self) -> None:
        measurement_keys = surfaces.prefix_cache_measurement_keys()
        pairs = []
        for pair_index in range(1, 6):
            baseline = {
                "cold_ttft_ms": 24_100.0,
                "hit_ttft_ms": 9.15,
                "ttft_speedup": 2633.879781420765,
                "cold_decode_tps": 28.30,
                "hit_decode_tps": 28.29,
                "decode_retention": 0.9996466431095405,
            }
            candidate = {
                "cold_ttft_ms": 24_300.0,
                "hit_ttft_ms": 9.16,
                "ttft_speedup": 2652.8384279475985,
                "cold_decode_tps": 31.00,
                "hit_decode_tps": 31.005,
                "decode_retention": 1.0001612903225807,
            }
            self.assertEqual(tuple(baseline), measurement_keys)
            self.assertEqual(tuple(candidate), measurement_keys)
            pairs.append(
                {
                    "pair_index": pair_index,
                    "measurements": {
                        "baseline": baseline,
                        "candidate": candidate,
                    },
                    "candidate_over_baseline": {
                        "ttft_speedup": (
                            candidate["ttft_speedup"]
                            / baseline["ttft_speedup"]
                        ),
                        "decode_retention": (
                            candidate["decode_retention"]
                            / baseline["decode_retention"]
                        ),
                    },
                }
            )
        result = surfaces.summarize_paired_prefix_cache(
            pairs, required_pair_count=5
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertGreaterEqual(
            result["paired_candidate_over_baseline_medians"][
                "ttft_speedup"
            ],
            1.0,
        )
        self.assertGreaterEqual(
            result["paired_candidate_over_baseline_medians"][
                "decode_retention"
            ],
            1.0,
        )

    def test_paired_prefix_resume_binds_load_report_and_empty_stderr(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "candidate.json"
            load_report = report.with_suffix(".load.json")
            stderr = report.with_suffix(".stderr.txt")
            load_report.write_text("{}\n", encoding="utf-8")
            stderr.write_text("", encoding="utf-8")
            payload = {
                "qualification": {
                    "load_report_sha256": surfaces.sha256(load_report),
                    "stderr_sha256": surfaces.sha256(stderr),
                }
            }
            self.assertTrue(
                surfaces.paired_prefix_artifacts_valid(report, payload)
            )

            stderr.write_text("unexpected warning\n", encoding="utf-8")
            self.assertFalse(
                surfaces.paired_prefix_artifacts_valid(report, payload)
            )
            stderr.write_text("", encoding="utf-8")
            load_report.write_text('{"changed":true}\n', encoding="utf-8")
            self.assertFalse(
                surfaces.paired_prefix_artifacts_valid(report, payload)
            )


if __name__ == "__main__":
    unittest.main()
