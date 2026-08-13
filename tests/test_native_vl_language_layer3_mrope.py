from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "benchmarks/results/native-vl-language-layer3-mrope-v0.1.0.json"
)


class NativeVlLanguageLayer3MropeTest(unittest.TestCase):
    def test_device_mrope_table_preserves_the_text_rotary_api(self) -> None:
        header = (ROOT / "native/include/aima/native_pointwise.h").read_text(
            encoding="utf-8"
        )
        source = (ROOT / "native/src/native_pointwise.hip.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("launch_prefill_rotary_table", header)
        self.assertIn("launch_prefill_mrope_rotary_table", header)
        self.assertIn("launch_full_attention_head_norm_mrope_prefill", header)
        self.assertIn("const void* positions_i64", header)
        self.assertIn("prefill_mrope_rotary_table_kernel", source)
        self.assertIn("mrope_axis_for_pair", source)
        self.assertIn("triton_bf16_product_rtz", source)
        self.assertIn(
            "launch_full_attention_head_norm_rope_prefill_impl<false>",
            source,
        )
        self.assertIn(
            "launch_full_attention_head_norm_rope_prefill_impl<true>",
            source,
        )
        self.assertIn("__float2bfloat16(cosf(angle))", source)
        self.assertIn("__float2bfloat16(sinf(angle))", source)

    def test_isolated_probe_attributes_table_and_head_norm_rope(self) -> None:
        probe = (
            ROOT
            / "native/tools/vl_language_layer3_mrope_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vl-language-layer3-mrope-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("generated_effective_cos", probe)
        self.assertIn("generated_effective_sin", probe)
        self.assertIn("head_norm_q", probe)
        self.assertIn("head_norm_k", probe)
        self.assertIn("oracle_table_rotary_q", probe)
        self.assertIn("oracle_table_rotary_k", probe)
        self.assertIn("generated_table_rotary_q", probe)
        self.assertIn("generated_table_rotary_k", probe)
        self.assertIn("kMeasuredRuns = 5", probe)
        self.assertIn("runtime_python", probe)
        self.assertIn("native_pointwise.hip.cpp", build)
        self.assertIn("--offload-arch=gfx1151", build)

    def test_probe_is_bound_to_the_five_case_capture_schema(self) -> None:
        probe = (
            ROOT
            / "native/tools/vl_language_layer3_mrope_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "vl-language-layer3-mrope-diagnostic-oracle/v1", probe
        )
        self.assertIn("manifest.at(\"cases\").size() != 5", probe)
        self.assertIn("json::array({11, 11, 10})", probe)
        self.assertIn("relative_l2_maximum", probe)
        self.assertIn("cosine_minimum", probe)

    def test_qualification_is_exact_hash_bound_and_non_overclaiming(self) -> None:
        import hashlib
        import json

        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "764fd57a08105f79b2d86cdcba45f9c25b17a864",
        )
        for record in result["source"]["files"]:
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / record["path"]).read_bytes()
                ).hexdigest(),
                record["sha256"],
            )
        capture = result["reference_capture"]
        self.assertEqual(capture["component_payloads_exact"], 120)
        self.assertEqual(capture["component_payloads_total"], 120)
        self.assertTrue(
            capture["all_component_shapes_dtypes_and_payloads_repeat_exact"]
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / capture["capture_script"]["path"]).read_bytes()
            ).hexdigest(),
            capture["capture_script"]["sha256"],
        )

        run = result["qualification_run"]
        self.assertTrue(run["all_bit_exact"])
        self.assertTrue(run["all_outputs_repeat_deterministic"])
        self.assertTrue(run["all_cross_capture_outputs_byte_exact"])
        self.assertEqual(run["total_exact_elements"], run["total_elements"])
        self.assertEqual(
            sum(case["comparison_elements"] for case in run["cases"]),
            run["total_elements"],
        )
        self.assertEqual(
            {case["case_id"] for case in run["cases"]},
            {
                "image_local_png",
                "video_local_mp4",
                "multi_image",
                "multi_video",
                "mixed_image_video",
            },
        )
        for case in run["cases"]:
            self.assertEqual(
                case["exact_comparison_elements"],
                case["comparison_elements"],
            )
            self.assertTrue(case["all_eight_comparisons_bit_exact"])

        decision = result["decision"]
        self.assertEqual(
            decision["language_layer3_mrope_table_and_qk_consumption"],
            "passed",
        )
        for gate in (
            "g1_full_vl_functional_parity",
            "g2_vl_correctness_parity",
            "g3_text_product_no_regression",
            "g4_native_vl_performance",
            "g5_native_release_product",
        ):
            self.assertFalse(decision[gate])


if __name__ == "__main__":
    unittest.main()
