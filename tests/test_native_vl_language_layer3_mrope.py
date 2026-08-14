from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "benchmarks/results/native-vl-language-layer3-mrope-v0.1.0.json"
)
FULL_RESULT = (
    ROOT / "benchmarks/results/native-vl-language-full-v0.1.0.json"
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

    def test_resident_request_selects_mrope_without_changing_text_dispatch(self) -> None:
        resident_header = (
            ROOT / "native/include/aima/native_resident_engine.h"
        ).read_text(encoding="utf-8")
        resident = (
            ROOT / "native/src/native_resident_engine.hip.cpp"
        ).read_text(encoding="utf-8")
        full_header = (
            ROOT / "native/include/aima/native_full_prefill.h"
        ).read_text(encoding="utf-8")
        full = (ROOT / "native/src/native_full_prefill.hip.cpp").read_text(
            encoding="utf-8"
        )
        decode_header = (
            ROOT / "native/include/aima/native_decode_runner.h"
        ).read_text(encoding="utf-8")
        decode = (
            ROOT / "native/src/native_decode_runner.hip.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("std::optional<NativeMropePlan> mrope_plan", resident_header)
        self.assertIn("mrope_position_state_bytes", resident_header)
        self.assertIn("hipMalloc resident M-RoPE positions", resident)
        self.assertIn("hipMemcpyAsync resident M-RoPE positions", resident)
        self.assertIn("hipMemsetAsync resident M-RoPE padding", resident)
        self.assertIn("attention_options.mrope_positions_i64", resident)
        self.assertIn("segment.input_offset", resident)
        self.assertIn("native_mrope_decode_position", resident)

        self.assertIn("const void* mrope_positions_i64", full_header)
        self.assertIn("const bool use_mrope", full)
        self.assertIn("launch_prefill_mrope_rotary_table", full)
        self.assertIn("launch_full_attention_head_norm_mrope_prefill", full)
        self.assertIn("launch_prefill_rotary_table", full)
        self.assertIn("launch_full_attention_head_norm_rope_prefill", full)
        self.assertIn("const bool use_vl_unified_attention = use_mrope", full)
        self.assertIn(
            "provider.launch(q, attention_k, attention_v, attention_f32, tokens",
            full,
        )

        self.assertIn("std::size_t rotary_position", decode_header)
        self.assertIn("position, position, input_token_id", decode)
        self.assertIn("decode_rotary_kernel", decode)
        self.assertIn("rotary_position, static_cast<float*>(cosine)", decode)

    def test_unified_attention_artifact_is_embedded_and_hash_bound(self) -> None:
        import hashlib
        import json

        closure = (
            ROOT / "native/aot/gfx1151/vl-unified-attention-v0.1.0"
        )
        manifest = json.loads(
            (closure / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["kernel_count"], 1)
        kernel = manifest["kernels"][0]
        self.assertEqual(kernel["symbol"], "kernel_unified_attention_2d")
        self.assertEqual(kernel["metadata"]["num_warps"], 4)
        self.assertEqual(kernel["metadata"]["shared"], 32768)
        image = closure / kernel["image"]["path"]
        self.assertEqual(
            hashlib.sha256(image.read_bytes()).hexdigest(),
            kernel["image"]["sha256"],
        )

        source = (
            ROOT / "native/src/native_vl_unified_attention.hip.cpp"
        ).read_text(encoding="utf-8")
        runtime_build = (ROOT / "scripts/build-native-runtime.sh").read_text(
            encoding="utf-8"
        )
        resident = (
            ROOT / "native/src/native_resident_engine.hip.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn(kernel["kernel_hash"], source)
        self.assertIn('"kernel_unified_attention_2d"', source)
        self.assertIn("vl-unified-attention-v0.1.0/manifest.json", runtime_build)
        self.assertIn("native_vl_unified_attention.hip.cpp", runtime_build)
        self.assertIn("std::make_unique<NativeVlUnifiedAttentionPlan>", resident)
        self.assertIn("attention_options.vl_unified_attention", resident)

    def test_composed_probe_executes_layers_zero_through_three(self) -> None:
        probe = (
            ROOT
            / "native/tools/vl_language_layer3_composed_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vl-language-layer3-composed-probe.sh"
        ).read_text(encoding="utf-8")
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        capture = (
            ROOT / "scripts/capture-vllm-vl-language-prefix-diagnostics.py"
        ).read_text(encoding="utf-8")

        self.assertIn("constexpr std::size_t kBucketTokens = 1024", probe)
        self.assertIn("for (std::size_t layer_index = 0; layer_index < 3", probe)
        self.assertIn("full_options.layer_index = 3", probe)
        self.assertIn("full_options.mrope_positions_i64", probe)
        self.assertIn("full_options.comparison_tokens = prompt_tokens", probe)
        self.assertIn("moe_options.chain_output_oracle_label", probe)
        self.assertIn(
            "moe_options.chain_output_oracle_dir = prefix_oracle_dir", probe
        )
        self.assertIn('"same_request_layer_output"', probe)
        self.assertIn("layer1_exact_input_moe_diagnostic", probe)
        self.assertIn("layer1_exact_input_attention_diagnostic", probe)
        self.assertIn(
            'attention_diagnostic_options.layer_input_oracle_label =', probe
        )
        self.assertIn("diagnostic_options.seed_post_attention = true", probe)
        self.assertIn("return-layer_body-router_indices", probe)
        self.assertIn(
            '"layer-003-return-layer_body-router_indices"', probe
        )
        self.assertIn("kMeasuredRuns = 5", probe)
        self.assertIn("relative_l2_error <= 0.002", probe)
        self.assertIn("cosine_similarity >= 0.999", probe)
        self.assertIn("execute_full_language", probe)
        self.assertIn("for (std::size_t layer_index = 0; layer_index < 40", probe)
        self.assertIn('boundaries.at("language_final_norm")', probe)
        self.assertIn('boundaries.at("full_vocabulary_logits")', probe)
        self.assertIn("kl_divergence < 0.005", probe)
        self.assertIn("native_vl_unified_attention_launches == 10", probe)
        self.assertIn('std::string_view(argv[12]) != "full-language"', probe)
        self.assertIn("runtime_python", probe)
        self.assertIn("q1024-output1", build)
        self.assertIn("native_full_prefill.hip.cpp", build)
        self.assertIn("--offload-arch=gfx1151", build)
        self.assertIn("build-native-vl-language-layer3-composed-probe", makefile)
        self.assertIn("vl-language-prefix-diagnostic-oracle/v3", capture)
        self.assertIn("LINEAR_LAYERS = (0, 1, 2)", capture)
        self.assertIn("FULL_ATTENTION_LAYER = 3", capture)
        self.assertIn("MOE_DIAGNOSTIC_SUFFIXES", capture)
        self.assertIn("router.select_experts = wrapped", capture)
        self.assertIn("attention.forward = wrapped", capture)
        self.assertIn('"attention_core_output"', capture)
        self.assertIn('"routed_moe_output"', capture)
        self.assertIn("layer_003_attention_input", capture)
        self.assertIn("layer_003_output", capture)
        self.assertIn("frozen_layer0_comparison", capture)
        self.assertIn("frozen_layer3_attention_input_comparison", capture)
        self.assertIn("frozen_layer3_output_comparison", capture)
        self.assertIn("source must be a clean commit", capture)

    def test_qualification_is_exact_hash_bound_and_non_overclaiming(self) -> None:
        import hashlib
        import json

        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "c44d1997c93349c1eec71c5f1a2b678a8439864c",
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

    def test_full_language_qualification_is_hash_bound_and_threshold_complete(
        self,
    ) -> None:
        import hashlib
        import json

        result = json.loads(FULL_RESULT.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "c44d1997c93349c1eec71c5f1a2b678a8439864c",
        )
        for record in result["source"]["files"]:
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / record["path"]).read_bytes()
                ).hexdigest(),
                record["sha256"],
            )

        references = result["reference_oracles"]
        full_model = references["full_model"]
        self.assertEqual(
            hashlib.sha256(
                (ROOT / full_model["path"]).read_bytes()
            ).hexdigest(),
            full_model["sha256"],
        )
        capture_script = references["all_layer_diagnostics"][
            "capture_script"
        ]
        self.assertEqual(
            hashlib.sha256(
                (ROOT / capture_script["path"]).read_bytes()
            ).hexdigest(),
            capture_script["sha256"],
        )

        thresholds = result["reference_thresholds"]
        run = result["qualification_run"]
        aggregate = run["aggregate"]
        cases = run["cases"]
        self.assertEqual(len(cases), 5)
        self.assertEqual(sum(case["prompt_tokens"] for case in cases), 585)
        self.assertEqual(
            sum(case["final_norm_elements"] for case in cases),
            aggregate["final_norm_elements"],
        )
        self.assertEqual(
            sum(case["final_norm_exact_elements"] for case in cases),
            aggregate["final_norm_exact_elements"],
        )
        self.assertEqual(
            sum(case["selected_logits_elements"] for case in cases),
            aggregate["selected_logits_elements"],
        )
        self.assertEqual(
            aggregate["selected_logits_exact_elements"],
            aggregate["selected_logits_elements"],
        )
        self.assertEqual(
            aggregate["top1_matches"], aggregate["top1_rows"]
        )
        for case in cases:
            self.assertTrue(case["repeat_deterministic"])
            self.assertLessEqual(
                case["final_norm_relative_l2_error"],
                thresholds["maximum_relative_l2_error"],
            )
            self.assertGreaterEqual(
                case["final_norm_cosine_similarity"],
                thresholds["minimum_cosine_similarity"],
            )
            self.assertEqual(
                case["selected_logits_exact_elements"],
                case["selected_logits_elements"],
            )
            self.assertLess(
                case["maximum_logits_kl_divergence"],
                thresholds["maximum_logits_kl_divergence"],
            )
            self.assertEqual(case["top1_matches"], case["selected_logits_rows"])

        diagnostic = run["single_image_layer_diagnostic"]
        self.assertTrue(diagnostic["all_40_layer_outputs_bit_exact"])
        self.assertTrue(diagnostic["all_40_router_expert_sets_exact"])
        self.assertEqual(
            diagnostic["layer_output_exact_elements"],
            diagnostic["layer_output_elements"],
        )
        self.assertEqual(
            diagnostic["router_expert_set_rows_exact"],
            diagnostic["router_expert_set_rows"],
        )

        decision = result["decision"]
        self.assertEqual(decision["language_boundary_gate"], "passed")
        self.assertEqual(
            decision["teacher_forced_full_vocabulary_logits_gate"],
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
