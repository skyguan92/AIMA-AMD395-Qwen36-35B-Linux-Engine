#!/usr/bin/env python3
"""Contracts for the native VL language layer-0 qualification boundary."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE_MANIFEST = ROOT / "benchmarks/results/vl-oracle-manifest.json"
QUALIFICATION_RESULT = (
    ROOT / "benchmarks/results/native-vl-language-layer0-v0.1.0.json"
)


class NativeVlLanguageLayer0Test(unittest.TestCase):
    def test_probe_executes_the_product_q1024_layer_without_oracle_seeds(self) -> None:
        probe = (
            ROOT / "native/tools/vl_language_layer0_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        moe_header = (
            ROOT / "native/include/aima/native_moe_prefill.h"
        ).read_text(encoding="utf-8")
        moe_source = (
            ROOT / "native/src/native_moe_prefill.hip.cpp"
        ).read_text(encoding="utf-8")
        linear_header = (
            ROOT / "native/include/aima/native_linear_prefill.h"
        ).read_text(encoding="utf-8")
        linear_source = (
            ROOT / "native/src/native_linear_prefill.hip.cpp"
        ).read_text(encoding="utf-8")
        resident_source = (
            ROOT / "native/src/native_resident_engine.hip.cpp"
        ).read_text(encoding="utf-8")
        pointwise_source = (
            ROOT / "native/src/native_pointwise.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vl-language-layer0-probe.sh"
        ).read_text(encoding="utf-8")
        shape_lab = (
            ROOT / "benchmarks/shape-lab/four_layer_mini_engine.py"
        ).read_text(encoding="utf-8")
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("constexpr std::size_t kBucketTokens = 1024", probe)
        self.assertIn("native_prefill_layer_input_pointer", probe)
        self.assertIn("native_prefill_layer_output_pointer", probe)
        self.assertIn("reference_layer0_first_tensor_pointer", probe)
        self.assertIn('tensor_pointer(sequence, "v_new")', probe)
        self.assertIn("linear_options.seed_layer_input = false", probe)
        self.assertIn("linear_options.collect_oracle_comparisons = false", probe)
        self.assertIn("linear_options.active_tokens = 0", probe)
        self.assertIn("linear_options.comparison_tokens = prompt_tokens", probe)
        self.assertIn("linear_options.exact_b_projection_tokens", probe)
        self.assertIn("moe_options.seed_post_attention = false", probe)
        self.assertIn("moe_options.collect_oracle_comparisons = false", probe)
        self.assertIn("moe_options.active_tokens = 0", probe)
        self.assertIn("moe_options.comparison_tokens = prompt_tokens", probe)
        self.assertIn("linear_options.sequence_oracle_dir", probe)
        self.assertIn("moe_options.chain_output_oracle_dir", probe)
        self.assertIn('"first_failed_diagnostic_stage"', probe)
        self.assertIn('"diagnostic_comparisons"', probe)
        self.assertIn('"first_failed_seeded_moe_stage"', probe)
        self.assertIn('"seeded_moe_diagnostic_comparisons"', probe)
        self.assertIn(
            'post_attention_h2_oracle_label = "diagnostic-h2"', probe
        )
        self.assertIn('"diagnostic-router_indices"', probe)
        self.assertIn("diagnostic.router_expert_sets_exact", probe)
        self.assertIn('"router_expert_set_rows_exact"', probe)
        self.assertIn('"seeded_router_expert_set_rows_exact"', probe)
        self.assertIn('"linear_projection_fused_full_sequence"', probe)
        self.assertIn('"diagnostic-qkv"', linear_source)
        self.assertIn('"diagnostic-fused-input"', linear_source)
        self.assertIn('"linear_convolution_full_sequence"', probe)
        self.assertIn('"fla_beta_full_sequence"', probe)
        self.assertIn('"diagnostic-conv"', linear_source)
        self.assertIn("std::size_t active_tokens = 0", moe_header)
        self.assertIn("std::size_t active_tokens = 0", linear_header)
        self.assertIn('"num_valid_tokens"', moe_source)
        self.assertIn("tokens > bucket_tokens", moe_source)
        self.assertIn("tokens > bucket_tokens", linear_source)
        self.assertIn("kMeasuredRuns = 5", probe)
        self.assertIn("q1024-output1", build)
        self.assertIn('Q8192_DIR="${ROOT}/native/aot/gfx1151/q8192-output2"', build)
        self.assertIn('--schedule "${Q8192_DIR}/decode-schedule.json"', build)
        self.assertIn("build-native-vl-language-layer0-probe", makefile)

        self.assertIn("const bool q1024_official_fla", linear_source)
        self.assertIn("merge_16x16_to_64x64_inverse_kernel", linear_source)
        self.assertIn("launch_prefill_rmsnorm_2048(", linear_source)
        self.assertIn("launch_prefill_add_rmsnorm_2048(", linear_source)
        self.assertIn("exact_linear_b_projection_kernel", linear_source)
        self.assertIn("exact_b_tokens != 0", linear_source)
        self.assertIn("fmaf(", linear_source)
        self.assertIn("request.multimodal_cache_namespace.empty()", resident_source)
        self.assertIn("if (layer_index == 0", resident_source)
        self.assertIn("segment.input_tokens <= 64", resident_source)
        self.assertIn(
            "attention_options.exact_b_projection_tokens",
            resident_source,
        )

        eager_rmsnorm = pointwise_source.split(
            "__global__ void prefill_rmsnorm_2048_kernel(", 1
        )[1].split("__global__ void prefill_add_rmsnorm_2048_kernel", 1)[0]
        self.assertIn("constexpr unsigned kVectorWidth = 4", eager_rmsnorm)
        self.assertIn("constexpr unsigned kRowsPerBlock = 16", eager_rmsnorm)
        self.assertIn("constexpr unsigned kRowThreads = 32", eager_rmsnorm)
        self.assertIn("float accumulator[kVectorWidth]", eager_rmsnorm)
        self.assertIn("volatile float squared", eager_rmsnorm)
        self.assertIn(
            "for (unsigned offset = 1; offset < kRowThreads; offset <<= 1)",
            eager_rmsnorm,
        )
        self.assertIn("volatile float variance_with_epsilon", eager_rmsnorm)

        prefill_conv = shape_lab.split(
            "def triton_prefill_direct_conv_kernel(", 1
        )[1].split("    @triton.jit", 1)[0]
        self.assertNotIn(".to(tl.float32)", prefill_conv)
        self.assertIn(
            "acc += (state_values + raw_values) * weight_values[None, :]",
            prefill_conv,
        )
        self.assertIn("each BF16", prefill_conv)

        gated_norm = pointwise_source.split(
            "__global__ void linear_gated_norm_fused_kernel(", 1
        )[1].split("__global__ void bf16_rowwise_variance_128_pytorch_kernel", 1)[0]
        self.assertIn("const float normalized", gated_norm)
        self.assertNotIn("const __hip_bfloat16 normalized", gated_norm)
        self.assertIn("__float2bfloat16(normalized * silu)", gated_norm)
        self.assertIn("const unsigned lane_in_row = lane & 15U", gated_norm)
        self.assertIn("row_repeat < 2", gated_norm)
        self.assertIn("__shfl_xor(square_sum, offset, 16)", gated_norm)
        self.assertIn("rsqrtf(fmaf(square_sum", gated_norm)
        self.assertIn("exp2f(-gate_value", gated_norm)
        self.assertNotIn("pytorch_rounded_rsqrtf", gated_norm)

        shared_activation = moe_source.split(
            "__global__ void shared_silu_multiply_batched_kernel(", 1
        )[1].split("__global__ void shared_sigmoid_scale_batched_kernel", 1)[0]
        self.assertIn("const __hip_bfloat16 silu_bf16", shared_activation)
        self.assertIn("__bfloat162float(silu_bf16) * up_value", shared_activation)

        router = moe_source.split(
            "__global__ void router_topk8_softmax_256_kernel(", 1
        )[1].split("__global__ void moe_align_block32_256_kernel", 1)[0]
        self.assertIn("std::int32_t* indices_i32, float* weights", router)
        self.assertIn("constexpr int kRouterWave = 32", router)
        self.assertIn(
            "constexpr int kValuesPerLane = kExperts / kRouterWave", router
        )
        self.assertIn("__shfl_xor(row_sum, mask, kRouterWave)", router)
        self.assertIn("row_chunk[value] *= reciprocal_row_sum", router)
        self.assertIn("selected_sum += max_probability", router)
        self.assertIn(
            "weights[base + rank] = selected_probabilities[rank] / denominator",
            router,
        )
        self.assertNotIn("__float2bfloat16", router)
        self.assertIn(
            "router_topk8_softmax_256_kernel, dim3(tokens), dim3(32)",
            moe_source,
        )

        routing_weights = shape_lab.split(
            "    def routing_weights(scores: Any) -> Any:", 1
        )[1].split("    def expanded_routed_moe", 1)[0]
        self.assertIn('if mode == "prefill"', routing_weights)
        self.assertIn("return normalized", routing_weights)
        self.assertIn("FusedTopKRouter", routing_weights)
        self.assertIn(
            'compare_stage_if_present("router_weights", "float32"',
            moe_source,
        )

        capture = (ROOT / "scripts/capture-vllm-vl-oracles.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _first_tensor", capture)
        self.assertIn('"language_layer_0": language.model.layers[0]', capture)

    def test_q1024_closure_uses_the_serving_bt64_fla_chain(self) -> None:
        closure = ROOT / "native/aot/gfx1151/q1024-output1"
        manifest = json.loads(
            (closure / "manifest.json").read_text(encoding="utf-8")
        )
        schedule = json.loads(
            (closure / "prefill-schedule.json").read_text(encoding="utf-8")
        )
        self.assertEqual(schedule["request"]["context_tokens"], 1024)
        self.assertEqual(manifest["kernel_count"], 13)

        expected_prefix = [
            "triton_rmsnorm_kernel",
            "triton_prefill_direct_conv_kernel",
            "_fused_post_conv_kernel",
            "chunk_local_cumsum_scalar_kernel",
            "chunk_scaled_dot_kkt_fwd_kernel",
            "merge_16x16_to_64x64_inverse_kernel",
            "recompute_w_u_fwd_kernel",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            "chunk_fwd_kernel_o",
            "triton_prefill_fused_add_rmsnorm_kernel",
            "fused_moe_kernel",
            "fused_moe_kernel",
        ]
        self.assertEqual(
            [launch["symbol"] for launch in schedule["schedule"][:12]],
            expected_prefix,
        )

        kernels = {kernel["kernel_hash"]: kernel for kernel in manifest["kernels"]}
        official_fla = set(expected_prefix[3:9])
        for launch in schedule["schedule"]:
            if (
                launch["layer_type"] == "linear_attention"
                and launch["symbol"] in official_fla
            ):
                self.assertEqual(
                    kernels[launch["kernel_hash"]]["compile_constants"]["BT"],
                    64,
                )
        for kernel in manifest["kernels"]:
            image = closure / kernel["image"]["path"]
            self.assertTrue(image.is_file())
            self.assertEqual(
                hashlib.sha256(image.read_bytes()).hexdigest(),
                kernel["image"]["sha256"],
            )

    def test_multicontext_registry_keeps_q8192_compatibility_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prefill-registry.cpp"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/generate-native-decode-registry.py"),
                    "--phase",
                    "prefill",
                    "--schedule",
                    str(
                        ROOT
                        / "native/aot/gfx1151/q1024-output1/prefill-schedule.json"
                    ),
                    "--aot-manifest",
                    str(ROOT / "native/aot/gfx1151/q1024-output1/manifest.json"),
                    "--schedule",
                    str(
                        ROOT
                        / "native/aot/gfx1151/q8192-output2/prefill-schedule.json"
                    ),
                    "--aot-manifest",
                    str(ROOT / "native/aot/gfx1151/q8192-output2/manifest.json"),
                    "--output-cpp",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            generated = output.read_text(encoding="utf-8")
        self.assertIn("return native_prefill_schedule(8192, count);", generated)
        self.assertIn("return native_prefill_schedule_sha256(8192);", generated)

    def test_layer0_qualification_is_hash_bound_and_threshold_complete(self) -> None:
        result = json.loads(QUALIFICATION_RESULT.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "764fd57a08105f79b2d86cdcba45f9c25b17a864",
        )
        for record in result["source"]["files"]:
            self.assertEqual(
                hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest(),
                record["sha256"],
            )
        for dependency in result["dependencies"].values():
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / dependency["path"]).read_bytes()
                ).hexdigest(),
                dependency["sha256"],
            )

        thresholds = result["reference_thresholds"]
        run = result["qualification_run"]
        cases = run["cases"]
        self.assertEqual(len(cases), 5)
        self.assertEqual(sum(case["prompt_tokens"] for case in cases), 585)
        self.assertEqual(
            sum(case["elements"] for case in cases),
            run["aggregate"]["output_elements"],
        )
        for case in cases:
            self.assertEqual(case["diagnostic_comparison_count"], 33)
            self.assertTrue(case["input_norm_bit_exact"])
            self.assertTrue(case["repeat_deterministic"])
            self.assertLessEqual(
                case["relative_l2_error"],
                thresholds["maximum_relative_l2_error"],
            )
            self.assertGreaterEqual(
                case["cosine_similarity"],
                thresholds["minimum_cosine_similarity"],
            )
            self.assertEqual(
                case["main_router_expert_set_rows_exact"],
                case["prompt_tokens"],
            )
            self.assertEqual(
                case["seeded_router_expert_set_rows_exact"],
                case["prompt_tokens"],
            )

        aggregate = run["aggregate"]
        self.assertEqual(
            aggregate["main_router_expert_set_rows_exact"],
            aggregate["main_router_expert_set_rows"],
        )
        self.assertEqual(
            aggregate["seeded_router_expert_set_rows_exact"],
            aggregate["seeded_router_expert_set_rows"],
        )
        self.assertTrue(aggregate["all_required_diagnostics_passed"])

        closure = result["runtime_closure"]
        self.assertTrue(closure["prefill_schedule_probe"]["complete"])
        self.assertTrue(closure["decode_schedule_probe"]["complete"])
        self.assertTrue(closure["aot_closure_probe"]["complete"])
        self.assertEqual(closure["aot_closure_probe"]["loaded_count"], 59)
        self.assertFalse(closure["portable_package_qualified"])

        decision = result["decision"]
        self.assertEqual(decision["language_layer0_boundary"], "passed")
        for gate in (
            "g1_full_vl_functional_parity",
            "g2_vl_correctness_parity",
            "g3_text_product_no_regression",
            "g4_native_vl_performance",
            "g5_native_release_product",
        ):
            self.assertFalse(decision[gate])

    def test_frozen_layer0_boundaries_cover_all_blocking_cases(self) -> None:
        manifest = json.loads(ORACLE_MANIFEST.read_text(encoding="utf-8"))
        expected = {
            "image_local_png": (81, "730f078bc97c7553f40f2f9f1c92c72608152f7cac7b18afea133b55e583a3cb"),
            "video_local_mp4": (63, "bc822e132eeeee9824c6ceeb43aabcafe0a31843ca88f990ff18734250bc192e"),
            "multi_image": (182, "7603077857c6ff64b7c4fd03fc76dfb5bbfabb9ac85be2d387b32df433531e25"),
            "multi_video": (128, "69a4f890bbee2bf79f979b0198fb6710e515cc79aae3c67580c957059ad5910c"),
            "mixed_image_video": (131, "277c5cbc0833fbd4a40f845669a6beab43478b54a0fe27d1f64406687a8cc446"),
        }
        self.assertEqual({case["case_id"] for case in manifest["cases"]}, set(expected))
        for case in manifest["cases"]:
            tokens, digest = expected[case["case_id"]]
            injected = case["boundaries"]["injected_embeddings"]
            layer0 = case["boundaries"]["language_layer_0"]
            self.assertEqual(injected["shape"], [tokens, 2048])
            self.assertEqual(layer0["shape"], [tokens, 2048])
            self.assertEqual(layer0["sha256"], digest)
            self.assertEqual(layer0["bytes"], tokens * 2048 * 2)


if __name__ == "__main__":
    unittest.main()
