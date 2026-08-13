#!/usr/bin/env python3
"""Contracts for the native VL language layer-0 qualification boundary."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE_MANIFEST = ROOT / "benchmarks/results/vl-oracle-manifest.json"


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
        self.assertIn("linear_options.active_tokens = prompt_tokens", probe)
        self.assertIn("moe_options.seed_post_attention = false", probe)
        self.assertIn("moe_options.collect_oracle_comparisons = false", probe)
        self.assertIn("moe_options.active_tokens = prompt_tokens", probe)
        self.assertIn("linear_options.sequence_oracle_dir", probe)
        self.assertIn("moe_options.chain_output_oracle_dir", probe)
        self.assertIn('"first_failed_diagnostic_stage"', probe)
        self.assertIn('"diagnostic_comparisons"', probe)
        self.assertIn('"first_failed_seeded_moe_stage"', probe)
        self.assertIn('"seeded_moe_diagnostic_comparisons"', probe)
        self.assertIn(
            'post_attention_h2_oracle_label = "diagnostic-h2"', probe
        )
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

        shared_activation = moe_source.split(
            "__global__ void shared_silu_multiply_batched_kernel(", 1
        )[1].split("__global__ void shared_sigmoid_scale_batched_kernel", 1)[0]
        self.assertIn("const __hip_bfloat16 silu_bf16", shared_activation)
        self.assertIn("__bfloat162float(silu_bf16) * up_value", shared_activation)

        capture = (ROOT / "scripts/capture-vllm-vl-oracles.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _first_tensor", capture)
        self.assertIn('"language_layer_0": language.model.layers[0]', capture)

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
