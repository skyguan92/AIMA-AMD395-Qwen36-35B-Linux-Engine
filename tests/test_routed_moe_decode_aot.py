from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/trace-vllm-routed-moe-decode-aot.py"
PROBE = ROOT / "native/tools/routed_moe_decode_aot_probe.hip.cpp"
BUILD = ROOT / "scripts/build-native-routed-moe-decode-aot-probe.sh"
AOT_ROOT = ROOT / "native/aot/gfx1151/routed-moe-decode-v0.1.0"
EXACT_HYBRID_ROOT = (
    ROOT / "native/aot/gfx1151/routed-moe-exact-hybrid-v0.1.0"
)
EVIDENCE_ROOT = ROOT / "benchmarks/runs/routed-moe-decode-aot-v0.1.0"
GATE_UP_HASH = (
    "3ef02098201cfc66fd82896bf7c1abc4fe406c41fb911c2b58e0023e1eca1c99"
)
DOWN_HASH = (
    "775c54180f9368197b9493aa15e604d3f7622519a20fc322e827e2d51a979b75"
)
HYBRID_GATE_UP_HASH = (
    "2aabec08044ef14f7f5f08e4854473bd85f15e17feb31d52b10cdf94a801a4ce"
)
SPARSE_CORRECTION_HASH = (
    "0e6da1f589b4787c411264ea8288e26cb4259c61da6f88e1a1f7d8b4e3e74dab"
)


class RoutedMoeDecodeAotTests(unittest.TestCase):
    def test_trace_driver_freezes_current_singleton_geometry(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn("fused_topk", source)
        self.assertIn("fused_experts", source)
        self.assertIn("hidden_size = 2_048", source)
        self.assertIn("intermediate_size = 512", source)
        self.assertIn("experts = 256", source)
        self.assertIn("top_k = 8", source)
        self.assertIn("renormalize=True", source)
        self.assertIn('"topk_weights_dtype": str(topk_weights.dtype)', source)
        self.assertIn('"qualified_for_aot_capture": True', source)
        self.assertIn(
            '"qualified_for_native_decode_replacement": False', source
        )
        self.assertIn("not_a_model_weight_correctness_oracle", source)
        self.assertIn("not_a_promotion_result", source)

    def test_model_weight_probe_freezes_every_routed_stage(self) -> None:
        source = PROBE.read_text(encoding="utf-8")
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("launch_bf16_wvsplitk", source)
        self.assertIn("router_topk8_softmax_256_kernel", source)
        self.assertIn("const __hip_bfloat16 silu_bf16", source)
        self.assertIn("routed_sum8_kernel", source)
        self.assertIn("AotLaunchConfig{512, 1, 1, 4, 32, 16384}", source)
        self.assertIn("AotLaunchConfig{1024, 1, 1, 4, 32, 16384}", source)
        for name in (
            "router_logits",
            "router_weights",
            "router_indices",
            "routed_gate_up_projection",
            "routed_activation",
            "routed_weighted_expert_outputs",
            "routed_moe_output",
        ):
            self.assertIn(f'\"{name}\"', source)
        self.assertIn('"qualified_for_native_decode_replacement", false', source)
        self.assertIn('"promotion_result", false', source)
        self.assertIn('"end_to_end_router_outputs_consumed", true', source)
        self.assertIn('"end_to_end_routed_moe_output"', source)
        self.assertIn("routed_moe_decode_aot_probe.hip.cpp", build)
        self.assertIn("native_weight_store.hip.cpp", build)

    def test_exported_closure_and_product_integration_are_pinned(self) -> None:
        manifest = json.loads(
            (AOT_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["schema"], "aima-amd395-qwen36/native-aot-closure/v1"
        )
        self.assertEqual(manifest["kernel_count"], 2)
        self.assertEqual(manifest["kernel_symbol_count"], 1)
        kernels = {kernel["kernel_hash"]: kernel for kernel in manifest["kernels"]}
        self.assertEqual(set(kernels), {GATE_UP_HASH, DOWN_HASH})
        expected_images = {
            GATE_UP_HASH: "86cc097b8f6ca7dd4f239d91b3a8368ec7db6e4a9c85bc5f748848797540750b",
            DOWN_HASH: "780ad77c75c0b7db065603bae69718050051f879ec5ee9f9be89aadd0355c358",
        }
        for kernel_hash, kernel in kernels.items():
            image = AOT_ROOT / kernel["image"]["path"]
            self.assertEqual(
                hashlib.sha256(image.read_bytes()).hexdigest(),
                expected_images[kernel_hash],
            )
            self.assertEqual(kernel["symbol"], "fused_moe_kernel")
            arguments = kernel["launch_variants"][0]["arguments"]
            topk_weights = next(
                value for value in arguments if value["name"] == "topk_weights_ptr"
            )
            self.assertEqual(topk_weights["abi_type"], "*fp32")

        hybrid_manifest = json.loads(
            (EXACT_HYBRID_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            hybrid_manifest["schema"],
            "aima-amd395-qwen36/native-aot-closure/v1",
        )
        self.assertEqual(
            hybrid_manifest["status"],
            "qualified_exact_hybrid_routed_gate_up",
        )
        hybrid_kernels = {
            kernel["kernel_hash"]: kernel
            for kernel in hybrid_manifest["kernels"]
        }
        self.assertEqual(
            set(hybrid_kernels),
            {HYBRID_GATE_UP_HASH, SPARSE_CORRECTION_HASH},
        )
        hybrid_images = {
            HYBRID_GATE_UP_HASH:
                "d3e5ce8a26568e9707e525df1660fcac7ba43ce8e894c411eeec77d2cf3085b4",
            SPARSE_CORRECTION_HASH:
                "40958cc9168155be48eaaed800efb657c0683b56b44b2db71012702ae380e700",
        }
        for kernel_hash, kernel in hybrid_kernels.items():
            image = EXACT_HYBRID_ROOT / kernel["image"]["path"]
            self.assertEqual(
                hashlib.sha256(image.read_bytes()).hexdigest(),
                hybrid_images[kernel_hash],
            )
        self.assertEqual(
            hybrid_kernels[HYBRID_GATE_UP_HASH]["metadata"][
                "error_coefficient"
            ],
            0.000002,
        )
        self.assertTrue(
            hybrid_kernels[HYBRID_GATE_UP_HASH]["metadata"][
                "correct_subnormals"
            ]
        )

        runtime_build = (
            ROOT / "scripts/build-native-runtime.sh"
        ).read_text(encoding="utf-8")
        runtime = (
            ROOT / "native/src/native_routed_moe.hip.cpp"
        ).read_text(encoding="utf-8")
        workspace = (
            ROOT / "native/src/native_decode_workspace.hip.cpp"
        ).read_text(encoding="utf-8")
        linear = (
            ROOT / "native/src/native_linear_layer.hip.cpp"
        ).read_text(encoding="utf-8")
        full = (ROOT / "native/src/native_full_layer.hip.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("ROUTED_MOE_DECODE_MANIFEST", runtime_build)
        self.assertIn("ROUTED_MOE_EXACT_HYBRID_MANIFEST", runtime_build)
        self.assertIn("native_routed_moe.hip.cpp", runtime_build)
        self.assertIn(HYBRID_GATE_UP_HASH, runtime)
        self.assertIn(SPARSE_CORRECTION_HASH, runtime)
        self.assertIn(DOWN_HASH, runtime)
        self.assertIn("router_weights_fp32", runtime)
        self.assertIn("buffers.activation_bf16, executor, stream", runtime)
        self.assertIn("*num_tokens_post_padded = 128", runtime)
        self.assertIn("const __hip_bfloat16 silu_bf16", runtime)
        self.assertIn("native.decode.routed_weighted", workspace)
        self.assertIn("run_native_decode_routed_moe", linear)
        self.assertIn("run_native_decode_routed_moe", full)
        frozen_loop = "for (std::size_t offset = 6; offset < 10; ++offset)"
        linear_frozen_start = linear.index("if (!use_current_vllm_projections)")
        linear_frozen_end = linear.index("return metrics;", linear_frozen_start)
        linear_native_start = linear.index(
            "run_native_decode_routed_moe", linear_frozen_end
        )
        self.assertIn(
            frozen_loop, linear[linear_frozen_start:linear_frozen_end]
        )
        self.assertGreater(linear_native_start, linear_frozen_end)

        full_routed_start = full.index("void* routed_output")
        full_native_start = full.index(
            "run_native_decode_routed_moe", full_routed_start
        )
        full_frozen_start = full.index("} else {", full_native_start)
        self.assertIn(frozen_loop, full[full_frozen_start:])
        self.assertLess(full_native_start, full_frozen_start)

    def test_four_model_weight_boundaries_are_bit_exact_non_promotion_evidence(
        self,
    ) -> None:
        trace = json.loads(
            (EVIDENCE_ROOT / "trace-result.json").read_text(encoding="utf-8")
        )
        self.assertTrue(trace["complete"])
        self.assertTrue(trace["qualified_for_aot_capture"])
        self.assertFalse(trace["qualified_for_native_decode_replacement"])
        self.assertEqual(trace["geometry"]["topk_weights_dtype"], "torch.float32")

        paths = sorted(EVIDENCE_ROOT.glob("model-probe-*.json"))
        self.assertEqual(len(paths), 4)
        identities = set()
        for path in paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(result["complete"])
            identities.add((result["case_id"], result["tail_set"]))
            self.assertTrue(result["expert_aot_seeded_from_reference_router"])
            self.assertTrue(result["router_evaluated_from_reference_hidden_state"])
            self.assertTrue(result["end_to_end_router_outputs_consumed"])
            self.assertTrue(result["decision"]["model_weight_numerical_closure"])
            self.assertFalse(
                result["decision"]["qualified_for_native_decode_replacement"]
            )
            self.assertFalse(result["decision"]["promotion_result"])
            self.assertEqual(
                result["gate_up_image_sha256"],
                "86cc097b8f6ca7dd4f239d91b3a8368ec7db6e4a9c85bc5f748848797540750b",
            )
            self.assertEqual(
                result["down_image_sha256"],
                "780ad77c75c0b7db065603bae69718050051f879ec5ee9f9be89aadd0355c358",
            )
            self.assertEqual(len(result["comparisons"]), 11)
            for comparison in result["comparisons"].values():
                self.assertTrue(comparison["bit_exact"])
                self.assertEqual(
                    comparison["exact_elements"], comparison["elements"]
                )
        self.assertEqual(
            identities,
            {
                ("tool_forced_image", "first_decode_layer0_tail"),
                ("tool_forced_image", "layer0_tail"),
                ("tool_auto_image", "first_decode_layer0_tail"),
                ("tool_auto_image", "layer0_tail"),
            },
        )


if __name__ == "__main__":
    unittest.main()
