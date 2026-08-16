from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/trace-vllm-unified-attention-decode-aot.py"
PROBE = ROOT / "native/tools/unified_attention_decode_aot_probe.hip.cpp"
BUILD = ROOT / "scripts/build-native-unified-attention-decode-aot-probe.sh"
AOT_ROOT = (
    ROOT / "native/aot/gfx1151/unified-attention-decode-v0.1.0"
)
EVIDENCE_ROOT = (
    ROOT / "benchmarks/runs/unified-attention-decode-aot-v0.1.0"
)
ATTENTION_HASH = (
    "57514aea3981e5fba3e25d46b9dd62fb311a3acfef9ca9ad8fa99b0076c61402"
)
REDUCE_HASH = (
    "6ecf435e2f5f8cfa2805d7433192f64a5e5e749e7a93c3ed4ae39e50921fe078"
)


class UnifiedAttentionDecodeAotTests(unittest.TestCase):
    def test_trace_driver_freezes_current_singleton_geometry(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn("triton_unified_attention import", source)
        self.assertIn("unified_attention", source)
        self.assertIn("query_heads = 16", source)
        self.assertIn("kv_heads = 2", source)
        self.assertIn("head_size = 256", source)
        self.assertIn("block_size = 1_056", source)
        self.assertIn("sequence_threshold_3d = 64", source)
        self.assertIn("softmax_segments = 16", source)
        self.assertIn("segment_output", source)
        self.assertIn('"segmented_3d_plus_reduce"', source)
        self.assertIn("1.0 / math.sqrt(head_size)", source)
        self.assertIn("output_finite", source)
        self.assertIn("repeat_exact", source)
        self.assertIn('"qualified_for_aot_capture": True', source)
        self.assertIn(
            '"qualified_for_native_decode_replacement": False', source
        )
        self.assertIn("not_a_model_weight_correctness_oracle", source)
        self.assertIn("not_yet_integrated_into_native_decode", source)
        self.assertIn("not_a_promotion_result", source)

    def test_model_tensor_probe_freezes_current_abi_without_promotion(self) -> None:
        source = PROBE.read_text(encoding="utf-8")
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("kCacheBlockTokens = 1056", source)
        self.assertIn('"kernel_unified_attention_3d"', source)
        self.assertIn('"reduce_segments"', source)
        self.assertIn(
            "AotLaunchConfig{1, 2, 16, 4, 32, 16384}", source
        )
        self.assertIn(
            "AotLaunchConfig{1, 16, 1, 4, 32, 2048}", source
        )
        self.assertIn("attention_parameters.size() != 30", source)
        self.assertIn("reduce_parameters.size() != 10", source)
        for name in (
            "query",
            "key_cache",
            "value_cache",
            "block_table",
            "sequence_lengths",
            "query_starts",
            "k_descale",
            "v_descale",
            "output",
        ):
            self.assertIn(f'"{name}"', source)
        self.assertIn('"model_tensor_numerical_closure"', source)
        self.assertIn(
            '"qualified_for_native_decode_replacement", false', source
        )
        self.assertIn('"promotion_result", false', source)
        self.assertIn("unified_attention_decode_aot_probe.hip.cpp", build)
        self.assertIn("aot_kernel.hip.cpp", build)

    def test_exported_segmented_closure_freezes_both_regular_abis(self) -> None:
        manifest = json.loads(
            (AOT_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["schema"],
            "aima-amd395-qwen36/native-aot-closure/v1",
        )
        self.assertEqual(manifest["kernel_count"], 2)
        self.assertEqual(manifest["kernel_symbol_count"], 2)
        kernels = {
            kernel["kernel_hash"]: kernel for kernel in manifest["kernels"]
        }
        self.assertEqual(set(kernels), {ATTENTION_HASH, REDUCE_HASH})
        expected = {
            ATTENTION_HASH: (
                "kernel_unified_attention_3d",
                "39e2b17f68f8c23ae6ed757a7878277ea53bc10caf62328bd3da470c4e9a119f",
                30,
            ),
            REDUCE_HASH: (
                "reduce_segments",
                "a5339fd0563297b377dba12b2493e35622660843febfe41556b6bdafb2e9d184",
                10,
            ),
        }
        for kernel_hash, (symbol, image_sha256, abi_count) in expected.items():
            kernel = kernels[kernel_hash]
            image = AOT_ROOT / kernel["image"]["path"]
            self.assertEqual(kernel["symbol"], symbol)
            self.assertEqual(len(kernel["regular_abi"]), abi_count)
            self.assertEqual(
                hashlib.sha256(image.read_bytes()).hexdigest(), image_sha256
            )
        attention_abi = [
            value["name"] for value in kernels[ATTENTION_HASH]["regular_abi"]
        ]
        self.assertNotIn("output_stride_0", attention_abi)
        self.assertNotIn("output_stride_1", attention_abi)
        self.assertEqual(attention_abi[15], "qq_bias_stride_0")
        reduce_abi = [
            value["name"] for value in kernels[REDUCE_HASH]["regular_abi"]
        ]
        self.assertEqual(reduce_abi[6:8], ["output_stride_0", "output_stride_1"])

    def test_four_model_tensor_boundaries_are_bit_exact_non_promotion_evidence(
        self,
    ) -> None:
        trace = json.loads(
            (EVIDENCE_ROOT / "trace-result.json").read_text(encoding="utf-8")
        )
        self.assertTrue(trace["complete"])
        self.assertTrue(trace["qualified_for_aot_capture"])
        self.assertFalse(trace["qualified_for_native_decode_replacement"])
        self.assertEqual(
            trace["geometry"]["attention_path"],
            "segmented_3d_plus_reduce",
        )

        probe_paths = sorted(EVIDENCE_ROOT.glob("model-probe-*.json"))
        self.assertEqual(len(probe_paths), 4)
        identities = set()
        for path in probe_paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(result["complete"])
            identities.add((result["case_id"], result["attention_set"]))
            self.assertTrue(
                result["decision"]["model_tensor_numerical_closure"]
            )
            self.assertFalse(
                result["decision"]["qualified_for_native_decode_replacement"]
            )
            self.assertFalse(result["decision"]["promotion_result"])
            self.assertEqual(
                result["attention_image_sha256"],
                "39e2b17f68f8c23ae6ed757a7878277ea53bc10caf62328bd3da470c4e9a119f",
            )
            self.assertEqual(
                result["reduce_image_sha256"],
                "a5339fd0563297b377dba12b2493e35622660843febfe41556b6bdafb2e9d184",
            )
            self.assertEqual(set(result["comparisons"]), {"output", "repeat_output"})
            for comparison in result["comparisons"].values():
                self.assertTrue(comparison["bit_exact"])
                self.assertEqual(comparison["exact_elements"], 4096)
                self.assertEqual(comparison["elements"], 4096)
        self.assertEqual(
            identities,
            {
                ("tool_forced_image", "first_decode_full_attention"),
                ("tool_forced_image", "full_attention"),
                ("tool_auto_image", "first_decode_full_attention"),
                ("tool_auto_image", "full_attention"),
            },
        )


if __name__ == "__main__":
    unittest.main()
