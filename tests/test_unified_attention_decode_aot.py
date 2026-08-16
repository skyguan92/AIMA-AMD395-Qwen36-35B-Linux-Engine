from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/trace-vllm-unified-attention-decode-aot.py"
PROBE = ROOT / "native/tools/unified_attention_decode_aot_probe.hip.cpp"
BUILD = ROOT / "scripts/build-native-unified-attention-decode-aot-probe.sh"


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


if __name__ == "__main__":
    unittest.main()
