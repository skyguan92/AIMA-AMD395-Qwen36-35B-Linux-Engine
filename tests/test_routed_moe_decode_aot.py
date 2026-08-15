from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/trace-vllm-routed-moe-decode-aot.py"
PROBE = ROOT / "native/tools/routed_moe_decode_aot_probe.hip.cpp"
BUILD = ROOT / "scripts/build-native-routed-moe-decode-aot-probe.sh"


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


if __name__ == "__main__":
    unittest.main()
