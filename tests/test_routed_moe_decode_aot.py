from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/trace-vllm-routed-moe-decode-aot.py"


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


if __name__ == "__main__":
    unittest.main()
