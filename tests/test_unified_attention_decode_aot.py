from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/trace-vllm-unified-attention-decode-aot.py"


class UnifiedAttentionDecodeAotTests(unittest.TestCase):
    def test_trace_driver_freezes_current_singleton_geometry(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn("triton_unified_attention import", source)
        self.assertIn("unified_attention", source)
        self.assertIn("query_heads = 16", source)
        self.assertIn("kv_heads = 2", source)
        self.assertIn("head_size = 256", source)
        self.assertIn("block_size = 16", source)
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


if __name__ == "__main__":
    unittest.main()
