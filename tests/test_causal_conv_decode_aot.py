from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/trace-vllm-causal-conv-decode-aot.py"
TRACE_RESULT = (
    ROOT
    / "benchmarks/runs/causal-conv-decode-aot-v0.1.0/trace-result.json"
)


class CausalConvDecodeAotTests(unittest.TestCase):
    def test_trace_driver_freezes_direct_production_parity(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn("causal_conv1d_update", source)
        self.assertIn("null_block_id=None", source)
        self.assertIn("production_output_equal", source)
        self.assertIn("production_state_equal", source)
        if TRACE_RESULT.exists():
            result = json.loads(TRACE_RESULT.read_text(encoding="utf-8"))
            self.assertTrue(result["complete"])
            self.assertTrue(result["qualified_for_native_decode_replacement"])
            self.assertTrue(result["checks"]["direct_state_changed"])
            self.assertTrue(result["checks"]["output_finite"])
            self.assertTrue(result["checks"]["state_finite"])
            self.assertTrue(result["checks"]["production_output_equal"])
            self.assertTrue(result["checks"]["production_state_equal"])
            self.assertEqual(
                result["checks"]["direct_output_sha256"],
                result["checks"]["production_output_sha256"],
            )
            self.assertEqual(
                result["checks"]["direct_state_sha256"],
                result["checks"]["production_state_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
