from __future__ import annotations

from pathlib import Path
import unittest

from aima_engine.vl_generation_oracle import CASE_ORDER
from aima_engine.vl_prefill_state_oracle import (
    CONV_STATE_SHAPE,
    RECURRENT_STATE_SHAPE,
    STATE_COMPONENT_NAMES,
    VL_PREFILL_STATE_ORACLE_SCHEMA,
    validate_vl_prefill_state_oracle_manifest,
)
from aima_engine.vl_reference import seal_manifest


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "scripts/capture-vllm-vl-prefill-state-oracles.py"
RESIDENT_HEADER = ROOT / "native/include/aima/native_resident_engine.h"
RESIDENT = ROOT / "native/src/native_resident_engine.hip.cpp"
HTTP = ROOT / "native/src/native_http_server.cpp"


def component(name: str) -> dict[str, object]:
    recurrent = name.endswith("_recurrent_state")
    shape = RECURRENT_STATE_SHAPE if recurrent else CONV_STATE_SHAPE
    element_size = 4 if recurrent else 2
    elements = 1
    for value in shape:
        elements *= value
    return {
        "path": f"case/components/{name}.bin",
        "shape": list(shape),
        "dtype": "torch.float32" if recurrent else "torch.bfloat16",
        "element_size": element_size,
        "bytes": elements * element_size,
        "sha256": "0" * 64,
    }


class VlPrefillStateOracleTest(unittest.TestCase):
    def test_manifest_freezes_both_complete_state_sets(self) -> None:
        cases = [
            {
                "case_id": case_id,
                "passed": True,
                "capture_decode_call": 1,
                "components": {
                    name: component(name) for name in STATE_COMPONENT_NAMES
                },
                "oracle_jsonl": {"path": f"{case_id}/oracle.jsonl"},
            }
            for case_id in CASE_ORDER
        ]
        manifest = seal_manifest(
            {
                "schema": VL_PREFILL_STATE_ORACLE_SCHEMA,
                "complete": True,
                "qualified_for_state_attribution": True,
                "generation_oracle": {"sha256": "1" * 64},
                "cases": cases,
                "decision": {
                    "two_prompt_prefixes_exact": True,
                    "two_prefill_state_sets_captured": True,
                    "g1_passed": False,
                    "g2_passed": False,
                    "g3_passed": False,
                    "g4_passed": False,
                    "g5_passed": False,
                },
            }
        )
        self.assertEqual(validate_vl_prefill_state_oracle_manifest(manifest), [])
        manifest["cases"][0]["components"]["layer_000_conv_state"][
            "shape"
        ] = [1]
        self.assertIn(
            "VL prefill state shape changed: tool_forced_image/"
            "layer_000_conv_state",
            validate_vl_prefill_state_oracle_manifest(manifest),
        )

    def test_capture_uses_first_decode_entry_and_one_model_load(self) -> None:
        source = CAPTURE.read_text(encoding="utf-8")
        self.assertIn('llm_kwargs["skip_mm_profiling"] = True', source)
        self.assertIn("max_tokens=2", source)
        self.assertIn("original_forward_core", source)
        self.assertIn("non_spec_state_indices_tensor", source)
        self.assertIn("context.attn_metadata[attention.prefix]", source)
        self.assertIn("for case_id in CASE_ORDER", source)

    def test_native_observer_is_qualification_only(self) -> None:
        header = RESIDENT_HEADER.read_text(encoding="utf-8")
        resident = RESIDENT.read_text(encoding="utf-8")
        http = HTTP.read_text(encoding="utf-8")
        self.assertIn("NativePrefillLinearStateObserver", header)
        self.assertIn("prefill_linear_state_observer", resident)
        self.assertIn('item.contains("reference_prefill_state_dir")', http)
        self.assertIn("prefill_state_comparisons.size() !=", http)


if __name__ == "__main__":
    unittest.main()
