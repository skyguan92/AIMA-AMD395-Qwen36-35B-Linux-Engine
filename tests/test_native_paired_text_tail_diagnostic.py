from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/diagnose-native-paired-text-tails.py"
SPEC = importlib.util.spec_from_file_location("text_tail_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class NativePairedTextTailDiagnosticTest(unittest.TestCase):
    def test_runtime_dependency_set_is_explicit(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for filename in (
            "libaima-fmha-aotriton.so",
            "libaima-fmha-ck.so",
            "libaima-fmha-q16384-hybrid.so",
            "aima-vision-attention.hsaco",
        ):
            self.assertIn(filename, source)
        self.assertIn("require_aotriton_closure", source)

    def test_context_parser_is_fail_closed(self) -> None:
        self.assertEqual(
            diagnostic.parse_contexts("7168,7680,8191"),
            (7168, 7680, 8191),
        )
        for value in ("", "0", "262144", "7168,7168", "oops"):
            with self.assertRaises(argparse.ArgumentTypeError):
                diagnostic.parse_contexts(value)

    def test_pair_order_alternates(self) -> None:
        self.assertEqual(diagnostic.pair_order(1), ("left", "right"))
        self.assertEqual(diagnostic.pair_order(2), ("right", "left"))
        with self.assertRaises(ValueError):
            diagnostic.pair_order(0)

    def test_report_requires_exact_identity_and_idle_text_path(self) -> None:
        request = {
            "prompt_tokens": 7168,
            "completion_tokens": 1,
            "first_token_certified": True,
            "all_decode_tokens_certified": True,
            "oracle_tensor_reads": 0,
            "mrope_enabled": False,
            "mrope_position_upload_bytes": 0,
            "mrope_full_attention_launches": 0,
            "mrope_decode_steps": 0,
            "prefill_vl_unified_attention_launches": 0,
            "vl_logical_projections_enabled": False,
            "vl_logical_projection_tokens": 0,
            "vl_logical_projection_plan_count": 0,
            "vl_logical_projection_workspace_bytes": 0,
            "vl_logical_projection_plan_build_wall_ms": 0.0,
        }
        payload = {
            "schema": "aima-amd395-qwen36/native-resident-session-probe/v1",
            "complete": True,
            "model_loads": 1,
            "request_count": 1,
            "runtime_python": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
            "requests": [request],
            "qualification": {
                "engine_role": "left",
                "engine_sha256": "a" * 64,
                "pair_index": 1,
                "pair_order": ["left", "right"],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(
                diagnostic.report_complete(
                    path,
                    context=7168,
                    pair_index=1,
                    role="left",
                    engine_sha256="a" * 64,
                )
            )
            payload["requests"][0]["mrope_enabled"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(
                diagnostic.report_complete(
                    path,
                    context=7168,
                    pair_index=1,
                    role="left",
                    engine_sha256="a" * 64,
                )
            )


if __name__ == "__main__":
    unittest.main()
