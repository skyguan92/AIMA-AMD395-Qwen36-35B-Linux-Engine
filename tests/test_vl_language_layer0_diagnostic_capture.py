from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/capture-vllm-vl-language-layer0-diagnostics.py"


def load_capture_module():
    spec = importlib.util.spec_from_file_location(
        "test_vl_language_layer0_diagnostic_capture_module", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load layer-0 diagnostic capture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VlLanguageLayer0DiagnosticCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.capture = load_capture_module()

    def test_case_set_matches_blocking_vl_matrix(self) -> None:
        self.assertEqual(
            self.capture.CASE_IDS,
            (
                "image_local_png",
                "video_local_mp4",
                "multi_image",
                "multi_video",
                "mixed_image_video",
            ),
        )

    def test_diagnostic_ledger_spans_attention_and_moe(self) -> None:
        self.assertTrue(
            {
                "input_norm",
                "gdn_core",
                "gdn_gated_norm",
                "linear_attention_output",
                "attention_residual",
                "post_attention_norm",
                "shared_moe_output",
                "routed_moe_output",
                "combined_moe_output",
                "layer_output",
            }.issubset(self.capture.REQUIRED_COMPONENTS)
        )
        self.assertEqual(
            self.capture.ORACLE_LABELS["launch-010-norm_out"],
            "post_attention_norm",
        )
        self.assertEqual(
            self.capture.ORACLE_LABELS["diagnostic-moe_out"],
            "combined_moe_output",
        )

    def test_capture_is_bound_to_exact_reference_sources(self) -> None:
        self.assertEqual(len(self.capture.VL_ORACLE_SHA256), 64)
        self.assertEqual(
            set(self.capture.SOURCE_HASHES),
            {
                "vllm.model_executor.models.qwen3_5",
                "vllm.model_executor.models.qwen3_next",
                "vllm.model_executor.layers.mamba.gdn_linear_attn",
                "vllm.model_executor.layers.layernorm",
                "vllm.model_executor.layers.linear",
                "vllm.model_executor.layers.fused_moe.shared_fused_moe",
            },
        )
        self.assertTrue(
            all(len(value) == 64 for value in self.capture.SOURCE_HASHES.values())
        )

    def test_native_compatibility_labels_are_complete(self) -> None:
        expected = {
            "launch-000-out",
            "launch-008-o",
            "return-linear_attention-gated_out",
            "return-linear-attention-output",
            "launch-010-residual_out",
            "launch-010-norm_out",
            "diagnostic-h2",
            "diagnostic-shared_out",
            "diagnostic-routed_moe",
            "diagnostic-moe_out",
            "diagnostic-output",
        }
        self.assertEqual(set(self.capture.ORACLE_LABELS), expected)


if __name__ == "__main__":
    unittest.main()
