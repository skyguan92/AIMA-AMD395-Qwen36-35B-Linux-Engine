from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/capture-vllm-vl-language-layer3-mrope-diagnostics.py"
)


def load_capture_module():
    spec = importlib.util.spec_from_file_location(
        "test_vl_language_layer3_mrope_capture_module", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load layer-3 M-RoPE diagnostic capture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VlLanguageLayer3MropeDiagnosticCaptureTest(unittest.TestCase):
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

    def test_interleaved_axis_contract_is_11_11_10(self) -> None:
        axes = [
            self.capture.mrope_axis_for_pair(pair)
            for pair in range(self.capture.ROTARY_PAIRS)
        ]
        self.assertEqual(axes.count(0), 11)
        self.assertEqual(axes.count(1), 11)
        self.assertEqual(axes.count(2), 10)
        self.assertEqual(axes[-1], 1)
        self.assertEqual(axes[:9], [0, 1, 2, 0, 1, 2, 0, 1, 2])
        with self.assertRaises(ValueError):
            self.capture.mrope_axis_for_pair(-1)
        with self.assertRaises(ValueError):
            self.capture.mrope_axis_for_pair(32)

    def test_capture_covers_rotary_inputs_outputs_and_layer_boundary(self) -> None:
        self.assertTrue(
            {
                "positions",
                "q_gate_projection",
                "raw_k",
                "raw_v",
                "q_norm_input",
                "k_norm_input",
                "normalized_q",
                "normalized_k",
                "axis_cos",
                "axis_sin",
                "effective_cos",
                "effective_sin",
                "rotary_q",
                "rotary_k",
                "attention_output",
                "attention_residual",
                "post_attention_norm",
                "layer_output",
            }.issubset(self.capture.REQUIRED_COMPONENTS)
        )

    def test_native_compatibility_labels_target_layer3(self) -> None:
        self.assertEqual(
            self.capture.ORACLE_LABELS[
                "layer-003-return-full_attention-q"
            ],
            "rotary_q",
        )
        self.assertEqual(
            self.capture.ORACLE_LABELS[
                "layer-003-return-full_attention-k"
            ],
            "rotary_k",
        )
        self.assertEqual(
            self.capture.ORACLE_LABELS[
                "layer-003-return-layer_body-output"
            ],
            "layer_output",
        )

    def test_capture_is_bound_to_exact_reference_sources(self) -> None:
        self.assertEqual(len(self.capture.VL_ORACLE_SHA256), 64)
        self.assertEqual(len(self.capture.SOURCE_HASHES), 7)
        self.assertTrue(
            all(
                len(value) == 64
                for value in self.capture.SOURCE_HASHES.values()
            )
        )
        self.assertIn(
            "vllm.model_executor.layers.rotary_embedding.mrope",
            self.capture.SOURCE_HASHES,
        )


if __name__ == "__main__":
    unittest.main()
