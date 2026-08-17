from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/capture-vllm-vl-task-quality-divergence.py"
REFERENCE = ROOT / "benchmarks/results/vl-task-quality-reference-v0.1.0.json"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "vl_task_quality_divergence_capture_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VlTaskQualityDivergenceCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.reference = json.loads(REFERENCE.read_text(encoding="utf-8"))

    def test_targets_bind_first_observed_reference_divergences(self) -> None:
        cases = {case["case_id"]: case for case in self.reference["cases"]}
        self.assertEqual(
            self.module.TARGETS,
            {
                "image_central_red_circle": 122,
                "video_blue_square_moves_down": 172,
            },
        )
        self.assertEqual(
            [
                cases[case_id]["output_token_ids"][target]
                for case_id, target in self.module.TARGETS.items()
            ],
            [62366, 318],
        )

    def test_capture_fail_closes_on_prompt_prefix_and_top1_drift(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("task-quality prompt drifted", source)
        self.assertIn("frozen task-quality prefix changed", source)
        self.assertIn("task-quality logit call count changed", source)
        self.assertIn("task-quality raw top-1 changed", source)
        self.assertIn('"expected_prefix_token_ids"', source)
        self.assertIn('"reference_logits_output_index"', source)

    def test_capture_is_explicitly_diagnostic(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("This is an attribution tool, not release evidence", source)
        self.assertIn("task-quality divergence runtime pin mismatch", source)
        self.assertIn('"files": source_components()', source)
        self.assertIn('"product_runtime_dependency": False', source)
        self.assertIn('"g1_passed": False', source)
        self.assertIn('"g2_passed": False', source)

    def test_layer_attribution_overrides_are_fail_closed(self) -> None:
        self.assertEqual(
            self.module.parse_case_int_overrides(
                ["image_central_red_circle=23"],
                option="--case-full-attention-layer",
                maximum=40,
            ),
            {"image_central_red_circle": 23},
        )
        for values in (
            ["unknown=3"],
            ["image_central_red_circle=-1"],
            ["image_central_red_circle=40"],
            ["image_central_red_circle=3", "image_central_red_circle=7"],
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.module.parse_case_int_overrides(
                        values,
                        option="--case-full-attention-layer",
                        maximum=40,
                    )

        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("InstallGenerationLayerHooks", source)
        self.assertIn('"decode_layers"', source)
        self.assertIn('"reference_decode_boundary_dir"', source)
        self.assertIn('"reference_decode_full_attention_dir"', source)
        self.assertIn('"reference_decode_linear_boundary_dir"', source)


if __name__ == "__main__":
    unittest.main()
