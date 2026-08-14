from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/capture-vllm-vl-http-language-layers.py"


class VlHttpLanguageLayerCaptureTest(unittest.TestCase):
    def test_capture_is_bound_to_http_prompt_and_diagnostic_scope(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("validate_http_oracle_manifest", source)
        self.assertIn('chat_template_content_format="string"', source)
        self.assertIn('llm_kwargs["skip_mm_profiling"] = True', source)
        self.assertIn("prompt_token_ids != expected_ids", source)
        self.assertIn("InstallLanguageLayerOutputHooks", source)
        self.assertIn("FinalizeLanguageLayerOutputHooks", source)
        self.assertIn("cloudpickle.register_pickle_by_value", source)
        self.assertIn("http_oracle_final_norm_comparison", source)
        self.assertIn('"diagnostic_only": True', source)
        self.assertIn('"g2_passed": False', source)


if __name__ == "__main__":
    unittest.main()
