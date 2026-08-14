from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/capture-vllm-vl-http-language-layers.py"
LAYER_HOOKS = ROOT / "scripts/capture-vllm-vl-http-language-attribution.py"
NATIVE_DIAGNOSTIC = (
    ROOT / "native/tools/vl_http_language_diagnostic_probe.hip.cpp"
)
NATIVE_DIAGNOSTIC_BUILD = (
    ROOT / "scripts/build-native-vl-http-language-diagnostic-probe.sh"
)


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
        self.assertIn("layers.base,", source)
        self.assertIn("http_oracle_final_norm_comparison", source)
        self.assertIn('"diagnostic_only": True', source)
        self.assertIn('"g2_passed": False', source)

    def test_capture_attributes_first_drift_and_router_discontinuities(self) -> None:
        hooks = LAYER_HOOKS.read_text(encoding="utf-8")
        self.assertIn("ATTENTION_DIAGNOSTIC_LAYER = 1", hooks)
        self.assertIn("DETAILED_ROUTER_LAYERS = (21, 30, 31)", hooks)
        self.assertIn('"router_logits", "router_scores", "router_weights"', hooks)
        self.assertIn('"fused_input_projection"', hooks)
        self.assertIn('"gdn_core"', hooks)
        self.assertIn('"gdn_gated"', hooks)
        self.assertIn("diagnostic_layer.linear_attn.register_forward_hook", hooks)
        self.assertIn(
            "diagnostic_layer.post_attention_layernorm.register_forward_hook",
            hooks,
        )
        native = NATIVE_DIAGNOSTIC.read_text(encoding="utf-8")
        build = NATIVE_DIAGNOSTIC_BUILD.read_text(encoding="utf-8")
        self.assertIn("http_language_diagnostic_case_dir", native)
        self.assertIn("diagnostic_options.sequence_oracle_dir", native)
        self.assertIn("pending_attention_comparisons", native)
        self.assertIn("router_logits.bf16.bin", native)
        self.assertIn("router_indices.i64.bin", native)
        self.assertIn(
            '#include "vl_language_layer3_composed_oracle_probe.hip.cpp"',
            native,
        )
        self.assertIn("vl_http_language_diagnostic_probe.hip.cpp", build)


if __name__ == "__main__":
    unittest.main()
