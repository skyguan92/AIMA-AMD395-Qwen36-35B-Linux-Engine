from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeVlLanguageLayer3MropeTest(unittest.TestCase):
    def test_device_mrope_table_preserves_the_text_rotary_api(self) -> None:
        header = (ROOT / "native/include/aima/native_pointwise.h").read_text(
            encoding="utf-8"
        )
        source = (ROOT / "native/src/native_pointwise.hip.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("launch_prefill_rotary_table", header)
        self.assertIn("launch_prefill_mrope_rotary_table", header)
        self.assertIn("launch_full_attention_head_norm_mrope_prefill", header)
        self.assertIn("const void* positions_i64", header)
        self.assertIn("prefill_mrope_rotary_table_kernel", source)
        self.assertIn("mrope_axis_for_pair", source)
        self.assertIn("triton_bf16_product_rtz", source)
        self.assertIn(
            "launch_full_attention_head_norm_rope_prefill_impl<false>",
            source,
        )
        self.assertIn(
            "launch_full_attention_head_norm_rope_prefill_impl<true>",
            source,
        )
        self.assertIn("__float2bfloat16(cosf(angle))", source)
        self.assertIn("__float2bfloat16(sinf(angle))", source)

    def test_isolated_probe_attributes_table_and_head_norm_rope(self) -> None:
        probe = (
            ROOT
            / "native/tools/vl_language_layer3_mrope_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vl-language-layer3-mrope-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("generated_effective_cos", probe)
        self.assertIn("generated_effective_sin", probe)
        self.assertIn("head_norm_q", probe)
        self.assertIn("head_norm_k", probe)
        self.assertIn("oracle_table_rotary_q", probe)
        self.assertIn("oracle_table_rotary_k", probe)
        self.assertIn("generated_table_rotary_q", probe)
        self.assertIn("generated_table_rotary_k", probe)
        self.assertIn("kMeasuredRuns = 5", probe)
        self.assertIn("runtime_python", probe)
        self.assertIn("native_pointwise.hip.cpp", build)
        self.assertIn("--offload-arch=gfx1151", build)

    def test_probe_is_bound_to_the_five_case_capture_schema(self) -> None:
        probe = (
            ROOT
            / "native/tools/vl_language_layer3_mrope_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "vl-language-layer3-mrope-diagnostic-oracle/v1", probe
        )
        self.assertIn("manifest.at(\"cases\").size() != 5", probe)
        self.assertIn("json::array({11, 11, 10})", probe)
        self.assertIn("relative_l2_maximum", probe)
        self.assertIn("cosine_minimum", probe)


if __name__ == "__main__":
    unittest.main()
