from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "native/aot/gfx1151/packed-linear-decode-v0.1.0/manifest.json"
)
TRACE_RESULT = (
    ROOT
    / "benchmarks/runs/packed-linear-decode-aot-v0.1.0/trace-result.json"
)
KERNEL_HASH = (
    "361b24af7b3fc502598ffb5fd1e191c9b82afc437361f92f4056bf8772a960dc"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PackedLinearDecodeAotTests(unittest.TestCase):
    def test_aot_closure_is_exact_and_path_clean(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema"], "aima-amd395-qwen36/native-aot-closure/v1"
        )
        self.assertEqual(manifest["target"]["arch"], "gfx1151")
        self.assertEqual(manifest["kernel_count"], 1)
        self.assertEqual(manifest["kernel_symbol_count"], 1)
        self.assertEqual(manifest["launch_variant_count"], 1)
        kernel = manifest["kernels"][0]
        self.assertEqual(kernel["kernel_hash"], KERNEL_HASH)
        self.assertEqual(
            kernel["symbol"],
            "fused_recurrent_gated_delta_rule_packed_decode_kernel",
        )
        self.assertEqual(
            [item["name"] for item in kernel["regular_abi"]],
            [
                "mixed_qkv",
                "a",
                "b",
                "A_log",
                "dt_bias",
                "o",
                "h0",
                "ht",
                "ssm_state_indices",
                "scale",
            ],
        )
        self.assertEqual(kernel["launch_variants"][0]["grid"], [4, 32, 1])
        self.assertEqual(kernel["metadata"]["num_warps"], 1)
        self.assertEqual(kernel["metadata"]["shared"], 64)
        self.assertEqual(kernel["compile_constants"]["stride_init_state_token"], 524288)
        self.assertEqual(kernel["compile_constants"]["stride_final_state_token"], 524288)
        image = MANIFEST.parent / kernel["image"]["path"]
        self.assertEqual(image.stat().st_size, kernel["image"]["bytes"])
        self.assertEqual(sha256(image), kernel["image"]["sha256"])
        payload = image.read_bytes()
        for marker in (b"/home/", b"/data/", b"site-packages"):
            self.assertNotIn(marker, payload)

    def test_trace_driver_freezes_in_place_state_contract(self) -> None:
        driver = (
            ROOT / "scripts/trace-vllm-packed-linear-decode-aot.py"
        ).read_text(encoding="utf-8")
        self.assertIn("fused_recurrent_gated_delta_rule_packed_decode", driver)
        self.assertIn("guard_state_unchanged", driver)
        self.assertIn("selected_state_changed", driver)
        if TRACE_RESULT.exists():
            result = json.loads(TRACE_RESULT.read_text(encoding="utf-8"))
            self.assertTrue(result["complete"])
            self.assertTrue(result["qualified_for_native_decode_replacement"])
            self.assertTrue(result["checks"]["guard_state_unchanged"])
            self.assertTrue(result["checks"]["selected_state_changed"])
            self.assertTrue(result["checks"]["output_finite"])

    def test_native_runtime_replaces_only_the_historical_recurrent_launch(self) -> None:
        schedule = json.loads(
            (
                ROOT / "native/aot/gfx1151/q8192-output2/decode-schedule.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            schedule["layer_templates"]["linear_attention"][2],
            "fused_sigmoid_gating_delta_rule_update_kernel",
        )
        runtime = (ROOT / "native/src/native_linear_layer.hip.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn(KERNEL_HASH, runtime)
        self.assertIn("launch_packed_recurrent", runtime)
        self.assertIn("native.linear.packed_ssm_state_base", runtime)
        self.assertIn("native.linear.packed_ssm_state_indices", runtime)
        invocation = (
            ROOT / "native/src/native_decode_invocation.cpp"
        ).read_text(encoding="utf-8")
        self.assertNotIn(
            'swap_named(launches_[base + 2], "h0", "ht")', invocation
        )
        self.assertIn("if (swaps != 30)", invocation)

    def test_default_build_embeds_packed_kernel_manifest(self) -> None:
        build = (ROOT / "scripts/build-native-runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("PACKED_LINEAR_DECODE_MANIFEST", build)
        self.assertIn("packed-linear-decode-v0.1.0/manifest.json", build)


if __name__ == "__main__":
    unittest.main()
