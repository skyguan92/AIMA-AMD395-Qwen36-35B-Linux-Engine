from __future__ import annotations

import json
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/trace-vllm-causal-conv-decode-aot.py"
TRACE_RESULT = (
    ROOT
    / "benchmarks/runs/causal-conv-decode-aot-v0.1.0/trace-result.json"
)
MANIFEST = (
    ROOT / "native/aot/gfx1151/causal-conv-decode-v0.1.0/manifest.json"
)
KERNEL_HASH = (
    "ab71972380fed224052336c248656eb49e8d2ccd89acc4bebbee193e2c6a699c"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CausalConvDecodeAotTests(unittest.TestCase):
    def test_aot_closure_is_exact_and_path_clean(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema"], "aima-amd395-qwen36/native-aot-closure/v1"
        )
        self.assertEqual(manifest["target"]["arch"], "gfx1151")
        self.assertEqual(manifest["kernel_count"], 1)
        kernel = manifest["kernels"][0]
        self.assertEqual(kernel["kernel_hash"], KERNEL_HASH)
        self.assertEqual(kernel["symbol"], "_causal_conv1d_update_kernel")
        self.assertEqual(
            [item["name"] for item in kernel["regular_abi"]],
            [
                "x_ptr",
                "w_ptr",
                "conv_state_ptr",
                "conv_state_indices_ptr",
                "o_ptr",
                "batch",
            ],
        )
        self.assertEqual(kernel["launch_variants"][0]["grid"], [1, 32, 1])
        self.assertEqual(kernel["metadata"]["num_warps"], 4)
        self.assertEqual(kernel["metadata"]["shared"], 2048)
        self.assertFalse(kernel["compile_constants"]["HAS_NULL_BLOCK"])
        self.assertTrue(kernel["compile_constants"]["SILU_ACTIVATION"])
        image = MANIFEST.parent / kernel["image"]["path"]
        self.assertEqual(image.stat().st_size, kernel["image"]["bytes"])
        self.assertEqual(sha256(image), kernel["image"]["sha256"])
        payload = image.read_bytes()
        for marker in (b"/home/", b"/data/", b"site-packages"):
            self.assertNotIn(marker, payload)

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

    def test_native_runtime_uses_current_projection_and_conv_primitives(self) -> None:
        runtime = (ROOT / "native/src/native_linear_layer.hip.cpp").read_text(
            encoding="utf-8"
        )
        provider = (ROOT / "native/src/bf16_wvsplitk.hip.cpp").read_text(
            encoding="utf-8"
        )
        parity = (ROOT / "scripts/check-native-wvsplitk-parity.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(KERNEL_HASH, runtime)
        self.assertIn("launch_current_causal_conv", runtime)
        self.assertIn("in_proj_qkv.weight", runtime)
        self.assertIn("in_proj_z.weight", runtime)
        self.assertIn("in_proj_a.weight", runtime)
        self.assertIn("in_proj_b.weight", runtime)
        self.assertNotIn("executor.launch(launches[base + 1]", runtime)
        for m in (8192, 4096, 32):
            self.assertIn(f"probe_case({m}, 2048, cu_count)", provider)
            self.assertIn(f"({m}, 2048)", parity)

    def test_default_build_embeds_causal_conv_manifest(self) -> None:
        build = (ROOT / "scripts/build-native-runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("CAUSAL_CONV_DECODE_MANIFEST", build)
        self.assertIn("causal-conv-decode-v0.1.0/manifest.json", build)


if __name__ == "__main__":
    unittest.main()
