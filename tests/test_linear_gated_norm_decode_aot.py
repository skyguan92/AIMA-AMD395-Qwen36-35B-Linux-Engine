from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/trace-vllm-linear-gated-norm-decode-aot.py"
TRACE_RESULT = (
    ROOT
    / "benchmarks/runs/linear-gated-norm-decode-aot-v0.1.0/trace-result.json"
)
MANIFEST = (
    ROOT
    / "native/aot/gfx1151/linear-gated-norm-decode-v0.1.0/manifest.json"
)
KERNEL_HASH = (
    "2c40422c776225912e71c6cd74fb90ea37001e24b57e6cc135af84c048a791db"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LinearGatedNormDecodeAotTests(unittest.TestCase):
    def test_aot_closure_matches_current_vllm_decode_specialization(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema"], "aima-amd395-qwen36/native-aot-closure/v1"
        )
        self.assertEqual(manifest["target"]["arch"], "gfx1151")
        self.assertEqual(manifest["kernel_count"], 1)
        kernel = manifest["kernels"][0]
        self.assertEqual(kernel["kernel_hash"], KERNEL_HASH)
        self.assertEqual(kernel["symbol"], "layer_norm_fwd_kernel")
        self.assertEqual(
            [item["name"] for item in kernel["regular_abi"]],
            [
                "X",
                "Y",
                "W",
                "Z",
                "Rstd",
                "stride_x_row",
                "stride_y_row",
                "stride_z_row",
                "M",
                "eps",
            ],
        )
        self.assertEqual(kernel["launch_variants"][0]["grid"], [32, 1, 1])
        self.assertEqual(kernel["metadata"]["num_warps"], 1)
        self.assertEqual(kernel["metadata"]["shared"], 0)
        constants = kernel["compile_constants"]
        self.assertEqual(constants["BLOCK_N"], 128)
        self.assertEqual(constants["ROWS_PER_BLOCK"], 1)
        self.assertTrue(constants["IS_RMS_NORM"])
        self.assertTrue(constants["NORM_BEFORE_GATE"])
        self.assertEqual(constants["ACTIVATION"], "silu")
        image = MANIFEST.parent / kernel["image"]["path"]
        self.assertEqual(image.stat().st_size, kernel["image"]["bytes"])
        self.assertEqual(sha256(image), kernel["image"]["sha256"])
        payload = image.read_bytes()
        for marker in (b"/home/", b"/data/", b"site-packages"):
            self.assertNotIn(marker, payload)

    def test_trace_driver_freezes_production_geometry(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn("rmsnorm_fn", source)
        self.assertIn("norm_before_gate=True", source)
        self.assertIn('activation="silu"', source)
        result = json.loads(TRACE_RESULT.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified_for_native_decode_replacement"])
        self.assertTrue(result["checks"]["input_unchanged"])
        self.assertTrue(result["checks"]["gate_unchanged"])
        self.assertTrue(result["checks"]["output_finite"])
        self.assertEqual(result["geometry"]["rows"], 32)
        self.assertEqual(result["geometry"]["head_dimension"], 128)

    def test_native_runtime_replaces_only_historical_gated_norm(self) -> None:
        runtime = (ROOT / "native/src/native_linear_layer.hip.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn(KERNEL_HASH, runtime)
        self.assertIn("launch_current_linear_gated_norm", runtime)
        self.assertIn("attention_output.device_pointer", runtime)
        self.assertNotIn("executor.launch(launches[base + 3]", runtime)

    def test_default_build_embeds_gated_norm_manifest(self) -> None:
        build = (ROOT / "scripts/build-native-runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("LINEAR_GATED_NORM_DECODE_MANIFEST", build)
        self.assertIn(
            "linear-gated-norm-decode-v0.1.0/manifest.json", build
        )


if __name__ == "__main__":
    unittest.main()
