#!/usr/bin/env python3
"""Contract tests for the frozen native visual weight layout."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "engine" / "native-visual-weight-manifest.json"
HEADER_PATH = ROOT / "native" / "generated" / "visual_model_layout.h"


class NativeVisualLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_capture_is_bound_to_the_frozen_checkpoint(self) -> None:
        self.assertEqual(
            hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            "abc5b3a0cc0881ba2d3e815b472eebe3404a6e3bc6438a430faccfbe8093c0aa",
        )
        self.assertEqual(
            self.manifest["schema"],
            "aima-amd395-qwen36/native-visual-weight-manifest/v1",
        )
        self.assertTrue(self.manifest["complete"])
        self.assertEqual(
            self.manifest["model"]["revision"],
            "995ad96eacd98c81ed38be0c5b274b04031597b0",
        )
        self.assertEqual(
            self.manifest["model"]["config_sha256"],
            "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
        )
        self.assertEqual(
            self.manifest["model"]["checkpoint_index_sha256"],
            "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        )

    def test_visual_layout_closes_the_fixed_architecture(self) -> None:
        entries = self.manifest["entries"]
        layout = self.manifest["layout"]
        self.assertEqual(len(entries), 333)
        self.assertEqual(layout["tensor_count"], 333)
        self.assertEqual(layout["payload_bytes"], 893_142_496)
        self.assertEqual(layout["payload_xor_u64"], "0xedae72382a609c99")
        self.assertEqual(layout["payload_sum_u64"], "0xcbf21fe7cde36cdf")
        names = [entry["name"] for entry in entries]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            hashlib.sha256("\n".join(names).encode()).hexdigest(),
            layout["active_names_sha256"],
        )
        self.assertEqual(
            sum(entry["payload_bytes"] for entry in entries),
            layout["payload_bytes"],
        )
        for entry in entries:
            self.assertEqual(entry["dtype"], "BF16")
            self.assertEqual(
                entry["payload_bytes"], math.prod(entry["shape"]) * 2
            )
        for block in range(27):
            prefix = f"model.visual.blocks.{block}."
            self.assertEqual(sum(name.startswith(prefix) for name in names), 12)
        by_name = {entry["name"]: entry for entry in entries}
        self.assertEqual(
            by_name["model.visual.patch_embed.proj.weight"]["shape"],
            [1152, 3, 2, 16, 16],
        )
        self.assertEqual(
            by_name["model.visual.merger.linear_fc1.weight"]["shape"],
            [4608, 4608],
        )

    def test_visual_sources_match_the_reference_manifest(self) -> None:
        reference = json.loads(
            (ROOT / "benchmarks/results/vl-reference-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        files = reference["model"]["files"]
        self.assertEqual(
            self.manifest["source_files"],
            {
                name: {"bytes": files[name]["bytes"], "sha256": files[name]["sha256"]}
                for name in self.manifest["source_files"]
            },
        )
        for shard_name, source in self.manifest["source_files"].items():
            ranges = sorted(
                (
                    entry["source_offset_bytes"],
                    entry["source_offset_bytes"] + entry["payload_bytes"],
                )
                for entry in self.manifest["entries"]
                if entry["source_shard"] == shard_name
            )
            self.assertTrue(ranges)
            self.assertLessEqual(ranges[-1][1], source["bytes"])
            self.assertTrue(
                all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))
            )

    def test_generated_visual_layout_is_current(self) -> None:
        result = subprocess.run(
            ["python3", "scripts/generate-native-visual-layout.py", "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        header = HEADER_PATH.read_text(encoding="utf-8")
        self.assertIn("std::array<TensorSpec, 333>", header)
        self.assertIn("kPayloadBytes = 893142496ULL", header)
        self.assertIn("kExpectedPayloadXor = 17126752018491284633ULL", header)
        self.assertIn("std::array<std::uint32_t, 5> shape", header)

    def test_resident_loader_uses_the_visual_contract(self) -> None:
        loader = (ROOT / "native/src/native_weight_store.hip.cpp").read_text(
            encoding="utf-8"
        )
        resident = (ROOT / "native/src/native_resident_engine.hip.cpp").read_text(
            encoding="utf-8"
        )
        build = (ROOT / "scripts/build-native-visual-weight-probe.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('#include "visual_model_layout.h"', loader)
        self.assertIn("NativeWeightStore::load_visual", loader)
        self.assertIn("generated::visual::kExpectedPayloadXor", loader)
        self.assertIn("weights.load_resident", resident)
        self.assertIn("NativeWeightStore::load_resident", loader)
        self.assertIn("generated::kExpectedPayloadXor ^", loader)
        self.assertIn("torch_owned_safetensors_loader.hip.cpp", build)

    def test_patch_embedding_is_native_and_oracle_qualified(self) -> None:
        header = (ROOT / "native/include/aima/native_vision_encoder.h").read_text(
            encoding="utf-8"
        )
        source = (ROOT / "native/src/native_vision_encoder.hip.cpp").read_text(
            encoding="utf-8"
        )
        gemm = (ROOT / "native/src/bf16_gemm.hip.cpp").read_text(
            encoding="utf-8"
        )
        build = (ROOT / "scripts/build-native-runtime.sh").read_text(
            encoding="utf-8"
        )
        probe = (ROOT / "scripts/build-native-vision-patch-probe.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("class NativeVisionPatchEmbedPlan", header)
        self.assertIn("model.visual.patch_embed.proj.weight", source)
        self.assertIn("model.visual.patch_embed.proj.bias", source)
        self.assertIn("launch_with_bias", source)
        self.assertIn("HIPBLASLT_EPILOGUE_BIAS", gemm)
        self.assertIn("native_vision_encoder.hip.cpp", build)
        self.assertIn("vision_patch_oracle_probe.hip.cpp", probe)


if __name__ == "__main__":
    unittest.main()
