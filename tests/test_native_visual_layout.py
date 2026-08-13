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
PATCH_RESULT_PATH = ROOT / "benchmarks/results/native-vision-patch-v0.1.0.json"
WITHDRAWN_POSITION_RESULT_PATH = (
    ROOT / "benchmarks/results/native-vision-position-v0.1.0.json"
)
POSITION_RESULT_PATH = (
    ROOT / "benchmarks/results/native-vision-position-v0.2.0.json"
)
VISION_BLOCK_ORACLE_RESULT_PATH = (
    ROOT / "benchmarks/results/native-vision-block-oracle-v0.1.0.json"
)
VISION_BLOCK_PREFIX_RESULT_PATH = (
    ROOT / "benchmarks/results/native-vision-block-prefix-v0.1.0.json"
)


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

        result = json.loads(PATCH_RESULT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(result["source"]["commit"], "9d29d9de9b4c68fd932b8616ef2bee6d65794266")
        self.assertEqual(len(result["cases"]), 4)
        self.assertTrue(result["decision"]["overall_pass"])
        self.assertEqual(
            result["decision"]["total_elements"],
            result["decision"]["total_exact_elements"],
        )
        self.assertTrue(
            all(case["expected_sha256"] == case["actual_sha256"] for case in result["cases"])
        )

    def test_position_interpolation_has_a_frozen_native_boundary(self) -> None:
        header = (ROOT / "native/include/aima/native_vision_encoder.h").read_text(
            encoding="utf-8"
        )
        source = (ROOT / "native/src/native_vision_encoder.hip.cpp").read_text(
            encoding="utf-8"
        )
        capture = (
            ROOT / "scripts/capture-vllm-vision-position-oracles.py"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT / "native/tools/vision_position_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vision-position-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("class NativeVisionPositionPlan", header)
        self.assertIn("launch_add", header)
        self.assertIn("triton_position_coordinate", source)
        self.assertIn("triton_position_fraction", source)
        self.assertIn("triton_bf16_product", source)
        self.assertIn("v_dot2_bf16_bf16", source)
        self.assertIn("fmaf", source)
        self.assertIn("model.visual.pos_embed.weight", source)
        self.assertIn("kNativeVlMergeSize", source)
        self.assertIn("triton_pos_embed_interpolate", capture)
        self.assertIn(
            "8ba3592a0fb481a959d6952af25a721cfaeab966558ac11214304e5cf7524d1a",
            capture,
        )
        self.assertIn("concatenated_exact_elements", probe)
        self.assertIn("zero_add_exact_elements", probe)
        self.assertIn("vision_position_oracle_probe.hip.cpp", build)

        withdrawn = json.loads(
            WITHDRAWN_POSITION_RESULT_PATH.read_text(encoding="utf-8")
        )
        self.assertFalse(withdrawn["complete"])
        self.assertEqual(withdrawn["status"], "withdrawn")
        self.assertTrue(withdrawn["withdrawal"]["serving_path_has_triton"])
        self.assertFalse(withdrawn["decision"]["overall_pass"])

        result = json.loads(POSITION_RESULT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "851605b0bfc3336123493aaab3f966f912def73a",
        )
        self.assertEqual(
            result["correction"]["withdrawn_result"],
            "benchmarks/results/native-vision-position-v0.1.0.json",
        )
        for source_name in (
            "build_script",
            "reference_capture",
            "vision_encoder",
            "probe",
        ):
            source_record = result["source"][source_name]
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / source_record["path"]).read_bytes()
                ).hexdigest(),
                source_record["sha256"],
            )
        self.assertEqual(
            result["oracle"]["reference_function"],
            "vllm.model_executor.models.qwen3_vl.triton_pos_embed_interpolate",
        )
        self.assertTrue(result["oracle"]["has_triton"])
        self.assertEqual(result["oracle"]["independent_identical_captures"], 2)
        self.assertEqual(len(result["cases"]), 4)
        self.assertTrue(result["decision"]["overall_pass"])
        self.assertEqual(
            result["decision"]["total_elements"],
            result["decision"]["total_exact_elements"],
        )
        self.assertEqual(
            result["decision"]["total_elements"],
            sum(case["elements"] for case in result["cases"]),
        )
        self.assertEqual(
            result["decision"]["total_concatenated_exact_elements"],
            2 * result["decision"]["total_elements"],
        )
        self.assertEqual(
            result["decision"]["total_zero_add_exact_elements"],
            2 * result["decision"]["total_elements"],
        )
        self.assertTrue(
            all(
                case["first_mismatch_index"] == -1
                and case["expected_sha256"] == case["actual_sha256"]
                for case in result["cases"]
            )
        )

    def test_vision_block_capture_uses_full_model_serving_hooks(self) -> None:
        capture = (
            ROOT / "scripts/capture-vllm-vision-block-oracles.py"
        ).read_text(encoding="utf-8")
        self.assertIn("InstallVisionBlockHooks", capture)
        self.assertIn("FinalizeVisionBlockHooks", capture)
        self.assertIn("root.visual.blocks[0]", capture)
        self.assertIn("block.attn.qkv.register_forward_hook", capture)
        self.assertIn("block.attn.attn.register_forward_pre_hook", capture)
        self.assertIn("block.mlp.act_fn.register_forward_hook", capture)
        self.assertIn("independent block capture is not byte-identical", capture)
        self.assertIn("VLLM_ALLOW_INSECURE_SERIALIZATION", capture)
        self.assertIn('llm_kwargs["skip_mm_profiling"] = True', capture)
        self.assertIn(
            "87dcdf76b7251f78da01a2a5f4312a9fb5c7d07a1ca2b2420566e77930f23d44",
            capture,
        )
        self.assertIn(
            "9d316fd6904764f88cd5f25726ecaed33d95bb6cfb4bbe21454c909d66c5d9f6",
            capture,
        )
        result = json.loads(
            VISION_BLOCK_ORACLE_RESULT_PATH.read_text(encoding="utf-8")
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "23ddda59502b6ba807ee374fb2b730d2c835cce3",
        )
        self.assertEqual(
            result["source"]["capture_script_sha256"],
            hashlib.sha256(
                (ROOT / result["source"]["capture_script"]).read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(result["oracle"]["case_count"], 2)
        self.assertEqual(result["oracle"]["component_count_per_case"], 16)
        expected_components = {
            "block_input",
            "norm1",
            "qkv_linear",
            "rotary_cos",
            "rotary_sin",
            "query_rotated",
            "key_rotated",
            "value",
            "attention",
            "attention_projection",
            "attention_residual",
            "norm2",
            "mlp_fc1",
            "mlp_activation",
            "mlp_fc2",
            "block_output",
        }
        self.assertTrue(result["decision"]["overall_pass"])
        for case in result["cases"]:
            self.assertEqual(set(case["components"]), expected_components)
            self.assertTrue(case["independent_full_model_output_exact"])
            self.assertTrue(
                all(
                    len(component["sha256"]) == 64
                    for component in case["components"].values()
                )
            )

    def test_vision_block_prefix_has_an_isolated_native_probe(self) -> None:
        header = (
            ROOT / "native/include/aima/native_vision_encoder.h"
        ).read_text(encoding="utf-8")
        source = (
            ROOT / "native/src/native_vision_block_prefix.hip.cpp"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT / "native/tools/vision_block_prefix_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vision-block-prefix-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("class NativeVisionBlockPrefixPlan", header)
        self.assertIn("vision_layer_norm_kernel", source)
        self.assertIn("WelfordState", source)
        self.assertIn("kVisionLayerNormEpsilon = 1.0e-6f", source)
        self.assertIn('"attn.qkv.weight"', source)
        self.assertIn('"attn.qkv.bias"', source)
        self.assertIn("qkv_gemm.launch_with_bias", source)
        self.assertIn("native-vision-block-prefix-oracle/v1", probe)
        self.assertIn("norm1.passed() && qkv.passed()", probe)
        self.assertIn("vision_block_prefix_oracle_probe.hip.cpp", build)
        self.assertIn("native_vision_block_prefix.hip.cpp", build)

        result = json.loads(
            VISION_BLOCK_PREFIX_RESULT_PATH.read_text(encoding="utf-8")
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "73498285c1538097f2c7ade11a49b6ce3936e481",
        )
        for source_name in ("build_script", "block_prefix", "probe", "gemm"):
            source_record = result["source"][source_name]
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / source_record["path"]).read_bytes()
                ).hexdigest(),
                source_record["sha256"],
            )
        oracle_record = json.loads(
            (ROOT / result["oracle"]["public_record"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / result["oracle"]["public_record"]).read_bytes()
            ).hexdigest(),
            result["oracle"]["public_record_sha256"],
        )
        self.assertEqual(
            oracle_record["oracle"]["manifest_sha256"],
            result["oracle"]["raw_manifest_sha256"],
        )
        self.assertEqual(len(result["cases"]), 2)
        for case in result["cases"]:
            self.assertEqual(set(case["comparisons"]), {"norm1", "qkv_linear"})
            for comparison in case["comparisons"].values():
                self.assertTrue(comparison["passed"])
                self.assertEqual(
                    comparison["finite_elements"], comparison["elements"]
                )
                self.assertLessEqual(comparison["relative_l2_error"], 0.002)
                self.assertGreaterEqual(comparison["cosine_similarity"], 0.999)
        decision = result["decision"]
        self.assertTrue(decision["overall_pass"])
        self.assertFalse(decision["full_vision_block_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])

    def test_vision_rotary_has_an_isolated_native_probe(self) -> None:
        header = (
            ROOT / "native/include/aima/native_vision_rotary.h"
        ).read_text(encoding="utf-8")
        source = (
            ROOT / "native/src/native_vision_rotary.hip.cpp"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT / "native/tools/vision_rotary_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vision-rotary-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("class NativeVisionRotaryPlan", header)
        self.assertIn("vision_rotary_kernel", source)
        self.assertIn("kVisionRotaryHalfDimension = 36", source)
        self.assertIn("fmaf", source)
        self.assertIn("native-vision-rotary-oracle/v1", probe)
        self.assertIn("value_comparison.exact_elements", probe)
        self.assertIn("vision_rotary_oracle_probe.hip.cpp", build)
        self.assertIn("native_vision_rotary.hip.cpp", build)


if __name__ == "__main__":
    unittest.main()
