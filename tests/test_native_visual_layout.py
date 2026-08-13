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
VISION_ROTARY_RESULT_PATH = (
    ROOT / "benchmarks/results/native-vision-rotary-v0.1.0.json"
)
VISION_ATTENTION_RESULT_PATH = (
    ROOT / "benchmarks/results/native-vision-segmented-attention-v0.1.0.json"
)
VISION_BLOCK_SUFFIX_RESULT_PATH = (
    ROOT / "benchmarks/results/native-vision-block-suffix-v0.1.0.json"
)
VISION_BLOCK_RESULT_PATH = (
    ROOT / "benchmarks/results/native-vision-block-v0.1.0.json"
)
VISION_DEPTH_ORACLE_RESULT_PATH = (
    ROOT / "benchmarks/results/native-vision-depth-oracle-v0.1.0.json"
)
VISION_REPRESENTATIVE_BLOCKS_RESULT_PATH = (
    ROOT / "benchmarks/results/native-vision-representative-blocks-v0.1.0.json"
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

        result = json.loads(
            VISION_DEPTH_ORACLE_RESULT_PATH.read_text(encoding="utf-8")
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "41d6f6ba79b5c916d072af7bd9311b0fc87abb26",
        )
        capture_record = result["source"]["capture_script"]
        self.assertEqual(
            hashlib.sha256((ROOT / capture_record["path"]).read_bytes()).hexdigest(),
            capture_record["sha256"],
        )
        oracle = result["oracle"]
        self.assertEqual(oracle["block_indices"], [0, 13, 26])
        self.assertEqual(oracle["independent_identical_captures"], 2)
        self.assertEqual(oracle["binary_file_count"], 36)
        self.assertTrue(oracle["qualified_for_native_boundary_comparison"])
        self.assertEqual(len(result["cases"]), 2)
        for case in result["cases"]:
            self.assertEqual(
                [block["block_index"] for block in case["blocks"]], [0, 13, 26]
            )
            self.assertTrue(
                all(block["full_model_output_exact"] for block in case["blocks"])
            )
            self.assertTrue(
                all(
                    len(block["input_sha256"]) == 64
                    and len(block["output_sha256"]) == 64
                    for block in case["blocks"]
                )
            )
        decision = result["decision"]
        self.assertTrue(decision["representative_block_inputs_frozen"])
        self.assertEqual(decision["qualified_block_indices"], [0])
        self.assertFalse(decision["all_representative_native_blocks_qualified"])
        self.assertFalse(decision["full_vision_encoder_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])
        self.assertTrue(decision["overall_pass"])
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

    def test_vision_depth_capture_hooks_all_representative_blocks(self) -> None:
        capture = (
            ROOT / "scripts/capture-vllm-vision-depth-oracles.py"
        ).read_text(encoding="utf-8")
        self.assertIn("InstallVisionDepthHooks", capture)
        self.assertIn("FinalizeVisionDepthHooks", capture)
        self.assertIn("BLOCK_INDICES = (0, 13, 26)", capture)
        self.assertIn("root.visual.blocks[block_index]", capture)
        self.assertIn("block.register_forward_pre_hook", capture)
        self.assertIn("block.attn.apply_rotary_emb.register_forward_pre_hook", capture)
        self.assertIn("block.attn.attn.register_forward_pre_hook", capture)
        self.assertIn("block.register_forward_hook", capture)
        self.assertIn("full_model_output_comparison", capture)
        self.assertIn("vision_block_{block_index}", capture)
        self.assertIn("VLLM_ALLOW_INSECURE_SERIALIZATION", capture)
        self.assertIn('llm_kwargs["skip_mm_profiling"] = True', capture)

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

        result = json.loads(VISION_ROTARY_RESULT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "b35d5dbe7ce77856c0e970d0c9060833d91d419e",
        )
        for source_name in ("build_script", "header", "rotary", "probe"):
            source_record = result["source"][source_name]
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / source_record["path"]).read_bytes()
                ).hexdigest(),
                source_record["sha256"],
            )
        self.assertEqual(len(result["cases"]), 2)
        for case in result["cases"]:
            self.assertEqual(
                set(case["comparisons"]),
                {"query_rotated", "key_rotated", "value"},
            )
            for comparison in case["comparisons"].values():
                self.assertTrue(comparison["passed"])
                self.assertEqual(
                    comparison["exact_elements"], comparison["elements"]
                )
                self.assertEqual(
                    comparison["actual_sha256"], comparison["expected_sha256"]
                )
        decision = result["decision"]
        self.assertEqual(
            decision["total_exact_elements"], decision["total_elements"]
        )
        self.assertTrue(decision["all_sha256_exact"])
        self.assertTrue(decision["overall_pass"])
        self.assertFalse(decision["full_vision_block_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])

    def test_segmented_attention_has_an_isolated_native_probe(self) -> None:
        header = (
            ROOT / "native/include/aima/native_vision_segmented_attention.h"
        ).read_text(encoding="utf-8")
        source = (
            ROOT / "native/src/native_vision_segmented_attention.hip.cpp"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT
            / "native/tools/vision_segmented_attention_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT
            / "scripts/build-native-vision-segmented-attention-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("class NativeVisionSegmentedAttentionPlan", header)
        self.assertIn("vision_segmented_attention_kernel", source)
        self.assertIn("kSoftmaxScaleLog2", source)
        self.assertIn("probabilities[kAttentionKeyBlock]", source)
        self.assertNotIn("patch_count * patch_count", source)
        self.assertIn("segment_isolation_exact_elements", probe)
        self.assertIn("native-vision-segmented-attention-oracle/v1", probe)
        self.assertIn("vision_segmented_attention_oracle_probe.hip.cpp", build)
        self.assertIn("native_vision_segmented_attention.hip.cpp", build)

        result = json.loads(
            VISION_ATTENTION_RESULT_PATH.read_text(encoding="utf-8")
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "e86b76b07bb66e590456b13347ffd43c8c3422b9",
        )
        for source_name in ("build_script", "header", "attention", "probe"):
            source_record = result["source"][source_name]
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / source_record["path"]).read_bytes()
                ).hexdigest(),
                source_record["sha256"],
            )
        self.assertEqual(len(result["cases"]), 2)
        for case in result["cases"]:
            self.assertTrue(case["passed"])
            self.assertEqual(case["finite_elements"], case["elements"])
            self.assertLessEqual(case["relative_l2_error"], 0.002)
            self.assertGreaterEqual(case["cosine_similarity"], 0.999)
            self.assertEqual(case["workspace_bytes"], 8 * case["patches"])
        video = next(
            case for case in result["cases"]
            if case["case_id"] == "video_local_mp4"
        )
        self.assertEqual(video["cu_seqlens"], [0, 64, 128])
        self.assertTrue(video["segment_isolation_applicable"])
        self.assertEqual(
            video["segment_isolation_exact_elements"],
            video["segment_isolation_elements"],
        )
        decision = result["decision"]
        self.assertTrue(decision["all_applicable_segment_isolation_exact"])
        self.assertTrue(decision["overall_pass"])
        self.assertFalse(decision["full_vision_block_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])

    def test_vision_block_suffix_has_isolated_boundaries(self) -> None:
        header = (
            ROOT / "native/include/aima/native_vision_block_suffix.h"
        ).read_text(encoding="utf-8")
        source = (
            ROOT / "native/src/native_vision_block_suffix.hip.cpp"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT / "native/tools/vision_block_suffix_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vision-block-suffix-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("class NativeVisionBlockSuffixPlan", header)
        for method in (
            "launch_attention_projection",
            "launch_residual",
            "launch_norm2",
            "launch_mlp_fc1",
            "launch_gelu",
            "launch_mlp_fc2",
        ):
            self.assertIn(method, header)
        self.assertIn("vision_exact_gelu_kernel", source)
        self.assertIn("erff", source)
        self.assertIn('"mlp.linear_fc1.weight"', source)
        self.assertIn('"mlp.linear_fc2.weight"', source)
        self.assertIn("block_output_isolated", probe)
        self.assertIn("block_output_chained", probe)
        self.assertIn("native-vision-block-suffix-oracle/v1", probe)
        self.assertIn("vision_block_suffix_oracle_probe.hip.cpp", build)

        result = json.loads(
            VISION_BLOCK_SUFFIX_RESULT_PATH.read_text(encoding="utf-8")
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "8e278f80205588ec8b00ce2c23c782f14656cd16",
        )
        for source_name in ("build_script", "header", "suffix", "probe", "gemm"):
            source_record = result["source"][source_name]
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / source_record["path"]).read_bytes()
                ).hexdigest(),
                source_record["sha256"],
            )
        self.assertEqual(len(result["cases"]), 2)
        expected_comparisons = {
            "attention_projection",
            "attention_residual",
            "norm2",
            "mlp_fc1",
            "mlp_activation",
            "mlp_fc2",
            "block_output_isolated",
            "block_output_chained",
        }
        for case in result["cases"]:
            self.assertEqual(set(case["comparisons"]), expected_comparisons)
            for comparison in case["comparisons"].values():
                self.assertTrue(comparison["passed"])
                self.assertEqual(
                    comparison["finite_elements"], comparison["elements"]
                )
                self.assertLessEqual(comparison["relative_l2_error"], 0.002)
                self.assertGreaterEqual(comparison["cosine_similarity"], 0.999)
            for name in expected_comparisons - {"norm2", "block_output_chained"}:
                comparison = case["comparisons"][name]
                self.assertEqual(
                    comparison["exact_elements"], comparison["elements"]
                )
        decision = result["decision"]
        self.assertTrue(decision["overall_pass"])
        self.assertFalse(decision["full_vision_block_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])

    def test_complete_vision_block_reuses_external_temporary_storage(self) -> None:
        header = (
            ROOT / "native/include/aima/native_vision_block.h"
        ).read_text(encoding="utf-8")
        source = (
            ROOT / "native/src/native_vision_block.hip.cpp"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT / "native/tools/vision_block_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vision-block-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("class NativeVisionBlockPlan", header)
        self.assertIn("temporary_device", header)
        self.assertIn("NativeVisionBlockPrefixPlan prefix", source)
        self.assertIn("NativeVisionRotaryPlan rotary", source)
        self.assertIn("NativeVisionSegmentedAttentionPlan attention", source)
        self.assertIn("NativeVisionBlockSuffixPlan suffix", source)
        self.assertIn("arena_large_bytes", source)
        self.assertIn("repeat_deterministic", probe)
        self.assertIn("native-vision-block-oracle/v1", probe)
        self.assertIn("vision_block_oracle_probe.hip.cpp", build)

        result = json.loads(
            VISION_BLOCK_RESULT_PATH.read_text(encoding="utf-8")
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "e1b4680a1f7348b2820a1b79c7efab335cba1ce0",
        )
        for source_name in (
            "build_script", "header", "block", "probe", "prefix", "rotary",
            "attention", "suffix", "gemm",
        ):
            source_record = result["source"][source_name]
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / source_record["path"]).read_bytes()
                ).hexdigest(),
                source_record["sha256"],
            )
        self.assertEqual(len(result["cases"]), 2)
        for case in result["cases"]:
            self.assertTrue(case["passed"])
            self.assertEqual(case["finite_elements"], case["elements"])
            self.assertLessEqual(case["relative_l2_error"], 0.002)
            self.assertGreaterEqual(case["cosine_similarity"], 0.999)
            self.assertEqual(case["temporary_bytes"], 19520 * case["patches"])
            self.assertEqual(
                case["actual_sha256"], case["repeat_actual_sha256"]
            )
            self.assertTrue(case["repeat_deterministic"])
        video = next(
            case for case in result["cases"]
            if case["case_id"] == "video_local_mp4"
        )
        self.assertEqual(video["cu_seqlens"], [0, 64, 128])
        decision = result["decision"]
        self.assertEqual(decision["qualified_block_indices"], [0])
        self.assertEqual(
            decision["required_representative_block_indices"], [0, 13, 26]
        )
        self.assertFalse(decision["all_representative_blocks_qualified"])
        self.assertFalse(decision["full_vision_encoder_qualified"])
        self.assertFalse(decision["vision_merger_qualified"])
        self.assertFalse(decision["serving_integration_qualified"])
        self.assertTrue(decision["overall_pass"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])

    def test_depth_block_probe_selects_a_frozen_block_index(self) -> None:
        probe = (
            ROOT / "native/tools/vision_depth_block_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vision-depth-block-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("BLOCK_INDEX BLOCK_INPUT", probe)
        self.assertIn("block_index >= kVisionBlockCount", probe)
        self.assertIn(
            "NativeVisionBlockPlan plan(weights, block_index, patches", probe
        )
        self.assertIn("plan.block_index() != block_index", probe)
        self.assertIn("native-vision-depth-block-oracle/v1", probe)
        self.assertIn("vision_depth_block_oracle_probe.hip.cpp", build)
        self.assertIn("native_vision_block.hip.cpp", build)

        result = json.loads(
            VISION_REPRESENTATIVE_BLOCKS_RESULT_PATH.read_text(encoding="utf-8")
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["source"]["clean"])
        self.assertEqual(
            result["source"]["commit"],
            "3681adbdf767f30fb30282d22736becafbdf67a5",
        )
        for source_name in (
            "build_script", "probe", "header", "block", "prefix", "rotary",
            "attention", "suffix", "gemm",
        ):
            source_record = result["source"][source_name]
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / source_record["path"]).read_bytes()
                ).hexdigest(),
                source_record["sha256"],
            )
        self.assertEqual(len(result["cases"]), 2)
        for case in result["cases"]:
            self.assertEqual(
                [block["block_index"] for block in case["blocks"]], [0, 13, 26]
            )
            for block in case["blocks"]:
                self.assertTrue(block["passed"])
                self.assertEqual(block["finite_elements"], block["elements"])
                self.assertLessEqual(block["relative_l2_error"], 0.002)
                self.assertGreaterEqual(block["cosine_similarity"], 0.999)
                self.assertEqual(
                    block["actual_sha256"], block["repeat_actual_sha256"]
                )
                self.assertTrue(block["repeat_deterministic"])
        decision = result["decision"]
        self.assertEqual(decision["passed_comparisons"], 6)
        self.assertEqual(decision["qualified_block_indices"], [0, 13, 26])
        self.assertTrue(decision["all_representative_native_blocks_qualified"])
        self.assertFalse(decision["full_vision_encoder_qualified"])
        self.assertFalse(decision["vision_merger_qualified"])
        self.assertFalse(decision["serving_integration_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])
        self.assertTrue(decision["overall_pass"])

    def test_vision_block_stack_reuses_one_intermediate_and_one_arena(self) -> None:
        header = (
            ROOT / "native/include/aima/native_vision_block_stack.h"
        ).read_text(encoding="utf-8")
        source = (
            ROOT / "native/src/native_vision_block_stack.hip.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("class NativeVisionBlockStackPlan", header)
        self.assertIn("constexpr std::size_t kVisionBlockCount = 27", source)
        self.assertIn("blocks.reserve(kVisionBlockCount)", source)
        self.assertIn("blocks.emplace_back(weights, block_index", source)
        self.assertIn("intermediate + impl_->intermediate_bytes", source)
        self.assertIn("(last_block_index - block_index) % 2 == 0", source)
        self.assertIn("impl_->blocks[block_index].launch", source)
        probe = (
            ROOT / "native/tools/vision_block_stack_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vision-block-stack-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("NativeVisionBlockStackPlan plan", probe)
        self.assertIn("plan.block_count() != 27", probe)
        self.assertIn("plan.launch_through(last_block_index", probe)
        self.assertIn("native-vision-block-stack-oracle/v1", probe)
        self.assertIn("vision_block_stack_oracle_probe.hip.cpp", build)
        self.assertIn("native_vision_block_stack.hip.cpp", build)

    def test_vision_attention_aot_trace_replays_frozen_raw_tensors(self) -> None:
        trace = (
            ROOT / "scripts/trace-vllm-vision-attention-aot.py"
        ).read_text(encoding="utf-8")
        self.assertIn("context_attention_fwd", trace)
        self.assertIn('is_causal=False', trace)
        self.assertIn('torch.bfloat16', trace)
        self.assertIn('query_rotated.bin', trace)
        self.assertIn('key_rotated.bin', trace)
        self.assertIn('value.bin', trace)
        self.assertIn('attention.bin', trace)
        self.assertIn('actual != expected', trace)

        manifest_path = (
            ROOT / "native/aot/gfx1151/vision-attention-v0.1.0/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["kernel_count"], 1)
        kernel = manifest["kernels"][0]
        self.assertEqual(kernel["kernel_hash"], "e45a4026e641c9f89b75bc5e5c22072d45ff39c42046ce4edec8d0c617d1584c")
        self.assertEqual(kernel["symbol"], "_fwd_kernel")
        self.assertEqual(kernel["compile_constants"]["BLOCK_M"], 128)
        self.assertEqual(kernel["compile_constants"]["BLOCK_N"], 128)
        self.assertEqual(kernel["compile_constants"]["Lk"], 72)
        self.assertFalse(kernel["compile_constants"]["IS_CAUSAL"])
        image = kernel["image"]
        self.assertEqual(
            hashlib.sha256(
                (manifest_path.parent / image["path"]).read_bytes()
            ).hexdigest(),
            image["sha256"],
        )

        header = (
            ROOT / "native/include/aima/native_vision_aot_attention.h"
        ).read_text(encoding="utf-8")
        source = (
            ROOT / "native/src/native_vision_aot_attention.hip.cpp"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT / "native/tools/vision_aot_attention_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("class NativeVisionAotAttentionPlan", header)
        self.assertIn(kernel["image"]["sha256"], source)
        self.assertIn("AotKernel::from_file", source)
        self.assertIn("config.shared_memory_bytes = 32768", source)
        self.assertIn("native-vision-aot-attention-oracle/v1", probe)

    def test_vision_aot_block_stack_shares_one_exact_attention_plan(self) -> None:
        block = (
            ROOT / "native/src/native_vision_aot_block.hip.cpp"
        ).read_text(encoding="utf-8")
        stack = (
            ROOT / "native/src/native_vision_aot_block_stack.hip.cpp"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT / "native/tools/vision_aot_block_stack_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vision-aot-block-stack-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("impl_->attention->launch", block)
        self.assertNotIn("NativeVisionSegmentedAttentionPlan", block)
        self.assertIn("impl_->norm1.launch", block)
        self.assertIn("impl_->norm2.launch", block)
        self.assertIn("NativeVisionLayerNormReciprocal::kFastAmdReciprocal", block)
        self.assertIn("std::make_shared<NativeVisionAotAttentionPlan>", stack)
        self.assertIn("constexpr std::size_t kVisionBlockCount = 27", stack)
        self.assertIn("blocks.emplace_back(weights, block_index, patches, attention)", stack)
        self.assertIn("plan.launch_through(last_block_index", probe)
        self.assertIn("native-vision-aot-block-stack-oracle/v1", probe)
        self.assertIn("native_vision_aot_attention.hip.cpp", build)

    def test_exact_vision_layer_norm_freezes_upstream_reduction_modes(self) -> None:
        source = (
            ROOT / "native/src/native_vision_exact_layer_norm.hip.cpp"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT / "native/tools/vision_exact_layer_norm_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("8514f05131610dab50233027b2fab9c01235081b", source)
        self.assertIn("constexpr int kVectorSize = 4", source)
        self.assertIn("constexpr int kWarpsPerBlock = 8", source)
        self.assertIn("__builtin_amdgcn_rcpf", source)
        self.assertIn("welford_combine<FastReciprocal>", source)
        self.assertIn("native-vision-exact-layer-norm-oracle/v1", probe)
        self.assertIn("const bool complete = fast_result.exact", probe)

    def test_exact_aot_vision_encoder_evidence_is_hash_bound(self) -> None:
        layer_norm = json.loads(
            (
                ROOT
                / "benchmarks/results/native-vision-exact-layer-norm-v0.1.0.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(layer_norm["complete"])
        self.assertEqual(
            layer_norm["implementation_contract"]["selected_reciprocal"],
            "fast_amd_reciprocal",
        )
        for case in layer_norm["cases"]:
            self.assertTrue(case["passed"])
            self.assertTrue(case["candidates"]["fast_amd_reciprocal"]["exact"])

        encoder = json.loads(
            (
                ROOT / "benchmarks/results/native-vision-aot-encoder-v0.1.0.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(encoder["complete"])
        for source in encoder["source"]["files"]:
            self.assertEqual(
                hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(),
                source["sha256"],
            )
        self.assertEqual(encoder["attention_aot"]["shared_plan_count"], 1)
        self.assertEqual(len(encoder["cases"]), 2)
        for case in encoder["cases"]:
            self.assertEqual(
                [item["block_index"] for item in case["checkpoints"]],
                [0, 13, 26],
            )
            for checkpoint in case["checkpoints"]:
                self.assertTrue(checkpoint["passed"])
                self.assertEqual(
                    checkpoint["exact_elements"], checkpoint["elements"]
                )
                self.assertEqual(checkpoint["relative_l2_error"], 0.0)
                self.assertEqual(checkpoint["cosine_similarity"], 1.0)
                self.assertEqual(
                    checkpoint["expected_sha256"], checkpoint["actual_sha256"]
                )
                self.assertEqual(
                    checkpoint["actual_sha256"],
                    checkpoint["repeat_actual_sha256"],
                )
        decision = encoder["decision"]
        self.assertTrue(decision["all_outputs_bit_exact"])
        self.assertTrue(decision["all_27_blocks_executed"])
        self.assertTrue(decision["representative_image_video_encoder_qualified"])
        self.assertFalse(decision["vision_merger_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])

    def test_native_vision_merger_preserves_contiguous_merge_order(self) -> None:
        header = (
            ROOT / "native/include/aima/native_vision_merger.h"
        ).read_text(encoding="utf-8")
        source = (
            ROOT / "native/src/native_vision_merger.hip.cpp"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT / "native/tools/vision_merger_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vision-merger-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("class NativeVisionMergerPlan", header)
        self.assertIn("kMergerHidden = kVisionHidden * kSpatialMergeArea", source)
        self.assertIn("NativeVisionLayerNormReciprocal::kFastAmdReciprocal", source)
        self.assertIn("launch_norm(input_device, arena_a", source)
        self.assertIn("launch_fc1(arena_a, arena_b", source)
        self.assertIn("launch_gelu(arena_b, arena_a", source)
        self.assertIn("launch_fc2(arena_a, output_device", source)
        self.assertIn("native-vision-merger-oracle/v1", probe)
        self.assertIn("native_vision_merger.hip.cpp", build)

    def test_native_vision_merger_evidence_is_hash_bound(self) -> None:
        evidence = json.loads(
            (
                ROOT / "benchmarks/results/native-vision-merger-v0.1.0.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(evidence["complete"])
        for source in evidence["source"]["files"]:
            self.assertEqual(
                hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(),
                source["sha256"],
            )
        self.assertEqual(
            [case["case_id"] for case in evidence["cases"]],
            [
                "image_local_png",
                "video_local_mp4",
                "multi_image",
                "multi_video",
                "mixed_image_video",
            ],
        )
        for case in evidence["cases"]:
            self.assertTrue(case["passed"])
            self.assertEqual(case["exact_elements"], case["elements"])
            self.assertEqual(case["finite_elements"], case["elements"])
            self.assertEqual(case["relative_l2_error"], 0.0)
            self.assertEqual(case["cosine_similarity"], 1.0)
            self.assertEqual(case["expected_sha256"], case["actual_sha256"])
            self.assertEqual(
                case["actual_sha256"], case["repeat_actual_sha256"]
            )
        decision = evidence["decision"]
        self.assertEqual(decision["total_elements"], 884736)
        self.assertEqual(decision["exact_elements"], 884736)
        self.assertTrue(decision["all_outputs_bit_exact"])
        self.assertTrue(decision["all_five_oracle_shapes_qualified"])
        self.assertTrue(decision["vision_merger_qualified"])
        self.assertFalse(decision["full_vision_pipeline_qualified"])
        self.assertFalse(decision["media_embedding_injection_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])

    def test_native_vision_pipeline_composes_frozen_stages(self) -> None:
        header = (
            ROOT / "native/include/aima/native_vision_pipeline.h"
        ).read_text(encoding="utf-8")
        source = (
            ROOT / "native/src/native_vision_pipeline.hip.cpp"
        ).read_text(encoding="utf-8")
        probe = (
            ROOT / "native/tools/vision_pipeline_oracle_probe.hip.cpp"
        ).read_text(encoding="utf-8")
        build = (
            ROOT / "scripts/build-native-vision-pipeline-probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("class NativeVisionEncoderMetadataPlan", header)
        self.assertIn("class NativeVisionPipelinePlan", header)
        self.assertIn("kVisionRotaryMaximumPosition = 8192", source)
        self.assertIn("kVisionInverseFrequencyBits", source)
        self.assertIn("vision_rotary_metadata_kernel", source)
        self.assertIn("cosf(angle)", source)
        self.assertIn("result.cu_seqlens.push_back", source)
        self.assertIn("cos_sha256_value = sha256_bytes", source)
        self.assertIn("rotary_cos_sha256", probe)
        self.assertIn("impl_->patch.launch", source)
        self.assertIn("impl_->position.launch_add", source)
        self.assertIn("impl_->blocks.launch", source)
        self.assertIn("impl_->merger.launch", source)
        self.assertIn("launch_encoder_through", header)
        self.assertIn("impl_->blocks.launch_through", source)
        self.assertIn("native-vision-pipeline-oracle/v1", probe)
        self.assertIn("read_concatenated_files", probe)
        self.assertIn("native_vision_pipeline.hip.cpp", build)

    def test_multimedia_block_capture_reuses_frozen_hooks(self) -> None:
        capture = (
            ROOT
            / "scripts/capture-vllm-vision-multimedia-block-oracles.py"
        ).read_text(encoding="utf-8")
        self.assertIn("capture-vllm-vision-block-oracles.py", capture)
        self.assertIn('(\"multi_image\", \"multi_video\")', capture)
        self.assertIn("vision-multimedia-block-oracle/v1", capture)
        self.assertIn("cloudpickle.register_pickle_by_value(module)", capture)

    def test_full_native_vision_pipeline_evidence_is_hash_bound(self) -> None:
        multimedia = json.loads(
            (
                ROOT
                / "benchmarks/results/"
                "native-vision-multimedia-block-oracle-v0.1.0.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(multimedia["complete"])
        for source in multimedia["source"]["files"]:
            self.assertEqual(
                hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(),
                source["sha256"],
            )
        self.assertEqual(
            [case["case_id"] for case in multimedia["cases"]],
            ["multi_image", "multi_video"],
        )
        self.assertTrue(
            multimedia["decision"]["all_independent_block_outputs_bit_exact"]
        )

        evidence = json.loads(
            (
                ROOT / "benchmarks/results/native-vision-pipeline-v0.1.0.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(evidence["complete"])
        for source in evidence["source"]["files"]:
            self.assertEqual(
                hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(),
                source["sha256"],
            )
        for dependency in ("encoder_qualification", "merger_qualification"):
            record = evidence["dependencies"][dependency]
            self.assertEqual(
                hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest(),
                record["sha256"],
            )
        multimedia_dependency = evidence["dependencies"][
            "multimedia_block_oracle"
        ]
        self.assertEqual(
            hashlib.sha256(
                (ROOT / multimedia_dependency["path"]).read_bytes()
            ).hexdigest(),
            multimedia_dependency["public_record_sha256"],
        )
        self.assertEqual(
            [case["case_id"] for case in evidence["cases"]],
            [
                "image_local_png",
                "video_local_mp4",
                "multi_image",
                "multi_video",
                "mixed_image_video",
            ],
        )
        boundary_elements = 0
        exact_elements = 0
        for case in evidence["cases"]:
            self.assertTrue(case["passed"])
            self.assertEqual(case["relative_l2_error"], 0.0)
            self.assertEqual(case["cosine_similarity"], 1.0)
            self.assertEqual(len(case["boundaries"]), 3)
            for boundary in case["boundaries"]:
                boundary_elements += boundary["elements"]
                exact_elements += boundary["exact_elements"]
                self.assertEqual(
                    boundary["exact_elements"], boundary["elements"]
                )
        decision = evidence["decision"]
        self.assertEqual(boundary_elements, 4866048)
        self.assertEqual(exact_elements, 4866048)
        self.assertEqual(decision["total_boundary_elements"], boundary_elements)
        self.assertEqual(decision["exact_boundary_elements"], exact_elements)
        self.assertTrue(decision["all_metadata_hashes_exact"])
        self.assertTrue(decision["all_boundaries_bit_exact"])
        self.assertTrue(decision["full_visual_pipeline_qualified"])
        self.assertFalse(decision["media_embedding_injection_qualified"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])


if __name__ == "__main__":
    unittest.main()
