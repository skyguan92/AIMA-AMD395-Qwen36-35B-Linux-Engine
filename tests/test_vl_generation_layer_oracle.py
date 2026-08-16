from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from aima_engine.vl_generation_layer_oracle import (
    BOUNDARY_NAMES,
    FIRST_DECODE_LINEAR_OUTPUT_INDEX,
    GENERATION_LAYER_ORACLE_SCHEMA,
    HIDDEN_SIZE,
    LAYER0_TAIL_BOUNDARY_SPECS,
    LINEAR_ATTENTION_BOUNDARY_SPECS,
    NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES,
    validate_generation_layer_oracle_manifest,
)
from aima_engine.vl_generation_oracle import CASE_CONTRACTS, CASE_ORDER
from aima_engine.vl_oracle import TENSOR_SCHEMA
from aima_engine.vl_reference import file_component, seal_manifest


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "scripts/capture-vllm-vl-generation-layer-oracles.py"
RUNNER_HEADER = ROOT / "native/include/aima/native_decode_runner.h"
RUNNER = ROOT / "native/src/native_decode_runner.hip.cpp"
RESIDENT = ROOT / "native/src/native_resident_engine.hip.cpp"
HTTP = ROOT / "native/src/native_http_server.cpp"


class VlGenerationLayerOracleTest(unittest.TestCase):
    def test_manifest_validates_hash_bound_boundary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            raw = b"\x00" * (HIDDEN_SIZE * 2)
            digest = hashlib.sha256(raw).hexdigest()

            def write_boundary_set(
                case_id: str,
                directory: str,
                target_decode_call: int,
                specs: dict[str, tuple[list[int], str, int]],
            ) -> dict[str, object]:
                components = {}
                ledger_lines = []
                for name, (shape, dtype, element_size) in specs.items():
                    elements = 1
                    for dimension in shape:
                        elements *= dimension
                    boundary_raw = b"\x00" * (elements * element_size)
                    relative = (
                        Path(case_id)
                        / directory
                        / "components"
                        / f"{name}.bin"
                    )
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(boundary_raw)
                    components[name] = {
                        "schema": TENSOR_SCHEMA,
                        "path": relative.as_posix(),
                        "shape": shape,
                        "dtype": dtype,
                        "element_size": element_size,
                        "bytes": len(boundary_raw),
                        "sha256": hashlib.sha256(boundary_raw).hexdigest(),
                    }
                    ledger_lines.append(
                        json.dumps(
                            {
                                "event": "native_layer_oracle_tensor",
                                "label": name,
                                "file": f"components/{name}.bin",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                ledger = root / case_id / directory / "oracle.jsonl"
                ledger.write_text(
                    "\n".join(ledger_lines) + "\n", encoding="utf-8"
                )
                return {
                    "target_decode_call": target_decode_call,
                    "components": components,
                    "oracle_jsonl": file_component(
                        ledger, f"{case_id}/{directory}/oracle.jsonl"
                    ),
                }

            for case_id in CASE_ORDER:
                contract = CASE_CONTRACTS[case_id]
                components = {}
                ledger_lines = []
                for name in BOUNDARY_NAMES:
                    relative = Path(case_id) / "components" / f"{name}.bin"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                    components[name] = {
                        "schema": TENSOR_SCHEMA,
                        "path": relative.as_posix(),
                        "shape": [HIDDEN_SIZE],
                        "dtype": "torch.bfloat16",
                        "element_size": 2,
                        "bytes": len(raw),
                        "sha256": digest,
                    }
                    ledger_lines.append(
                        json.dumps(
                            {
                                "event": "native_layer_oracle_tensor",
                                "label": name,
                                "file": f"components/{name}.bin",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                ledger = root / case_id / "oracle.jsonl"
                ledger.write_text(
                    "\n".join(ledger_lines) + "\n", encoding="utf-8"
                )
                cases.append(
                    {
                        "case_id": case_id,
                        "passed": True,
                        "target_output_index": contract[
                            "divergence_output_index"
                        ],
                        "target_token_id": contract["reference_token_id"],
                        "captured_logits_output_index": contract[
                            "divergence_output_index"
                        ],
                        "target_layer_decode_call": contract[
                            "divergence_output_index"
                        ],
                        "components": components,
                        "oracle_jsonl": file_component(
                            ledger, f"{case_id}/oracle.jsonl"
                        ),
                        "linear_attention": write_boundary_set(
                            case_id,
                            "linear",
                            contract["divergence_output_index"],
                            LINEAR_ATTENTION_BOUNDARY_SPECS,
                        ),
                        "first_decode_linear_attention": (
                            write_boundary_set(
                                case_id,
                                "first-decode-linear",
                                FIRST_DECODE_LINEAR_OUTPUT_INDEX,
                                LINEAR_ATTENTION_BOUNDARY_SPECS,
                            )
                        ),
                        "layer0_tail": write_boundary_set(
                            case_id,
                            "layer0-tail",
                            contract["divergence_output_index"],
                            LAYER0_TAIL_BOUNDARY_SPECS,
                        ),
                        "first_decode_layer0_tail": write_boundary_set(
                            case_id,
                            "first-decode-layer0-tail",
                            FIRST_DECODE_LINEAR_OUTPUT_INDEX,
                            LAYER0_TAIL_BOUNDARY_SPECS,
                        ),
                    }
                )
            manifest = seal_manifest(
                {
                    "schema": GENERATION_LAYER_ORACLE_SCHEMA,
                    "complete": True,
                    "qualified_for_decode_attribution": True,
                    "generation_oracle": {"sha256": "0" * 64},
                    "cases": cases,
                    "decision": {
                        "two_target_prefixes_exact": True,
                        "two_target_logits_bound": True,
                        "two_decode_boundary_sets_captured": True,
                        "two_layer0_linear_attention_boundary_sets_captured": True,
                        "two_first_decode_layer0_linear_attention_boundary_sets_captured": True,
                        "two_layer0_tail_boundary_sets_captured": True,
                        "two_first_decode_layer0_tail_boundary_sets_captured": True,
                        "two_routed_moe_stage_sets_captured": True,
                        "g1_passed": False,
                        "g2_passed": False,
                        "g3_passed": False,
                        "g4_passed": False,
                        "g5_passed": False,
                    },
                }
            )
            self.assertEqual(
                validate_generation_layer_oracle_manifest(
                    manifest, oracle_root=root
                ),
                [],
            )
            manifest["cases"][0]["layer0_tail"]["components"][
                "router_weights"
            ]["sha256"] = "e" * 64
            self.assertIn(
                "raw tensor SHA-256 mismatch: "
                "tool_forced_image/layer0-tail/components/router_weights.bin",
                validate_generation_layer_oracle_manifest(
                    manifest, oracle_root=root
                ),
            )
            manifest["cases"][0]["components"]["layer_000_output"][
                "sha256"
            ] = "f" * 64
            self.assertIn(
                "raw tensor SHA-256 mismatch: "
                "tool_forced_image/components/layer_000_output.bin",
                validate_generation_layer_oracle_manifest(
                    manifest, oracle_root=root
                ),
            )

    def test_capture_targets_one_decode_step_with_one_model_load(self) -> None:
        source = CAPTURE.read_text(encoding="utf-8")
        self.assertIn('llm_kwargs["skip_mm_profiling"] = True', source)
        self.assertIn("cloudpickle.register_pickle_by_value", source)
        self.assertIn("max_tokens=target_index + 1", source)
        self.assertIn("target_logits_sha256", source)
        self.assertIn("boundary_singleton_calls", source)
        self.assertIn("first_decode_captures", source)
        self.assertIn('"first-decode",', source)
        self.assertIn("linear_singleton_calls", source)
        self.assertIn("first_decode_linear_captures", source)
        self.assertIn("first_decode_layer0_tail_captures", source)
        self.assertIn("instrumented_router_select_experts", source)
        self.assertIn("instrumented_apply_moe_activation", source)
        self.assertIn("modular_moe_module.apply_moe_activation", source)
        self.assertIn("original_modular_apply_moe_activation", source)
        self.assertIn("instrumented_moe_sum", source)
        self.assertIn('"routed_weighted_expert_outputs"', source)
        self.assertIn('"first-decode-layer0-tail"', source)
        self.assertIn("FULL_ATTENTION_DECODE_COMPONENT_NAMES", source)
        self.assertIn("instrumented_unified_attention", source)
        self.assertIn('"first-decode-full-attention"', source)
        self.assertIn(
            '"two_diagnostic_layer3_unified_attention_sets_captured"',
            source,
        )
        self.assertIn(
            '"two_first_decode_layer3_unified_attention_sets_captured"',
            source,
        )
        self.assertIn(
            "triton_attention_module.unified_attention = state[", source
        )
        self.assertIn("FIRST_DECODE_LINEAR_OUTPUT_INDEX", source)
        self.assertIn("--diagnostic-output-index", source)
        self.assertIn("vl-generation-layer-diagnostic/v1", source)
        self.assertIn('"promotion_oracle": False', source)
        self.assertIn(
            '"two_diagnostic_routed_moe_stage_sets_captured"', source
        )
        self.assertIn("instrumented_causal_conv1d_update", source)
        self.assertIn("instrumented_packed_decode", source)
        self.assertIn("for case_id in CASE_ORDER", source)

    def test_native_observer_is_opt_in_and_captures_final_norm(self) -> None:
        header = RUNNER_HEADER.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        resident = RESIDENT.read_text(encoding="utf-8")
        http = HTTP.read_text(encoding="utf-8")
        self.assertIn("NativeDecodeLayerObserver", header)
        self.assertIn("NativeDecodeLinearLayer0Observer", header)
        self.assertIn("layer_observer = nullptr", header)
        self.assertIn('workspace.find("rmsnorm_final_output")', runner)
        self.assertIn("(*layer_observer)(40", runner)
        self.assertIn("decode_layer_observer_output_index", resident)
        self.assertIn('item.contains("reference_decode_boundary_dir")', http)
        self.assertIn("decode_boundary_comparisons.size() != 41", http)
        self.assertIn("reference_decode_linear_boundary_dir", http)
        self.assertIn("reference_decode_layer0_tail_boundary_dir", http)
        self.assertIn("decode_linear_layer0_observer", resident)
        self.assertIn("decode_layer0_tail_observer", resident)
        self.assertEqual(len(NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES), 13)
        self.assertEqual(len(LAYER0_TAIL_BOUNDARY_SPECS), 15)
        tail_contract = http.split(
            "kDecodeLayer0TailBoundaryContracts{{", 1
        )[1].split("}};", 1)[0]
        self.assertEqual(
            re.findall(r'\{"([^"]+)"', tail_contract),
            list(LAYER0_TAIL_BOUNDARY_SPECS),
        )


if __name__ == "__main__":
    unittest.main()
