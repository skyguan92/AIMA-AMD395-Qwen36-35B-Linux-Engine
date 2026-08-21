from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate-native-vl-g2-qualification.py"
RESULT = ROOT / "benchmarks/results/vl-correctness-v0.1.0.json"
CANDIDATE_COMMIT = "a" * 40


def load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "aima_native_vl_g2_qualification", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the G2 qualification generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def full_language_case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "complete": True,
        "prompt_tokens": 10,
        "production_operation_shape": True,
        "repeat_deterministic": True,
        "final_norm": {
            "passed": True,
            "elements": 20,
            "exact_elements": 20,
        },
        "full_vocabulary_logits": {
            "rows": [
                {
                    "passed": True,
                    "top1_match": True,
                    "kl_divergence": 0.0,
                    "tensor": {"elements": 30, "exact_elements": 30},
                }
            ]
        },
    }


class NativeVlG2QualificationLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator_module()

    def test_empty_boolean_collections_cannot_pass(self) -> None:
        self.assertFalse(self.generator.all_true({}))
        self.assertFalse(self.generator.all_true([]))
        self.assertTrue(self.generator.all_true({"outer": {"inner": True}}))

    def test_all_language_probe_sources_are_content_bound(self) -> None:
        self.assertEqual(
            {path.name for path in self.generator.QUALIFICATION_FILES},
            {
                "vl_language_layer0_oracle_probe.hip.cpp",
                "vl_language_layer3_mrope_oracle_probe.hip.cpp",
                "vl_language_layer3_composed_oracle_probe.hip.cpp",
                "build-native-vl-language-layer0-probe.sh",
                "build-native-vl-language-layer3-mrope-probe.sh",
                "build-native-vl-language-layer3-composed-probe.sh",
                "generate-native-vl-g2-qualification.py",
            },
        )

    def test_vision_pipeline_requires_all_named_depth_boundaries(self) -> None:
        payload = {
            "schema": (
                "aima-amd395-qwen36/"
                "native-vision-pipeline-qualification/v2"
            ),
            "complete": True,
            "qualified": True,
            "source": {"commit": CANDIDATE_COMMIT},
            "candidate_binary_sha256": "b" * 64,
            "cases": [
                {
                    "case_id": case_id,
                    "boundaries": [
                        {"name": name}
                        for name in self.generator.VISION_BOUNDARY_ORDER
                    ],
                }
                for case_id in self.generator.CASE_ORDER
            ],
            "checks": {"aggregate": True},
            "decision": {
                "boundary_comparison_count": 20,
                "total_boundary_elements": 6_856_704,
                "exact_boundary_elements": 6_856_704,
                "all_boundaries_bit_exact": True,
                "total_visual_output_elements": 884_736,
                "all_repeats_deterministic": True,
                "all_27_blocks_executed": True,
                "full_visual_pipeline_qualified": True,
            },
        }
        checks = self.generator.validate_vision_pipeline(
            payload, CANDIDATE_COMMIT, "b" * 64
        )
        self.assertTrue(all(checks.values()))

        payload["cases"][0]["boundaries"].pop(1)
        failed_checks = self.generator.validate_vision_pipeline(
            payload, CANDIDATE_COMMIT, "b" * 64
        )
        self.assertFalse(failed_checks["required_boundary_order_exact"])

        payload["cases"][0]["boundaries"].insert(
            1, {"name": "vision_block_13"}
        )
        payload["schema"] = (
            "aima-amd395-qwen36/native-vision-pipeline-qualification/v1"
        )
        stale_checks = self.generator.validate_vision_pipeline(
            payload, CANDIDATE_COMMIT, "b" * 64
        )
        self.assertFalse(stale_checks["schema_exact"])

    def test_full_language_requires_exact_case_order_and_rows(self) -> None:
        payload = {
            "schema": (
                "aima-amd395-qwen36/"
                "native-vl-language-full-qualification-run/v1"
            ),
            "complete": True,
            "source_commit": CANDIDATE_COMMIT,
            "single_resident_weight_load": True,
            "cases": [
                full_language_case(case_id)
                for case_id in self.generator.CASE_ORDER
            ],
        }
        summary, checks = self.generator.summarize_full_language(
            payload, CANDIDATE_COMMIT, self.generator.CASE_ORDER
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(summary["case_count"], 5)
        self.assertEqual(summary["selected_logits_rows"], 5)
        self.assertEqual(summary["selected_logits_elements"], 150)
        self.assertEqual(summary["selected_logits_exact_elements"], 150)

        payload["cases"] = list(reversed(payload["cases"]))
        _, reordered_checks = self.generator.summarize_full_language(
            payload, CANDIDATE_COMMIT, self.generator.CASE_ORDER
        )
        self.assertFalse(reordered_checks["case_order_exact"])

    def test_deep_language_requires_all_layers_and_router_rows(self) -> None:
        payload = {
            "complete": True,
            "source_commit": CANDIDATE_COMMIT,
            "case_selector": "multi_video",
            "cases": [
                {
                    "case_id": "multi_video",
                    "complete": True,
                    "layer_diagnostics": {
                        "provided": True,
                        "comparisons": [{"passed": True} for _ in range(111)],
                        "router_expert_sets": [
                            {"rows": 131, "exact_rows": 131}
                            for _ in range(40)
                        ],
                        "all_router_expert_sets_exact": True,
                    },
                }
            ],
        }
        summary, checks = self.generator.summarize_deep_language(
            payload, CANDIDATE_COMMIT
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(summary["tensor_comparisons"], 111)
        self.assertEqual(summary["router_layer_sets"], 40)
        self.assertEqual(summary["router_rows"], 5240)
        self.assertEqual(summary["exact_router_rows"], 5240)

        payload["cases"][0]["layer_diagnostics"]["router_expert_sets"][0][
            "exact_rows"
        ] = 130
        _, failed_checks = self.generator.summarize_deep_language(
            payload, CANDIDATE_COMMIT
        )
        self.assertFalse(failed_checks["all_router_rows_exact"])

    def test_layer_boundaries_require_exact_totals(self) -> None:
        primary_comparisons = [
            {
                "label": "input_norm_full_sequence" if index == 0 else f"d{index}",
                "passed": True,
                "elements": 20,
                "exact_elements": 20,
                "finite_elements": 20,
                "relative_l2_error": 0.0,
                "cosine_similarity": 1.0,
            }
            for index in range(24)
        ]
        seeded_comparisons = [
            {
                "label": f"s{index}",
                "passed": True,
                "elements": 20,
                "exact_elements": 20,
                "finite_elements": 20,
                "relative_l2_error": 0.0,
                "cosine_similarity": 1.0,
            }
            for index in range(9)
        ]
        layer0_cases = [
            {
                "case_id": case_id,
                "prompt_tokens": 10,
                "elements": 20,
                "finite_elements": 20,
                "repeat_deterministic": True,
                "diagnostic_complete": True,
                "diagnostic_comparisons": primary_comparisons,
                "seeded_moe_diagnostic_complete": True,
                "seeded_moe_diagnostic_comparisons": seeded_comparisons,
                "router_expert_set_rows": 10,
                "router_expert_set_rows_exact": 10,
                "seeded_router_expert_set_rows": 10,
                "seeded_router_expert_set_rows_exact": 10,
            }
            for case_id in self.generator.CASE_ORDER
        ]
        layer0 = {
            "schema": (
                "aima-amd395-qwen36/"
                "native-vl-language-layer0-qualification-run/v1"
            ),
            "complete": True,
            "source_commit": CANDIDATE_COMMIT,
            "single_resident_weight_load": True,
            "all_bit_exact": True,
            "total_elements": 100,
            "total_exact_elements": 100,
            "cases": layer0_cases,
        }
        summary, checks = self.generator.summarize_layer0(
            layer0, CANDIDATE_COMMIT
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(summary["diagnostic_comparisons"], 165)

        layer3 = {
            "schema": (
                "aima-amd395-qwen36/"
                "native-vl-language-layer3-mrope-qualification-run/v1"
            ),
            "complete": True,
            "source_commit": CANDIDATE_COMMIT,
            "capture_source_commit": "b" * 40,
            "capture_manifest_sha256": "c" * 64,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            "rotary_dimension": 64,
            "all_bit_exact": True,
            "total_elements": 200,
            "total_exact_elements": 200,
            "runtime_python": False,
            "runtime_numpy": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
            "cases": [
                {"case_id": case_id}
                for case_id in self.generator.CASE_ORDER
            ],
        }
        _, layer3_checks = self.generator.summarize_layer3(
            layer3, CANDIDATE_COMMIT
        )
        self.assertTrue(all(layer3_checks.values()))


class NativeVlG2QualificationArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = RESULT.read_bytes()
        cls.result = json.loads(cls.payload)

    def test_result_is_exact_and_reproducible(self) -> None:
        result = self.result
        qualification = result["qualification"]
        probes = qualification["probe_binaries"]
        inputs = result["inputs"]
        command = [
            "python3",
            str(SCRIPT),
            "--candidate-source-commit",
            result["candidate"]["source_commit"],
            "--candidate-binary-sha256",
            result["candidate"]["binary_sha256"],
            "--qualification-source-commit",
            qualification["source_commit"],
            "--layer0-probe-sha256",
            probes["layer0"]["sha256"],
            "--layer0-probe-bytes",
            str(probes["layer0"]["bytes"]),
            "--layer3-probe-sha256",
            probes["layer3"]["sha256"],
            "--layer3-probe-bytes",
            str(probes["layer3"]["bytes"]),
            "--full-language-probe-sha256",
            probes["full_language"]["sha256"],
            "--full-language-probe-bytes",
            str(probes["full_language"]["bytes"]),
            "--recorded-on",
            result["recorded_on"],
        ]
        for name in (
            "envelope",
            "vision_pipeline",
            "task_quality",
            "generation",
            "error_limits",
            "layer0",
            "layer3_a",
            "layer3_b",
            "full_private",
            "full_http",
            "full_http_deep",
        ):
            command.extend(
                [f"--{name.replace('_', '-')}", inputs[name]["path"]]
            )
        command.extend(["--output", str(RESULT), "--check"])
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("native VL G2 qualification: PASS", completed.stdout)

    def test_result_and_qualification_inputs_are_hash_bound(self) -> None:
        digest = hashlib.sha256(self.payload).hexdigest()
        sidecar = RESULT.with_name(RESULT.name + ".sha256")
        self.assertEqual(
            sidecar.read_text(encoding="utf-8"),
            f"{digest}  {RESULT.name}\n",
        )
        for component in self.result["qualification"]["files"]:
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                component["sha256"],
            )

    def test_all_g2_gates_pass_without_threshold_widening(self) -> None:
        self.assertTrue(self.result["complete"])
        self.assertTrue(self.result["qualified"])
        self.assertTrue(self.result["decision"]["g2_passed"])
        self.assertTrue(self.result["decision"]["exact_binary_bound"])
        self.assertTrue(self.result["decision"]["no_threshold_widening"])
        self.assertTrue(
            all(
                all(gate["checks"].values())
                for gate in self.result["gates"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
