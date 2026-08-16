from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from aima_engine.vl_generation_oracle import CASE_CONTRACTS, CASE_ORDER
from aima_engine.vl_generation_layer_oracle import (
    BOUNDARY_NAMES,
    LAYER0_TAIL_BOUNDARY_SPECS,
    NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES,
)
from aima_engine.vl_prefill_state_oracle import STATE_COMPONENT_NAMES


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify-native-vl-generation.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "native_vl_generation_qualification_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeVlGenerationQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_materialize_request_verifies_fixture_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "image.png"
            fixture.write_bytes(b"fixed")
            identity = {
                "fixture": fixture.name,
                "transport": "local",
                "bytes": 5,
                "sha256": __import__("hashlib").sha256(b"fixed").hexdigest(),
            }
            request = {"image_url": {"url": identity}}
            actual = self.module.materialize_request(request, root)
            self.assertEqual(
                actual["image_url"]["url"], fixture.resolve().as_uri()
            )
            fixture.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "fixture changed"):
                self.module.materialize_request(request, root)

    def test_probe_cases_bind_target_layer0_tail_oracles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixtures"
            fixture_root.mkdir()
            fixture = fixture_root / "image.png"
            fixture.write_bytes(b"fixed")
            fixture_identity = {
                "fixture": fixture.name,
                "transport": "local",
                "bytes": 5,
                "sha256": __import__("hashlib").sha256(b"fixed").hexdigest(),
            }
            oracle_cases = []
            layer_cases = []
            for case_id in CASE_ORDER:
                contract = CASE_CONTRACTS[case_id]
                target = contract["divergence_output_index"]
                oracle_cases.append(
                    {
                        "case_id": case_id,
                        "divergence_output_index": target,
                        "generation": {
                            "output_token_ids": [0] * target
                            + [contract["reference_token_id"]]
                        },
                        "request": {"image_url": {"url": fixture_identity}},
                        "reference_logits": {
                            "component": {"path": f"{case_id}.bin"}
                        },
                    }
                )
                layer_cases.append(
                    {
                        "case_id": case_id,
                        "target_output_index": target,
                        "target_logits_component": {
                            "path": f"{case_id}/target-logits.bin"
                        },
                        "linear_attention": {
                            "metadata": {"layer_index": 0}
                        },
                        "full_attention": {
                            "metadata": {
                                "layer_index": (
                                    11
                                    if case_id == "tool_forced_image"
                                    else 3
                                )
                            }
                        },
                    }
                )
            layer_root = root / "layers"
            cases = self.module.build_probe_cases(
                {"cases": oracle_cases},
                root,
                fixture_root,
                {"cases": layer_cases},
                layer_root,
            )["cases"]
            self.assertEqual(
                [case["reference_logits_output_index"] for case in cases],
                [
                    CASE_CONTRACTS[case_id]["divergence_output_index"]
                    for case_id in CASE_ORDER
                ],
            )
            self.assertEqual(
                [case["reference_decode_output_index"] for case in cases],
                [
                    CASE_CONTRACTS[case_id]["divergence_output_index"]
                    for case_id in CASE_ORDER
                ],
            )
            self.assertEqual(
                [
                    case["reference_decode_linear_layer_index"]
                    for case in cases
                ],
                [0, 0],
            )
            self.assertEqual(
                [Path(case["reference_logits"]) for case in cases],
                [
                    (layer_root / case_id / "target-logits.bin").resolve()
                    for case_id in CASE_ORDER
                ],
            )
            self.assertEqual(
                [
                    Path(case["reference_decode_layer0_tail_boundary_dir"])
                    for case in cases
                ],
                [
                    (layer_root / case_id / "layer0-tail").resolve()
                    for case_id in CASE_ORDER
                ],
            )
            self.assertEqual(
                [
                    Path(case["reference_decode_full_attention_dir"])
                    for case in cases
                ],
                [
                    (layer_root / case_id / "full-attention").resolve()
                    for case_id in CASE_ORDER
                ],
            )
            self.assertEqual(
                [
                    case["reference_decode_full_attention_layer_index"]
                    for case in cases
                ],
                [11, 3],
            )

            layer_cases[0]["target_output_index"] += 1
            with self.assertRaisesRegex(
                RuntimeError, "layer oracle output index differs"
            ):
                self.module.build_probe_cases(
                    {"cases": oracle_cases},
                    root,
                    fixture_root,
                    {"cases": layer_cases},
                    layer_root,
                )

    def test_prefix_divergence_diagnostic_is_explicit_and_non_default(self) -> None:
        source = (ROOT / "native/src/native_http_server.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("diagnostic_allow_prefix_divergence", source)
        self.assertIn("reference_logits_output_index", source)
        self.assertIn("reference_decode_output_index", source)
        self.assertIn("reference_decode_linear_layer_index", source)
        self.assertIn("decode_linear_observer_layer_index", source)
        self.assertIn("reference logits output index is misaligned", source)
        self.assertIn("decode reference output index is misaligned", source)
        self.assertIn(
            'item.value("diagnostic_allow_prefix_divergence", false)', source
        )
        self.assertIn("output_index=", source)
        self.assertIn("reference_decode_layer0_tail_boundary_dir", source)
        self.assertIn("reference_decode_full_attention_dir", source)
        self.assertIn("decode_layer0_tail_boundaries", source)

    def test_qualification_binds_decode_norm_owners(self) -> None:
        source = (ROOT / "scripts/qualify-native-vl-generation.py").read_text(
            encoding="utf-8"
        )
        for relative in (
            "native/include/aima/native_pointwise.h",
            "native/src/native_full_layer.hip.cpp",
            "native/src/native_linear_layer.hip.cpp",
            "native/src/native_pointwise.hip.cpp",
        ):
            self.assertIn(f'ROOT / "{relative}"', source)

    def test_checks_separate_setup_from_current_native_mismatch(self) -> None:
        oracle_cases = []
        probe_cases = []
        for case_id in CASE_ORDER:
            contract = CASE_CONTRACTS[case_id]
            prompt_sha = case_id.ljust(64, "0")[:64]
            component_sha = case_id.ljust(64, "1")[:64]
            oracle_cases.append(
                {
                    "case_id": case_id,
                    "prompt_token_ids_sha256": prompt_sha,
                    "reference_logits": {"component": {"sha256": component_sha}},
                }
            )
            probe_cases.append(
                {
                    "case_id": case_id,
                    "prefix_exact": True,
                    "selected_native_token_id": contract[
                        "previous_native_token_id"
                    ],
                    "native_top1_exact": False,
                    "request_metrics": {
                        "prompt_token_ids_sha256": prompt_sha,
                        "vl": {"enabled": True},
                        "mrope": {"enabled": True},
                    },
                    "reference_logits": {
                        "expected_sha256": component_sha,
                        "reference_top1_token_id": contract[
                            "reference_token_id"
                        ],
                        "elements": self.module.MODEL_VOCABULARY_SIZE,
                        "finite_elements": self.module.MODEL_VOCABULARY_SIZE,
                        "top1_match": False,
                        "kl_divergence": 0.001,
                    },
                }
            )
        checks = self.module.qualification_checks(
            {
                "schema": self.module.PROBE_SCHEMA,
                "complete": True,
                "qualified_for_attribution": True,
                "model_loads": 1,
                "cases": probe_cases,
            },
            {"cases": oracle_cases},
        )
        self.assertTrue(checks["probe_attribution_qualified"])
        self.assertTrue(checks["tool_auto_image_prefix_exact"])
        self.assertTrue(checks["tool_forced_image_reference_row_bound"])
        self.assertFalse(checks["tool_auto_image_native_top1_exact"])
        self.assertFalse(checks["tool_forced_image_selected_token_exact"])
        self.assertTrue(checks["tool_auto_image_kld_under_0_005"])

    def test_layer_checks_bind_every_finite_decode_boundary(self) -> None:
        oracle_cases = []
        layer_cases = []
        probe_cases = []
        for case_id in CASE_ORDER:
            contract = CASE_CONTRACTS[case_id]
            prompt_sha = case_id.ljust(64, "0")[:64]
            logits_sha = case_id.ljust(64, "1")[:64]
            components = {
                name: {"sha256": f"{index:064x}"}
                for index, name in enumerate(BOUNDARY_NAMES, start=1)
            }
            boundaries = [
                {
                    "label": name,
                    "elements": 2_048,
                    "finite_elements": 2_048,
                    "expected_sha256": components[name]["sha256"],
                }
                for name in BOUNDARY_NAMES
            ]
            linear_components = {
                name: {"sha256": f"{index + 100:064x}"}
                for index, name in enumerate(
                    NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES, start=1
                )
            }
            linear_boundaries = [
                {
                    "label": name,
                    "elements": 1,
                    "finite_elements": 1,
                    "expected_sha256": linear_components[name]["sha256"],
                }
                for name in NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES
            ]
            tail_components = {
                name: {"sha256": f"{index + 200:064x}"}
                for index, name in enumerate(
                    LAYER0_TAIL_BOUNDARY_SPECS, start=1
                )
            }
            tail_boundaries = [
                {
                    "label": name,
                    "elements": 1,
                    "finite_elements": 1,
                    "expected_sha256": tail_components[name]["sha256"],
                }
                for name in LAYER0_TAIL_BOUNDARY_SPECS
            ]
            oracle_cases.append(
                {
                    "case_id": case_id,
                    "prompt_token_ids_sha256": prompt_sha,
                    "reference_logits": {"component": {"sha256": logits_sha}},
                }
            )
            layer_cases.append(
                {
                    "case_id": case_id,
                    "components": components,
                    "linear_attention": {"components": linear_components},
                    "layer0_tail": {"components": tail_components},
                }
            )
            probe_cases.append(
                {
                    "case_id": case_id,
                    "prefix_exact": True,
                    "selected_native_token_id": contract["reference_token_id"],
                    "native_top1_exact": True,
                    "decode_boundaries_complete": True,
                    "decode_boundaries_finite": True,
                    "decode_boundaries": boundaries,
                    "decode_linear_boundaries_complete": True,
                    "decode_linear_boundaries_finite": True,
                    "decode_linear_boundaries": linear_boundaries,
                    "decode_layer0_tail_boundaries_complete": True,
                    "decode_layer0_tail_boundaries_finite": True,
                    "decode_layer0_tail_boundaries": tail_boundaries,
                    "request_metrics": {
                        "prompt_token_ids_sha256": prompt_sha,
                        "vl": {"enabled": True},
                        "mrope": {"enabled": True},
                    },
                    "reference_logits": {
                        "expected_sha256": logits_sha,
                        "reference_top1_token_id": contract[
                            "reference_token_id"
                        ],
                        "elements": self.module.MODEL_VOCABULARY_SIZE,
                        "finite_elements": self.module.MODEL_VOCABULARY_SIZE,
                        "top1_match": True,
                        "kl_divergence": 0.0,
                    },
                }
            )
        checks = self.module.qualification_checks(
            {
                "schema": self.module.PROBE_SCHEMA,
                "complete": True,
                "qualified_for_attribution": True,
                "model_loads": 1,
                "cases": probe_cases,
            },
            {"cases": oracle_cases},
            {"cases": layer_cases},
        )
        for case_id in CASE_ORDER:
            self.assertTrue(checks[f"{case_id}_decode_boundaries_complete"])
            self.assertTrue(checks[f"{case_id}_decode_boundaries_finite"])
            self.assertTrue(checks[f"{case_id}_decode_boundary_rows_bound"])
            self.assertTrue(
                checks[f"{case_id}_decode_linear_boundary_rows_bound"]
            )
            self.assertTrue(
                checks[f"{case_id}_decode_layer0_tail_boundaries_complete"]
            )
            self.assertTrue(
                checks[f"{case_id}_decode_layer0_tail_boundaries_finite"]
            )
            self.assertTrue(
                checks[f"{case_id}_decode_layer0_tail_boundary_rows_bound"]
            )

    def test_prefill_state_checks_bind_all_linear_layer_states(self) -> None:
        oracle_cases = []
        state_cases = []
        probe_cases = []
        for case_id in CASE_ORDER:
            contract = CASE_CONTRACTS[case_id]
            prompt_sha = case_id.ljust(64, "0")[:64]
            logits_sha = case_id.ljust(64, "1")[:64]
            components = {
                name: {"sha256": f"{index:064x}"}
                for index, name in enumerate(STATE_COMPONENT_NAMES, start=1)
            }
            states = [
                {
                    "label": name,
                    "elements": (
                        32 * 128 * 128
                        if name.endswith("_recurrent_state")
                        else 8_192 * 3
                    ),
                    "finite_elements": (
                        32 * 128 * 128
                        if name.endswith("_recurrent_state")
                        else 8_192 * 3
                    ),
                    "expected_sha256": components[name]["sha256"],
                }
                for name in STATE_COMPONENT_NAMES
            ]
            oracle_cases.append(
                {
                    "case_id": case_id,
                    "prompt_token_ids_sha256": prompt_sha,
                    "reference_logits": {"component": {"sha256": logits_sha}},
                }
            )
            state_cases.append({"case_id": case_id, "components": components})
            probe_cases.append(
                {
                    "case_id": case_id,
                    "prefix_exact": True,
                    "selected_native_token_id": contract["reference_token_id"],
                    "native_top1_exact": True,
                    "prefill_states_complete": True,
                    "prefill_states_finite": True,
                    "prefill_states": states,
                    "request_metrics": {
                        "prompt_token_ids_sha256": prompt_sha,
                        "vl": {"enabled": True},
                        "mrope": {"enabled": True},
                    },
                    "reference_logits": {
                        "expected_sha256": logits_sha,
                        "reference_top1_token_id": contract[
                            "reference_token_id"
                        ],
                        "elements": self.module.MODEL_VOCABULARY_SIZE,
                        "finite_elements": self.module.MODEL_VOCABULARY_SIZE,
                        "top1_match": True,
                        "kl_divergence": 0.0,
                    },
                }
            )
        checks = self.module.qualification_checks(
            {
                "schema": self.module.PROBE_SCHEMA,
                "complete": True,
                "qualified_for_attribution": True,
                "model_loads": 1,
                "cases": probe_cases,
            },
            {"cases": oracle_cases},
            None,
            {"cases": state_cases},
        )
        for case_id in CASE_ORDER:
            self.assertTrue(checks[f"{case_id}_prefill_states_complete"])
            self.assertTrue(checks[f"{case_id}_prefill_states_finite"])
            self.assertTrue(checks[f"{case_id}_prefill_state_rows_bound"])


if __name__ == "__main__":
    unittest.main()
