from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from aima_engine.vl_generation_oracle import CASE_CONTRACTS, CASE_ORDER


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


if __name__ == "__main__":
    unittest.main()
