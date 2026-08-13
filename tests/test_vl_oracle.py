from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from aima_engine.vl_oracle import (
    ORACLE_SCHEMA,
    REQUIRED_BOUNDARIES,
    REQUIRED_ORACLE_CASES,
    validate_oracle_manifest,
    verify_raw_tensor,
    write_raw_tensor,
)
from aima_engine.vl_reference import seal_manifest


class VlOracleTest(unittest.TestCase):
    @staticmethod
    def _case(case_id: str, boundaries: set[str]) -> dict[str, object]:
        token_ids = [1, 2, 3]
        output_ids = [4]
        components = {name: {"schema": "unused"} for name in boundaries}
        if "mrope_positions" in components:
            components["mrope_positions"]["position_delta"] = -1
        if "full_vocabulary_logits" in components:
            components["full_vocabulary_logits"].update(
                {
                    "selected_rows": [0],
                    "teacher_forced_target_token_ids": [2],
                }
            )
        return {
            "case_id": case_id,
            "passed": True,
            "processor": {
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": (
                    "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4"
                ),
                "placeholders": {"image": [{"offset": 1, "length": 1}]},
                "tensors": {
                    "pixel_values": {"schema": "unused"},
                    "image_grid_thw": {"schema": "unused"},
                },
            },
            "boundaries": components,
            "generation": {
                "prompt_token_ids_sha256": (
                    "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4"
                ),
                "output_token_ids": output_ids,
                "output_token_ids_sha256": (
                    "46b1884167c4edd308bcf0c04163dd02d05c9742b35e86b57b5f7ed1b82f3850"
                ),
            },
        }

    def test_raw_tensor_round_trip_and_tamper_detection(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is only available in the target runtime")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            component = write_raw_tensor(
                root, "case/hidden", torch.arange(12, dtype=torch.float32).view(3, 4)
            )
            self.assertEqual(verify_raw_tensor(component, root), [])
            (root / component["path"]).write_bytes(b"tampered")
            errors = verify_raw_tensor(component, root)
            self.assertTrue(any("size mismatch" in error for error in errors))
            self.assertTrue(any("SHA-256 mismatch" in error for error in errors))

    def test_complete_oracle_contract(self) -> None:
        manifest = seal_manifest({
            "schema": ORACLE_SCHEMA,
            "complete": True,
            "qualified_for_native_boundary_comparison": True,
            "reference_manifest": {"sha256": "a" * 64},
            "cases": [
                self._case(case_id, REQUIRED_BOUNDARIES)
                for case_id in REQUIRED_ORACLE_CASES
            ],
        })
        self.assertEqual(validate_oracle_manifest(manifest), [])

    def test_missing_boundary_is_rejected(self) -> None:
        case_id = next(iter(REQUIRED_ORACLE_CASES))
        manifest = seal_manifest({
            "schema": ORACLE_SCHEMA,
            "complete": True,
            "qualified_for_native_boundary_comparison": True,
            "reference_manifest": {"sha256": "a" * 64},
            "cases": [self._case(case_id, set())],
        })
        errors = validate_oracle_manifest(manifest)
        self.assertTrue(any("missing oracle cases" in error for error in errors))
        self.assertTrue(any("missing boundaries" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
