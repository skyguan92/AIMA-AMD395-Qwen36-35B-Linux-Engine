from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_generation_oracle import (
    CASE_CONTRACTS,
    CASE_ORDER,
    GENERATION_ORACLE_SCHEMA,
    validate_generation_oracle_manifest,
)
from aima_engine.vl_reference import seal_manifest


ROOT = Path(__file__).resolve().parents[1]
CURRENT_HEAD_RESULT = (
    ROOT / "benchmarks/results/native-vl-current-head-serving-v0.1.0.json"
)


class VlGenerationOracleContractTest(unittest.TestCase):
    def test_frozen_divergence_contract_is_high_signal(self) -> None:
        self.assertEqual(
            tuple(CASE_CONTRACTS),
            CASE_ORDER,
        )
        forced = CASE_CONTRACTS["tool_forced_image"]
        auto = CASE_CONTRACTS["tool_auto_image"]
        self.assertEqual(forced["divergence_output_index"], 14)
        self.assertEqual(auto["divergence_output_index"], 93)
        self.assertNotEqual(
            forced["reference_token_id"], forced["previous_native_token_id"]
        )
        self.assertNotEqual(
            auto["reference_token_id"], auto["previous_native_token_id"]
        )

    def test_validator_rejects_missing_raw_logits(self) -> None:
        cases = []
        for case_id in CASE_ORDER:
            contract = CASE_CONTRACTS[case_id]
            ids = [0] * contract["completion_tokens"]
            ids[contract["divergence_output_index"]] = contract[
                "reference_token_id"
            ]
            # Deliberately use the frozen digest so this synthetic record gets
            # as far as the raw-component validation.
            cases.append(
                {
                    "case_id": case_id,
                    "passed": True,
                    "divergence_output_index": contract[
                        "divergence_output_index"
                    ],
                    "generation": {
                        "output_token_ids": ids,
                        "output_token_ids_sha256": contract[
                            "output_token_ids_sha256"
                        ],
                        "completion_tokens": len(ids),
                    },
                    "reference_logits": {
                        "selected_token_id": contract["reference_token_id"],
                        "raw_top_tokens": [{"rank": 1, "token_id": 0}],
                    },
                }
            )
        payload = seal_manifest(
            {
                "schema": GENERATION_ORACLE_SCHEMA,
                "complete": True,
                "qualified_for_native_generation_comparison": True,
                "cases": cases,
                "decision": {
                    "two_tool_generations_exact": True,
                    "two_prompt_vectors_exact": True,
                    "two_divergence_logits_captured": True,
                    "g1_passed": False,
                    "g2_passed": False,
                    "g3_passed": False,
                    "g4_passed": False,
                    "g5_passed": False,
                },
            }
        )
        errors = validate_generation_oracle_manifest(payload)
        self.assertTrue(any("component is missing" in error for error in errors))

    def test_current_head_serving_snapshot_is_sealed(self) -> None:
        payload = CURRENT_HEAD_RESULT.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        sidecar = CURRENT_HEAD_RESULT.with_name(CURRENT_HEAD_RESULT.name + ".sha256")
        self.assertEqual(
            sidecar.read_text(encoding="utf-8"),
            f"{digest}  {CURRENT_HEAD_RESULT.name}\n",
        )
        result = json.loads(payload)
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertEqual(
            result["source"]["commit"],
            "cbd9868d727c66cf03976466db2b3ab42016b3cf",
        )
        self.assertEqual(
            result["binary"]["sha256"],
            "63d4b5fd3e51ba5d60421b81c74a9ce5fb32d08e062cc07cfff74b5b0698e21e",
        )
        self.assertEqual(len(result["oracle_cases"]), 5)
        self.assertTrue(all(case["passed"] for case in result["oracle_cases"]))
        self.assertTrue(all(result["cache_correctness"]["checks"].values()))
        self.assertEqual(result["raw"]["stderr"]["bytes"], 0)
        for gate in ("g1_passed", "g2_passed", "g3_passed", "g4_passed", "g5_passed"):
            self.assertFalse(result["decision"][gate])


if __name__ == "__main__":
    unittest.main()
