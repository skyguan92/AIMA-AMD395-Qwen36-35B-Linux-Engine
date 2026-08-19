from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize-vl-performance-matrix-pair.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_vl_performance_matrix_pair", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def cache_cell(sequence: str, digest: str = "c" * 64) -> dict:
    return {
        "cache_sequence": sequence,
        "contract": {
            "template_sha256": "a" * 64,
            "prompt_nonce_sha256": "b" * 64,
        },
        "response_audit": {
            role: {
                "content_sha256": digest,
                "content_bytes": 1,
                "semantic_chunks": ["1"],
            }
            for role in ("reference", "candidate")
        },
    }


class VlPerformanceMatrixPairSummaryTest(unittest.TestCase):
    def test_a_b_a_contract_requires_stable_a_request_and_output(self) -> None:
        cells = [cache_cell(name) for name in ("A1", "A2", "B", "A3")]
        self.assertTrue(
            summary.cache_a_b_a_contract_exact(cells, "candidate")
        )
        cells[-1]["response_audit"]["candidate"]["content_sha256"] = "d" * 64
        self.assertFalse(
            summary.cache_a_b_a_contract_exact(cells, "candidate")
        )

    def test_a_b_a_contract_rejects_missing_or_duplicate_sequence(self) -> None:
        cells = [cache_cell(name) for name in ("A1", "A2", "B", "B")]
        self.assertFalse(
            summary.cache_a_b_a_contract_exact(cells, "reference")
        )


if __name__ == "__main__":
    unittest.main()
