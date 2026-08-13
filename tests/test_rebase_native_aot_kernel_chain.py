#!/usr/bin/env python3
"""Contract tests for deterministic AOT trace rebasing."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rebase-native-aot-kernel-chain.py"
SPEC = importlib.util.spec_from_file_location("rebase_native_aot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RebaseNativeAotKernelChainTest(unittest.TestCase):
    def test_rebase_preserves_only_base_pointer_identity(self) -> None:
        base = {
            "event": "triton_launch",
            "sequence": 17,
            "arguments": [
                {
                    "name": "input",
                    "kind": "tensor",
                    "data_ptr": 100,
                    "storage_offset": 4,
                },
                {"name": "block", "kind": "scalar", "value": 32},
            ],
        }
        replacement = {
            "event": "triton_launch",
            "sequence": 99,
            "arguments": [
                {
                    "name": "input",
                    "kind": "tensor",
                    "data_ptr": 900,
                    "storage_data_ptr": 800,
                    "storage_offset": 0,
                },
                {"name": "block", "kind": "scalar", "value": 64},
            ],
        }

        rebased = MODULE.rebase_launch(base, replacement)

        self.assertEqual(rebased["sequence"], 17)
        tensor = rebased["arguments"][0]
        self.assertEqual(tensor["data_ptr"], 100)
        self.assertEqual(tensor["storage_offset"], 4)
        self.assertNotIn("storage_data_ptr", tensor)
        self.assertEqual(rebased["arguments"][1]["value"], 64)

    def test_rebase_rejects_unmapped_replacement_tensor(self) -> None:
        base = {"arguments": []}
        replacement = {
            "arguments": [
                {"name": "input", "kind": "tensor", "data_ptr": 900}
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "no base pointer identity"):
            MODULE.rebase_launch(base, replacement)


if __name__ == "__main__":
    unittest.main()
