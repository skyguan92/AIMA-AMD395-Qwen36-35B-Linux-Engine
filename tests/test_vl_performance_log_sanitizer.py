#!/usr/bin/env python3
"""Tests for public G4 reference failure log sanitization."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest

from aima_engine.public_hygiene import scan_bytes


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sanitize-vl-performance-reference-log.py"
SPEC = importlib.util.spec_from_file_location("vl_log_sanitizer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sanitizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sanitizer)


class VlPerformanceLogSanitizerTest(unittest.TestCase):
    def test_redacts_private_deployment_paths_and_addresses(self) -> None:
        raw = (
            b"/ho" + b"me/runner/venv /data/models/private/model "
            b"/tmp/aima-native-vl-final-abc/run 192." + b"168.1.20 \r\n"
        )
        published, metadata = sanitizer.sanitize(raw)
        self.assertEqual(metadata["source_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(scan_bytes("published.txt", published.encode()), [])
        for marker in (
            "/ho" + "me/",
            "/data/models/",
            "/tmp/aima-native",
            "192." + "168.",
        ):
            self.assertNotIn(marker, published)
        self.assertIn("${AIMA_REMOTE_HOME}", published)
        self.assertIn("${AIMA_MODEL_DIR}", published)
        self.assertIn("${AIMA_BENCH_ROOT}", published)
        self.assertIn("${AIMA_PRIVATE_IPV4}", published)
        self.assertNotIn("\r", published)
        self.assertNotIn(" \n", published)
        self.assertEqual(
            metadata["normalization_counts"],
            {"carriage_returns": 1, "trailing_whitespace_lines": 1},
        )

    def test_metadata_header_is_deterministic(self) -> None:
        raw = b"HIP error: unspecified launch failure\nError code 719\n"
        first, _ = sanitizer.sanitize(raw)
        second, _ = sanitizer.sanitize(raw)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith(sanitizer.PREFIX))


if __name__ == "__main__":
    unittest.main()
