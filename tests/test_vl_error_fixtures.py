from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from aima_engine.vl_reference import verify_manifest_integrity


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate-vl-error-fixtures.py"
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-error-v0.1.0"
FIXTURE = FIXTURE_ROOT / "video-12f-0.002fps-192x128.avi"
MANIFEST = FIXTURE_ROOT / "fixtures-manifest.json"


class VlErrorFixturesTest(unittest.TestCase):
    def test_committed_fixture_is_reproducible_and_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aima-vl-error-fixtures-") as raw:
            generated_root = Path(raw)
            subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--output",
                    str(generated_root),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                (generated_root / FIXTURE.name).read_bytes(),
                FIXTURE.read_bytes(),
            )
            self.assertEqual(
                (generated_root / MANIFEST.name).read_bytes(),
                MANIFEST.read_bytes(),
            )

        payload = MANIFEST.read_bytes()
        manifest = json.loads(payload)
        self.assertEqual(
            manifest["schema"], "aima-amd395-qwen36/vl-error-fixtures/v1"
        )
        self.assertTrue(manifest["complete"])
        self.assertEqual(verify_manifest_integrity(manifest), [])
        self.assertEqual(
            MANIFEST.with_name(MANIFEST.name + ".sha256").read_text(
                encoding="utf-8"
            ),
            f"{hashlib.sha256(payload).hexdigest()}  {MANIFEST.name}\n",
        )
        fixture = manifest["fixtures"][0]
        self.assertEqual(fixture["bytes"], FIXTURE.stat().st_size)
        self.assertEqual(
            fixture["sha256"], hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        )

    def test_only_avi_timing_headers_change(self) -> None:
        source = (
            ROOT
            / "benchmarks/fixtures/vl-capability-v0.1.0"
            / "video-12f-6fps-192x128.avi"
        ).read_bytes()
        derived = FIXTURE.read_bytes()
        self.assertEqual(len(source), len(derived))
        changed = [
            index
            for index, (before, after) in enumerate(zip(source, derived))
            if before != after
        ]
        self.assertEqual(changed, [32, 33, 34, 35, 128, 129, 132])
        self.assertEqual(struct.unpack_from("<I", derived, 32)[0], 500_000_000)
        self.assertEqual(struct.unpack_from("<I", derived, 128)[0], 500)
        self.assertEqual(struct.unpack_from("<I", derived, 132)[0], 1)


if __name__ == "__main__":
    unittest.main()
