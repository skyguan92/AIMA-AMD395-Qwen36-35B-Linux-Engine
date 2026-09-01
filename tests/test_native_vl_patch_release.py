from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate-native-vl-patch-product-qualification.py"
HTTP_QUALIFIER = ROOT / "scripts/qualify-native-http-control-plane.py"
G5_GENERATOR = ROOT / "scripts/generate-native-vl-patch-g5-qualification.py"
CONTRACT = ROOT / "native/product-contract-v1.5.1-native-vl.5.json"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "native_vl_patch_product_qualification_test", GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeVlPatchReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator()
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_patch_contract_binds_exact_candidate_and_inheritance_limit(self) -> None:
        contract = self.contract
        self.assertEqual(contract["release"], "1.5.1-native-vl.5")
        self.assertEqual(contract["release_tag"], "v1.5.1-native-vl.5")
        self.assertEqual(
            contract["candidate"]["native_source_commit"],
            self.generator.NATIVE_SOURCE_COMMIT,
        )
        self.assertEqual(
            contract["candidate"]["native_engine_sha256"],
            self.generator.ENGINE_SHA256,
        )
        self.assertEqual(
            set(contract["patch_scope"]["allowed_runtime_paths"]),
            self.generator.ALLOWED_RUNTIME_DELTA,
        )
        self.assertIn("never", contract["patch_scope"]["inheritance_rule"])
        self.assertFalse(contract["target"]["model_weights_in_archive"])
        self.assertEqual(
            contract["target"]["maximum_resident_memory_bytes"], 96 * 1024**3
        )

    def test_runtime_delta_is_exactly_the_declared_cpu_control_patch(self) -> None:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "diff",
                "--name-only",
                (
                    f"{self.generator.BASELINE_NATIVE_SOURCE_COMMIT}.."
                    f"{self.generator.NATIVE_SOURCE_COMMIT}"
                ),
                "--",
                *self.generator.RUNTIME_PATHS,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            set(completed.stdout.splitlines()),
            self.generator.ALLOWED_RUNTIME_DELTA,
        )
        self.assertFalse(
            any(
                path.startswith(("native/aot/", "native/generated/"))
                for path in completed.stdout.splitlines()
            )
        )

    def test_patch_release_tooling_is_checked_and_does_not_publish_hostnames(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn(GENERATOR.name, makefile)
        self.assertIn(G5_GENERATOR.name, makefile)
        self.assertIn(HTTP_QUALIFIER.name, makefile)
        http_source = HTTP_QUALIFIER.read_text(encoding="utf-8")
        chat_source = (
            ROOT / "scripts/qualify-native-chat-protocol.py"
        ).read_text(encoding="utf-8")
        self.assertIn("fingerprint_sha256", http_source)
        self.assertIn("fingerprint_sha256", chat_source)
        self.assertNotIn('"hostname":', http_source)
        self.assertNotIn('"hostname":', chat_source)
        self.assertIn("zero_timeout_incomplete_read_is_interruptible", http_source)
        self.assertIn("two_chats_execute_serially_without_rejection", http_source)
        g5_source = G5_GENERATOR.read_text(encoding="utf-8")
        self.assertIn("inherited_two_host_portable_userspace", g5_source)
        self.assertIn("not exact .5 measurements", g5_source)


if __name__ == "__main__":
    unittest.main()
