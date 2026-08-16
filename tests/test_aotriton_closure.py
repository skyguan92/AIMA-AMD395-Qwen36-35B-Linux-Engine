from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aima_engine import aotriton_closure


ROOT = Path(__file__).resolve().parents[1]
VL_QUALIFIERS = (
    "qualify-native-vl-capabilities.py",
    "qualify-native-vl-envelope.py",
    "qualify-native-vl-error-limits.py",
    "qualify-native-vl-g1-extension.py",
    "qualify-native-vl-generation.py",
    "qualify-native-vl-serving.py",
    "qualify-native-vl-transport-cache.py",
)


class AotritonClosureTest(unittest.TestCase):
    def materialize(self, root: Path) -> tuple[Path, Path, Path]:
        provider = root / "libaima-fmha-aotriton.so"
        runtime = root / aotriton_closure.AOTRITON_RUNTIME_SONAME
        image = root / aotriton_closure.AOTRITON_IMAGE_RELATIVE
        provider.write_bytes(b"provider")
        runtime.write_bytes(b"runtime")
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        return provider, runtime, image

    def frozen_test_hashes(self):
        return mock.patch.multiple(
            aotriton_closure,
            FROZEN_AOTRITON_RUNTIME_SHA256=hashlib.sha256(b"runtime").hexdigest(),
            FROZEN_AOTRITON_IMAGE_SHA256=hashlib.sha256(b"image").hexdigest(),
        )

    def test_resolves_exact_provider_runtime_and_single_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.frozen_test_hashes():
            provider, runtime, image = self.materialize(Path(directory))
            closure = aotriton_closure.resolve_aotriton_closure(provider)
            self.assertEqual(closure.provider, provider.resolve())
            self.assertEqual(closure.runtime.resolve(), runtime.resolve())
            self.assertEqual(closure.image.resolve(), image.resolve())

    def test_missing_or_changed_dependencies_fail_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.frozen_test_hashes():
            provider, runtime, image = self.materialize(Path(directory))
            image.unlink()
            with self.assertRaisesRegex(RuntimeError, "image is missing"):
                aotriton_closure.resolve_aotriton_closure(provider)
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "image differs"):
                aotriton_closure.resolve_aotriton_closure(provider)
            image.write_bytes(b"image")
            runtime.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "runtime differs"):
                aotriton_closure.resolve_aotriton_closure(provider)

    def test_extra_code_object_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.frozen_test_hashes():
            provider, _runtime, image = self.materialize(Path(directory))
            (image.parent / "extra.aks2").write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "exactly the frozen image"):
                aotriton_closure.resolve_aotriton_closure(provider)

    def test_qualifier_failure_is_concise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = Path(directory) / "missing-provider.so"
            with self.assertRaisesRegex(
                SystemExit,
                "invalid AOTriton qualification closure: "
                "AOTriton FMHA provider is missing",
            ):
                aotriton_closure.require_aotriton_closure(provider)

    def test_every_vl_qualifier_binds_the_complete_closure(self) -> None:
        for name in VL_QUALIFIERS:
            with self.subTest(qualifier=name):
                source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn("require_aotriton_closure", source)
                self.assertIn('"aima_engine/aotriton_closure.py"', source)
                self.assertIn('"aotriton_runtime"', source)
                self.assertIn('"aotriton_image"', source)


if __name__ == "__main__":
    unittest.main()
