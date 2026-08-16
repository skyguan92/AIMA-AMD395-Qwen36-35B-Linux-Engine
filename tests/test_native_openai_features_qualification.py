from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "qualify-native-openai-features.py"
SPEC = importlib.util.spec_from_file_location("native_openai_features", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
features = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(features)


class NativeOpenaiFeaturesQualificationTest(unittest.TestCase):
    def test_frozen_protocol_requires_q8192(self) -> None:
        features.require_qualified_context(8192)
        for context_tokens in (1024, 2048, 4096, 16384):
            with self.subTest(context_tokens=context_tokens):
                with self.assertRaisesRegex(ValueError, "requires.*8192"):
                    features.require_qualified_context(context_tokens)


if __name__ == "__main__":
    unittest.main()
