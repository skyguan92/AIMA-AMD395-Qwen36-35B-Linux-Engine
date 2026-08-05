from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
QUALIFIER = ROOT / "scripts/qualify-native-eval.py"


def load_qualifier():
    spec = importlib.util.spec_from_file_location("native_eval_qualifier", QUALIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import native eval qualifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeEvalQualifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qualifier = load_qualifier()

    def test_answer_parser_is_bounded_to_standalone_choices(self) -> None:
        self.assertEqual(self.qualifier.parse_answer("Answer: C"), "C")
        self.assertEqual(self.qualifier.parse_answer("D\n"), "D")
        self.assertIsNone(self.qualifier.parse_answer("BAD"))
        self.assertIsNone(self.qualifier.parse_answer("no answer"))

    def test_frozen_request_is_hash_checked_and_prompt_free(self) -> None:
        values = [248045, 846, 198, 32]
        digest = hashlib.sha256(
            ",".join(str(value) for value in values).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "requests/item.json"
            request.parent.mkdir()
            request.write_text(
                json.dumps({"prompt_token_ids": values, "max_tokens": 2}),
                encoding="utf-8",
            )
            item = {
                "item_id": "fixture-1",
                "request_path": "requests/item.json",
                "prompt_tokens": len(values),
                "prompt_token_ids_sha256": digest,
                "requested_output_tokens": 2,
            }
            payload, source = self.qualifier.request_for(item, root)
            self.assertEqual(payload["prompt_token_ids"], values)
            self.assertEqual(source["prompt_token_ids_sha256"], digest)
            self.assertNotIn("question", source)
            self.assertNotIn("prompt_token_ids", source)
            item["prompt_token_ids_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "prompt token hash changed"):
                self.qualifier.request_for(item, root)


if __name__ == "__main__":
    unittest.main()
