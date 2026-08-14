from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify-native-vl-capabilities.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "native_vl_qualifier_logic_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forced_case() -> dict:
    return {
        "case_id": "tool_forced_image",
        "surfaces": ["tool", "image"],
        "accepted": True,
        "passed": True,
        "status_code": 200,
        "response": {
            "aima_amd395": {
                "model_loads": 1,
                "oracle_tensor_reads": 0,
                "runtime": "native-resident-q349",
                "prompt_tokens": 349,
                "prompt_token_ids_sha256": "a" * 64,
                "structured_decoding": {
                    "enabled": True,
                    "token_selections": 18,
                    "token_mask_upload_bytes": 18 * 248_320,
                },
                "vl": {
                    "enabled": True,
                    "media_count": 1,
                    "vision_patches": 256,
                    "visual_tokens": 64,
                },
                "mrope": {"enabled": True},
            },
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "inspect_visual",
                                    "arguments": '{"label":"stripes"}',
                                }
                            }
                        ]
                    }
                }
            ],
        },
    }


class NativeVlQualifierLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_named_tool_binds_render_vector_and_mask_accounting(self) -> None:
        case = forced_case()
        checks = self.module.case_checks(
            case,
            {"status_code": 200},
            {"prompt_tokens": 349, "prompt_token_ids_sha256": "a" * 64},
        )
        self.assertTrue(all(checks.values()), checks)

    def test_render_or_mask_drift_fails_closed(self) -> None:
        case = forced_case()
        case["response"]["aima_amd395"]["structured_decoding"][
            "token_mask_upload_bytes"
        ] -= 1
        checks = self.module.case_checks(
            case,
            {"status_code": 200},
            {"prompt_tokens": 349, "prompt_token_ids_sha256": "b" * 64},
        )
        self.assertFalse(checks["structured_token_mask_accounting_exact"])
        self.assertFalse(checks["render_prompt_token_ids_exact"])


if __name__ == "__main__":
    unittest.main()
