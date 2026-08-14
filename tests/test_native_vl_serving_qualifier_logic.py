from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from aima_engine.vl_reference import sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify-native-vl-serving.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "native_vl_serving_qualifier_logic_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs() -> tuple[dict, dict, dict]:
    content = "striped"
    case = {
        "case_id": "image_local_png",
        "processor": {
            "prompt_token_ids": [1, 2],
            "prompt_token_ids_sha256": "a" * 64,
            "placeholders": {"image": [{"num_embeds": 64}]},
            "tensors": {"pixel_values": {"shape": [256, 3, 14, 14]}},
        },
        "boundaries": {"mrope_positions": {"position_delta": -7}},
        "generation": {
            "completion_tokens": 1,
            "output_token_ids_sha256": "b" * 64,
            "output_text_sha256": sha256_bytes(content.encode("utf-8")),
            "finish_reason": "stop",
        },
    }
    render = {
        "prompt_tokens": 82,
        "prompt_token_ids_sha256": "c" * 64,
        "private_prompt_tokens": 2,
        "private_prompt_token_ids_sha256": "a" * 64,
        "private_prompt_matches_real_http": False,
    }
    response = {
        "usage": {"prompt_tokens": 82, "completion_tokens": 1},
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ],
        "aima_amd395": {
            "prompt_token_ids_sha256": "c" * 64,
            "output_token_ids_canonical_sha256": "b" * 64,
            "model_loads": 1,
            "oracle_tensor_reads": 0,
            "vl": {
                "enabled": True,
                "vision_patches": 256,
                "visual_tokens": 64,
            },
            "mrope": {"enabled": True, "position_delta": -7},
        },
    }
    return case, render, response


class NativeVlServingQualifierLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_real_http_prompt_and_private_generation_are_separate(self) -> None:
        case, render, response = inputs()
        result = self.module.oracle_case_result(
            case, render, 200, response, 1.0
        )
        self.assertTrue(result["passed"], result["checks"])

    def test_real_http_prompt_drift_fails_closed(self) -> None:
        case, render, response = inputs()
        response["aima_amd395"]["prompt_token_ids_sha256"] = "d" * 64
        result = self.module.oracle_case_result(
            case, render, 200, response, 1.0
        )
        self.assertFalse(
            result["checks"]["real_http_prompt_token_ids_sha256_exact"]
        )


if __name__ == "__main__":
    unittest.main()
