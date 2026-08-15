from __future__ import annotations

import copy
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


def cache_inputs() -> list[dict]:
    def observation(
        case_id: str,
        *,
        hits: int,
        misses: int,
        prefix: str,
        output: str = "e" * 64,
        vision_ms: float = 1.0,
        plan_hit: bool = False,
    ) -> dict:
        return {
            "case_id": case_id,
            "content": "The",
            "output_token_ids_sha256": output,
            "prefix_lookup": prefix,
            "vl": {
                "media_cache_hits": hits,
                "media_cache_misses": misses,
                "media_decode_wall_ms": 0.0,
                "processor_wall_ms": 0.0,
                "vision_plan_cache_hit": plan_hit,
                "vision_encode_wall_ms": vision_ms,
            },
        }

    return [
        observation("image_local_a", hits=0, misses=1, prefix="miss"),
        observation(
            "image_local_b",
            hits=0,
            misses=1,
            prefix="miss",
            plan_hit=True,
        ),
        observation(
            "image_local_a_restored",
            hits=1,
            misses=0,
            prefix="exact",
            vision_ms=0.0,
        ),
        observation(
            "image_data_a_equivalent",
            hits=1,
            misses=0,
            prefix="exact",
            vision_ms=0.0,
        ),
        observation(
            "image_data_a_prompt_variant",
            hits=1,
            misses=0,
            prefix="miss",
            plan_hit=True,
        ),
        observation("image_http_a", hits=0, misses=1, prefix="miss"),
        observation(
            "image_http_b",
            hits=0,
            misses=1,
            prefix="miss",
            plan_hit=True,
        ),
        observation(
            "image_http_a_restored",
            hits=1,
            misses=0,
            prefix="exact",
            vision_ms=0.0,
        ),
        observation("video_local_cold", hits=0, misses=1, prefix="miss"),
        observation(
            "video_data_equivalent",
            hits=1,
            misses=0,
            prefix="exact",
            vision_ms=0.0,
        ),
        observation("mixed_local_cold", hits=0, misses=2, prefix="miss"),
        observation(
            "mixed_local_exact",
            hits=2,
            misses=0,
            prefix="exact",
            vision_ms=0.0,
        ),
    ]


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

    def test_extended_cache_contract_accepts_only_exact_safe_reuse(self) -> None:
        checks = self.module.cache_correctness_checks(cache_inputs())
        self.assertTrue(all(checks.values()), checks)

    def test_http_video_and_mixed_cache_drift_fails_closed(self) -> None:
        observations = cache_inputs()
        by_id = {item["case_id"]: item for item in observations}
        by_id["image_http_b"]["vl"]["media_cache_misses"] = 0
        by_id["image_http_b"]["vl"]["vision_plan_cache_hit"] = False
        by_id["video_data_equivalent"]["output_token_ids_sha256"] = "f" * 64
        by_id["mixed_local_exact"]["vl"]["media_cache_hits"] = 1
        checks = self.module.cache_correctness_checks(observations)
        self.assertFalse(checks["same_http_url_b_processor_miss"])
        self.assertFalse(checks["same_http_url_b_prefix_miss"])
        self.assertFalse(checks["video_data_local_output_exact"])
        self.assertFalse(checks["mixed_exact_two_media_hits"])

    def test_cache_case_ids_must_be_unique(self) -> None:
        observations = cache_inputs()
        observations.append(copy.deepcopy(observations[0]))
        with self.assertRaisesRegex(RuntimeError, "not unique"):
            self.module.cache_correctness_checks(observations)


if __name__ == "__main__":
    unittest.main()
