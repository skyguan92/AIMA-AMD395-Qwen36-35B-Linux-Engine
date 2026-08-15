from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from aima_engine.vl_g1_extension import (
    CASE_ORDER,
    build_cases,
    normalize_contract_request,
    request_media_counts,
)


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
CAPTURE = ROOT / "scripts/capture-vllm-vl-g1-extension.py"
QUALIFIER = ROOT / "scripts/qualify-native-vl-g1-extension.py"
FIXTURES = ROOT / "benchmarks/fixtures/vl-capability-v0.1.0"


def load_probe():
    spec = importlib.util.spec_from_file_location("vl_g1_extension_probe", PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PROBE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name, path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VlG1ExtensionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        probe = load_probe()
        fixtures = probe.Fixtures(FIXTURES, "http://127.0.0.1:1")
        cls.specs = build_cases(fixtures, "test-model")
        cls.capture = load_script(CAPTURE, "vl_g1_extension_capture")
        cls.qualifier = load_script(QUALIFIER, "vl_g1_extension_qualifier")

    def test_case_order_and_success_contract_are_frozen(self) -> None:
        self.assertEqual(
            tuple(item["case_id"] for item in self.specs), CASE_ORDER
        )
        self.assertTrue(all(item["expected_accept"] for item in self.specs))

    def test_frozen_reference_runtime_inputs_are_still_valid(self) -> None:
        launch, reference, version = self.capture.frozen_reference_inputs(
            ROOT / "benchmarks/results/vl-reference-launch.json",
            ROOT / "benchmarks/results/vl-reference-manifest.json",
        )
        self.assertEqual(reference["launch"], launch)
        self.assertTrue(version.startswith("0.19.1rc1.dev300+g29e5d1020"))

    def test_multi_item_mixed_orders_are_distinct(self) -> None:
        first = self.specs[0]["payload"]
        second = self.specs[1]["payload"]
        self.assertEqual(request_media_counts(first), {"image": 2, "video": 1})
        self.assertEqual(request_media_counts(second), {"image": 1, "video": 2})
        first_types = [
            item["type"] for item in first["messages"][0]["content"]
        ]
        second_types = [
            item["type"] for item in second["messages"][0]["content"]
        ]
        self.assertNotEqual(first_types, second_types)

    def test_video_and_mixed_history_span_prior_turns(self) -> None:
        video = self.specs[2]["payload"]
        mixed = self.specs[3]["payload"]
        self.assertEqual(
            [item["role"] for item in video["messages"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(request_media_counts(video), {"image": 0, "video": 2})
        self.assertEqual(
            [item["role"] for item in mixed["messages"]],
            ["user", "assistant", "user"],
        )
        self.assertEqual(request_media_counts(mixed), {"image": 1, "video": 1})

    def test_mixed_stream_and_model_normalization_are_explicit(self) -> None:
        stream = self.specs[4]["payload"]
        self.assertTrue(stream["stream"])
        self.assertEqual(stream["max_tokens"], 4)
        self.assertEqual(request_media_counts(stream), {"image": 1, "video": 1})
        normalized = normalize_contract_request(stream)
        self.assertEqual(normalized["model"], "${AIMA_SERVED_MODEL}")
        self.assertEqual(stream["model"], "test-model")

    def test_native_case_comparison_is_fail_closed(self) -> None:
        request = normalize_contract_request(self.specs[3]["payload"])
        response = {
            "choices": [
                {
                    "message": {"content": "The"},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 131,
                "completion_tokens": 1,
                "total_tokens": 132,
            },
        }
        case = {
            "surfaces": self.specs[3]["surfaces"],
            "passed": True,
            "status_code": 200,
            "request": request,
            "response": response,
        }
        reference = {
            "status_code": 200,
            "request": request,
            "response": response,
            "render": {
                "prompt_tokens": 131,
                "prompt_token_ids_sha256": "a" * 64,
            },
        }
        metrics = {
            "model_loads": 1,
            "oracle_tensor_reads": 0,
            "runtime": "native-resident-q1024",
            "prompt_tokens": 131,
            "prompt_token_ids_sha256": "a" * 64,
            "vl": {
                "enabled": True,
                "image_count": 1,
                "video_count": 1,
                "media_count": 2,
                "vision_patches": 8,
                "visual_tokens": 2,
            },
            "mrope": {"enabled": True},
        }
        checks = self.qualifier.case_checks(case, reference, metrics)
        self.assertTrue(all(checks.values()), checks)
        case["response"] = {
            **response,
            "choices": [
                {
                    "message": {"content": "A"},
                    "finish_reason": "length",
                }
            ],
        }
        checks = self.qualifier.case_checks(case, reference, metrics)
        self.assertFalse(checks["generated_content_exact"])


if __name__ == "__main__":
    unittest.main()
