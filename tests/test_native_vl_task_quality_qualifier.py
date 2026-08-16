from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

from aima_engine.vl_task_quality import score_text


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify-native-vl-task-quality.py"
CAPTURE = ROOT / "scripts/capture-vllm-vl-task-quality.py"
HTTP_SERVER = ROOT / "native/src/native_http_server.cpp"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "native_vl_task_quality_qualifier_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs() -> tuple[dict, dict, dict]:
    rubric = [
        {"id": "red", "any_of": ["red"]},
        {"id": "circle", "any_of": ["circle"]},
    ]
    content = "The image contains a red circle."
    response = {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }
    request = {"model": "${AIMA_SERVED_MODEL}", "max_tokens": 192}
    reference = {
        "case_id": "image_central_red_circle",
        "modality": "image",
        "status_code": 200,
        "request": request,
        "response": copy.deepcopy(response),
        "render": {
            "prompt_tokens": 10,
            "prompt_token_ids_sha256": "a" * 64,
        },
        "rubric": rubric,
        "output_text": content,
        "output_token_ids_sha256": "b" * 64,
        "score": score_text(content, rubric),
    }
    case = {
        "case_id": reference["case_id"],
        "passed": True,
        "status_code": 200,
        "request": request,
        "response": response,
        "score": score_text(content, rubric),
    }
    metrics = {
        "model_loads": 1,
        "oracle_tensor_reads": 0,
        "runtime": "native-resident-q1024",
        "prompt_tokens": 10,
        "prompt_token_ids_sha256": "a" * 64,
        "completion_tokens": 8,
        "output_token_ids_canonical_sha256": "b" * 64,
        "vl": {
            "enabled": True,
            "image_count": 1,
            "video_count": 0,
            "media_count": 1,
            "vision_patches": 256,
            "visual_tokens": 64,
        },
        "mrope": {"enabled": True},
        "structured_decoding": {"enabled": False},
    }
    return case, reference, metrics


class NativeVlTaskQualityQualifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_exact_long_generation_and_quality_pass(self) -> None:
        case, reference, metrics = inputs()
        checks = self.module.case_checks(case, reference, metrics)
        self.assertTrue(all(checks.values()), checks)

    def test_output_token_or_prompt_drift_fails_closed(self) -> None:
        case, reference, metrics = inputs()
        metrics["output_token_ids_canonical_sha256"] = "c" * 64
        metrics["prompt_token_ids_sha256"] = "d" * 64
        checks = self.module.case_checks(case, reference, metrics)
        self.assertFalse(checks["render_prompt_token_ids_exact"])
        diagnostics = self.module.parity_diagnostics(case, reference, metrics)
        self.assertFalse(diagnostics["output_token_ids_reference_exact"])

    def test_semantically_equivalent_text_is_quality_not_exact_generation(
        self,
    ) -> None:
        case, reference, metrics = inputs()
        replacement = "A red circle is visible in the image."
        case["response"]["choices"][0]["message"]["content"] = replacement
        case["score"] = score_text(replacement, reference["rubric"])
        metrics["output_token_ids_canonical_sha256"] = "c" * 64
        checks = self.module.case_checks(case, reference, metrics)
        diagnostics = self.module.parity_diagnostics(case, reference, metrics)
        self.assertTrue(all(checks.values()), checks)
        self.assertFalse(diagnostics["generated_content_reference_exact"])
        self.assertFalse(diagnostics["output_token_ids_reference_exact"])

    def test_score_is_recomputed_and_compared_as_exact_rational(self) -> None:
        case, reference, metrics = inputs()
        case["score"] = {
            "earned_points": 1,
            "total_points": 2,
            "score_millionths": 500_000,
            "criteria": [],
        }
        checks = self.module.case_checks(case, reference, metrics)
        self.assertFalse(checks["score_recomputed"])
        self.assertFalse(checks["task_quality_not_below_reference"])

    def test_cli_contract_rejects_non_loopback_and_unbounded_timeout(self) -> None:
        with self.assertRaisesRegex(SystemExit, "loopback"):
            self.module.validate_cli_contract(
                host="0.0.0.0",
                port=18166,
                ready_timeout_seconds=1,
                request_timeout_seconds=1,
            )
        with self.assertRaisesRegex(SystemExit, "cannot exceed 600"):
            self.module.validate_cli_contract(
                host="127.0.0.1",
                port=18166,
                ready_timeout_seconds=1,
                request_timeout_seconds=601,
            )

    def test_reference_capture_uses_fail_closed_tokenize_reconstruction(self) -> None:
        source = CAPTURE.read_text(encoding="utf-8")
        self.assertIn('endpoint + "/tokenize"', source)
        self.assertIn("complete_output_token_ids", source)

    def test_native_metrics_expose_completion_accounting(self) -> None:
        source = HTTP_SERVER.read_text(encoding="utf-8")
        self.assertIn('{"completion_tokens", metrics.completion_tokens}', source)


if __name__ == "__main__":
    unittest.main()
