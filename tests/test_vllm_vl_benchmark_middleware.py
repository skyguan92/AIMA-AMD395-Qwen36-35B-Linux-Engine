from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "vllm_vl_benchmark_middleware.py"
SPEC = importlib.util.spec_from_file_location(
    "vllm_vl_benchmark_middleware", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
middleware_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(middleware_module)


class VllmVlBenchmarkMiddlewareLogicTest(unittest.TestCase):
    def test_overlapping_media_intervals_use_wall_union(self) -> None:
        intervals = [
            {"start_ns": 0, "end_ns": 10},
            {"start_ns": 5, "end_ns": 15},
            {"start_ns": 20, "end_ns": 24},
        ]
        self.assertEqual(middleware_module._merged_interval_ns(intervals), 19)

    def test_mm_stats_match_official_suffix_and_worker_max_merge(self) -> None:
        processor = {
            "renderer0-mm-1": {
                "apply_hf_processor_secs": 0.4,
                "preprocessor_total_secs": 0.5,
            }
        }
        workers = [
            {
                "renderer0-mm-1-0": {
                    "encoder_forward_secs": 0.2,
                    "num_encoder_calls": 1,
                }
            },
            {
                "renderer0-mm-1-0": {
                    "encoder_forward_secs": 0.3,
                    "num_encoder_calls": 2,
                }
            },
        ]
        merged = middleware_module._merge_mm_stats(processor, workers)
        self.assertEqual(set(merged), {"renderer0-mm-1"})
        self.assertEqual(merged["renderer0-mm-1"]["encoder_forward_secs"], 0.3)
        self.assertEqual(merged["renderer0-mm-1"]["num_encoder_calls"], 2)

    def test_marked_request_logs_stages_without_request_content(self) -> None:
        class FakeRegistry:
            def __init__(self) -> None:
                self.calls = 0

            def stat(self):
                self.calls += 1
                if self.calls == 1:
                    return {"stale": {"preprocessor_total_secs": 9.0}}
                return {
                    "renderer0-mm-2": {
                        "apply_hf_processor_secs": 0.4,
                        "preprocessor_total_secs": 0.5,
                    }
                }

        class FakeEngine:
            def __init__(self) -> None:
                self.renderer = types.SimpleNamespace(
                    _mm_timing_registry=FakeRegistry()
                )
                self.calls = 0

            async def collective_rpc(self, method):
                self.calls += 1
                self_method = method
                if self_method != "get_encoder_timing_stats":
                    raise AssertionError(self_method)
                if self.calls == 1:
                    return [{"stale-0": {"encoder_forward_secs": 9.0}}]
                return [
                    {
                        "renderer0-mm-2-0": {
                            "encoder_forward_secs": 0.25,
                            "num_encoder_calls": 1,
                        }
                    }
                ]

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            await send(
                {
                    "type": "http.response.body",
                    "body": b"safe-response",
                    "more_body": False,
                }
            )

        async def receive():
            return {"type": "http.request", "body": b"secret-prompt"}

        sent = []

        async def send(message):
            sent.append(message)

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "stages.jsonl"
            env = {middleware_module.LOG_ENV: str(log_path)}
            with mock.patch.object(
                middleware_module, "_install_media_patch"
            ), mock.patch.dict(os.environ, env, clear=False):
                wrapped = middleware_module.VlBenchmarkMetricsMiddleware(app)
                scope = {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "headers": [
                        (
                            middleware_module.BENCHMARK_HEADER,
                            b"image-minimum.pair-1.reference",
                        )
                    ],
                    "app": types.SimpleNamespace(
                        state=types.SimpleNamespace(engine_client=FakeEngine())
                    ),
                }
                asyncio.run(wrapped(scope, receive, send))

            self.assertEqual(len(sent), 2)
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["status_code"], 200)
            self.assertEqual(record["response_bytes"], len(b"safe-response"))
            self.assertIsNone(record["request_error"])
            self.assertIsNone(record["stats_error"])
            merged = record["multimodal"]["merged"]
            self.assertEqual(set(merged), {"renderer0-mm-2"})
            self.assertEqual(
                merged["renderer0-mm-2"]["encoder_forward_secs"], 0.25
            )
            serialized = log_path.read_text(encoding="utf-8")
            self.assertNotIn("secret-prompt", serialized)
            self.assertNotIn("safe-response", serialized)


if __name__ == "__main__":
    unittest.main()
