from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capture-vl-performance-request.py"
SPEC = importlib.util.spec_from_file_location(
    "capture_vl_performance_request", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class VlPerformanceRequestCaptureLogicTest(unittest.TestCase):
    def test_prometheus_parser_aggregates_labels_without_buckets(self) -> None:
        source = """
# HELP ignored ignored
vllm:request_prefill_time_seconds_sum{engine="0"} 0.25
vllm:request_prefill_time_seconds_sum{engine="1"} 0.5
vllm:request_prefill_time_seconds_count{engine="0"} 1
vllm:request_prefill_time_seconds_bucket{le="1"} 99
vllm:generation_tokens{engine="0"} 8
"""
        totals = capture.metric_totals(source)
        self.assertEqual(
            totals["vllm:request_prefill_time_seconds_sum"], 0.75
        )
        self.assertEqual(
            totals["vllm:request_prefill_time_seconds_count"], 1.0
        )
        self.assertEqual(totals["vllm:generation_tokens"], 8.0)

    def test_semantic_delta_skips_role_and_accepts_reasoning_or_tools(self) -> None:
        role = {"choices": [{"delta": {"role": "assistant"}}]}
        reasoning = {
            "choices": [{"delta": {"reasoning_content": "think"}}]
        }
        tools = {"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]}
        self.assertEqual(capture.semantic_delta(role), "")
        self.assertEqual(capture.semantic_delta(reasoning), "think")
        self.assertTrue(capture.semantic_delta(tools))

    def test_stream_error_is_preserved_without_copying_unknown_payloads(self) -> None:
        error = capture.normalized_stream_error(
            {
                "error": {
                    "message": "native embedded launch failed",
                    "type": "server_error",
                    "code": "native_failure",
                    "param": None,
                    "nested": {"private": "diagnostic"},
                }
            }
        )
        self.assertEqual(
            error,
            {
                "message": "native embedded launch failed",
                "type": "server_error",
                "code": "native_failure",
                "param": None,
            },
        )
        opaque = capture.normalized_stream_error({"error": {"nested": 1}})
        self.assertEqual(set(opaque or {}), {"payload_sha256"})
        self.assertIsNone(capture.normalized_stream_error({"choices": []}))

    def test_capture_error_records_only_kind_type_and_message_digest(self) -> None:
        error = capture.normalized_capture_error(
            "metrics_after_error", ConnectionRefusedError("private detail")
        )
        self.assertEqual(error["kind"], "metrics_after_error")
        self.assertEqual(error["exception_type"], "ConnectionRefusedError")
        self.assertEqual(len(error["message_sha256"]), 64)
        self.assertNotIn("private detail", str(error))

    def test_request_summary_excludes_content_but_counts_surfaces(self) -> None:
        payload = {
            "temperature": 0,
            "stream": True,
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "x"}},
                        {"type": "video_url", "video_url": {"url": "y"}},
                        {"type": "text", "text": "secret"},
                    ],
                }
            ],
        }
        summary = capture.request_summary(payload)
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(summary["video_count"], 1)
        self.assertEqual(summary["text_characters"], 6)
        self.assertNotIn("secret", str(summary))

    def test_media_root_substitution_is_recursive_and_narrow(self) -> None:
        value = {
            "messages": [
                {
                    "url": "file://${AIMA_VL_MEDIA_ROOT}/image.png",
                    "text": "${UNRELATED}",
                }
            ]
        }
        replaced = capture.substitute_media_root(value, Path("/srv/media"))
        self.assertEqual(
            replaced["messages"][0]["url"], "file:///srv/media/image.png"
        )
        self.assertEqual(replaced["messages"][0]["text"], "${UNRELATED}")

    def test_prompt_nonce_requires_one_placeholder_and_safe_value(self) -> None:
        value = {
            "messages": [
                {
                    "text": "benchmark ${AIMA_VL_PROMPT_NONCE}",
                    "url": "file://${AIMA_VL_MEDIA_ROOT}/image.png",
                }
            ]
        }
        replaced = capture.substitute_prompt_nonce(value, "pair-02.cell-7")
        self.assertEqual(
            replaced["messages"][0]["text"],
            "benchmark pair-02.cell-7",
        )
        self.assertEqual(
            replaced["messages"][0]["url"],
            "file://${AIMA_VL_MEDIA_ROOT}/image.png",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            capture.substitute_prompt_nonce({"messages": []}, "pair-02")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            capture.substitute_prompt_nonce(value, "contains spaces")

    def test_text_padding_appends_exact_frozen_single_token_units(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Earlier"},
                        {"type": "image_url", "image_url": {"url": "x"}},
                        {"type": "text", "text": "Final"},
                    ],
                }
            ]
        }
        capture.append_text_padding(payload, 3)
        self.assertEqual(
            payload["messages"][0]["content"][-1]["text"],
            "Final x x x",
        )
        self.assertEqual(capture.TEXT_PADDING_TOKEN_ID, 830)
        with self.assertRaisesRegex(ValueError, "outside the model window"):
            capture.append_text_padding(payload, 262_145)
        with self.assertRaisesRegex(ValueError, "no text field"):
            capture.append_text_padding(
                {"messages": [{"content": [{"type": "image_url"}]}]}, 1
            )

    def test_media_components_bind_ordered_bytes_inside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "nested/image.png"
            video = root / "video.mp4"
            image.parent.mkdir()
            image.write_bytes(b"image-bytes")
            video.write_bytes(b"video-bytes")
            payload = {
                "messages": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image.resolve().as_uri()},
                            },
                            {
                                "type": "video_url",
                                "video_url": {"url": video.resolve().as_uri()},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image.resolve().as_uri()},
                            },
                        ]
                    }
                ]
            }
            components = capture.media_components(payload, root)

        self.assertEqual([item["index"] for item in components], [0, 1, 2])
        self.assertEqual(
            [item["modality"] for item in components],
            ["image", "video", "image"],
        )
        self.assertEqual(
            components[0]["path"],
            "${AIMA_VL_MEDIA_ROOT}/nested/image.png",
        )
        self.assertEqual(components[0]["sha256"], components[2]["sha256"])
        self.assertEqual(components[0]["bytes"], len(b"image-bytes"))

    def test_media_components_reject_nonlocal_or_escaped_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, \
             tempfile.TemporaryDirectory() as outside_temporary:
            root = Path(temporary)
            outside = Path(outside_temporary) / "outside-media.bin"
            outside.write_bytes(b"outside")
            escaped = {
                "messages": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": outside.resolve().as_uri()
                                },
                            }
                        ]
                    }
                ]
            }
            with self.assertRaisesRegex(ValueError, "escaped"):
                capture.media_components(escaped, root)
            remote = {
                "messages": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.invalid/image.png"
                                },
                            }
                        ]
                    }
                ]
            }
            with self.assertRaisesRegex(ValueError, "local file URL"):
                capture.media_components(remote, root)


if __name__ == "__main__":
    unittest.main()
