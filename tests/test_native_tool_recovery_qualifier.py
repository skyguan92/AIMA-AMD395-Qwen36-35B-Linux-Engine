from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from aima_engine import cli

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tool_recovery_qualifier", ROOT / "scripts/qualify-native-chat-protocol.py"
)
assert SPEC is not None and SPEC.loader is not None
QUALIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFIER)


class ToolRecoveryQualifierTest(unittest.TestCase):
    def error_event(self):
        return {
            "error": {"code": "tool_call_no_progress", "message": "Change strategy.",
                      "type": "invalid_request_error"},
            "aima_amd395": {"tool_progress": {
                "no_progress": True, "history_no_progress_results": 5,
                "history_no_progress_streak": 2,
            }},
        }

    def test_nonstream_error_is_not_summarized_as_completion(self):
        result = QUALIFIER.response_summary(self.error_event())
        self.assertEqual(result["error"]["code"], "tool_call_no_progress")
        self.assertIsNone(result["finish_reason"])
        self.assertIsNone(result["content"])
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(result["tool_progress"]["history_no_progress_streak"], 2)

    def stream(self, events, done=True):
        raw = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        if done:
            raw += b"data: [DONE]\n\n"
        reader = io.BytesIO(raw)
        response = Mock(status=200)
        response.getheader.side_effect = lambda key: {
            "Content-Type": "text/event-stream", "Transfer-Encoding": "chunked"
        }.get(key)
        response.readline.side_effect = reader.readline
        connection = Mock()
        connection.getresponse.return_value = response
        with patch.object(QUALIFIER.http.client, "HTTPConnection", return_value=connection):
            result = QUALIFIER.request_stream(1, {"messages": []})
        connection.close.assert_called_once()
        return result

    def test_sse_error_is_retained_despite_http_200_and_done(self):
        result = self.stream([
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            self.error_event(),
        ])
        self.assertEqual(result["status"], 200)
        self.assertTrue(result["done"])
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["error"]["code"], "tool_call_no_progress")
        self.assertIsNone(result["finish_reason"])
        self.assertEqual(result["tool_calls"], [])
        summary = QUALIFIER.stream_summary(result)
        self.assertIsNone(summary["output_token_ids_sha256"])
        self.assertIsNone(summary["thinking"])
        self.assertEqual(summary["tool_progress"], self.error_event()["aima_amd395"]["tool_progress"])

    def test_incomplete_error_stream_does_not_manufacture_done(self):
        result = self.stream([self.error_event()], done=False)
        self.assertFalse(result["done"])
        self.assertEqual(result["error_count"], 1)
        self.assertIsNone(result["finish_reason"])

    def test_successful_stream_keeps_terminal_reason_and_metrics(self):
        result = self.stream([
            {"choices": [{"delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "aima_amd395": {"output_token_ids_sha256": "abc", "thinking": {"mode": "default"}}},
        ])
        self.assertIsNone(result["error"])
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(QUALIFIER.stream_summary(result)["content"], "hello")

    def assert_cli_failed_turn(self, json_output):
        wire = b"data: " + json.dumps(self.error_event()).encode() + b"\n\ndata: [DONE]\n\n"
        response = io.BytesIO(wire)
        response.headers = Mock()
        response.headers.get_content_type.return_value = "text/event-stream"
        arguments = ["aima-engine", "chat", "hello", "--stream"]
        if json_output:
            arguments.append("--json")
        stderr = io.StringIO()
        with patch.object(cli.urlrequest, "urlopen", return_value=response), \
             patch("sys.argv", arguments), patch("sys.stderr", stderr), \
             patch("sys.stdout", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("stream failed: Change strategy.", stderr.getvalue())

    def test_plain_cli_exits_nonzero_on_policy_error(self):
        self.assert_cli_failed_turn(False)

    def test_json_cli_exits_nonzero_on_policy_error(self):
        self.assert_cli_failed_turn(True)


if __name__ == "__main__":
    unittest.main()
