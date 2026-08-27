from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/native-chat-protocol-v0.1.0.json"
RESULT_SIDECAR = RESULT.with_name(RESULT.name + ".sha256")
QUALIFIED_COMMIT = "8c54852592922ad489ef968d4b79a7c3df9c3deb"
QUALIFIED_BINARY_SHA256 = (
    "1570fc41ddb8eacb6efd83d2631e6258cd5c5a466ed2b758dcbcf336e98f0053"
)
CHECKS = {
    "artifact_paths_sanitized",
    "bounded_history_no_progress_stream_parity",
    "different_arguments_remain_parallel",
    "invalid_and_unsupported_fields_fail_closed",
    "omitted_thinking_is_backward_compatible",
    "parallel_false_admits_at_most_one",
    "same_response_duplicate_suppressed",
    "thinking_before_tool_call",
    "thinking_stream_nonstream_and_history",
    "vl_thinking_prompt_and_response_split",
}


def tool_signatures(observation: dict) -> list[tuple[str, dict]]:
    return [
        (
            call["function"]["name"],
            json.loads(call["function"]["arguments"]),
        )
        for call in observation["tool_calls"]
    ]


class NativeChatProtocolQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = RESULT.read_bytes()
        cls.result = json.loads(cls.payload)
        cls.observations = cls.result["observations"]

    def assert_sealed_file(self, path: Path, expected_sha256: str) -> None:
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(digest, expected_sha256)
        sidecar = path.with_name(path.name + ".sha256")
        self.assertEqual(
            sidecar.read_text(encoding="utf-8"),
            f"{digest}  {path.name}\n",
        )

    def test_primary_result_is_canonically_sealed(self) -> None:
        digest = hashlib.sha256(self.payload).hexdigest()
        self.assertEqual(
            RESULT_SIDECAR.read_text(encoding="utf-8"),
            f"{digest}  {RESULT.name}\n",
        )
        canonical = {
            key: value
            for key, value in self.result.items()
            if key != "integrity"
        }
        canonical_bytes = json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(
            self.result["integrity"],
            {
                "algorithm": "sha256",
                "canonical_payload_sha256": hashlib.sha256(
                    canonical_bytes
                ).hexdigest(),
            },
        )

    def test_candidate_identity_is_exact_and_resolvable(self) -> None:
        self.assertEqual(
            self.result["schema"],
            "aima.native-chat-protocol-qualification.v0.1.0",
        )
        self.assertTrue(self.result["qualified"])
        self.assertEqual(self.result["engine"]["path"], "${AIMA_ENGINE}")
        self.assertEqual(
            self.result["engine"]["sha256"], QUALIFIED_BINARY_SHA256
        )
        self.assertEqual(
            self.result["engine"]["build_info"],
            {"version": "1.5.1-native", "source_commit": QUALIFIED_COMMIT},
        )
        resolved = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", f"{QUALIFIED_COMMIT}^{{commit}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(resolved, QUALIFIED_COMMIT)

    def test_all_checks_are_explicit_and_qualified(self) -> None:
        self.assertEqual(set(self.result["checks"]), CHECKS)
        self.assertTrue(all(self.result["checks"].values()))
        ready = self.result["server"]["ready"]
        self.assertEqual(ready["event"], "ready")
        self.assertTrue(ready["native_vl"])
        self.assertEqual(ready["static_prefill_tokens"], 8192)
        for runtime in ("python", "torch", "triton", "vllm"):
            self.assertFalse(ready[f"runtime_{runtime}"])

    def test_raw_artifacts_are_content_bound_and_sealed(self) -> None:
        for name in (
            "load_report",
            "language_load_report",
            "visual_load_report",
            "stderr",
        ):
            component = self.result["server"][name]
            path = ROOT / "benchmarks/results" / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assert_sealed_file(path, component["sha256"])

        load = json.loads(
            (
                ROOT
                / "benchmarks/results"
                / self.result["server"]["load_report"]["path"]
            ).read_bytes()
        )
        self.assertTrue(load["complete"])
        self.assertTrue(load["cleanup_complete"])
        self.assertTrue(load["gpu_payload_checksum_equal"])
        self.assertEqual(load["tensor_count"], 1026)
        self.assertEqual(load["shard_count"], 26)
        self.assertEqual(self.result["server"]["stderr"]["bytes"], 0)

    def test_thinking_default_history_stream_and_vl_contracts(self) -> None:
        default = self.observations["default"]
        disabled = self.observations["disabled"]
        self.assertEqual(default["content"], disabled["content"])
        self.assertEqual(
            default["output_token_ids_sha256"],
            disabled["output_token_ids_sha256"],
        )
        self.assertFalse(default["reasoning_content_present"])
        self.assertFalse(disabled["reasoning_content_present"])
        self.assertEqual(default["thinking"]["mode"], "default")
        self.assertEqual(disabled["thinking"]["mode"], "disabled")

        history = self.observations["thinking_history_nonstream"]
        history_stream = self.observations["thinking_history_stream"]
        self.assertTrue(history["reasoning_content_present"])
        self.assertGreater(history["reasoning_content_chars"], 0)
        for field in (
            "content",
            "reasoning_content_chars",
            "reasoning_content_sha256",
            "output_token_ids_sha256",
            "usage",
        ):
            self.assertEqual(history[field], history_stream[field])
        first_content = history_stream["delta_order"].index("content")
        self.assertTrue(
            all(
                delta == "reasoning_content"
                for delta in history_stream["delta_order"][:first_content]
            )
        )

        vl_disabled = self.observations["vl_disabled"]
        vl_enabled = self.observations["vl_enabled"]
        self.assertEqual(
            vl_disabled["usage"]["prompt_tokens"],
            vl_enabled["usage"]["prompt_tokens"] + 2,
        )
        self.assertFalse(vl_disabled["reasoning_content_present"])
        self.assertTrue(vl_enabled["reasoning_content_present"])
        self.assertGreater(vl_enabled["reasoning_content_chars"], 0)

    def test_thinking_precedes_identical_streamed_tool_call(self) -> None:
        ordinary = self.observations["thinking_tool_nonstream"]
        streamed = self.observations["thinking_tool_stream"]
        self.assertGreater(ordinary["reasoning_content_chars"], 0)
        self.assertEqual(
            ordinary["reasoning_content_sha256"],
            streamed["reasoning_content_sha256"],
        )
        self.assertEqual(
            ordinary["output_token_ids_sha256"],
            streamed["output_token_ids_sha256"],
        )
        self.assertEqual(tool_signatures(ordinary), tool_signatures(streamed))
        first_tool = streamed["delta_order"].index("tool_calls")
        self.assertTrue(
            all(
                delta == "reasoning_content"
                for delta in streamed["delta_order"][:first_tool]
            )
        )

    def test_tool_progress_policy_is_bounded_and_stream_exact(self) -> None:
        duplicate = self.observations["duplicate_nonstream"]
        duplicate_stream = self.observations["duplicate_stream"]
        self.assertEqual(len(tool_signatures(duplicate)), 1)
        self.assertEqual(tool_signatures(duplicate), tool_signatures(duplicate_stream))
        self.assertEqual(duplicate["tool_progress"]["parsed_calls"], 2)
        self.assertEqual(
            duplicate["tool_progress"]["duplicate_calls_suppressed"], 1
        )
        self.assertEqual(
            duplicate["tool_progress"], duplicate_stream["tool_progress"]
        )

        different = self.observations["different_arguments"]
        self.assertEqual(
            tool_signatures(different),
            [
                ("get_weather", {"city": "Paris"}),
                ("get_weather", {"city": "Tokyo"}),
            ],
        )
        serial = self.observations["parallel_false"]
        self.assertEqual(len(tool_signatures(serial)), 1)
        self.assertEqual(
            serial["tool_progress"]["parallel_calls_suppressed"], 1
        )

        exhausted = self.observations["exhausted_nonstream"]
        exhausted_stream = self.observations["exhausted_stream"]
        self.assertEqual(tool_signatures(exhausted), [])
        self.assertEqual(tool_signatures(exhausted_stream), [])
        self.assertEqual(
            exhausted["tool_progress"], exhausted_stream["tool_progress"]
        )
        progress = exhausted["tool_progress"]
        self.assertEqual(progress["same_signature_retry_limit"], 1)
        self.assertEqual(progress["history_signature_occurrences"], 2)
        self.assertEqual(progress["history_no_progress_results"], 2)
        self.assertEqual(progress["exhausted_history_calls_suppressed"], 1)
        self.assertTrue(progress["no_progress"])
        self.assertEqual(
            progress["caller_action"], "change_strategy_or_return_blocked"
        )

    def test_invalid_fields_fail_closed_with_exact_errors(self) -> None:
        self.assertEqual(
            self.observations["invalid_requests"],
            {
                "invalid_type": {
                    "status": 400,
                    "message": "thinking.type must be enabled or disabled",
                },
                "zero_budget": {
                    "status": 400,
                    "message": (
                        "thinking.budget_tokens must be a positive integer"
                    ),
                },
                "budget_above_max": {
                    "status": 400,
                    "message": (
                        "thinking.budget_tokens must not exceed max_tokens"
                    ),
                },
                "unsupported_sampling": {
                    "status": 400,
                    "message": (
                        "unsupported generation field: frequency_penalty"
                    ),
                },
                "raw_prompt_thinking": {
                    "status": 400,
                    "message": (
                        "thinking cannot be combined with prompt_token_ids"
                    ),
                },
            },
        )

    def test_committed_evidence_contains_no_private_paths_or_markers(self) -> None:
        files = [RESULT, RESULT_SIDECAR]
        raw_dir = RESULT.with_name(RESULT.stem + "-raw")
        files.extend(path for path in raw_dir.iterdir() if path.is_file())
        serialized = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in files
        )
        for private_prefix in ("/home/", "/Users/", "/data/", "/tmp/"):
            self.assertNotIn(private_prefix, serialized)
        observations = json.dumps(
            self.observations, ensure_ascii=False, sort_keys=True
        )
        for marker in ("<think>", "</think>", "<tool_call>"):
            self.assertNotIn(marker, observations)


if __name__ == "__main__":
    unittest.main()
