from __future__ import annotations

import unittest

from aima_engine.vl_capability import (
    API_RENDER_MAX_TOKENS,
    API_RENDER_MEDIA_COUNTS,
    API_RENDER_SCHEMA,
    API_RENDER_TOOL_CASES,
    API_RENDER_USAGELESS_CASES,
    EXPECTED_MAX_ITEMS_PER_PROMPT,
    EXPECTED_MAX_MODEL_LEN,
    EXPECTED_MAX_TOKENS_PER_ITEM,
    EXPECTED_FORCED_TOOL_STRUCTURED_OUTPUTS,
    PROCESSOR_PROBE_SCHEMA,
    REQUIRED_API_CASES,
    REQUIRED_API_RENDER_CASES,
    REQUIRED_API_SURFACES,
    REQUIRED_IMAGE_CASES,
    REQUIRED_VIDEO_RESIZE_CASES,
    REQUIRED_VIDEO_SAMPLING_CASES,
    validate_capability_manifest,
    validate_api_render_manifest,
    validate_processor_probe,
)
from aima_engine.vl_reference import (
    CAPABILITY_SCHEMA,
    MODEL_REVISION,
    PINNED_PACKAGES,
    canonical_json_sha256,
    seal_manifest,
)


def valid_probe() -> dict:
    def cases(names: set[str]) -> list[dict]:
        return [
            {"case_id": name, "outcome": "accepted", "result": {}}
            for name in sorted(names)
        ]

    return {
        "schema": PROCESSOR_PROBE_SCHEMA,
        "complete": True,
        "qualified": True,
        "model": {"revision": MODEL_REVISION},
        "runtime": {"packages": dict(PINNED_PACKAGES)},
        "vllm_budget": {
            "max_model_len": EXPECTED_MAX_MODEL_LEN,
            "supported_limits": {"image": None, "video": None},
            "max_tokens_per_item": dict(EXPECTED_MAX_TOKENS_PER_ITEM),
            "max_items_per_prompt": dict(EXPECTED_MAX_ITEMS_PER_PROMPT),
            "max_items_per_batch": dict(EXPECTED_MAX_ITEMS_PER_PROMPT),
            "encoder_budget_tokens": EXPECTED_MAX_MODEL_LEN,
        },
        "image_resize_cases": cases(REQUIRED_IMAGE_CASES),
        "video_resize_cases": cases(REQUIRED_VIDEO_RESIZE_CASES),
        "video_sampling_cases": cases(REQUIRED_VIDEO_SAMPLING_CASES),
        "deterministic_processor_fixtures": [
            {
                "modality": modality,
                "outputs": {"input_ids": {"sha256": "a" * 64}},
            }
            for modality in ("image", "video")
        ],
    }


def valid_capability_manifest() -> dict:
    surface_cycle = sorted(REQUIRED_API_SURFACES)
    cases = []
    for index, (case_id, expected) in enumerate(REQUIRED_API_CASES.items()):
        cases.append(
            {
                "case_id": case_id,
                "surfaces": [surface_cycle[index % len(surface_cycle)]],
                "expected_accept": expected,
                "passed": True,
                "status_code": 200 if expected else 400,
                "request_sha256": "a" * 64,
                "response_sha256": "b" * 64,
            }
        )
    return {
        "schema": CAPABILITY_SCHEMA,
        "complete": True,
        "qualified": True,
        "bindings": {
            "processor_probe": {"sha256": "c" * 64},
            "fixture_manifest": {"sha256": "d" * 64},
        },
        "cases": cases,
    }


def valid_api_render_manifest() -> dict:
    cases = []
    for case_id in REQUIRED_API_RENDER_CASES:
        token_ids = [248045]
        placeholders: dict[str, list[dict[str, int | str]]] = {}
        for modality, count in API_RENDER_MEDIA_COUNTS[case_id].items():
            placeholders[modality] = []
            pad_token = 248056 if modality == "image" else 248057
            for _ in range(count):
                offset = len(token_ids)
                token_ids.extend([pad_token, 248044])
                placeholders[modality].append(
                    {
                        "offset": offset,
                        "length": 1,
                        "pad_token_count": 1,
                        "token_ids_sha256": canonical_json_sha256([pad_token]),
                    }
                )
        token_ids.append(248046)
        usage_delta = (
            None
            if case_id in API_RENDER_USAGELESS_CASES
            else 1
            if case_id in API_RENDER_TOOL_CASES
            else 0
        )
        request = {
            "model": "qwen36-vl-reference",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0,
            "max_tokens": API_RENDER_MAX_TOKENS[case_id],
            "stream": case_id in {"stream_image", "stream_video"},
        }
        cases.append(
            {
                "case_id": case_id,
                "surfaces": (
                    ["tool", "image"]
                    if case_id in API_RENDER_TOOL_CASES
                    else ["api"]
                ),
                "request": request,
                "request_sha256": canonical_json_sha256(request),
                "reference_transport_request_sha256": "a" * 64,
                "render_transport_request_sha256": "f" * 64,
                "prompt_tokens": len(token_ids),
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": canonical_json_sha256(token_ids),
                "mm_placeholders": placeholders,
                "reference_usage_prompt_tokens": (
                    None if usage_delta is None else len(token_ids) + usage_delta
                ),
                "reference_usage_delta": usage_delta,
                "max_tokens": API_RENDER_MAX_TOKENS[case_id],
                "structured_outputs": (
                    EXPECTED_FORCED_TOOL_STRUCTURED_OUTPUTS
                    if case_id == "tool_forced_image"
                    else None
                ),
            }
        )
    return seal_manifest(
        {
            "schema": API_RENDER_SCHEMA,
            "complete": True,
            "qualified": True,
            "scope": "fixed-vllm-openai-gpu-less-render-token-boundary",
            "host": {"label": "amd395", "hostname": "test-amd395"},
            "source": {
                "commit": "c" * 40,
                "dirty": False,
                "status_sha256": "d" * 64,
                "files": [
                    {"path": path, "bytes": 1, "sha256": "e" * 64}
                    for path in (
                        "aima_engine/vl_capability.py",
                        "aima_engine/vl_reference.py",
                        "scripts/probe-vllm-vl-api-capabilities.py",
                        "scripts/capture-vllm-vl-api-render.py",
                    )
                ],
            },
            "runtime": {
                "vllm": PINNED_PACKAGES["vllm"],
                "endpoint": {
                    "scheme": "http",
                    "host": "127.0.0.1",
                    "port": 18126,
                },
            },
            "bindings": {
                "capability_manifest": {
                    "path": "benchmarks/results/vl-capability-manifest.json",
                    "bytes": 1,
                    "sha256": "b" * 64,
                },
                "fixture_manifest": {
                    "path": (
                        "benchmarks/fixtures/vl-capability-v0.1.0/"
                        "fixtures-manifest.json"
                    ),
                    "bytes": 1,
                    "sha256": "b" * 64,
                },
                "reference_launch": {
                    "path": "benchmarks/results/vl-reference-launch.json",
                    "bytes": 1,
                    "sha256": "b" * 64,
                },
                "reference_manifest": {
                    "path": "benchmarks/results/vl-reference-manifest.json",
                    "bytes": 1,
                    "sha256": "b" * 64,
                },
            },
            "contract": {
                "content_format": "auto-resolved-string",
                "request_identity": "fixture-normalized-reference-request",
                "tool_normalization": "ChatCompletionRequest-Pydantic-model_dump",
                "render_runtime_uses_gpu": False,
            },
            "cases": cases,
            "decision": {
                "success_render_cases_20_of_20": True,
                "non_tool_non_stream_render_matches_full_usage": True,
                "tool_full_server_usage_offset_one": True,
                "named_tool_json_schema_bound": True,
                "g1_passed": False,
                "g2_passed": False,
            },
        }
    )


class VlCapabilityTest(unittest.TestCase):
    def test_complete_processor_probe_is_accepted(self) -> None:
        self.assertEqual(validate_processor_probe(valid_probe()), [])

    def test_budget_drift_is_rejected(self) -> None:
        probe = valid_probe()
        probe["vllm_budget"]["max_items_per_prompt"]["image"] = 15
        self.assertIn(
            "vLLM derived media-count boundary drifted",
            validate_processor_probe(probe),
        )

    def test_missing_discrete_boundary_is_rejected(self) -> None:
        probe = valid_probe()
        probe["image_resize_cases"] = [
            item
            for item in probe["image_resize_cases"]
            if item["case_id"] != "factor_plus_one"
        ]
        errors = validate_processor_probe(probe)
        self.assertIn("missing image resize cases: factor_plus_one", errors)

    def test_missing_video_fixture_is_rejected(self) -> None:
        probe = valid_probe()
        probe["deterministic_processor_fixtures"] = [
            probe["deterministic_processor_fixtures"][0]
        ]
        self.assertIn(
            "deterministic processor fixtures must cover image and video",
            validate_processor_probe(probe),
        )

    def test_complete_api_capability_manifest_is_accepted(self) -> None:
        self.assertEqual(validate_capability_manifest(valid_capability_manifest()), [])

    def test_missing_or_false_api_case_is_rejected(self) -> None:
        manifest = valid_capability_manifest()
        manifest["cases"] = manifest["cases"][1:]
        manifest["cases"][0]["passed"] = False
        errors = validate_capability_manifest(manifest)
        self.assertTrue(any("missing API capability cases" in error for error in errors))
        self.assertTrue(any("did not pass" in error for error in errors))

    def test_complete_api_render_manifest_is_accepted(self) -> None:
        self.assertEqual(validate_api_render_manifest(valid_api_render_manifest()), [])

    def test_api_render_prompt_or_case_drift_is_rejected(self) -> None:
        manifest = valid_api_render_manifest()
        manifest["cases"][0]["prompt_token_ids"][0] = 7
        manifest["cases"] = manifest["cases"][1:]
        errors = validate_api_render_manifest(manifest)
        self.assertTrue(any("canonical payload" in error for error in errors))
        self.assertIn("API render case order or membership changed", errors)

    def test_api_render_decision_cannot_hide_usage_drift(self) -> None:
        manifest = valid_api_render_manifest()
        manifest.pop("integrity")
        case = manifest["cases"][0]
        case["reference_usage_prompt_tokens"] += 1
        case["reference_usage_delta"] += 1
        errors = validate_api_render_manifest(seal_manifest(manifest))
        self.assertTrue(
            any("full-server usage offset changed" in error for error in errors)
        )
        self.assertIn(
            "API render decision is inconsistent: "
            "non_tool_non_stream_render_matches_full_usage",
            errors,
        )
        self.assertIn(
            "API render qualification is inconsistent with its decisions",
            errors,
        )

    def test_api_render_requires_exact_forced_tool_schema(self) -> None:
        manifest = valid_api_render_manifest()
        manifest.pop("integrity")
        forced = next(
            case
            for case in manifest["cases"]
            if case["case_id"] == "tool_forced_image"
        )
        forced["structured_outputs"] = {"json": {"type": "object"}}
        errors = validate_api_render_manifest(seal_manifest(manifest))
        self.assertTrue(any("structured output changed" in error for error in errors))
        self.assertIn("named tool render is not bound to a JSON schema", errors)

    def test_api_render_stream_usage_absence_is_frozen(self) -> None:
        manifest = valid_api_render_manifest()
        manifest.pop("integrity")
        stream = next(
            case
            for case in manifest["cases"]
            if case["case_id"] == "stream_image"
        )
        stream["reference_usage_prompt_tokens"] = stream["prompt_tokens"]
        stream["reference_usage_delta"] = 0
        errors = validate_api_render_manifest(seal_manifest(manifest))
        self.assertIn("API render stream usage must be absent: stream_image", errors)

    def test_api_render_video_placeholder_may_include_wrapper_tokens(self) -> None:
        manifest = valid_api_render_manifest()
        manifest.pop("integrity")
        video = next(
            case
            for case in manifest["cases"]
            if case["case_id"] == "video_local_mp4"
        )
        placeholder = video["mm_placeholders"]["video"][0]
        offset = placeholder["offset"]
        video["prompt_token_ids"][offset : offset + 1] = [27, 248057, 29]
        placeholder["length"] = 3
        placeholder["token_ids_sha256"] = canonical_json_sha256(
            [27, 248057, 29]
        )
        video["prompt_tokens"] = len(video["prompt_token_ids"])
        video["prompt_token_ids_sha256"] = canonical_json_sha256(
            video["prompt_token_ids"]
        )
        video["reference_usage_prompt_tokens"] = video["prompt_tokens"]
        self.assertEqual(validate_api_render_manifest(seal_manifest(manifest)), [])


if __name__ == "__main__":
    unittest.main()
