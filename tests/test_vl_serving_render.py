from __future__ import annotations

import unittest

from aima_engine.vl_reference import (
    PINNED_PACKAGES,
    canonical_json_sha256,
    seal_manifest,
)
from aima_engine.vl_serving_render import (
    SERVING_RENDER_CASES,
    SERVING_RENDER_SCHEMA,
    SERVING_RENDER_SCOPE,
    validate_serving_render_manifest,
)


def valid_manifest() -> dict:
    cases = []
    for case_id in SERVING_RENDER_CASES:
        token_ids = [248045, 248056, 248046]
        cases.append(
            {
                "case_id": case_id,
                "oracle_request_sha256": "a" * 64,
                "render_transport_request_sha256": "b" * 64,
                "prompt_tokens": len(token_ids),
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": canonical_json_sha256(token_ids),
                "mm_placeholders": {
                    "image": [
                        {
                            "offset": 1,
                            "length": 1,
                            "pad_token_count": 1,
                            "token_ids_sha256": canonical_json_sha256(
                                [248056]
                            ),
                        }
                    ]
                },
                "private_prompt_tokens": 2,
                "private_prompt_token_ids_sha256": "c" * 64,
                "private_prompt_matches_real_http": False,
                "max_tokens": 8,
            }
        )
    return seal_manifest(
        {
            "schema": SERVING_RENDER_SCHEMA,
            "captured_at": "2026-08-14T00:00:00+00:00",
            "complete": True,
            "qualified": True,
            "scope": SERVING_RENDER_SCOPE,
            "host": {"label": "amd395", "hostname": "test-amd395"},
            "source": {
                "commit": "d" * 40,
                "dirty": False,
                "status_sha256": "e" * 64,
                "files": [
                    {"path": path, "bytes": 1, "sha256": "f" * 64}
                    for path in (
                        "aima_engine/vl_reference.py",
                        "aima_engine/vl_serving_render.py",
                        "scripts/capture-vllm-vl-serving-render.py",
                        "scripts/qualify-native-vl-serving.py",
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
                "fixture_manifest": {
                    "path": (
                        "benchmarks/fixtures/vl-capability-v0.1.0/"
                        "fixtures-manifest.json"
                    ),
                    "bytes": 1,
                    "sha256": "1" * 64,
                },
                "oracle_manifest": {
                    "path": "benchmarks/results/vl-oracle-manifest.json",
                    "bytes": 1,
                    "sha256": "2" * 64,
                },
            },
            "contract": {
                "content_format": "auto-resolved-string",
                "request_identity": "oracle-request-with-model-id-only-translation",
                "request_serialization": "native-serving-client-json-separators",
                "render_runtime_uses_gpu": False,
            },
            "cases": cases,
            "decision": {
                "five_serving_render_cases_5_of_5": True,
                "private_preprocessor_boundary_distinguished_5_of_5": True,
                "g1_passed": False,
                "g2_passed": False,
            },
        }
    )


class VlServingRenderTest(unittest.TestCase):
    def test_complete_manifest_is_accepted(self) -> None:
        self.assertEqual(validate_serving_render_manifest(valid_manifest()), [])

    def test_prompt_drift_is_rejected(self) -> None:
        manifest = valid_manifest()
        manifest["cases"][0]["prompt_token_ids"][0] = 7
        errors = validate_serving_render_manifest(manifest)
        self.assertTrue(any("canonical payload" in error for error in errors))
        self.assertTrue(any("prompt digest mismatch" in error for error in errors))

    def test_private_boundary_decision_cannot_hide_drift(self) -> None:
        manifest = valid_manifest()
        manifest.pop("integrity")
        case = manifest["cases"][0]
        case["private_prompt_tokens"] = case["prompt_tokens"]
        case["private_prompt_token_ids_sha256"] = case[
            "prompt_token_ids_sha256"
        ]
        errors = validate_serving_render_manifest(seal_manifest(manifest))
        self.assertTrue(any("private prompt comparison drifted" in e for e in errors))
        self.assertTrue(any("decision is inconsistent" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
