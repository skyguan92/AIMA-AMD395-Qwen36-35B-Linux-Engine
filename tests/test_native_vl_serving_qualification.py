from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify-native-vl-serving.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "native_vl_serving_qualification_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeVlServingQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_frozen_request_materialization_is_content_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.png"
            fixture.write_bytes(b"fixed image bytes")
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            case = {
                "request": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe."},
                        {
                            "type": "image",
                            "fixture": fixture.name,
                            "bytes": fixture.stat().st_size,
                            "sha256": digest,
                        },
                    ],
                }
            }
            request = self.module.build_request(case, root)
            self.assertEqual(request["model"], self.module.MODEL_ID)
            self.assertEqual(request["max_tokens"], 8)
            media = request["messages"][0]["content"][1]
            self.assertEqual(media["type"], "image_url")
            self.assertEqual(
                media["image_url"]["url"], fixture.resolve().as_uri()
            )
            fixture.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "fixture changed"):
                self.module.build_request(case, root)

    def test_oracle_result_requires_both_canonical_token_hashes(self) -> None:
        prompt_ids = [1, 2, 3]
        output_ids = [4, 5]
        canonical = lambda values: hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode()
        ).hexdigest()
        text = "answer"
        case = {
            "case_id": "synthetic",
            "processor": {
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": canonical(prompt_ids),
                "placeholders": {"image": [{"num_embeds": 1}]},
                "tensors": {
                    "image_grid_thw": {"shape": [1, 3]},
                    "pixel_values": {"shape": [4, 1536]},
                },
            },
            "boundaries": {"mrope_positions": {"position_delta": -1}},
            "generation": {
                "completion_tokens": 2,
                "output_token_ids_sha256": canonical(output_ids),
                "output_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "finish_reason": "length",
            },
        }
        response = {
            "choices": [
                {
                    "message": {"content": text},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "aima_amd395": {
                "prompt_token_ids_sha256": canonical(prompt_ids),
                "output_token_ids_canonical_sha256": canonical(output_ids),
                "model_loads": 1,
                "oracle_tensor_reads": 0,
                "vl": {
                    "enabled": True,
                    "vision_patches": 4,
                    "visual_tokens": 1,
                },
                "mrope": {"enabled": True, "position_delta": -1},
            },
        }
        result = self.module.oracle_case_result(case, 200, response, 1.0)
        self.assertTrue(result["passed"])
        response["aima_amd395"]["prompt_token_ids_sha256"] = "0" * 64
        result = self.module.oracle_case_result(case, 200, response, 1.0)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["prompt_token_ids_sha256_exact"])

    def test_cache_variant_is_deterministic_same_shape_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            self.module.write_cache_variant_png(first)
            self.module.write_cache_variant_png(second)
            payload = first.read_bytes()
            self.assertEqual(payload, second.read_bytes())
            self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(
                self.module.struct.unpack(">II", payload[16:24]),
                (160, 320),
            )
            cache_a = (
                ROOT
                / "benchmarks/fixtures/vl-capability-v0.1.0"
                / "image-transparent-160x320.png"
            )
            self.assertNotEqual(payload, cache_a.read_bytes())

    def test_publicize_replaces_qualified_dependency_paths(self) -> None:
        value = {
            "command": [
                "/qualified/fmha.so",
                "/qualified/vision.hsaco",
            ],
            "ready": {"fmha_provider_path": "/qualified/fmha.so"},
        }
        replaced = self.module.publicize(
            value,
            [
                ("/qualified/fmha.so", "${AIMA_FMHA_PROVIDER}"),
                (
                    "/qualified/vision.hsaco",
                    "${AIMA_VISION_ATTENTION_IMAGE}",
                ),
            ],
        )
        serialized = json.dumps(replaced, sort_keys=True)
        self.assertNotIn("/qualified/", serialized)
        self.assertIn("${AIMA_FMHA_PROVIDER}", serialized)
        self.assertIn("${AIMA_VISION_ATTENTION_IMAGE}", serialized)


if __name__ == "__main__":
    unittest.main()
