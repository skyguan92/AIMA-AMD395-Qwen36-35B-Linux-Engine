from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify-native-vl-serving.py"
RESULT = ROOT / "benchmarks/results/native-vl-serving-v0.1.0.json"
RESULT_SIDECAR = RESULT.with_name(RESULT.name + ".sha256")


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

    def test_oracle_result_separates_http_prompt_and_private_output(self) -> None:
        prompt_ids = [1, 2, 3]
        render_prompt_ids = [1, 7, 2, 3]
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
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            "aima_amd395": {
                "prompt_token_ids_sha256": canonical(render_prompt_ids),
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
        render = {
            "prompt_tokens": 4,
            "prompt_token_ids_sha256": canonical(render_prompt_ids),
            "private_prompt_tokens": len(prompt_ids),
            "private_prompt_token_ids_sha256": canonical(prompt_ids),
            "private_prompt_matches_real_http": False,
        }
        result = self.module.oracle_case_result(
            case, render, 200, response, 1.0
        )
        self.assertTrue(result["passed"])
        response["aima_amd395"]["prompt_token_ids_sha256"] = "0" * 64
        result = self.module.oracle_case_result(
            case, render, 200, response, 1.0
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["checks"]["real_http_prompt_token_ids_sha256_exact"]
        )

    def test_cache_variant_is_deterministic_same_shape_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            changed = Path(directory) / "changed.png"
            self.module.write_cache_variant_png(first)
            self.module.write_cache_variant_png(second)
            self.module.write_cache_variant_png(changed, phase=1)
            payload = first.read_bytes()
            self.assertEqual(payload, second.read_bytes())
            self.assertNotEqual(payload, changed.read_bytes())
            self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(
                self.module.struct.unpack(">II", payload[16:24]),
                (160, 320),
            )
            self.assertEqual(
                self.module.struct.unpack(">II", changed.read_bytes()[16:24]),
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

    def test_clean_serving_evidence_is_hash_bound_and_complete(self) -> None:
        payload = RESULT.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(
            RESULT_SIDECAR.read_text(encoding="utf-8"),
            f"{digest}  {RESULT.name}\n",
        )
        result = json.loads(payload)
        integrity = result["integrity"]
        canonical_payload = {
            key: value for key, value in result.items() if key != "integrity"
        }
        canonical_bytes = json.dumps(
            canonical_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(integrity["algorithm"], "sha256")
        self.assertEqual(
            integrity["canonical_payload_sha256"],
            hashlib.sha256(canonical_bytes).hexdigest(),
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(
            result["source"]["commit"],
            "50289f1cbae150997ca82bbc054635932a2721c3",
        )
        self.assertEqual(
            result["build_info"]["source_commit"],
            result["source"]["commit"],
        )
        for component in result["source"]["files"]:
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / component["path"]).read_bytes()
                ).hexdigest(),
                component["sha256"],
            )
        self.assertEqual(
            result["binary"]["sha256"],
            "4bf377135bafe4dd0d449dc2c8563fa727ed47414eb4c7c7221ecb7e631711d0",
        )
        self.assertEqual(
            result["dependencies"]["fmha_provider"]["sha256"],
            "e5336b2d66b36c5f17aeb07ab780fa8f60a6092910f9b01b3ebf4bc31f766bb4",
        )
        for name in (
            "oracle_manifest",
            "serving_render_manifest",
            "fixture_manifest",
        ):
            component = result["dependencies"][name]
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                component["sha256"],
            )
        self.assertEqual(len(result["oracle_cases"]), 5)
        self.assertTrue(all(case["passed"] for case in result["oracle_cases"]))
        self.assertTrue(
            all(all(case["checks"].values()) for case in result["oracle_cases"])
        )
        self.assertTrue(all(result["cache_correctness"]["checks"].values()))
        self.assertTrue(all(result["launch"]["checks"].values()))
        observations = result["cache_correctness"]["observations"]
        self.assertEqual(
            [item["case_id"] for item in observations],
            [
                "image_local_a",
                "image_local_b",
                "image_local_a_restored",
                "image_data_a_equivalent",
                "image_data_a_prompt_variant",
                "image_http_a",
                "image_http_b",
                "image_http_a_restored",
                "video_local_cold",
                "video_data_equivalent",
                "mixed_local_cold",
                "mixed_local_exact",
            ],
        )
        self.assertEqual(
            result["launch"]["stopped"],
            {"event": "stopped", "model_loads": 1, "served": 17},
        )
        self.assertEqual(result["launch"]["ready"]["allowed_media_domains"], 1)
        self.assertEqual(result["raw"]["stderr"]["bytes"], 0)
        self.assertTrue(
            result["decision"]["five_private_oracle_generations_preserved"]
        )
        self.assertTrue(
            result["decision"]["five_real_http_prompt_hashes_exact"]
        )
        self.assertTrue(
            result["decision"]["five_private_prompt_boundaries_distinguished"]
        )
        self.assertTrue(
            result["decision"]["content_addressed_media_cache_qualified"]
        )
        self.assertTrue(
            result["decision"]["same_http_url_content_mutation_qualified"]
        )
        self.assertTrue(
            result["decision"]["video_transport_cache_equivalence_qualified"]
        )
        self.assertTrue(result["decision"]["mixed_cache_invariance_qualified"])
        self.assertTrue(result["decision"]["single_resident_model_load"])
        serialized = payload.decode("utf-8")
        for private_prefix in ("/home/", "/Users/", "/data/", "/tmp/"):
            self.assertNotIn(f'"{private_prefix}', serialized)
        for gate in (
            "g1_passed",
            "g2_passed",
            "g3_passed",
            "g4_passed",
            "g5_passed",
        ):
            self.assertFalse(result["decision"][gate])
        serialized = payload.decode("utf-8")
        for private_prefix in (
            "/home/",
            "/Users/",
            "/data/",
            "/tmp/aima-native",
        ):
            self.assertNotIn(private_prefix, serialized)


if __name__ == "__main__":
    unittest.main()
