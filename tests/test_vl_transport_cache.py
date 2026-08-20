from __future__ import annotations

import importlib.util
from pathlib import Path
import ssl
import subprocess
import tempfile
import unittest
import urllib.request

from aima_engine.vl_local_media_server import LocalMediaServers
from aima_engine.vl_transport_cache import (
    DISABLED_REPLAY,
    ENABLED_REPLAY,
    REFERENCE_CASE_ORDER,
    build_reference_cases,
    normalize_contract_request,
)


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
CAPTURE = ROOT / "scripts/capture-vllm-vl-transport-cache.py"
QUALIFIER = ROOT / "scripts/qualify-native-vl-transport-cache.py"
TLS_GENERATOR = ROOT / "scripts/generate-vl-test-tls-material.sh"
FIXTURES = ROOT / "benchmarks/fixtures/vl-capability-v0.1.0"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VlTransportCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = load_script(PROBE, "vl_transport_cache_test_probe")
        cls.capture = load_script(CAPTURE, "vl_transport_cache_test_capture")
        cls.qualifier = load_script(
            QUALIFIER, "vl_transport_cache_test_qualifier"
        )
        fixtures = cls.probe.Fixtures(FIXTURES, "http://127.0.0.1:1")
        cls.specs = build_reference_cases(
            fixtures,
            "test-model",
            "http://127.0.0.1:2",
            "https://127.0.0.1:3",
        )

    def test_reference_cases_and_replay_plans_are_frozen(self) -> None:
        self.assertEqual(
            tuple(item["case_id"] for item in self.specs),
            REFERENCE_CASE_ORDER,
        )
        known = set(REFERENCE_CASE_ORDER)
        for replay in (ENABLED_REPLAY, DISABLED_REPLAY):
            observation_ids = [item[0] for item in replay]
            self.assertEqual(len(observation_ids), len(set(observation_ids)))
            self.assertTrue(all(item[1] in known for item in replay))
        self.assertEqual(
            [item[1] for item in ENABLED_REPLAY].count("video_content_error"),
            1,
        )
        self.assertEqual(
            [item[1] for item in DISABLED_REPLAY].count("video_content_error"),
            1,
        )

    def test_request_scoped_sampling_and_same_url_content_are_explicit(self) -> None:
        cases = {item["case_id"]: item for item in self.specs}
        self.assertNotIn(
            "media_io_kwargs", cases["video_sampling_default"]["payload"]
        )
        self.assertEqual(
            cases["video_sampling_fps_1"]["payload"]["media_io_kwargs"],
            {"video": {"fps": 1.0, "video_backend": "opencv"}},
        )
        self.assertEqual(
            cases["video_sampling_num_frames_6"]["payload"][
                "media_io_kwargs"
            ],
            {"video": {"num_frames": 6, "video_backend": "opencv"}},
        )
        content_a = cases["video_content_a"]
        content_b = cases["video_content_b"]
        self.assertEqual(content_a["payload"], content_b["payload"])
        self.assertNotEqual(content_a["replacements"], content_b["replacements"])
        self.assertEqual(content_a["payload"]["max_tokens"], 8)

    def test_contract_normalization_removes_only_served_model_identity(self) -> None:
        original = self.specs[2]["payload"]
        normalized = normalize_contract_request(original)
        self.assertEqual(normalized["model"], "${AIMA_SERVED_MODEL}")
        self.assertEqual(
            normalized["media_io_kwargs"], original["media_io_kwargs"]
        )
        self.assertEqual(original["model"], "test-model")

    def test_loopback_https_fixture_uses_explicit_test_ca(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aima-vl-tls-") as temporary:
            root = Path(temporary)
            subprocess.run(
                [str(TLS_GENERATOR), str(root)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            certificate = root / "loopback-test-ca.pem"
            private_key = root / "loopback-test-server.key"
            with LocalMediaServers(
                FIXTURES, certificate, private_key
            ) as servers:
                context = ssl.create_default_context(cafile=str(certificate))
                with urllib.request.urlopen(
                    servers.https_base + "/image-rgb-256.png",
                    context=context,
                ) as response:
                    self.assertEqual(response.status, 200)
                    self.assertGreater(len(response.read()), 0)
                servers.set_mode("video_b")
                with urllib.request.urlopen(
                    servers.http_base + "/mutable-video"
                ) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(
                        response.headers["Content-Type"], "video/x-msvideo"
                    )
                self.assertEqual(
                    servers.request_counts, {"http": 1, "https": 1}
                )

    def test_native_replay_rotates_ephemeral_capture_ca(self) -> None:
        reference = {
            "runtime": {
                "test_ca": {
                    "bytes": 1024,
                    "sha256": "a" * 64,
                    "private_key_recorded": False,
                }
            }
        }
        provenance = self.qualifier.tls_ca_provenance(reference, "b" * 64)
        self.assertEqual(provenance["sha256"], "b" * 64)
        self.assertEqual(provenance["reference_capture_sha256"], "a" * 64)
        self.assertTrue(provenance["rotated_from_reference_capture"])
        self.assertFalse(provenance["private_key_recorded"])

        same = self.qualifier.tls_ca_provenance(reference, "a" * 64)
        self.assertFalse(same["rotated_from_reference_capture"])

    def test_native_success_comparison_fails_closed(self) -> None:
        request = normalize_contract_request(self.specs[1]["payload"])
        response = {
            "choices": [
                {"message": {"content": "The"}, "finish_reason": "length"}
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "total_tokens": 101,
            },
        }
        case = {
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
                "prompt_tokens": 100,
                "prompt_token_ids_sha256": "a" * 64,
            },
        }
        metrics = {
            "model_loads": 1,
            "oracle_tensor_reads": 0,
            "runtime": "native-resident-q1024",
            "prompt_tokens": 100,
            "prompt_token_ids_sha256": "a" * 64,
            "vl": {
                "enabled": True,
                "image_count": 0,
                "video_count": 1,
                "media_count": 1,
                "vision_patches": 8,
                "visual_tokens": 2,
            },
            "mrope": {"enabled": True},
        }
        checks = self.qualifier.successful_case_checks(
            case, reference, metrics
        )
        self.assertTrue(all(checks.values()), checks)
        metrics["prompt_token_ids_sha256"] = "b" * 64
        checks = self.qualifier.successful_case_checks(
            case, reference, metrics
        )
        self.assertFalse(checks["render_prompt_token_ids_exact"])

    def test_cache_decision_checks_cover_enabled_and_disabled_surfaces(self) -> None:
        def summary(
            output: str,
            hits: int,
            misses: int,
            lookup: str,
            *,
            media_count: int = 1,
            entries: int = 3,
        ) -> dict:
            return {
                "output_token_ids_sha256": output,
                "media_cache_hits": hits,
                "media_cache_misses": misses,
                "prefix_lookup": lookup,
                "media_count": media_count,
                "media_cache_entries": entries,
                "media_cache_resident_bytes": 0 if entries == 0 else 1,
            }

        enabled_rows = {
            "https_image_cold": ("https_image", summary("h", 0, 1, "miss")),
            "https_image_exact": ("https_image", summary("h", 1, 0, "exact")),
            "video_content_a_cold": (
                "video_content_a",
                summary("a", 0, 1, "miss"),
            ),
            "video_content_b_miss": (
                "video_content_b",
                summary("b", 0, 1, "miss"),
            ),
            "video_content_a_restored": (
                "video_content_a",
                summary("a", 1, 0, "exact"),
            ),
            "video_content_a_after_error": (
                "video_content_a",
                summary("a", 1, 0, "exact"),
            ),
            "video_sampling_default": (
                "video_sampling_default",
                summary("d", 1, 0, "miss"),
            ),
            "video_sampling_default_exact": (
                "video_sampling_default",
                summary("d", 1, 0, "exact"),
            ),
            "video_sampling_fps_1": (
                "video_sampling_fps_1",
                summary("f", 0, 1, "miss"),
            ),
            "video_sampling_default_restored": (
                "video_sampling_default",
                summary("d", 1, 0, "exact"),
            ),
            "video_sampling_num_frames_6": (
                "video_sampling_num_frames_6",
                summary("n", 0, 1, "miss"),
            ),
            "video_sampling_num_frames_6_exact": (
                "video_sampling_num_frames_6",
                summary("n", 1, 0, "exact"),
            ),
            "mixed_image_video": (
                "mixed_image_video",
                summary("m", 1, 1, "miss", media_count=2),
            ),
            "mixed_video_image_reordered": (
                "mixed_video_image",
                summary("r", 2, 0, "miss", media_count=2),
            ),
            "mixed_mutated_image_video": (
                "mixed_mutated_image_video",
                summary("u", 1, 1, "miss", media_count=2),
            ),
            "mixed_image_video_restored": (
                "mixed_image_video",
                summary("m", 2, 0, "exact", media_count=2),
            ),
        }
        enabled_cases = [
            {
                "observation_id": observation_id,
                "reference_case_id": reference_id,
                "cache": cache,
                "qualified": True,
            }
            for observation_id, (reference_id, cache) in enabled_rows.items()
        ]
        error = {"error": {"type": "bad_request", "code": None}}
        enabled_cases.append(
            {
                "observation_id": "video_content_error",
                "reference_case_id": "video_content_error",
                "status_code": 400,
                "response": error,
                "cache": None,
                "qualified": True,
            }
        )
        output_by_reference = {
            reference_id: cache["output_token_ids_sha256"]
            for reference_id, cache in enabled_rows.values()
        }
        disabled_cases = []
        for observation_id, reference_id, _mode in DISABLED_REPLAY:
            if reference_id == "video_content_error":
                disabled_cases.append(
                    {
                        "observation_id": observation_id,
                        "reference_case_id": reference_id,
                        "status_code": 400,
                        "response": error,
                        "cache": None,
                        "qualified": True,
                    }
                )
                continue
            media_count = 2 if reference_id == "mixed_image_video" else 1
            disabled_cases.append(
                {
                    "observation_id": observation_id,
                    "reference_case_id": reference_id,
                    "cache": summary(
                        output_by_reference[reference_id],
                        0,
                        media_count,
                        "miss",
                        media_count=media_count,
                        entries=0,
                    ),
                    "qualified": True,
                }
            )
        checks = self.qualifier.cache_correctness_checks(
            {"cases": enabled_cases}, {"cases": disabled_cases}
        )
        self.assertTrue(all(checks.values()), checks)
        disabled_cases[0]["cache"]["media_cache_hits"] = 1
        checks = self.qualifier.cache_correctness_checks(
            {"cases": enabled_cases}, {"cases": disabled_cases}
        )
        self.assertFalse(checks["disabled_cache_always_misses"])


if __name__ == "__main__":
    unittest.main()
