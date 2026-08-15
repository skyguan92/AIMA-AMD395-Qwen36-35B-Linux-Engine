from __future__ import annotations

import importlib.util
from pathlib import Path
import urllib.error
import urllib.request
import unittest

from aima_engine.vl_error_limits import (
    NATIVE_COMPATIBLE_ERROR,
    NATIVE_REPLAY,
    REFERENCE_CASE_ORDER,
    REFERENCE_ERROR_CONTRACT,
    build_reference_cases,
)
from aima_engine.vl_error_media_server import (
    LARGE_IMAGE_BYTES,
    ErrorLimitMediaServer,
)


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/probe-vllm-vl-api-capabilities.py"
QUALIFIER = ROOT / "scripts/qualify-native-vl-error-limits.py"
FIXTURES = ROOT / "benchmarks/fixtures/vl-capability-v0.1.0"
ERROR_FIXTURES = ROOT / "benchmarks/fixtures/vl-error-v0.1.0"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyMediaServer:
    http_base = "http://127.0.0.1:2"
    unreachable_base = "http://127.0.0.1:3"
    large_image_bytes = LARGE_IMAGE_BYTES


class VlErrorLimitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = load_script(PROBE, "vl_error_limits_test_probe")
        cls.qualifier = load_script(QUALIFIER, "vl_error_limits_test_qualifier")
        fixtures = cls.probe.Fixtures(FIXTURES, "http://127.0.0.1:1")
        cls.specs = build_reference_cases(
            fixtures,
            ERROR_FIXTURES,
            "test-model",
            DummyMediaServer(),
        )

    def test_reference_cases_and_native_replay_are_frozen(self) -> None:
        self.assertEqual(
            tuple(item["case_id"] for item in self.specs), REFERENCE_CASE_ORDER
        )
        observations = [item[0] for item in NATIVE_REPLAY]
        self.assertEqual(len(observations), len(set(observations)))
        self.assertEqual(len(NATIVE_REPLAY), 13)
        self.assertEqual(NATIVE_REPLAY[-1], ("rgba_red_after_errors", "rgba_background_red"))
        self.assertTrue(
            all(reference_id in REFERENCE_CASE_ORDER for _, reference_id in NATIVE_REPLAY)
        )

    def test_request_level_merge_and_rgba_surfaces_are_exact(self) -> None:
        cases = {item["case_id"]: item for item in self.specs}
        self.assertNotIn("media_io_kwargs", cases["rgba_default_white"]["payload"])
        self.assertEqual(
            cases["rgba_background_red"]["payload"]["media_io_kwargs"],
            {"image": {"rgba_background_color": [255, 0, 0]}},
        )
        self.assertNotIn(
            "media_io_kwargs", cases["video_sampling_default"]["payload"]
        )
        self.assertEqual(
            cases["video_sampling_empty_mapping"]["payload"]["media_io_kwargs"],
            {"video": {}},
        )
        self.assertEqual(
            cases["video_sampling_default"]["payload"]["messages"],
            cases["video_sampling_empty_mapping"]["payload"]["messages"],
        )
        long_replacement = next(
            iter(cases["video_long_duration"]["replacements"].values())
        )
        self.assertEqual(long_replacement["fixture"], "video_long_duration_low_fps")
        self.assertEqual(long_replacement["bytes"], 169_626)

    def test_all_five_error_boundaries_are_explicit(self) -> None:
        cases = {item["case_id"]: item for item in self.specs}
        for case_id in REFERENCE_CASE_ORDER[5:]:
            self.assertFalse(cases[case_id]["expected_accept"])
            self.assertIn("error", cases[case_id]["surfaces"])
        large = cases["oversize_image_remote"]
        metadata = next(iter(large["replacements"].values()))
        self.assertEqual(metadata["bytes"], 64 * 1024 * 1024 + 1)

    def test_loopback_error_server_is_bounded_and_observable(self) -> None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with ErrorLimitMediaServer(slow_image_seconds=0.01) as server:
            with opener.open(server.http_base + "/empty-image", timeout=1) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), b"")
            with opener.open(server.http_base + "/empty-video", timeout=1) as response:
                self.assertEqual(response.read(), b"")
            with opener.open(server.http_base + "/slow-image", timeout=1) as response:
                self.assertEqual(response.read(), b"x")
            with self.assertRaises(urllib.error.URLError):
                opener.open(server.unreachable_base + "/image", timeout=1)
            statistics = server.statistics
            self.assertEqual(
                statistics["requests"],
                {
                    "empty_image": 1,
                    "empty_video": 1,
                    "large_image": 0,
                    "slow_image": 1,
                },
            )
            self.assertEqual(statistics["bytes_sent"]["slow_image"], 1)

    def test_error_comparison_freezes_reference_and_native_contracts(self) -> None:
        request = {"model": "${AIMA_SERVED_MODEL}"}
        native_response = {
            "error": {
                "type": NATIVE_COMPATIBLE_ERROR[1],
                "code": NATIVE_COMPATIBLE_ERROR[2],
                "message": "invalid media",
            }
        }
        reference_contract = REFERENCE_ERROR_CONTRACT["empty_image_remote"]
        reference_response = {
            "error": {
                "type": reference_contract[1],
                "code": reference_contract[2],
                "message": "invalid media",
            }
        }
        case = {
            "passed": True,
            "status_code": NATIVE_COMPATIBLE_ERROR[0],
            "request": request,
            "response": native_response,
        }
        reference = {
            "case_id": "empty_image_remote",
            "status_code": reference_contract[0],
            "request": request,
            "response": reference_response,
        }
        checks = self.qualifier.rejected_case_checks(case, reference)
        self.assertTrue(all(checks.values()))
        reference["response"] = {
            "error": {
                "type": reference_contract[1],
                "code": 499,
                "message": "invalid media",
            }
        }
        self.assertFalse(
            self.qualifier.rejected_case_checks(case, reference)[
                "reference_contract_exact"
            ]
        )

    def test_cache_checks_fail_closed(self) -> None:
        def cache(
            *,
            hits: int,
            misses: int,
            lookup: str,
            output: str,
            entries: int,
            media_count: int = 1,
        ) -> dict[str, object]:
            return {
                "media_cache_hits": hits,
                "media_cache_misses": misses,
                "prefix_lookup": lookup,
                "output_token_ids_sha256": output,
                "media_cache_entries": entries,
                "media_count": media_count,
            }

        values = {
            "rgba_default_cold": cache(
                hits=0, misses=1, lookup="miss", output="white", entries=1
            ),
            "rgba_red_miss": cache(
                hits=0, misses=1, lookup="miss", output="red", entries=2
            ),
            "rgba_default_restored": cache(
                hits=1, misses=0, lookup="exact", output="white", entries=2
            ),
            "video_default_cold": cache(
                hits=0, misses=1, lookup="miss", output="video", entries=3
            ),
            "video_empty_mapping_exact": cache(
                hits=1, misses=0, lookup="exact", output="video", entries=3
            ),
            "video_default_restored": cache(
                hits=1, misses=0, lookup="exact", output="video", entries=3
            ),
            "video_long_duration": cache(
                hits=0, misses=1, lookup="miss", output="long", entries=4
            ),
            "rgba_red_after_errors": cache(
                hits=1, misses=0, lookup="exact", output="red", entries=4
            ),
        }
        cases = []
        for observation_id, _ in NATIVE_REPLAY:
            cases.append(
                {
                    "observation_id": observation_id,
                    "qualified": True,
                    "cache": values.get(observation_id),
                }
            )
        checks = self.qualifier.cache_correctness_checks({"cases": cases})
        self.assertTrue(all(checks.values()))
        cases[0]["qualified"] = False
        self.assertFalse(
            self.qualifier.cache_correctness_checks({"cases": cases})[
                "all_observations_reference_exact"
            ]
        )


if __name__ == "__main__":
    unittest.main()
