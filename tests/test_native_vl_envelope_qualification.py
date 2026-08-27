from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_execution import build_http_probe_specs
from tests.evidence_test_utils import assert_components_at_commit


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/native-vl-envelope-v0.1.0.json"
RESULT_SIDECAR = RESULT.with_name(RESULT.name + ".sha256")
ENVELOPE = ROOT / "benchmarks/results/vl-capability-envelope-v0.1.0.json"
FIXTURE_ROOT = ROOT / "benchmarks/fixtures/vl-envelope-v0.1.0"
FIXTURE_MANIFEST = FIXTURE_ROOT / "fixtures-manifest.json"
QUALIFIED_COMMIT = "bd012874027defa528279a357609b713e9069df4"
QUALIFIED_BINARY_SHA256 = (
    "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
)


class NativeVlEnvelopeQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = RESULT.read_bytes()
        cls.result = json.loads(cls.payload)
        envelope = json.loads(ENVELOPE.read_text(encoding="utf-8"))
        fixtures = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
        cls.expected_probes = build_http_probe_specs(
            envelope, fixtures, FIXTURE_ROOT
        )

    def test_evidence_sidecar_and_canonical_seal(self) -> None:
        digest = hashlib.sha256(self.payload).hexdigest()
        self.assertEqual(
            RESULT_SIDECAR.read_text(encoding="utf-8"),
            f"{digest}  {RESULT.name}\n",
        )
        canonical_payload = {
            key: value
            for key, value in self.result.items()
            if key != "integrity"
        }
        canonical_bytes = json.dumps(
            canonical_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(self.result["integrity"]["algorithm"], "sha256")
        self.assertEqual(
            self.result["integrity"]["canonical_payload_sha256"],
            hashlib.sha256(canonical_bytes).hexdigest(),
        )

    def test_evidence_is_bound_to_clean_source_binary_and_inputs(self) -> None:
        result = self.result
        self.assertEqual(
            result["schema"],
            "aima-amd395-qwen36/native-vl-envelope-qualification/v1",
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(result["source"]["commit"], QUALIFIED_COMMIT)
        self.assertEqual(
            result["build_info"]["source_commit"], QUALIFIED_COMMIT
        )
        self.assertEqual(
            result["binary"]["sha256"], QUALIFIED_BINARY_SHA256
        )
        assert_components_at_commit(
            self, result["source"]["files"], QUALIFIED_COMMIT
        )
        for name in ("capability_envelope", "fixture_manifest"):
            component = result["dependencies"][name]
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                component["sha256"],
            )

    def test_frozen_execution_matrix_is_complete(self) -> None:
        result = self.result
        plan = result["execution_plan"]
        matrix = result["matrix"]
        observations = matrix["observations"]
        self.assertEqual(plan["cells"], 23)
        self.assertEqual(plan["http_observations"], 23)
        self.assertEqual(plan["client_timeout_seconds"], 2400.0)
        self.assertEqual(matrix["required_cells"], 23)
        self.assertEqual(matrix["http_observations"], 23)
        self.assertEqual(matrix["successful_observations"], 17)
        self.assertEqual(matrix["error_observations"], 6)
        self.assertEqual(matrix["qualified_observations"], 23)
        self.assertEqual(
            [item["case_id"] for item in observations],
            [item["probe_id"] for item in self.expected_probes],
        )
        self.assertTrue(all(item["qualified"] for item in observations))
        self.assertTrue(
            all(
                all(item["qualification_checks"].values())
                for item in observations
            )
        )
        self.assertEqual(
            [item["status_code"] for item in observations if item["accepted"]],
            [200] * 17,
        )
        self.assertEqual(
            [
                item["status_code"]
                for item in observations
                if not item["accepted"]
            ],
            [400] * 6,
        )

    def test_full_budget_and_continuation_dispatch_are_exact(self) -> None:
        result = self.result
        vision = result["vision_probe"]["result"]
        self.assertEqual(vision["media_items"], 16)
        self.assertEqual(vision["visual_tokens"], 262_144)
        self.assertEqual(vision["vision_patches"], 1_048_576)
        self.assertEqual(vision["vision_batch_count"], 16)
        self.assertEqual(vision["vision_max_batch_tokens"], 16_384)
        self.assertEqual(vision["vision_max_batch_patches"], 65_536)
        self.assertEqual(
            vision["finite_output_elements"],
            vision["expected_finite_output_elements"],
        )
        self.assertTrue(vision["repeat_deterministic"])

        observations = {
            item["case_id"]: item
            for item in result["matrix"]["observations"]
        }
        expected_boundaries = {
            "image_maximum_pixels": (16_402, 16_384, 1, 3, 30),
            "mixed_cross_batch_boundary": (16_415, 16_388, 2, 3, 30),
            "image_near_window_maximum": (245_820, 245_760, 15, 31, 310),
            "video_full_item_budget": (258_252, 258_048, 21, 33, 330),
        }
        for case_id, expected in expected_boundaries.items():
            native = observations[case_id]["response"]["aima_amd395"]
            vl = native["vl"]
            mrope = native["mrope"]
            self.assertEqual(
                (
                    native["prompt_tokens"],
                    vl["visual_tokens"],
                    vl["vision_batch_count"],
                    native["aot_prefill_segments"],
                    mrope["full_attention_launches"],
                ),
                expected,
            )
            self.assertEqual(
                mrope["full_attention_launches"],
                mrope["fmha_launches"],
            )
            self.assertEqual(mrope["unified_attention_launches"], 0)

        for item in observations.values():
            if not item["accepted"]:
                continue
            native = item["response"]["aima_amd395"]
            mrope = native["mrope"]
            segments = native["aot_prefill_segments"]
            expected_unified = (
                10
                if segments == 1 and native["padded_prefill_tokens"] > 0
                else 0
            )
            self.assertEqual(mrope["full_attention_launches"], segments * 10)
            self.assertEqual(
                mrope["full_attention_launches"],
                mrope["fmha_launches"]
                + mrope["unified_attention_launches"],
            )
            self.assertEqual(
                mrope["unified_attention_launches"], expected_unified
            )

    def test_single_native_residency_and_fail_closed_checks_pass(self) -> None:
        result = self.result
        self.assertTrue(all(result["processor_probe"]["checks"].values()))
        self.assertTrue(all(result["vision_probe"]["checks"].values()))
        self.assertTrue(all(result["server"]["checks"].values()))
        ready = result["server"]["ready"]
        self.assertTrue(ready["native_vl"])
        self.assertEqual(ready["context_capacity"], 262_144)
        self.assertEqual(ready["static_prefill_tokens"], 262_143)
        self.assertEqual(ready["visual_model_tensor_count"], 333)
        self.assertEqual(ready["visual_model_payload_bytes"], 893_142_496)
        self.assertEqual(ready["fmha_provider_backend"], "CK-Tile")
        self.assertEqual(
            ready["secondary_fmha_provider_backend"], "AOTriton 0.11.1"
        )
        for runtime in ("python", "torch", "triton", "vllm"):
            self.assertFalse(ready[f"runtime_{runtime}"])
        self.assertEqual(
            result["server"]["stopped"],
            {"event": "stopped", "model_loads": 1, "served": 17},
        )
        self.assertEqual(result["raw"]["processor_stderr"]["bytes"], 0)
        self.assertEqual(result["raw"]["vision_stderr"]["bytes"], 0)
        self.assertEqual(result["raw"]["server_stderr"]["bytes"], 0)

    def test_decision_closes_only_the_execution_envelope(self) -> None:
        decision = self.result["decision"]
        for gate in (
            "native_execution_qualification_complete",
            "execution_cells_23_of_23",
            "http_observations_23_of_23",
            "processor_option_boundary_qualified",
            "full_encoder_budget_vision_qualified",
            "single_resident_model_load",
        ):
            self.assertTrue(decision[gate])
        for gate in (
            "g1_passed",
            "g2_passed",
            "g3_passed",
            "g4_passed",
            "g5_passed",
        ):
            self.assertFalse(decision[gate])

    def test_evidence_contains_no_private_machine_paths(self) -> None:
        serialized = self.payload.decode("utf-8")
        for private_prefix in (
            "/home/",
            "/Users/",
            "/data/",
            "/tmp/aima-native",
        ):
            self.assertNotIn(private_prefix, serialized)


if __name__ == "__main__":
    unittest.main()
