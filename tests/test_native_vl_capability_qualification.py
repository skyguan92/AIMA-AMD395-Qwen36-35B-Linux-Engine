from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from tests.evidence_test_utils import assert_components_at_commit


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify-native-vl-capabilities.py"
RESULT = ROOT / "benchmarks/results/native-vl-capability-v0.1.0.json"
RESULT_SIDECAR = RESULT.with_name(RESULT.name + ".sha256")
QUALIFIED_COMMIT = "bd012874027defa528279a357609b713e9069df4"
QUALIFIED_BINARY_SHA256 = (
    "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "native_vl_capability_qualification_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeVlCapabilityQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.payload = RESULT.read_bytes()
        cls.result = json.loads(cls.payload)

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
            "aima-amd395-qwen36/native-vl-capability-qualification/v1",
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
        for name in (
            "reference_capability_manifest",
            "api_render_manifest",
            "fixture_manifest",
        ):
            component = result["dependencies"][name]
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                component["sha256"],
            )

    def test_frozen_matrix_and_native_residency_are_complete(self) -> None:
        matrix = self.result["matrix"]
        cases = matrix["cases"]
        expected_ids = tuple(self.module.REQUIRED_API_CASES)
        self.assertEqual(tuple(case["case_id"] for case in cases), expected_ids)
        self.assertEqual(matrix["required_cases"], 30)
        self.assertEqual(matrix["success_cases"], 20)
        self.assertEqual(matrix["error_cases"], 10)
        self.assertEqual(matrix["reference_status_exact"], "30/30")
        self.assertEqual(matrix["reference_finish_reason_exact"], "20/20")
        self.assertEqual(matrix["render_prompt_tokens_exact"], "18/18")
        self.assertEqual(matrix["render_prompt_token_ids_exact"], "18/18")
        self.assertTrue(all(case["qualified"] for case in cases))
        self.assertTrue(
            all(all(case["qualification_checks"].values()) for case in cases)
        )
        self.assertEqual(
            [case["status_code"] for case in cases if case["accepted"]],
            [200] * 20,
        )
        self.assertEqual(
            [case["status_code"] for case in cases if not case["accepted"]],
            [400] * 10,
        )
        self.assertTrue(all(self.result["launch"]["checks"].values()))
        ready = self.result["launch"]["ready"]
        self.assertTrue(ready["native_vl"])
        self.assertEqual(ready["structured_token_mask_bytes"], 248_320)
        self.assertEqual(ready["visual_model_tensor_count"], 333)
        self.assertEqual(ready["visual_model_payload_bytes"], 893_142_496)
        for runtime in ("python", "torch", "triton", "vllm"):
            self.assertFalse(ready[f"runtime_{runtime}"])
        self.assertEqual(
            self.result["launch"]["stopped"],
            {"event": "stopped", "model_loads": 1, "served": 20},
        )

    def test_tool_calls_and_streams_are_structurally_complete(self) -> None:
        cases = {
            case["case_id"]: case for case in self.result["matrix"]["cases"]
        }
        for case_id in ("tool_forced_image", "tool_auto_image"):
            self.assertTrue(
                self.module.valid_inspect_visual_call(
                    cases[case_id]["response"]
                )
            )
            self.assertTrue(
                cases[case_id]["qualification_checks"][
                    "structured_tool_call"
                ]
            )
        forced = cases["tool_forced_image"]["response"]["aima_amd395"]
        self.assertEqual(forced["prompt_tokens"], 349)
        self.assertEqual(
            forced["prompt_token_ids_sha256"],
            "c00ccaf4063b7a0eb5f30ca053d3484cd2658aac57d1ef7ee79d38287d940566",
        )
        structured = forced["structured_decoding"]
        self.assertTrue(structured["enabled"])
        self.assertGreater(structured["token_selections"], 0)
        self.assertEqual(
            structured["token_mask_upload_bytes"],
            structured["token_selections"] * 248_320,
        )
        self.assertTrue(
            all(
                case["qualification_checks"].get(
                    "structured_decoding_boundary_exact", False
                )
                for case in cases.values()
                if case["accepted"]
            )
        )
        for case_id in ("stream_image", "stream_video"):
            self.assertTrue(
                cases[case_id]["qualification_checks"]["complete_sse"]
            )

    def test_decision_closes_surface_only_and_keeps_later_gates_open(self) -> None:
        decision = self.result["decision"]
        for gate in (
            "frozen_surface_matrix_30_of_30",
            "success_surfaces_20_of_20",
            "error_surfaces_10_of_10",
            "http_status_parity_30_of_30",
            "structured_vl_tool_calls_2_of_2",
            "complete_vl_sse_2_of_2",
            "single_resident_model_load",
            "render_prompt_vectors_exact_18_of_18",
            "structured_decoding_boundary_20_of_20",
            "named_tool_mask_accounting_exact",
        ):
            self.assertTrue(decision[gate])
        self.assertEqual(
            self.result["matrix"]["reference_usage_exact"], "16/18"
        )
        self.assertEqual(
            self.result["matrix"]["vl_reference_usage_exact"], "16/16"
        )
        self.assertEqual(
            self.result["matrix"]["text_vllm_usage_diagnostic"], "0/2"
        )
        self.assertTrue(decision["deterministic_vl_reference_usage_exact"])
        self.assertTrue(decision["text_usage_boundary_owned_by_g3_v151"])
        self.assertFalse(decision["deterministic_reference_usage_exact"])
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
