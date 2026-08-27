from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.release_evidence import NATIVE_VL_ARCHIVE_ONLY_COMPONENTS
from aima_engine.vl_generation_layer_oracle import (
    BOUNDARY_NAMES,
    LAYER0_TAIL_BOUNDARY_SPECS,
    NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES,
    validate_generation_layer_oracle_manifest,
)
from aima_engine.vl_generation_oracle import (
    CASE_ORDER,
    MODEL_VOCABULARY_SIZE,
    validate_generation_oracle_manifest,
)
from aima_engine.vl_prefill_state_oracle import (
    STATE_COMPONENT_NAMES,
    validate_vl_prefill_state_oracle_manifest,
)
from tests.evidence_test_utils import assert_components_at_commit


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
ORACLES = ROOT / "benchmarks/oracles"
GENERATION = RESULTS / "vl-generation-oracle-v0.1.0.json"
GENERATION_ROOT = ORACLES / "vl-generation-v0.1.0"
LAYER = RESULTS / "vl-generation-layer-oracle-v0.1.0.json"
LAYER_ROOT = ORACLES / "vl-generation-layer-v0.1.0"
PREFILL_STATE = RESULTS / "vl-prefill-state-oracle-v0.1.0.json"
PREFILL_STATE_ROOT = ORACLES / "vl-prefill-state-v0.1.0"
NATIVE = RESULTS / "native-vl-generation-current-head-v0.1.0.json"
NATIVE_RAW = RESULTS / "native-vl-generation-current-head-v0.1.0-raw"
QUALIFIED_COMMIT = "bd012874027defa528279a357609b713e9069df4"
QUALIFIED_BINARY_SHA256 = (
    "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
)
EXPECTED_SHA256 = {
    GENERATION: "954c6e55389cd90390cb517224df14719f2556555ce7bf44571cae1ad1812888",
    LAYER: "70cec2c7884b5e641884037212a34f6ffc6ac1944af08d59c19dc90353915188",
    PREFILL_STATE: (
        "ec4d23bef4058dd5f0189f703214caeb006159d3c4937b7e9ea14ba9bfc82782"
    ),
    NATIVE: "dd3e185d6b49c5c2c81ce53d886d46f74b44db3fce8cb163a9a3ab3e27f783d2",
}


def assert_sidecar_and_seal(
    test: unittest.TestCase, path: Path, result: dict
) -> None:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    test.assertEqual(digest, EXPECTED_SHA256[path])
    test.assertEqual(
        path.with_name(path.name + ".sha256").read_text(encoding="utf-8"),
        f"{digest}  {path.name}\n",
    )
    canonical = {key: value for key, value in result.items() if key != "integrity"}
    canonical_bytes = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    test.assertEqual(result["integrity"]["algorithm"], "sha256")
    test.assertEqual(
        result["integrity"]["canonical_payload_sha256"],
        hashlib.sha256(canonical_bytes).hexdigest(),
    )


def assert_component_current(
    test: unittest.TestCase, component: dict, root: Path = ROOT
) -> None:
    path = root / component["path"]
    if not path.is_file():
        repository_relative = path.relative_to(ROOT)
        test.assertEqual(
            NATIVE_VL_ARCHIVE_ONLY_COMPONENTS.get(repository_relative),
            (component["sha256"], component["bytes"]),
            component["path"],
        )
        return
    test.assertEqual(path.stat().st_size, component["bytes"])
    test.assertEqual(
        hashlib.sha256(path.read_bytes()).hexdigest(), component["sha256"]
    )


class NativeVlGenerationEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generation = json.loads(GENERATION.read_bytes())
        cls.layer = json.loads(LAYER.read_bytes())
        cls.prefill_state = json.loads(PREFILL_STATE.read_bytes())
        cls.native_bytes = NATIVE.read_bytes()
        cls.native = json.loads(cls.native_bytes)

    def test_reference_oracle_closure_is_complete_and_replayable(self) -> None:
        self.assertEqual(
            validate_generation_oracle_manifest(
                self.generation, oracle_root=GENERATION_ROOT
            ),
            [],
        )
        self.assertEqual(
            validate_generation_layer_oracle_manifest(
                self.layer, oracle_root=LAYER_ROOT
            ),
            [],
        )
        self.assertEqual(
            validate_vl_prefill_state_oracle_manifest(
                self.prefill_state, oracle_root=PREFILL_STATE_ROOT
            ),
            [],
        )
        for path, result in (
            (GENERATION, self.generation),
            (LAYER, self.layer),
            (PREFILL_STATE, self.prefill_state),
        ):
            assert_sidecar_and_seal(self, path, result)
            self.assertTrue(result["complete"])
            self.assertFalse(result["source"]["dirty"])
        self.assertTrue(
            self.generation["qualified_for_native_generation_comparison"]
        )
        self.assertTrue(self.layer["qualified_for_decode_attribution"])
        self.assertTrue(
            self.prefill_state["qualified_for_state_attribution"]
        )
        for result in (self.layer, self.prefill_state):
            self.assertEqual(
                result["generation_oracle"]["sha256"],
                EXPECTED_SHA256[GENERATION],
            )
        self.assertEqual(
            tuple(case["case_id"] for case in self.layer["cases"]),
            CASE_ORDER,
        )
        self.assertEqual(
            [case["target_output_index"] for case in self.layer["cases"]],
            [14, 93],
        )
        self.assertEqual(
            tuple(case["case_id"] for case in self.prefill_state["cases"]),
            CASE_ORDER,
        )

    def test_native_result_is_exactly_source_binary_and_input_bound(self) -> None:
        result = self.native
        assert_sidecar_and_seal(self, NATIVE, result)
        self.assertEqual(
            result["schema"],
            "aima-amd395-qwen36/native-vl-generation-qualification/v1",
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertFalse(result["source"]["dirty"])
        self.assertEqual(result["source"]["commit"], QUALIFIED_COMMIT)
        self.assertEqual(result["build_info"]["source_commit"], QUALIFIED_COMMIT)
        self.assertEqual(
            result["binary"]["sha256"], QUALIFIED_BINARY_SHA256
        )
        assert_components_at_commit(
            self, result["source"]["files"], QUALIFIED_COMMIT
        )
        for name in (
            "generation_oracle",
            "fixture_manifest",
            "generation_layer_oracle",
            "vl_prefill_state_oracle",
        ):
            assert_component_current(self, result["dependencies"][name])
        self.assertEqual(
            {
                name: result["dependencies"][name]["sha256"]
                for name in (
                    "fmha_provider",
                    "aotriton_runtime",
                    "aotriton_image",
                    "vision_attention",
                )
            },
            {
                "fmha_provider": (
                    "e5336b2d66b36c5f17aeb07ab780fa8f60a6092910f9b01b3ebf4bc31f766bb4"
                ),
                "aotriton_runtime": (
                    "e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5"
                ),
                "aotriton_image": (
                    "0f3a6a2f9dee6620443ee2145ee1f8257bde65a378589952840d99bf3d485c10"
                ),
                "vision_attention": (
                    "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e"
                ),
            },
        )
        for component in result["raw"].values():
            assert_component_current(self, component, RESULTS)
        self.assertEqual(
            json.loads((NATIVE_RAW / "probe.stdout.json").read_bytes()),
            result["probe"],
        )

    def test_current_head_processor_to_output_chain_is_qualified(self) -> None:
        result = self.native
        self.assertEqual(len(result["checks"]), 62)
        self.assertTrue(all(result["checks"].values()))
        probe = result["probe"]
        self.assertTrue(probe["complete"])
        self.assertTrue(probe["qualified_for_attribution"])
        self.assertEqual(probe["model_loads"], 1)
        self.assertEqual(
            tuple(case["case_id"] for case in probe["cases"]), CASE_ORDER
        )
        for case in probe["cases"]:
            case_id = case["case_id"]
            self.assertTrue(case["complete"], case_id)
            self.assertTrue(case["prefix_exact"], case_id)
            self.assertTrue(case["native_top1_exact"], case_id)
            self.assertTrue(case["selected_reference_token"], case_id)
            self.assertEqual(
                case["selected_native_token_id"],
                case["expected_reference_token_id"],
            )
            self.assertEqual(
                case["reference_logits"]["finite_elements"],
                MODEL_VOCABULARY_SIZE,
            )
            self.assertTrue(case["reference_logits"]["top1_match"], case_id)
            self.assertLess(
                case["reference_logits"]["kl_divergence"], 0.005
            )
            for key, count in (
                ("decode_boundaries", len(BOUNDARY_NAMES)),
                (
                    "decode_linear_boundaries",
                    len(NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES),
                ),
                (
                    "decode_layer0_tail_boundaries",
                    len(LAYER0_TAIL_BOUNDARY_SPECS),
                ),
                ("decode_full_attention", 6),
                ("prefill_states", len(STATE_COMPONENT_NAMES)),
            ):
                components = case[key]
                self.assertEqual(len(components), count, (case_id, key))
                self.assertTrue(
                    all(
                        item["finite_elements"] == item["elements"]
                        and item["exact_elements"] == item["elements"]
                        for item in components
                    ),
                    (case_id, key),
                )
        for decision in (
            "two_shared_prefixes_exact",
            "two_reference_rows_bound",
            "two_native_full_vocab_finite",
            "two_decode_boundary_sets_bound",
            "two_decode_linear_boundary_sets_bound",
            "two_decode_layer0_tail_boundary_sets_bound",
            "two_decode_full_attention_sets_bound",
            "two_decode_boundary_sets_bit_exact",
            "two_prefill_state_sets_bound",
            "two_prefill_state_sets_bit_exact",
            "two_native_generation_top1_exact",
            "two_generation_logits_kld_under_0_005",
            "g1_generation_closed",
        ):
            self.assertTrue(result["decision"][decision], decision)
        self.assertEqual(result["raw"]["stderr"]["bytes"], 0)
        load = json.loads((NATIVE_RAW / "native-weight-load.json").read_bytes())
        self.assertTrue(load["complete"])
        self.assertEqual(load["shard_count"], 26)
        self.assertEqual(load["tensor_count"], 1026)
        self.assertTrue(load["gpu_payload_checksum_equal"])

    def test_generation_evidence_is_public_and_does_not_promote_gates(
        self,
    ) -> None:
        for path, result in (
            (GENERATION, self.generation),
            (LAYER, self.layer),
            (PREFILL_STATE, self.prefill_state),
            (NATIVE, self.native),
        ):
            serialized = path.read_text(encoding="utf-8")
            for prefix in ("/home/", "/Users/", "/data/", "/tmp/"):
                self.assertNotIn(prefix, serialized)
            for gate in (
                "g1_passed",
                "g2_passed",
                "g3_passed",
                "g4_passed",
                "g5_passed",
            ):
                self.assertFalse(result["decision"][gate])


if __name__ == "__main__":
    unittest.main()
