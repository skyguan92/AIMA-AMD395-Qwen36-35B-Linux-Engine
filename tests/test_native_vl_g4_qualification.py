from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_reference import verify_manifest_integrity


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/vl-performance-v0.1.0.json"
RESULT_SIDECAR = RESULT.with_name(RESULT.name + ".sha256")
QUALIFIED_COMMIT = "bd012874027defa528279a357609b713e9069df4"
QUALIFIED_BINARY_SHA256 = (
    "fb5cae0ca5ffaa4bc3d418d5fb1630d822eae9d60f639ba6cc143e427c0cd1e9"
)


class NativeVlG4QualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = RESULT.read_bytes()
        cls.result = json.loads(cls.payload)

    def test_record_is_integrity_sealed_and_sidecar_bound(self) -> None:
        self.assertEqual(verify_manifest_integrity(self.result), [])
        digest = hashlib.sha256(self.payload).hexdigest()
        self.assertEqual(
            RESULT_SIDECAR.read_text(encoding="utf-8"),
            f"{digest}  {RESULT.name}\n",
        )

    def test_exact_candidate_and_fixed_reference_are_bound(self) -> None:
        identity = self.result["artifact_identity"]
        candidate = identity["candidate"]
        self.assertEqual(candidate["source_commit"], QUALIFIED_COMMIT)
        by_path = {item["path"]: item for item in candidate["files"]}
        expected_closure = {
            "aima-engine-native": QUALIFIED_BINARY_SHA256,
            "libaima-fmha-aotriton.so": (
                "e5336b2d66b36c5f17aeb07ab780fa8f60a6092910f9b01b3ebf4bc31f766bb4"
            ),
            "libaima-fmha-ck.so": (
                "0145e819869d3ea5b25661f8f11279f5e6bd3484b29e8c7910a8b30c927baa93"
            ),
            "libaima-fmha-q16384-hybrid.so": (
                "e6b8c50e76c3c7d49b8c208275234d7f4607faff250019826866f86e37fedd29"
            ),
            "libaotriton_v2.so.0.11.1": (
                "e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5"
            ),
            "aima-vision-attention.hsaco": (
                "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e"
            ),
            (
                "aotriton.images/amd-gfx11xx/flash/attn_fwd/"
                "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2"
            ): (
                "0f3a6a2f9dee6620443ee2145ee1f8257bde65a378589952840d99bf3d485c10"
            ),
        }
        self.assertEqual(
            {path: item["sha256"] for path, item in by_path.items()},
            expected_closure,
        )
        self.assertEqual(
            identity["reference"]["vllm_version"],
            "0.19.1rc1.dev300+g29e5d1020.rocm721",
        )
        self.assertEqual(
            identity["model"]["checkpoint_index_sha256"],
            "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        )
        self.assertTrue(all(identity["checks"].values()))

    def test_complete_comparable_partition_is_qualified(self) -> None:
        result = self.result
        self.assertEqual(
            result["schema"], "aima-amd395-qwen36/vl-performance/v1"
        )
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertEqual(
            result["decision"],
            "qualified_on_all_reference_available_cells",
        )
        self.assertGreaterEqual(result["pair_count"], 5)
        self.assertEqual(result["cell_count"], 20)
        self.assertEqual(result["comparable_cell_count"], 20)
        self.assertEqual(result["full_cell_count"], 23)
        self.assertEqual(result["reference_unavailable_cell_count"], 3)
        self.assertFalse(result["all_frozen_cells_performance_compared"])
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(all(result["gates"].values()))
        self.assertTrue(
            result["gates"][
                "every_decode_cell_gte_same_boundary_text_control"
            ]
        )

    def test_every_comparable_cell_passes_the_exact_thresholds(self) -> None:
        for cell in self.result["cells"]:
            with self.subTest(cell=cell["cell_id"]):
                self.assertTrue(cell["complete"])
                self.assertTrue(cell["qualified"])
                self.assertTrue(all(cell["checks"].values()))
                self.assertTrue(all(cell["gates"].values()))
                self.assertEqual(
                    len(cell["pairs"]), self.result["pair_count"]
                )
                medians = cell["paired_medians"]
                self.assertLessEqual(
                    medians["ttft_candidate_over_reference"], 1.0
                )
                self.assertLessEqual(
                    medians["total_candidate_over_reference"], 1.0
                )
                self.assertGreaterEqual(
                    medians["prefill_tps_candidate_over_reference"], 1.0
                )
                if cell["metric_applicability"]["vision_throughput"]:
                    self.assertGreaterEqual(
                        medians["vision_tps_candidate_over_reference"], 1.0
                    )
                else:
                    self.assertEqual(
                        medians["vision_cache_hit_candidate_seconds"], 0.0
                    )
                if cell["output_tokens"] > 1:
                    self.assertGreaterEqual(
                        medians["decode_tps_candidate_over_reference"], 1.0
                    )
                    self.assertGreaterEqual(
                        medians[
                            "decode_tps_candidate_over_same_boundary_text_control"
                        ],
                        1.0,
                    )
                for pair in cell["pairs"]:
                    for role in ("reference", "candidate"):
                        memory = pair["measurements"][role]["memory"]
                        self.assertGreater(memory["peak_host_rss_bytes"], 0)
                        self.assertGreater(memory["peak_gtt_used_bytes"], 0)

    def test_startup_and_reference_unavailable_ledger_are_explicit(self) -> None:
        for group in ("disabled", "enabled"):
            startup = self.result["candidate_startup"][group]
            self.assertEqual(
                len(startup["measurements_ms"]),
                self.result["pair_count"],
            )
            self.assertLessEqual(startup["median_ms"], 44_900.0)
            self.assertTrue(startup["qualified"])
        for cell in self.result["reference_unavailable"]:
            self.assertEqual(cell["status"], "reference_unavailable")
            self.assertEqual(
                cell["performance_decision"],
                "not_comparable_not_candidate_pass",
            )
        retention = self.result["text_decode_retention"]
        self.assertEqual(len(retention), 3)
        self.assertTrue(all(item["qualified"] for item in retention))

    def test_raw_pair_summaries_and_source_bindings_are_content_bound(self) -> None:
        for pair in self.result["raw_pairs"]:
            path = ROOT / pair["pair_dir"] / "summary.json"
            self.assertTrue(path.is_file())
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                pair["summary_sha256"],
            )
        for component in self.result["bindings"].values():
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                component["sha256"],
            )
        self.assertIn("diagnostic_summarizer", self.result["bindings"])
        self.assertIn("text_decode_control", self.result["bindings"])

    def test_public_record_contains_no_private_machine_paths(self) -> None:
        serialized = self.payload.decode("utf-8")
        for prefix in ("/home/", "/Users/", "/data/", "/tmp/"):
            self.assertNotIn(f'"{prefix}', serialized)


if __name__ == "__main__":
    unittest.main()
