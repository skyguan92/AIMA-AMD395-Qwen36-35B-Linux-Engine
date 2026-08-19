from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_envelope import build_envelope, validate_envelope


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/vl-capability-envelope-v0.1.0.json"
SIDECAR = RESULT.with_name(RESULT.name + ".sha256")


class VlCapabilityEnvelopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = RESULT.read_bytes()
        cls.result = json.loads(cls.payload)

    def test_hash_bound_envelope_rebuilds_from_frozen_inputs(self) -> None:
        digest = hashlib.sha256(self.payload).hexdigest()
        self.assertEqual(
            SIDECAR.read_text(encoding="utf-8"),
            f"{digest}  {RESULT.name}\n",
        )
        for component in self.result["bindings"].values():
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                component["sha256"],
            )
        rebuilt = build_envelope(
            json.loads(
                (
                    ROOT
                    / self.result["bindings"]["processor_probe"]["path"]
                ).read_text(encoding="utf-8")
            ),
            json.loads(
                (
                    ROOT
                    / self.result["bindings"]["api_capability_manifest"][
                        "path"
                    ]
                ).read_text(encoding="utf-8")
            ),
            self.result["bindings"],
            self.result["generated_at"],
        )
        self.assertEqual(rebuilt, self.result)
        self.assertEqual(validate_envelope(self.result), [])

    def test_all_processor_boundaries_and_execution_cells_are_frozen(self) -> None:
        boundaries = self.result["processor_boundaries"]
        self.assertEqual(len(boundaries["image_resize"]), 10)
        self.assertEqual(len(boundaries["video_resize"]), 8)
        self.assertEqual(len(boundaries["video_sampling"]), 7)
        self.assertEqual(len(boundaries["media_count"]), 4)
        cells = {
            cell["cell_id"]: cell for cell in self.result["execution_cells"]
        }
        self.assertEqual(len(cells), 23)
        expected = {
            "image_minimum": (64, 1),
            "image_maximum_pixels": (16_384, 1),
            "video_minimum": (4, 1),
            "video_maximum_feature_shape": (12_288, 1),
            "video_sampling_minimum": (128, 1),
            "video_sampling_typical": (640, 1),
            "video_sampling_maximum": (9_600, 1),
            "mixed_cross_batch_boundary": (16_388, 2),
            "image_count_maximum_small": (1_024, 1),
            "video_count_maximum_small": (84, 1),
            "image_near_window_maximum": (245_760, 15),
            "image_full_encoder_budget": (262_144, 16),
            "video_full_item_budget": (258_048, 21),
        }
        for case_id, (tokens, batches) in expected.items():
            self.assertEqual(cells[case_id]["aggregate_visual_tokens"], tokens)
            self.assertEqual(cells[case_id]["vision_batch_count"], batches)
        self.assertEqual(
            cells["video_typical"]["boundary_ids"],
            ["video.resize.typical"],
        )
        self.assertEqual(
            cells["video_sampling_option_conflict"]["qualification_layers"],
            ["processor"],
        )

    def test_native_contract_separates_aggregate_and_vision_batch_limits(self) -> None:
        header = (
            ROOT / "native/include/aima/native_vl_processor.h"
        ).read_text(encoding="utf-8")
        processor = (ROOT / "native/src/native_vl_processor.cpp").read_text(
            encoding="utf-8"
        )
        resident = (
            ROOT / "native/src/native_resident_engine.hip.cpp"
        ).read_text(encoding="utf-8")
        http = (ROOT / "native/src/native_http_server.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("kNativeVlAggregateTokenLimit = 262144", header)
        self.assertIn("kNativeVlVisionBatchTokenLimit = 16384", header)
        self.assertIn("native_qwen36_vision_batches", processor)
        self.assertIn(
            "for (const NativeVlVisionBatch& batch : vl_vision_batches)",
            resident,
        )
        self.assertIn("batch.visual_token_offset * kHidden", resident)
        self.assertIn("batch.patch_offset * kVisionPixelColumns", resident)
        self.assertIn("kVisionPlanCachePatchBudget", resident)
        self.assertIn("kVisionPlanCacheSharedPatchLimit", resident)
        self.assertIn("kVisionExecutionPlanCacheEntries = 8", resident)
        self.assertIn("kNativeVlVisionBatchPatchLimit", resident)
        self.assertIn(
            "kVisionRequestPlanPreparationMinPatches = 256", resident
        )
        self.assertIn(
            "kVisionRequestPlanPreparationPatchLimit = 2048", resident
        )
        self.assertIn("vision_preparation_embeddings", resident)
        self.assertIn("prepare_on_miss", resident)
        self.assertIn("retain_warmed_vision_execution_plans", resident)
        self.assertIn("warm_up_standard_vision", resident)
        self.assertIn(
            "const std::vector<NativeVlGrid> grids{{1, 64, 16}};",
            resident,
        )
        self.assertIn(
            "16, NativeVlGrid{1, 16, 16}",
            resident,
        )
        self.assertIn("vision_warmup_completed", http)
        self.assertIn("vision_image_count_warmup_patches", http)
        self.assertIn("vision_image_count_warmup_encode_wall_ms", http)
        release_all = resident.index("vision_plans.clear();")
        eviction = resident.index("vision_plans.erase(oldest);")
        construction = resident.index(
            "std::make_unique<NativeVisionPipelinePlan>"
        )
        self.assertLess(release_all, construction)
        self.assertLess(eviction, construction)
        self.assertIn('"vision_batch_count"', http)
        self.assertIn('"vision_max_batch_tokens"', http)
        self.assertIn('"vision_max_batch_patches"', http)

    def test_manifest_records_progress_without_overclaiming_gates(self) -> None:
        decision = self.result["decision"]
        self.assertTrue(decision["processor_boundary_manifest_complete"])
        self.assertFalse(decision["native_execution_qualification_complete"])
        self.assertFalse(decision["task_quality_qualification_complete"])
        self.assertFalse(decision["g1_passed"])
        self.assertFalse(decision["g2_passed"])

    def test_integrity_tamper_fails_closed(self) -> None:
        tampered = json.loads(json.dumps(self.result))
        tampered["limits"]["encoder_tokens"] -= 1
        errors = validate_envelope(tampered)
        self.assertTrue(any("canonical payload" in error for error in errors))
        self.assertIn("VL envelope encoder-token limit drifted", errors)


if __name__ == "__main__":
    unittest.main()
