from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify-native-paired-text-matrix.py"
SPEC = importlib.util.spec_from_file_location("paired_text_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
paired = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paired)


def request(
    *,
    index: int,
    context: int,
    output: int,
    prefill_tps: float,
    prefill_wall_ms: float,
    decode_tps: float,
    decode_wall_ms: float,
) -> dict[str, object]:
    return {
        "request_index": index,
        "prompt_tokens": context,
        "completion_tokens": output,
        "first_token_certified": True,
        "all_decode_tokens_certified": True,
        "oracle_tensor_reads": 0,
        "prefix_cache_lookup": "miss" if index == 1 else "exact",
        "prefix_cache_matched_tokens": 0 if index == 1 else context,
        "prefill_tokens_per_second": prefill_tps,
        "prefill_wall_ms": prefill_wall_ms,
        "decode_tokens_per_second": decode_tps,
        "decode_wall_ms": decode_wall_ms,
        "request_wall_ms": prefill_wall_ms + decode_wall_ms,
        "mrope_enabled": False,
        "mrope_position_upload_bytes": 0,
        "mrope_full_attention_launches": 0,
        "mrope_decode_steps": 0,
        "prefill_vl_unified_attention_launches": 0,
        "vl_logical_projections_enabled": False,
        "vl_logical_projection_tokens": 0,
        "vl_logical_projection_plan_count": 0,
        "vl_logical_projection_workspace_bytes": 0,
        "vl_logical_projection_plan_build_wall_ms": 0.0,
    }


def payload(
    *,
    role: str,
    pair_index: int,
    context: int,
    outputs: tuple[int, ...],
    engine_sha256: str,
    prefill_tps: float,
    prefill_wall_ms: float,
    decode_tps: float,
    decode_wall_ms: float,
) -> dict[str, object]:
    requests = [
        request(
            index=index + 1,
            context=context,
            output=output,
            prefill_tps=prefill_tps,
            prefill_wall_ms=prefill_wall_ms,
            decode_tps=decode_tps,
            decode_wall_ms=decode_wall_ms,
        )
        for index, output in enumerate(outputs)
    ]
    return {
        "schema": "aima-amd395-qwen36/native-resident-session-probe/v1",
        "complete": True,
        "model_loads": 1,
        "request_count": len(outputs),
        "runtime_python": False,
        "runtime_torch": False,
        "runtime_vllm": False,
        "runtime_triton": False,
        "load": {"command_to_ready_wall_ms": 40_000.0},
        "requests": requests,
        "qualification": {
            "engine_role": role,
            "pair_index": pair_index,
            "pair_order": list(paired.pair_order(pair_index)),
            "engine_sha256": engine_sha256,
        },
    }


class NativePairedTextMatrixTest(unittest.TestCase):
    def test_candidate_uses_automatic_context_provider_paths(self) -> None:
        build_engine = Path("/tmp/candidate/aima-engine-native")
        self.assertEqual(
            paired.automatic_runtime_path(
                build_engine, paired.FMHA_AOTRITON_FILENAME
            ),
            Path("/tmp/candidate/libaima-fmha-aotriton.so"),
        )
        portable_engine = Path(
            "/tmp/candidate/aima-engine-native-portable/libexec/"
            "aima-engine-native"
        )
        self.assertEqual(
            paired.automatic_runtime_path(
                portable_engine, paired.FMHA_CK_FILENAME
            ),
            Path(
                "/tmp/candidate/aima-engine-native-portable/lib/"
                "libaima-fmha-ck.so"
            ),
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            '"--candidate-long-context-fmha-provider-sha256"',
            source,
        )
        self.assertIn(
            '"--candidate-q16384-hybrid-provider-sha256"',
            source,
        )
        self.assertIn('"candidate": (),', source)

    def test_frozen_matrix_contains_exactly_nineteen_cells(self) -> None:
        self.assertEqual(len(paired.FROZEN_V151_FLOORS), 19)
        self.assertEqual(sum(len(outputs) for _, outputs in paired.jobs()), 19)
        self.assertEqual(paired.MINIMUM_PAIRS, 5)
        self.assertEqual(paired.DEFAULT_PAIRS, 6)
        self.assertIn((262143, 1), paired.FROZEN_V151_FLOORS)
        self.assertIn((261632, 512), paired.FROZEN_V151_FLOORS)
        self.assertIn((261120, 1024), paired.FROZEN_V151_FLOORS)

    def test_pair_order_alternates_release_and_candidate(self) -> None:
        self.assertEqual(paired.pair_order(1), ("baseline", "candidate"))
        self.assertEqual(paired.pair_order(2), ("candidate", "baseline"))
        self.assertEqual(paired.pair_order(5), ("baseline", "candidate"))
        with self.assertRaises(ValueError):
            paired.pair_order(0)

    def write_pairs(
        self,
        root: Path,
        *,
        candidate_prefill_tps: float,
        candidate_prefill_wall_ms: float,
        candidate_decode_tps: float,
        candidate_decode_wall_ms: float,
    ) -> dict[str, str]:
        identities = {"baseline": "b" * 64, "candidate": "c" * 64}
        context = 1024
        outputs = (512, 1024)
        for pair_index in range(1, 6):
            for role in ("baseline", "candidate"):
                path = paired.report_path(
                    root, context, outputs, pair_index, role
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                if role == "baseline":
                    values = (1700.0, 602.0, 34.0, 15_000.0)
                else:
                    values = (
                        candidate_prefill_tps,
                        candidate_prefill_wall_ms,
                        candidate_decode_tps,
                        candidate_decode_wall_ms,
                    )
                path.write_text(
                    json.dumps(
                        payload(
                            role=role,
                            pair_index=pair_index,
                            context=context,
                            outputs=outputs,
                            engine_sha256=identities[role],
                            prefill_tps=values[0],
                            prefill_wall_ms=values[1],
                            decode_tps=values[2],
                            decode_wall_ms=values[3],
                        )
                    ),
                    encoding="utf-8",
                )
        return identities

    def test_cell_passes_only_at_one_point_zero_paired_retention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = self.write_pairs(
                root,
                candidate_prefill_tps=1717.0,
                candidate_prefill_wall_ms=596.0,
                candidate_decode_tps=34.34,
                candidate_decode_wall_ms=14_850.0,
            )
            cell = paired.build_cell(
                output_dir=root,
                context=1024,
                outputs=(512, 1024),
                output_index=0,
                pair_count=5,
                engine_sha256=identities,
            )
            assert cell is not None
            self.assertTrue(cell["complete"])
            self.assertTrue(cell["qualified"])
            self.assertEqual(cell["pair_count"], 5)
            self.assertGreaterEqual(
                cell["paired_medians"][
                    "prefill_tps_candidate_over_baseline"
                ],
                1.0,
            )

    def test_legacy_floor_cannot_hide_a_paired_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = self.write_pairs(
                root,
                candidate_prefill_tps=1683.0,
                candidate_prefill_wall_ms=608.0,
                candidate_decode_tps=33.66,
                candidate_decode_wall_ms=15_150.0,
            )
            cell = paired.build_cell(
                output_dir=root,
                context=1024,
                outputs=(512, 1024),
                output_index=0,
                pair_count=5,
                engine_sha256=identities,
            )
            assert cell is not None
            self.assertTrue(all(cell["legacy_floor_checks"].values()))
            self.assertFalse(cell["paired_checks"]["prefill_tps"])
            self.assertFalse(cell["qualified"])

    def test_candidate_text_path_rejects_any_vl_work(self) -> None:
        clean = payload(
            role="candidate",
            pair_index=1,
            context=1024,
            outputs=(512, 1024),
            engine_sha256="c" * 64,
            prefill_tps=1700.0,
            prefill_wall_ms=602.0,
            decode_tps=34.0,
            decode_wall_ms=15_000.0,
        )
        self.assertTrue(paired.candidate_text_path_is_idle(clean))
        clean["requests"][0]["prefill_vl_unified_attention_launches"] = 1
        self.assertFalse(paired.candidate_text_path_is_idle(clean))

    def test_candidate_resume_is_bound_to_automatic_runtime_closure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = paired.report_path(
                root, 1024, (512, 1024), 1, "candidate"
            )
            path.parent.mkdir(parents=True)
            value = payload(
                role="candidate",
                pair_index=1,
                context=1024,
                outputs=(512, 1024),
                engine_sha256="c" * 64,
                prefill_tps=1700.0,
                prefill_wall_ms=602.0,
                decode_tps=34.0,
                decode_wall_ms=15_000.0,
            )
            value["qualification"].update(
                {
                    "runtime_policy": paired.CANDIDATE_RUNTIME_POLICY,
                    "runtime_binding_sha256": "d" * 64,
                }
            )
            path.write_text(json.dumps(value), encoding="utf-8")
            arguments = {
                "context": 1024,
                "outputs": (512, 1024),
                "role": "candidate",
                "pair_index": 1,
                "order": paired.pair_order(1),
                "engine_sha256": "c" * 64,
            }
            self.assertTrue(
                paired.report_complete(
                    path,
                    **arguments,
                    candidate_runtime_binding_sha256="d" * 64,
                )
            )
            self.assertFalse(
                paired.report_complete(
                    path,
                    **arguments,
                    candidate_runtime_binding_sha256="e" * 64,
                )
            )


if __name__ == "__main__":
    unittest.main()
