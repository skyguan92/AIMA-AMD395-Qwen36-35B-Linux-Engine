from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "native/aot/gfx1151/q1024-text-v151"
CURRENT = ROOT / "native/aot/gfx1151/q1024-output1"
GENERATOR_PATH = ROOT / "scripts/generate-native-decode-registry.py"
SPEC = importlib.util.spec_from_file_location("native_registry", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def routed_weight_abis(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for kernel in manifest["kernels"]:
        if kernel.get("symbol") != "fused_moe_kernel":
            continue
        arguments = kernel["launch_variants"][0]["arguments"]
        result.add(
            next(
                argument["abi_type"]
                for argument in arguments
                if argument["name"] == "topk_weights_ptr"
            )
        )
    return result


class NativeTextClosureIsolationTest(unittest.TestCase):
    def test_frozen_closure_identity_and_router_abi_are_immutable(self) -> None:
        self.assertEqual(
            sha256(FROZEN / "manifest.json"),
            registry.FROZEN_TEXT_Q1024_MANIFEST_SHA256,
        )
        self.assertEqual(
            sha256(FROZEN / "prefill-schedule.json"),
            registry.FROZEN_TEXT_Q1024_SCHEDULE_SHA256,
        )
        self.assertEqual(routed_weight_abis(FROZEN / "manifest.json"), {"*bf16"})
        self.assertEqual(routed_weight_abis(CURRENT / "manifest.json"), {"*fp32"})

    def test_registry_names_keep_frozen_and_vl_schedules_distinct(self) -> None:
        current_schedule = json.loads(
            (CURRENT / "prefill-schedule.json").read_text(encoding="utf-8")
        )
        frozen_schedule = json.loads(
            (FROZEN / "prefill-schedule.json").read_text(encoding="utf-8")
        )
        current = registry.generate_prefill_cpp(
            [(current_schedule, "c" * 64)]
        )
        frozen = registry.generate_prefill_cpp(
            [(frozen_schedule, "f" * 64)], "frozen-text"
        )
        self.assertIn("native_prefill_schedule(std::size_t context_tokens", current)
        self.assertNotIn("native_frozen_text_prefill_schedule", current)
        self.assertIn(
            "native_frozen_text_prefill_schedule(std::size_t context_tokens",
            frozen,
        )

    def test_frozen_registry_rejects_even_whitespace_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "prefill-schedule.json"
            changed.write_text(
                (FROZEN / "prefill-schedule.json").read_text(encoding="utf-8")
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "python3",
                    str(GENERATOR_PATH),
                    "--phase",
                    "prefill",
                    "--prefill-registry",
                    "frozen-text",
                    "--schedule",
                    str(changed),
                    "--aot-manifest",
                    str(FROZEN / "manifest.json"),
                    "--check",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("frozen text q1024 closure identity changed", completed.stderr)

    def test_runtime_uses_schedule_isolated_workspaces_and_request_owners(self) -> None:
        build = (ROOT / "scripts/build-native-runtime.sh").read_text(
            encoding="utf-8"
        )
        workspace = (
            ROOT / "native/src/native_prefill_workspace.hip.cpp"
        ).read_text(encoding="utf-8")
        invocation = (
            ROOT / "native/src/native_prefill_invocation.cpp"
        ).read_text(encoding="utf-8")
        linear = (
            ROOT / "native/src/native_linear_prefill.hip.cpp"
        ).read_text(encoding="utf-8")
        resident = (
            ROOT / "native/src/native_resident_engine.hip.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("q1024-text-v151", build)
        self.assertIn("--prefill-registry frozen-text", build)
        self.assertIn("FROZEN_TEXT_PREFILL_REGISTRY_CPP", build)
        self.assertIn("metrics.allocation_bytes == 669875456ULL", workspace)
        self.assertIn("includes_frozen_text_ = frozen_text_schedule", workspace)
        self.assertIn("NativePrefillScheduleKind::kFrozenText", invocation)
        q1024_owner = linear.split("const bool q1024_official_fla =", 1)[1].split(
            "const bool use_vl_rmsnorm", 1
        )[0]
        self.assertIn("launches[5].launch->symbol", q1024_owner)
        self.assertIn("merge_16x16_to_64x64_inverse_kernel", q1024_owner)
        self.assertIn("frozen_text_q1024_invocations", resident)
        self.assertIn("frozen_text_q1024_workspace", resident)
        self.assertIn(
            "owner.workspace = &frozen_text_q1024_workspace", resident
        )
        self.assertIn("hipMalloc native prefill workspace", workspace)
        self.assertNotIn("hipMemCreate", workspace)
        self.assertNotIn("release_allocation", workspace)
        self.assertIn(
            "shared_allocation_bytes < split_allocation_offset", workspace
        )
        self.assertIn(
            "tail_allocation_bytes_ = allocation_bytes_ - split_allocation_offset_",
            workspace,
        )
        self.assertIn("plan.offset >= split_allocation_offset_", workspace)
        self.assertIn("hipFree(tail_allocation_)", workspace)
        self.assertIn("owns_allocation_ = false", workspace)
        self.assertIn("owns_allocation_ = true", workspace)
        self.assertIn("owns_allocation_ && allocation_ != nullptr", workspace)
        self.assertIn("current_q1024_owner.workspace->allocation()", resident)
        self.assertIn(
            "kFrozenTextQ1024WorkspaceBytes = 669879552ULL", resident
        )
        self.assertIn("kCurrentQ1024WorkspaceBytes = 674090240ULL", resident)
        self.assertIn("kCurrentQ1024SplitOffset = 668730624ULL", resident)
        self.assertIn("kCurrentQ1024TailBytes", resident)
        self.assertIn("native frozen q1024 primary backing contract changed", resident)
        self.assertIn("native current q1024 split backing contract changed", resident)
        self.assertIn("owns_primary_allocation()", resident)
        self.assertIn("has_split_allocation()", resident)
        frozen_build = resident.index("build_frozen_text_q1024();")
        decode_workspace_build = resident.index(
            "const NativeDecodeWorkspaceMetrics decode_workspace_metrics"
        )
        plan_complete = resident.index(
            "const double plan_wall_ms = elapsed_ms(plan_started);"
        )
        current_split_build = resident.index(
            "current_q1024_owner.workspace->build("
        )
        visual_weight_load = resident.index("impl_->visual_weights.load_visual(")
        self.assertLess(frozen_build, decode_workspace_build)
        self.assertLess(plan_complete, current_split_build)
        self.assertLess(current_split_build, visual_weight_load)
        self.assertNotIn("rebind_workspace", invocation)
        self.assertNotIn("bucket->workspace.reset()", resident)
        workspace_metric = resident.split(
            "impl_->metrics.prefill_workspace_bytes =", 1
        )[1].split("impl_->metrics.mrope_position_state_bytes", 1)[0]
        self.assertIn(
            "frozen_text_q1024_workspace_metrics.physical_allocation_bytes",
            workspace_metric,
        )
        self.assertIn(
            "current_q1024_workspace_metrics.physical_allocation_bytes",
            resident,
        )
        self.assertEqual(resident.count("warm_up_q1024_text()"), 3)
        self.assertIn(
            "prefill_owner(segment.bucket_tokens, vl_input == nullptr)",
            resident,
        )
        self.assertIn("prefill_owner(segment.bucket_tokens, true)", resident)
        self.assertIn("prefill_owner(first_segment.bucket_tokens, false)", resident)

    def test_text_decode_keeps_the_v151_launch_topology(self) -> None:
        linear = (ROOT / "native/src/native_linear_layer.hip.cpp").read_text(
            encoding="utf-8"
        )
        full = (ROOT / "native/src/native_full_layer.hip.cpp").read_text(
            encoding="utf-8"
        )
        runner = (ROOT / "native/src/native_decode_runner.hip.cpp").read_text(
            encoding="utf-8"
        )
        invocation = (
            ROOT / "native/src/native_decode_invocation.cpp"
        ).read_text(encoding="utf-8")
        frozen_linear = linear.split(
            "if (!use_current_vllm_projections)", 1
        )[1].split("if (use_current_vllm_projections)", 1)[0]
        self.assertIn("for (std::size_t offset = 0; offset < 4; ++offset)", frozen_linear)
        self.assertIn("for (std::size_t offset = 6; offset < 10; ++offset)", frozen_linear)
        self.assertIn("launch_shared_silu_multiply_v151", frozen_linear)
        self.assertNotIn("run_native_decode_routed_moe", frozen_linear)
        self.assertIn("void* routed_output = frozen_routed_moe.device_pointer", full)
        self.assertIn("if (use_mrope)", full)
        self.assertIn("for (std::size_t offset = 6; offset < 10; ++offset)", full)
        self.assertIn("if (!use_mrope)", runner)
        self.assertIn("swap_linear_decode_recurrent_state_buffers", runner)
        self.assertIn("reset_linear_decode_recurrent_state_buffers", invocation)


if __name__ == "__main__":
    unittest.main()
