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

    def test_runtime_uses_one_union_workspace_and_request_scoped_owners(self) -> None:
        build = (ROOT / "scripts/build-native-runtime.sh").read_text(
            encoding="utf-8"
        )
        workspace = (
            ROOT / "native/src/native_prefill_workspace.hip.cpp"
        ).read_text(encoding="utf-8")
        invocation = (
            ROOT / "native/src/native_prefill_invocation.cpp"
        ).read_text(encoding="utf-8")
        resident = (
            ROOT / "native/src/native_resident_engine.hip.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("q1024-text-v151", build)
        self.assertIn("--prefill-registry frozen-text", build)
        self.assertIn("FROZEN_TEXT_PREFILL_REGISTRY_CPP", build)
        self.assertIn("metrics.allocation_bytes == 915552256ULL", workspace)
        self.assertIn("includes_frozen_text_ = include_frozen_text", workspace)
        self.assertIn("NativePrefillScheduleKind::kFrozenText", invocation)
        self.assertIn("frozen_text_q1024_invocations", resident)
        self.assertIn(
            "prefill_owner(segment.bucket_tokens, vl_input == nullptr)",
            resident,
        )
        self.assertIn("prefill_owner(segment.bucket_tokens, true)", resident)
        self.assertIn("prefill_owner(first_segment.bucket_tokens, false)", resident)


if __name__ == "__main__":
    unittest.main()
