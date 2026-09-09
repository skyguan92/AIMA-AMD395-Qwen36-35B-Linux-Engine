from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate-native-vl-patch-product-qualification.py"
HTTP_QUALIFIER = ROOT / "scripts/qualify-native-http-control-plane.py"
G5_GENERATOR = ROOT / "scripts/generate-native-vl-patch-g5-qualification.py"
CONTRACT = ROOT / "native/product-contract-v1.5.1-native-vl.5.json"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "native_vl_patch_product_qualification_test", GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeVlPatchReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator()
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_patch_contract_binds_exact_candidate_and_inheritance_limit(self) -> None:
        contract = self.contract
        self.assertEqual(contract["release"], "1.5.1-native-vl.5")
        self.assertEqual(contract["release_tag"], "v1.5.1-native-vl.5")
        self.assertEqual(
            contract["candidate"]["native_source_commit"],
            self.generator.NATIVE_SOURCE_COMMIT,
        )
        self.assertEqual(
            contract["candidate"]["native_engine_sha256"],
            self.generator.ENGINE_SHA256,
        )
        self.assertEqual(
            set(contract["patch_scope"]["allowed_runtime_paths"]),
            self.generator.ALLOWED_RUNTIME_DELTA,
        )
        self.assertIn("never", contract["patch_scope"]["inheritance_rule"])
        self.assertFalse(contract["target"]["model_weights_in_archive"])
        self.assertEqual(
            contract["target"]["maximum_resident_memory_bytes"], 96 * 1024**3
        )

    def test_runtime_delta_is_exactly_the_declared_cpu_control_patch(self) -> None:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "diff",
                "--name-only",
                (
                    f"{self.generator.BASELINE_NATIVE_SOURCE_COMMIT}.."
                    f"{self.generator.NATIVE_SOURCE_COMMIT}"
                ),
                "--",
                *self.generator.RUNTIME_PATHS,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            set(completed.stdout.splitlines()),
            self.generator.ALLOWED_RUNTIME_DELTA,
        )
        self.assertFalse(
            any(
                path.startswith(("native/aot/", "native/generated/"))
                for path in completed.stdout.splitlines()
            )
        )

    def test_patch_release_tooling_is_checked_and_does_not_publish_hostnames(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn(GENERATOR.name, makefile)
        self.assertIn(G5_GENERATOR.name, makefile)
        self.assertIn(HTTP_QUALIFIER.name, makefile)
        http_source = HTTP_QUALIFIER.read_text(encoding="utf-8")
        chat_source = (
            ROOT / "scripts/qualify-native-chat-protocol.py"
        ).read_text(encoding="utf-8")
        self.assertIn("fingerprint_sha256", http_source)
        self.assertIn("fingerprint_sha256", chat_source)
        self.assertNotIn('"hostname":', http_source)
        self.assertNotIn('"hostname":', chat_source)
        self.assertIn("zero_timeout_incomplete_read_is_interruptible", http_source)
        self.assertIn("two_chats_execute_serially_without_rejection", http_source)
        g5_source = G5_GENERATOR.read_text(encoding="utf-8")
        self.assertIn("inherited_two_host_portable_userspace", g5_source)
        self.assertIn("results are inherited baseline", g5_source)

    def test_checkpoint_public_evidence_binds_raw_files_and_both_capacities(self) -> None:
        from aima_engine.release_evidence import _verify_checkpoint_validation
        from aima_engine.vl_reference import atomic_json, file_component, seal_manifest

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {"engine_sha256": "e" * 64, "native_source_commit": "a" * 40}
            inputs = {"gates": dict.fromkeys(("exact_safe_prefix_generation_logits",
                      "exact_safe_prefix_one_owner", "exact_text_19_cell_matrix"), True),
                      "candidate_validation": {}}
            public = {}
            for name, capacity in (("prefix", 32768), ("one_owner", 262144), ("text_matrix", None)):
                directory = root / name
                directory.mkdir()
                raw = directory / "raw.json"
                raw.write_text('{"complete": true}\n')
                artifact = file_component(raw, raw.name)
                result = {"complete": True, "qualified": True}
                if capacity is None:
                    result.update(engine={"sha256": identity["engine_sha256"]},
                                  cells=[{"pass": True, "reports": [raw.name, raw.name],
                                          "report_sha256": [artifact["sha256"]] * 2}] * 19)
                else:
                    result.update(release_eligible=True, engine_sha256=identity["engine_sha256"],
                                  build_info={"source_commit": identity["native_source_commit"]},
                                  configuration={"cache_capacity": capacity}, artifacts=[artifact])
                    result = seal_manifest(result)
                summary = directory / "summary.json"
                atomic_json(summary, result)
                inputs["candidate_validation"][name] = file_component(
                    summary, f"candidate-validation/{name}/{summary.name}"
                )
                public[name] = file_component(summary, f"{name}/{summary.name}")
            self.assertEqual(_verify_checkpoint_validation(root, inputs, public, identity), [])
            for key in ("prefix", "one_owner", "text_matrix"):
                changed = copy.deepcopy(inputs)
                changed["candidate_validation"][key]["sha256"] = "0" * 64
                self.assertTrue(_verify_checkpoint_validation(root, changed, public, identity))
            (root / "prefix/raw.json").write_text("tampered\n")
            self.assertIn("safe-prefix raw artifact differs: prefix/raw.json",
                          _verify_checkpoint_validation(root, inputs, public, identity))


class NativeVlThinkingPatchReleaseTest(unittest.TestCase):
    def test_new_contract_retains_the_frozen_runtime_allowlist(self) -> None:
        generator = load_generator()
        contract_path = ROOT / "native/product-contract-v1.5.1-native-vl.6.json"
        generator.configure_release(contract_path)
        self.assertEqual(generator.RELEASE, "1.5.1-native-vl.6")
        delta = subprocess.check_output(
            ["git", "diff", "--name-only",
             f"{generator.BASELINE_NATIVE_SOURCE_COMMIT}..{generator.NATIVE_SOURCE_COMMIT}",
             "--", *generator.RUNTIME_PATHS], cwd=ROOT, text=True,
        )
        self.assertEqual(set(delta.splitlines()), generator.ALLOWED_RUNTIME_DELTA)

    def test_old_protocol_success_cannot_qualify_the_new_default_vl_fix(self) -> None:
        generator = load_generator()
        generator.configure_release(ROOT / "native/product-contract-v1.5.1-native-vl.6.json")
        old = json.loads((ROOT / "benchmarks/results/native-chat-protocol-v1.5.1-native-vl.5.json").read_text())
        for marker in (None, False):
            with self.subTest(marker=marker):
                candidate = copy.deepcopy(old)
                if marker is not None:
                    candidate["checks"]["vl_default_thinking_stream_nonstream_parity"] = marker
                with self.assertRaises(ValueError):
                    generator.require_chat_protocol(candidate)

    def test_userspace_inventory_rejects_changed_missing_and_extra_files(self) -> None:
        spec = importlib.util.spec_from_file_location("vl6_g5_test", G5_GENERATOR)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        baseline = json.loads((ROOT / "benchmarks/results/native-portable-manifest-v1.5.1-native-vl.5.json").read_text())
        self.assertTrue(all(generator.unchanged_userspace_checks(baseline).values()))
        for mode in ("changed", "missing", "extra"):
            with self.subTest(mode=mode):
                candidate = copy.deepcopy(baseline)
                record = next(item for item in candidate["files"] if item["path"] == "lib/libamdhip64.so.7")
                if mode == "changed":
                    record["sha256"] = "0" * 64
                elif mode == "missing":
                    candidate["files"].remove(record)
                else:
                    candidate["files"].append({**record, "path": "lib/unqualified.so"})
                self.assertFalse(all(generator.unchanged_userspace_checks(candidate).values()))


class NativeVlSafePrefixReleaseTest(unittest.TestCase):
    def test_checkpoint_release_has_its_own_exact_runtime_boundary(self) -> None:
        generator = load_generator()
        generator.configure_release(ROOT / "native/product-contract-v1.5.1-native-vl.7.json")
        delta = subprocess.check_output(
            ["git", "diff", "--name-only",
             f"{generator.BASELINE_NATIVE_SOURCE_COMMIT}..{generator.NATIVE_SOURCE_COMMIT}",
             "--", *generator.RUNTIME_PATHS], cwd=ROOT, text=True,
        )
        self.assertEqual(set(delta.splitlines()), generator.ALLOWED_RUNTIME_DELTA)
        self.assertGreater(len(generator.ALLOWED_RUNTIME_DELTA), len(generator.CPU_RUNTIME_DELTA))
        generator.configure_release(ROOT / "native/product-contract-v1.5.1-native-vl.6.json")
        self.assertEqual(generator.ALLOWED_RUNTIME_DELTA, generator.CPU_RUNTIME_DELTA)

    def test_prefix_gate_rejects_incomplete_unbound_and_out_of_tolerance_records(self) -> None:
        from aima_engine.qualification_runtime import expected_binding
        generator = load_generator()
        generator.configure_release(ROOT / "native/product-contract-v1.5.1-native-vl.7.json")
        names = ["short_cold", "short_exact", "divergent_chat", "divergent_exact_sse",
                 "shared_system", "multi_turn", "append_seed", "whole_prompt_append",
                 "checkpoint_as_complete_prompt", "full_owner_after_short_checkpoint", "evicted_short",
                 "media_a", "media_b_same_geometry", "media_a_restored", "text_after_media", "performance_seed"]
        names += [f"eviction_fill_{index}" for index in range(5)]
        names += [f"performance_partial_{index}" for index in range(5)]
        boundaries = [f"boundary_{index}_partial" for index in (31, 32, 33, 1023, 1024, 1025, 8192, 8193)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.json"
            raw.write_text("{}\n")
            value = {
                "schema": "aima-amd395-qwen36/native-safe-prefix-cache/v1",
                "complete": True, "qualified": True, "release_eligible": True,
                "engine_sha256": generator.ENGINE_SHA256,
                "runtime_binding": expected_binding(generator.ENGINE_SHA256),
                "build_info": {"source_commit": generator.NATIVE_SOURCE_COMMIT},
                "source": {"checkout_clean": True, "engine_runtime_matches_checkout": True,
                           "native_source_commit": generator.NATIVE_SOURCE_COMMIT,
                           "generator_sha256": generator.sha256(ROOT / "scripts/qualify-native-prefix-cache.py"),
                           "protocol_helper_sha256": generator.sha256(ROOT / "scripts/qualify-native-chat-protocol.py")},
                "configuration": {"context_tokens": 8192, "cache_capacity": 32768,
                                  "checkpoint_limit_per_entry": 3, "checkpoint_block_tokens": 32},
                "cached_peak_memory": {"gtt_bytes": 80 * 1024**3},
                "cases": [{"case_id": name, "pass": True, "checks": {key: True for key in generator.SAFE_PREFIX_CASE_CHECKS},
                           "prefix_cache": {"matched_tokens": 15}, "suffix_aot_tokens": 11} for name in names],
                "logits": {"qualified": True,
                           "cases": [{"case_id": name, "pass": True, "top1_match": True,
                                      "elements": 248320, "kl_divergence": 0.001}
                                     for name in boundaries + [f"other_{index}" for index in range(26)]],
                           "short_checkpoint_replay": {str(index): True for index in range(4)}},
                "performance": {"median_partial_ttft_speedup": 3, "median_decode_retention": 1},
                "artifacts": [{"path": "raw.json", "bytes": raw.stat().st_size,
                               "sha256": generator.sha256(raw)}],
            }
            def check(payload):
                with patch.object(generator, "load_object", return_value=generator.seal_manifest(payload)):
                    return generator.require_safe_prefix(root / "qualification.json", 32768)
            check(value)
            for field, replacement in (("release_eligible", False), ("engine_sha256", "0" * 64),
                                       ("cases", value["cases"][:-1]), ("artifacts", []),
                                       ("runtime_binding", None), ("runtime_binding", {})):
                mutated = copy.deepcopy(value)
                mutated[field] = replacement
                with self.subTest(field=field), self.assertRaises(ValueError):
                    check(mutated)
            for path, replacement in ((["logits", "cases", 0, "kl_divergence"], 0.005),
                                      (["source", "checkout_clean"], False),
                                      (["source", "generator_sha256"], "0" * 64),
                                      (["runtime_binding", "runtime_inventory_sha256"], "0" * 64),
                                      (["configuration", "checkpoint_block_tokens"], 16),
                                      (["performance", "median_decode_retention"], 0.969),
                                      (["performance", "median_partial_ttft_speedup"], float("nan")),
                                      (["cached_peak_memory", "gtt_bytes"], 97 * 1024**3)):
                mutated = copy.deepcopy(value)
                target = mutated
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                with self.subTest(path=path), self.assertRaises(ValueError):
                    check(mutated)
            raw.write_text("modified")
            with self.assertRaises(ValueError):
                check(value)


if __name__ == "__main__":
    unittest.main()
