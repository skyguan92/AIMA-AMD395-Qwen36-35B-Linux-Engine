from pathlib import Path
import copy
import importlib.util
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prefix_qualification", ROOT / "scripts/qualify-native-prefix-cache.py")
QUALIFICATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFICATION)


class NativePrefixCacheTest(unittest.TestCase):
    def test_resident_qualification_fails_closed(self) -> None:
        case = {"id": "divergent", "lookup": "prefix", "matched_tokens": 15}
        metrics = {"output_token_ids_sha256": "a" * 64, "completion_tokens": 64,
                   "prompt_tokens": 26, "aot_prefill_tokens": 11, "aot_prefill_bucket_tokens": 1024,
                   "cold_prompt_decode_tokens": 0, "ttft_ms": 100, "decode_tokens_per_second": 30,
                   "prefix_cache": {"lookup": "prefix", "matched_tokens": 15, "suffix_tokens": 11,
                                    "suffix_decode_tokens": 0, "restore_bytes": 64000000,
                                    "restore_wall_ms": 1}}
        cached = {"observations": [{"case_id": "divergent", "metrics": metrics}]}
        cold = copy.deepcopy(cached)
        cold["observations"][0]["metrics"]["prefix_cache"].update(lookup="disabled", matched_tokens=0)
        self.assertTrue(QUALIFICATION.compare_runs([case], cold, cached)[0]["pass"])
        for field, value in (("output_token_ids_sha256", "b" * 64), ("completion_tokens", 63),
                             ("aot_prefill_tokens", 26), ("cold_prompt_decode_tokens", 1)):
            mutated = copy.deepcopy(cached)
            mutated["observations"][0]["metrics"][field] = value
            self.assertFalse(QUALIFICATION.compare_runs([case], cold, mutated)[0]["pass"], field)
        for field, value in (("matched_tokens", 16), ("lookup", "exact"),
                             ("restore_bytes", 0), ("restore_wall_ms", 0), ("suffix_decode_tokens", 11)):
            mutated = copy.deepcopy(cached)
            mutated["observations"][0]["metrics"]["prefix_cache"][field] = value
            self.assertFalse(QUALIFICATION.compare_runs([case], cold, mutated)[0]["pass"], field)
        for cases, observed in (([], cached), ([case], {"observations": []}),
                                ([case, case], cached),
                                ([case], {"observations": cached["observations"] * 2})):
            with self.assertRaises(ValueError):
                QUALIFICATION.compare_runs(cases, cold, observed)

    def test_full_vocabulary_logits_gate(self) -> None:
        reference = [0.0] * 248320
        reference[42] = 10
        self.assertTrue(QUALIFICATION.compare_logits(reference, reference)["pass"])
        changed = reference.copy()
        changed[43] = 11
        self.assertFalse(QUALIFICATION.compare_logits(reference, changed)["pass"])
        changed = reference.copy()
        changed[42] = 20
        self.assertFalse(QUALIFICATION.compare_logits(reference, changed)["pass"])
        with self.assertRaises(ValueError):
            QUALIFICATION.compare_logits(reference, reference[:-1])
        changed[0] = float("nan")
        with self.assertRaises(ValueError):
            QUALIFICATION.compare_logits(reference, changed)

    def test_safe_prefix_selection_and_lru_without_gpu(self) -> None:
        compiler = shutil.which("g++") or shutil.which("clang++")
        self.assertIsNotNone(compiler, "a C++ compiler is required for prefix-cache checks")
        with tempfile.TemporaryDirectory(prefix="aima-prefix-cache-") as temporary:
            executable = Path(temporary) / "native_prefix_cache_test"
            compiled = subprocess.run(
                [compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                 "-O2", "-I", str(ROOT / "native/include"),
                 str(ROOT / "tests/native_multimodal_cache_test.cpp"),
                 str(ROOT / "native/src/native_multimodal_cache.cpp"),
                 str(ROOT / "native/src/native_media.cpp"),
                 str(ROOT / "native/src/native_remote_media.cpp"),
                 str(ROOT / "native/src/sha256.cpp"), "-lcurl", "-o", str(executable)],
                capture_output=True, text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
            self.assertIn("PASS", result.stdout)
