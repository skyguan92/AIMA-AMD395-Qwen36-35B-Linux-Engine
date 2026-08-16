from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify-native-correctness.py"
SPEC = importlib.util.spec_from_file_location("native_correctness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
correctness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(correctness)


class NativeCorrectnessQualificationTest(unittest.TestCase):
    def test_automatic_runtime_path_matches_product_discovery(self) -> None:
        self.assertEqual(
            correctness.automatic_runtime_path(
                Path("/tmp/candidate/aima-engine-native"),
                correctness.FMHA_AOTRITON_FILENAME,
            ),
            Path("/tmp/candidate/libaima-fmha-aotriton.so"),
        )
        self.assertEqual(
            correctness.automatic_runtime_path(
                Path("/tmp/portable/libexec/aima-engine.real"),
                correctness.FMHA_CK_FILENAME,
            ),
            Path("/tmp/portable/lib/libaima-fmha-ck.so"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            portable = Path(temporary)
            (portable / "bin").mkdir()
            (portable / "libexec").mkdir()
            (portable / "libexec" / "aima-engine.real").touch()
            self.assertEqual(
                correctness.automatic_runtime_path(
                    portable / "bin" / "aima-engine",
                    correctness.VISION_ATTENTION_FILENAME,
                ),
                portable / "lib" / "aima-vision-attention.hsaco",
            )

    def test_resume_rejects_a_different_runtime_binding(self) -> None:
        binding = "a" * 64
        payload = {
            "schema": "aima-amd395-qwen36/native-resident-session-probe/v1",
            "complete": True,
            "qualified": True,
            "correctness_claim": True,
            "runtime_python": False,
            "runtime_torch": False,
            "runtime_vllm": False,
            "runtime_triton": False,
            "model_loads": 1,
            "request_count": 1,
            "requests": [
                {
                    "prompt_tokens": 1024,
                    "completion_tokens": 1,
                    "first_token_certified": True,
                    "all_decode_tokens_certified": True,
                }
            ],
            "reference_logits": {
                "elements": 248320,
                "finite_elements": 248320,
                "top1_match": True,
                "qualified": True,
                "kl_divergence": 0.001,
            },
            "qualification": {
                "engine_sha256": "b" * 64,
                "oracle_sha256": "c" * 64,
                "input_token_period": [1],
                "runtime_binding_sha256": binding,
            },
        }
        arguments = {
            "context": 1024,
            "engine_sha256": "b" * 64,
            "oracle_sha256": "c" * 64,
            "input_cycle": (1,),
        }
        self.assertTrue(
            correctness.report_qualified(
                payload, runtime_binding_sha256=binding, **arguments
            )
        )
        self.assertFalse(
            correctness.report_qualified(
                payload, runtime_binding_sha256="d" * 64, **arguments
            )
        )

    def test_runtime_identity_arguments_are_part_of_the_runner_contract(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for option in (
            '"--aotriton-provider-sha256"',
            '"--ck-provider-sha256"',
            '"--q16384-hybrid-provider-sha256"',
            '"--vision-attention-sha256"',
        ):
            self.assertIn(option, source)
        self.assertIn('"runtime_dependencies": runtime_dependencies', source)
        self.assertIn('"runtime_binding_sha256": runtime_binding_sha256', source)


if __name__ == "__main__":
    unittest.main()
