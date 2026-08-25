from __future__ import annotations

import hashlib
import importlib.util
import json
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
    def frozen_reference(
        self, root: Path, oracle: Path, *, oracle_sha256: str | None = None
    ) -> Path:
        reference = root / "correctness.json"
        reference.write_text(
            json.dumps(
                {
                    "schema": correctness.CORRECTNESS_SCHEMA,
                    "complete": True,
                    "qualified": True,
                    "engine": {"sha256": "a" * 64},
                    "cases": [
                        {
                            "context_tokens": 1024,
                            "input_token_period": [1],
                            "oracle_sha256": oracle_sha256
                            or hashlib.sha256(oracle.read_bytes()).hexdigest(),
                            "qualified": True,
                        }
                    ],
                    "exact_completion": {
                        "context_tokens": 8192,
                        "input_token_id": 1000,
                        "completion_tokens": 128,
                        "expected_token_id": 1000,
                        "output_token_ids_sha256": "b" * 64,
                        "qualified": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        return reference

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
                "reference_correctness_sha256": "e" * 64,
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
        self.assertTrue(
            correctness.report_qualified(
                payload,
                runtime_binding_sha256=binding,
                reference_correctness_sha256="e" * 64,
                **arguments,
            )
        )
        self.assertFalse(
            correctness.report_qualified(
                payload,
                runtime_binding_sha256=binding,
                reference_correctness_sha256="f" * 64,
                **arguments,
            )
        )

    def test_frozen_reference_binds_oracle_period_and_exact_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle = root / "oracle.bin"
            oracle.write_bytes(b"frozen logits")
            reference = self.frozen_reference(root, oracle)
            binding = correctness.bind_reference_correctness(
                path=reference,
                cases=[(1024, (1,), oracle)],
                exact_context=8192,
                exact_token_id=1000,
                exact_completion_tokens=128,
            )
            self.assertEqual(binding["case_count"], 1)
            self.assertEqual(binding["sha256"], correctness.sha256(reference))
            self.assertEqual(binding["exact_output_token_ids_sha256"], "b" * 64)

    def test_frozen_reference_rejects_a_different_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle = root / "oracle.bin"
            oracle.write_bytes(b"changed logits")
            reference = self.frozen_reference(
                root, oracle, oracle_sha256="c" * 64
            )
            with self.assertRaisesRegex(SystemExit, "oracle SHA-256 changed"):
                correctness.bind_reference_correctness(
                    path=reference,
                    cases=[(1024, (1,), oracle)],
                    exact_context=8192,
                    exact_token_id=1000,
                    exact_completion_tokens=128,
                )

    def test_publicize_replaces_external_runner_paths(self) -> None:
        payload = {
            "command": [
                "/tmp/candidate/aima-engine-native",
                "--model-dir",
                "/private/model",
                "--reference-logits",
                "/private/oracle.bin",
                "--report",
                "/tmp/qualification/raw/report.json",
            ],
            "provider": "/tmp/candidate/libaima-fmha-ck.so",
        }
        replaced = correctness.publicize(
            payload,
            Path("/private/model"),
            (
                ("/private/oracle.bin", "${AIMA_ORACLE_Q1024}"),
                (
                    "/tmp/candidate/aima-engine-native",
                    "${AIMA_CANDIDATE_ENGINE}",
                ),
                (
                    "/tmp/candidate",
                    "${AIMA_CANDIDATE_ENGINE_DIR}",
                ),
                ("/tmp/qualification", "${AIMA_QUALIFICATION_DIR}"),
            ),
        )
        self.assertEqual(
            replaced["command"],
            [
                "${AIMA_CANDIDATE_ENGINE}",
                "--model-dir",
                "${AIMA_MODEL_DIR}",
                "--reference-logits",
                "${AIMA_ORACLE_Q1024}",
                "--report",
                "${AIMA_QUALIFICATION_DIR}/raw/report.json",
            ],
        )
        self.assertEqual(
            replaced["provider"],
            "${AIMA_CANDIDATE_ENGINE_DIR}/libaima-fmha-ck.so",
        )

    def test_runtime_identity_arguments_are_part_of_the_runner_contract(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for option in (
            '"--aotriton-provider-sha256"',
            '"--ck-provider-sha256"',
            '"--q16384-hybrid-provider-sha256"',
            '"--vision-attention-sha256"',
            '"--reference-correctness"',
        ):
            self.assertIn(option, source)
        self.assertIn('"runtime_dependencies": runtime_dependencies', source)
        self.assertIn('"runtime_binding_sha256": runtime_binding_sha256', source)
        self.assertIn(
            '"frozen_correctness_reference": reference_correctness', source
        )


if __name__ == "__main__":
    unittest.main()
