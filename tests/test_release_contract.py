from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from aima_engine import __version__
from aima_engine import cli


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "tools/aima_chat_contract.py"
RESIDENT_PATH = ROOT / "tools/amd395-qwen36-35b-a3b-bf16-resident-chat-completions-request.py"
SKELETON_PATH = ROOT / "benchmarks/shape-lab/run_full_model_skeleton.py"
CONTEXT_POLICY_PATH = ROOT / "tools/amd395-qwen36-35b-a3b-bf16-aotriton-context-policy.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = cli.load_config()
        cls.adapter = load_module("release_adapter_test", ADAPTER_PATH)
        cls.resident = load_module("release_resident_test", RESIDENT_PATH)
        cls.skeleton = load_module("release_skeleton_test", SKELETON_PATH)
        cls.context_policy = load_module("release_context_policy_test", CONTEXT_POLICY_PATH)

    def test_release_components_are_hash_qualified(self) -> None:
        self.assertEqual(__version__, "1.0.0")
        checks = cli.verify_components(self.config)
        self.assertGreaterEqual(len(checks), 10)
        self.assertTrue(all(item["passed"] for item in checks), checks)
        self.assertEqual(
            self.config["engine"]["sha256"],
            "79b5f070a30176af2a7a87a473fe578a15abd5177fb39b2ab9e188f66572fe0e",
        )

    def test_public_identity_and_request_subset(self) -> None:
        self.assertEqual(self.adapter.DEFAULT_MODEL_ID, "aima-amd395-qwen36-35b")
        self.adapter.validate_request(
            {
                "model": self.adapter.DEFAULT_MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 1,
            }
        )
        with self.assertRaises(SystemExit):
            self.adapter.validate_request(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                }
            )

    def _generation(self, *, requested: int, seed: int, seed_text: str, stops: list[int]) -> dict:
        result = {
            "contract": {"tokens": 4, "layers": []},
            "measurement": {
                "pipeline": {"resident_ms_per_iter": 2.0},
                "text_smoke": {
                    "generated_token_id": seed,
                    "generated_token_text": seed_text,
                    "resident_final_norm_lm_head": {"ms_per_iter": 1.0},
                },
                "decode_loop": None,
                "engine_stage_wall_time_ms": {},
            },
        }
        case = {
            "name": "prefill",
            "result": result,
            "metrics": self.skeleton.metrics_for_result(result),
            "subprocess_wall_time_ms": None,
        }
        generation = self.skeleton.generation_report(
            {
                "case_mode": "generation",
                "generation_token_count": requested,
                "prompt_token_ids": [1, 2, 3, 4],
                "decode_stop_token_ids": stops,
                "decode_sampling": "argmax",
            },
            [case],
        )
        self.assertIsInstance(generation, dict)
        return generation

    def test_terminal_stop_token_is_counted_but_not_rendered(self) -> None:
        generation = self._generation(requested=1, seed=99, seed_text="<eos>", stops=[99])
        self.resident.validate_generation_contract(generation)
        response = self.resident.build_response(
            adapter=self.adapter,
            request={"model": self.adapter.DEFAULT_MODEL_ID},
            created=1,
            output_dir=Path("/tmp/aima-release-test"),
            metadata={},
            generation=generation,
            resident_runs=[],
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"], {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5,
        })
        self.assertIn("aima_amd395", response)
        self.assertNotIn("amd395_highspeed", response)

    def test_context_policy_is_exact_length_and_prefix_aware(self) -> None:
        cold = self.context_policy.select_policy(
            prompt_tokens=65536,
            enabled=True,
            exact_prefix_cache=True,
            exact_prefix_cache_max_tokens=32768,
            fallback_layout="grouped",
        )
        self.assertTrue(cold["active"])
        self.assertEqual(cold["kv_layout"], "seq")
        self.assertEqual(cold["schedule_index"], 7)
        cache_eligible = self.context_policy.select_policy(
            prompt_tokens=32768,
            enabled=True,
            exact_prefix_cache=True,
            exact_prefix_cache_max_tokens=32768,
            fallback_layout="seq",
        )
        self.assertFalse(cache_eligible["active"])
        self.assertEqual(cache_eligible["reason"], "prefix_cache_eligible")

    def test_striped_manifest_contract_and_portable_materialization(self) -> None:
        template = json.loads(
            (ROOT / "engine/production-striped-image-manifest.json").read_text(encoding="utf-8")
        )
        self.assertTrue(template["complete"])
        self.assertEqual(len(template["entries"]), 693)
        self.assertEqual(template["layout"]["active_tensors"], 693)
        self.assertEqual(template["lanes"][0]["image_path"], "${AIMA_IMAGE_DIR}/lane0.bin")

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model = base / "model"
            model.mkdir()
            (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
            lane0 = base / "lane0.bin"
            lane1 = base / "lane1.bin"
            lane0.write_bytes(b"0")
            lane1.write_bytes(b"1")
            output = base / "manifest.json"
            cli.materialize_manifest(
                output_path=output,
                model_path=model,
                lane_paths=[lane0, lane1],
                config=self.config,
            )
            materialized = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(materialized["lanes"][0]["image_path"], str(lane0))
            self.assertEqual(materialized["lanes"][1]["image_path"], str(lane1))
            self.assertEqual(
                materialized["inputs"]["checkpoint_index"]["path"],
                str(model / "model.safetensors.index.json"),
            )

    def test_published_benchmark_boundary_is_explicit(self) -> None:
        result = json.loads((ROOT / "benchmarks/results/v1.0.0.json").read_text(encoding="utf-8"))
        matrix = result["cold_context_matrix"]
        self.assertEqual(len(matrix), 18)
        self.assertEqual({row["output_tokens"] for row in matrix}, {512, 1024})
        self.assertTrue(result["decision"]["all_blocking_context_floors_passed"])
        self.assertFalse(result["decision"]["raw_d275_engineering_target_complete"])
        self.assertEqual(result["correctness"]["http_usage_stop_prefix_checks"], "76/76")

    def test_cli_surface_and_native_symbols(self) -> None:
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args(["status"]).command, "status")
        serve = parser.parse_args(["serve", "--output-root", "/tmp/aima"])
        self.assertEqual(serve.output_root, "/tmp/aima")
        nm = subprocess.run(
            ["nm", "-D", str(ROOT / self.config["native"]["striped_image_loader"]["path"])],
            capture_output=True,
            text=True,
            check=False,
        )
        if nm.returncode == 0:
            self.assertIn("torch_owned_striped_tensor_scatter_ingest", nm.stdout)

    def test_runtime_python_preserves_virtualenv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            executable = base / "python3"
            executable.write_text("runtime\n", encoding="utf-8")
            shim = base / "python"
            shim.symlink_to(executable.name)
            args = cli.build_parser().parse_args(["doctor", "--runtime-python", str(shim)])
            selected = cli.runtime_python(args, self.config)
            self.assertEqual(selected, shim)
            self.assertTrue(selected.is_symlink())

    def test_no_private_host_or_credential_markers(self) -> None:
        forbidden = ["/home/" + "quings", "/data/home/" + "quings", "qujing" + "#$@21"]
        suffixes = {".c", ".cc", ".cpp", ".h", ".hip", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yml", ".yaml"}
        findings: list[str] = []
        excluded = {".git", "__pycache__", "build", "output", "state"}
        for path in ROOT.rglob("*"):
            if (
                not path.is_file()
                or any(part in excluded for part in path.parts)
                or path.suffix not in suffixes
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for marker in forbidden:
                if marker in text:
                    findings.append(f"{path.relative_to(ROOT)}: {marker}")
        self.assertEqual(findings, [])

    def test_json_and_relative_document_links_are_valid(self) -> None:
        excluded = {".git", "__pycache__", "build", "output", "state"}
        for path in ROOT.rglob("*.json"):
            if any(part in excluded for part in path.parts):
                continue
            with self.subTest(json=str(path.relative_to(ROOT))):
                json.loads(path.read_text(encoding="utf-8"))

        missing: list[str] = []
        link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
        for path in ROOT.rglob("*.md"):
            if any(part in excluded for part in path.parts):
                continue
            for target in link_pattern.findall(path.read_text(encoding="utf-8")):
                target = target.strip().split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                if not (path.parent / target).resolve().exists():
                    missing.append(f"{path.relative_to(ROOT)} -> {target}")
        self.assertEqual(missing, [])

    def test_apache_release_and_systemd_lifecycle_contract(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0, January 2004", license_text)
        notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("Copyright 2026 Approaching AI Authors", notice)
        service = (ROOT / "packaging/systemd/aima-engine.service").read_text(encoding="utf-8")
        self.assertIn("--output-root /var/lib/aima-qwen36", service)
        self.assertIn("ExecStop=", service)
        self.assertIn("--endpoint http://127.0.0.1:8000", service)


if __name__ == "__main__":
    unittest.main()
