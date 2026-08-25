from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from aima_engine import __version__
from aima_engine import cli
from aima_engine.package_qualification import (
    REQUIRED_COMPONENTS,
    verify_package_qualification,
)
from aima_engine.public_hygiene import scan_bytes, scan_public_tree
from aima_engine import release_evidence
from aima_engine.vl_reference import seal_manifest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "tools/aima_chat_contract.py"
RESIDENT_PATH = ROOT / "tools/amd395-qwen36-35b-a3b-bf16-resident-chat-completions-request.py"
SKELETON_PATH = ROOT / "benchmarks/shape-lab/run_full_model_skeleton.py"
CONTEXT_POLICY_PATH = ROOT / "tools/amd395-qwen36-35b-a3b-bf16-aotriton-context-policy.py"
PORTABLE_QUALIFIER_PATH = ROOT / "scripts/qualify-native-portable-bundle.py"
EVAL_QUALIFIER_PATH = ROOT / "scripts/qualify-native-eval.py"


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
        self.assertEqual(__version__, "1.5.1")
        checks = cli.verify_components(self.config)
        self.assertGreaterEqual(len(checks), 10)
        self.assertTrue(all(item["passed"] for item in checks), checks)
        self.assertEqual(
            self.config["engine"]["sha256"],
            "0d740895a9f88ea269b945e2339c97ca2afc904a6877b159621238e1c14a9d6a",
        )

    def test_package_inputs_are_bound_to_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            component_paths: dict[str, Path] = {}
            component_records: dict[str, object] = {
                "source": {
                    "release_tag": "v2.0.0",
                    "release_commit": "a" * 40,
                    "native_source_commit": "b" * 40,
                }
            }
            for index, name in enumerate(REQUIRED_COMPONENTS):
                payload = f"synthetic-component-{index}\n".encode()
                path = root / name
                path.write_bytes(payload)
                component_paths[name] = path
                component_records[name] = {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            qualification = root / "qualification.json"
            qualification.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "qualified": True,
                        "release": "2.0.0",
                        "components": component_records,
                    }
                ),
                encoding="utf-8",
            )
            arguments = {
                "release": "2.0.0",
                "release_tag": "v2.0.0",
                "source_commit": "a" * 40,
                "native_source_commit": "b" * 40,
                "components": component_paths,
            }
            self.assertEqual(
                verify_package_qualification(qualification, **arguments), []
            )
            native_mismatch = {
                **arguments,
                "native_source_commit": "c" * 40,
            }
            self.assertIn(
                "qualification native source commit does not match executable",
                verify_package_qualification(
                    qualification, **native_mismatch
                ),
            )
            component_paths["native_engine"].write_bytes(b"changed")
            errors = verify_package_qualification(qualification, **arguments)
            self.assertIn("qualification SHA-256 mismatch: native_engine", errors)

    def test_native_vl_package_inputs_require_a_sealed_clean_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            component_paths: dict[str, Path] = {}
            component_records: dict[str, object] = {
                "source": {
                    "release_tag": "v1.5.1-native-vl.3",
                    "release_commit": "a" * 40,
                    "native_source_commit": "b" * 40,
                    "native_source_dirty": False,
                }
            }
            for index, name in enumerate(REQUIRED_COMPONENTS):
                payload = f"native-vl-component-{index}\n".encode()
                path = root / name
                path.write_bytes(payload)
                component_paths[name] = path
                component_records[name] = {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            qualification = root / "qualification.json"
            sealed = seal_manifest(
                {
                    "schema": (
                        "aima-amd395-qwen36/"
                        "native-vl-product-qualification/v1"
                    ),
                    "release": "1.5.1-native-vl.3",
                    "complete": True,
                    "qualified": True,
                    "components": component_records,
                }
            )
            qualification.write_text(json.dumps(sealed), encoding="utf-8")
            arguments = {
                "release": "1.5.1-native-vl.3",
                "release_tag": "v1.5.1-native-vl.3",
                "source_commit": "a" * 40,
                "native_source_commit": "b" * 40,
                "components": component_paths,
            }
            self.assertEqual(
                verify_package_qualification(qualification, **arguments), []
            )

            sealed["components"]["source"]["native_source_dirty"] = True
            qualification.write_text(json.dumps(sealed), encoding="utf-8")
            errors = verify_package_qualification(qualification, **arguments)
            self.assertIn("qualification native source is not clean", errors)
            self.assertTrue(
                any(error.startswith("qualification integrity failed:") for error in errors)
            )

    def test_native_vl_evidence_archive_includes_provenance_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = root / "provenance.json"
            provenance.write_text(
                json.dumps(
                    {"immutable_records": {}, "public_evidence": {}}
                ),
                encoding="utf-8",
            )
            sidecar = provenance.with_name(provenance.name + ".sha256")
            sidecar.write_text(
                f"{hashlib.sha256(provenance.read_bytes()).hexdigest()}  "
                f"{provenance.name}\n",
                encoding="utf-8",
            )
            release = release_evidence.NATIVE_VL_RELEASE
            original = release_evidence.RELEASE_RECORDS[release]
            release_evidence.RELEASE_RECORDS[release] = {
                "provenance": Path(provenance.name)
            }
            try:
                paths = release_evidence.evidence_paths(root, release)
            finally:
                release_evidence.RELEASE_RECORDS[release] = original
            self.assertIn(provenance.resolve(), paths)
            self.assertIn(sidecar.resolve(), paths)

    def test_portable_qualifier_preserves_release_provenance(self) -> None:
        scripts = str(ROOT / "scripts")
        sys.path.insert(0, scripts)
        try:
            qualifier = load_module(
                "portable_qualifier_test", PORTABLE_QUALIFIER_PATH
            )
        finally:
            sys.path.remove(scripts)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            payload = bundle / "payload.bin"
            payload.write_bytes(b"qualified payload\n")
            source = {
                "release_tag": "v2.0.0",
                "commit": "a" * 40,
                "native_commit": "b" * 40,
                "dirty": False,
            }
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "test/native-portable-bundle/v1",
                        "complete": True,
                        "release": "2.0.0",
                        "source": source,
                        "payload_bytes_excluding_manifest": (
                            payload.stat().st_size
                        ),
                        "attention_providers": {},
                        "files": [
                            {
                                "path": payload.name,
                                "type": "file",
                                "bytes": payload.stat().st_size,
                                "sha256": hashlib.sha256(
                                    payload.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            verified = qualifier.verify_manifest(bundle)
            self.assertEqual(verified["release"], "2.0.0")
            self.assertEqual(verified["source"], source)

    def test_portable_host_fingerprint_tolerates_unreadable_dmi_uuid(
        self,
    ) -> None:
        scripts = str(ROOT / "scripts")
        sys.path.insert(0, scripts)
        try:
            qualifier = load_module(
                "portable_fingerprint_test", PORTABLE_QUALIFIER_PATH
            )
        finally:
            sys.path.remove(scripts)

        def identity_bytes(path: Path) -> bytes:
            if path == Path("/etc/machine-id"):
                return b"synthetic-machine-id\n"
            raise PermissionError(path)

        with mock.patch.object(
            qualifier.Path, "is_file", autospec=True, return_value=True
        ), mock.patch.object(
            qualifier.Path, "read_bytes", autospec=True, side_effect=identity_bytes
        ):
            actual = qualifier.host_fingerprint_sha256()
        expected = hashlib.sha256(
            b"aima/native-vl/host-fingerprint/v1\0"
            b"machine-id\0synthetic-machine-id"
        ).hexdigest()
        self.assertEqual(actual, expected)

    def test_eval_reference_comparison_is_prompt_hash_bound(self) -> None:
        qualifier = load_module(
            "eval_qualifier_test", EVAL_QUALIFIER_PATH
        )
        items = [
            {"item_id": "item-0", "correct_answer": "A"},
            {"item_id": "item-1", "correct_answer": "C"},
        ]
        records = [
            {
                "item_id": "item-0",
                "prompt_token_ids_sha256": "a" * 64,
                "output_token_ids_sha256": "b" * 64,
                "parsed_answer": "A",
                "correct": True,
            },
            {
                "item_id": "item-1",
                "prompt_token_ids_sha256": "c" * 64,
                "output_token_ids_sha256": "d" * 64,
                "parsed_answer": "C",
                "correct": True,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "test/reference/v1",
                        "served_model": "test-model",
                        "records": [
                            {
                                "item_id": "item-0",
                                "correct_answer": "A",
                                "prompt_token_ids_sha256": "a" * 64,
                                "completion_token_ids_sha256": "b" * 64,
                                "first_token_id": 32,
                            },
                            {
                                "item_id": "item-1",
                                "correct_answer": "C",
                                "prompt_token_ids_sha256": "c" * 64,
                                "completion_token_ids_sha256": "e" * 64,
                                "first_token_id": 35,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            comparison = qualifier.compare_reference(path, items, records)
        self.assertEqual(comparison["prompt_token_hash_matches"], 2)
        self.assertEqual(comparison["completion_token_hash_matches"], 1)
        self.assertEqual(comparison["answer_matches"], 1)
        self.assertEqual(comparison["reference_correct"], 1)
        self.assertEqual(comparison["current_correct"], 2)
        self.assertEqual(comparison["correct_delta"], 1)
        self.assertTrue(comparison["score_nonregression_pass"])

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
            self.assertEqual(
                materialized["lanes"][0]["image_path"], str(lane0.resolve())
            )
            self.assertEqual(
                materialized["lanes"][1]["image_path"], str(lane1.resolve())
            )
            self.assertEqual(
                materialized["inputs"]["checkpoint_index"]["path"],
                str((model / "model.safetensors.index.json").resolve()),
            )

    def test_direct_checkpoint_is_the_default_contract(self) -> None:
        direct = self.config["direct_checkpoint"]
        self.assertTrue(direct["default"])
        self.assertEqual(direct["tensor_count"], 693)
        self.assertEqual(direct["payload_bytes"], 69_321_221_376)
        self.assertEqual(direct["chunk_bytes"], 128 * 1024 * 1024)
        self.assertEqual(direct["workers"], 1)
        self.assertEqual(
            direct["plan"]["path"],
            self.config["striped_image"]["template"]["path"],
        )

    def test_published_benchmark_boundary_is_explicit(self) -> None:
        result = json.loads((ROOT / "benchmarks/results/v1.0.0.json").read_text(encoding="utf-8"))
        matrix = result["cold_context_matrix"]
        self.assertEqual(len(matrix), 18)
        self.assertEqual({row["output_tokens"] for row in matrix}, {512, 1024})
        self.assertTrue(result["decision"]["all_blocking_context_floors_passed"])
        self.assertFalse(result["decision"]["raw_d275_engineering_target_complete"])
        self.assertEqual(result["correctness"]["http_usage_stop_prefix_checks"], "76/76")

        direct = json.loads(
            (ROOT / "benchmarks/results/v1.1.0.json").read_text(encoding="utf-8")
        )
        self.assertEqual(direct["release"], "1.1.0")
        self.assertEqual(direct["load_mode"], "direct_safetensors")
        self.assertEqual(len(direct["startup"]["model_load_to_api_ready_ms"]), 3)
        self.assertTrue(direct["integrity"]["all_runs_gpu_payload_checksum_equal"])
        self.assertEqual(direct["integrity"]["extra_weight_copy_bytes"], 0)

    def test_cli_surface_and_native_symbols(self) -> None:
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args(["status"]).command, "status")
        chat = parser.parse_args(
            [
                "chat",
                "--stream",
                "--tools-json",
                "tools.json",
                "--tool-choice",
                "required",
                "--no-parallel-tool-calls",
                "--api-key-file",
                "client-key.txt",
                "hello",
            ]
        )
        self.assertTrue(chat.stream)
        self.assertEqual(chat.tools_json, "tools.json")
        self.assertEqual(chat.tool_choice, "required")
        self.assertFalse(chat.parallel_tool_calls)
        self.assertEqual(chat.api_key_file, "client-key.txt")
        history_chat = parser.parse_args(
            ["chat", "--messages-json", "messages.json"]
        )
        self.assertEqual(history_chat.messages_json, "messages.json")
        serve = parser.parse_args(["serve", "--output-root", "/tmp/aima"])
        self.assertEqual(serve.output_root, "/tmp/aima")
        self.assertEqual(serve.load_mode, "direct")
        self.assertIsNone(serve.image_manifest)
        self.assertEqual(cli.direct_worker_count(serve, self.config), 1)
        invalid_workers = parser.parse_args(["serve", "--load-workers", "0"])
        with self.assertRaises(cli.UserError):
            cli.direct_worker_count(invalid_workers, self.config)
        nm = subprocess.run(
            ["nm", "-D", str(ROOT / self.config["native"]["striped_image_loader"]["path"])],
            capture_output=True,
            text=True,
            check=False,
        )
        if nm.returncode == 0:
            self.assertIn("torch_owned_striped_tensor_scatter_ingest", nm.stdout)
        direct_nm = subprocess.run(
            ["nm", "-D", str(ROOT / self.config["native"]["direct_checkpoint_loader"]["path"])],
            capture_output=True,
            text=True,
            check=False,
        )
        if direct_nm.returncode == 0:
            self.assertIn("torch_owned_safetensors_tensor_scatter_ingest", direct_nm.stdout)

    def test_installed_wheel_exposes_only_dependency_free_client(self) -> None:
        original = cli.DEFAULT_CONFIG
        cli.DEFAULT_CONFIG = ROOT / "missing-wheel-runtime-config.json"
        try:
            parser = cli.build_parser()
        finally:
            cli.DEFAULT_CONFIG = original
        command_action = next(
            action for action in parser._actions if action.dest == "command"
        )
        self.assertEqual(
            set(command_action.choices),
            {"status", "models", "chat", "shutdown"},
        )

    def test_client_api_key_file_and_authorization_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "client-key"
            token = "synthetic-" + "client-token-value-0001"
            path.write_text(token + "\n", encoding="utf-8")
            path.chmod(0o600)
            self.assertEqual(cli.load_client_api_key(str(path)), token)
            path.chmod(0o644)
            with self.assertRaises(cli.UserError):
                cli.load_client_api_key(str(path))

            path.chmod(0o600)
            link = Path(directory) / "client-key-link"
            link.symlink_to(path.name)
            with self.assertRaises(cli.UserError):
                cli.load_client_api_key(str(link))

            captured: dict[str, object] = {}

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def read(self) -> bytes:
                    return b'{"status":"ok"}'

            def urlopen(request, timeout):
                captured["authorization"] = request.get_header("Authorization")
                captured["timeout"] = timeout
                return Response()

            original = cli.urlrequest.urlopen
            cli.urlrequest.urlopen = urlopen
            try:
                self.assertEqual(
                    cli.http_json(
                        "GET",
                        "http://127.0.0.1:8000",
                        "/v1/models",
                        timeout=4.0,
                        api_key=token,
                    ),
                    {"status": "ok"},
                )
            finally:
                cli.urlrequest.urlopen = original
            self.assertEqual(captured["authorization"], f"Bearer {token}")
            self.assertEqual(captured["timeout"], 4.0)

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
        self.assertEqual(scan_public_tree(ROOT), [])

    def test_public_hygiene_rules_use_synthetic_fixtures(self) -> None:
        fixture = b'password="test-only-placeholder"\n'
        self.assertEqual(scan_bytes("fixture.txt", fixture), [])
        findings = scan_bytes(
            "fixture.txt",
            b"pass" + b'word="not-a-real-secret-123!"\nssh user@192.' + b"168.10.20\n",
        )
        self.assertEqual(
            {finding.rule for finding in findings},
            {"literal-credential", "private-ipv4"},
        )

    def test_json_and_relative_document_links_are_valid(self) -> None:
        excluded = {".git", "__pycache__", "build", "dist", "output", "state"}
        for path in ROOT.rglob("*.json"):
            if any(part in excluded for part in path.parts):
                continue
            with self.subTest(json=str(path.relative_to(ROOT))):
                json.loads(path.read_text(encoding="utf-8"))

        missing: list[str] = []
        link_pattern = re.compile(r"\]\(([^)]+)\)")
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
        environment = (
            ROOT / "packaging/systemd/aima-engine.env.example"
        ).read_text(encoding="utf-8")
        self.assertIn("/opt/aima-engine/bin/aima-engine serve", service)
        self.assertIn("--context-tokens ${AIMA_CONTEXT_TOKENS}", service)
        self.assertIn("--host ${AIMA_HOST}", service)
        self.assertIn("--port ${AIMA_PORT}", service)
        self.assertIn("--api-key-file ${AIMA_API_KEY_FILE}", service)
        self.assertIn("--request-timeout-ms ${AIMA_REQUEST_TIMEOUT_MS}", service)
        self.assertIn(
            "--allowed-local-media-path ${AIMA_ALLOWED_LOCAL_MEDIA_PATH}",
            service,
        )
        self.assertIn("--disable-http-shutdown", service)
        self.assertIn("Type=notify", service)
        self.assertIn("NotifyAccess=main", service)
        self.assertNotIn("ExecStop=", service)
        self.assertIn("TimeoutStopSec=30", service)
        self.assertIn("AIMA_CONTEXT_TOKENS=8192", environment)
        self.assertIn("AIMA_API_KEY_FILE=/etc/aima-qwen36/api-key", environment)
        self.assertIn("AIMA_REQUEST_TIMEOUT_MS=15000", environment)
        self.assertIn(
            "AIMA_ALLOWED_LOCAL_MEDIA_PATH=/srv/aima-media", environment
        )
        self.assertNotIn("AIMA_LOAD_MODE=", environment)
        self.assertNotIn("AIMA_IMAGE_MANIFEST=", environment)


if __name__ == "__main__":
    unittest.main()
