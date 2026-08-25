from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify-native-temperature-sampling.py"
SPEC = importlib.util.spec_from_file_location(
    "native_temperature_sampling_qualification", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampling)


class NativeTemperatureSamplingTest(unittest.TestCase):
    def test_qualification_accepts_only_complete_exact_sampling_metrics(self) -> None:
        response = {
            "usage": {"completion_tokens": 3},
            "aima_amd395": {
                "sampling": {
                    "mode": "temperature-top-p",
                    "logits_source": "exact-bf16-full-vocabulary",
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "seed_provided": True,
                    "seed": 42,
                    "token_selections": 3,
                    "logits_device_to_host_bytes": (
                        3 * sampling.BF16_LOGIT_BYTES
                    ),
                    "wall_ms": 1.0,
                }
            },
        }
        self.assertTrue(
            sampling.sampling_metrics_pass(
                response, temperature=0.8, top_p=0.9, seed=42
            )
        )
        response["aima_amd395"]["sampling"]["logits_source"] = (
            "approximate"
        )
        self.assertFalse(
            sampling.sampling_metrics_pass(
                response, temperature=0.8, top_p=0.9, seed=42
            )
        )

    def test_runtime_keeps_greedy_and_sampling_owners_separate(self) -> None:
        resident = (
            ROOT / "native/src/native_resident_engine.hip.cpp"
        ).read_text(encoding="utf-8")
        http = (ROOT / "native/src/native_http_server.cpp").read_text(
            encoding="utf-8"
        )
        build = (ROOT / "scripts/build-native-runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("request.temperature > 0.0", resident)
        self.assertIn("launch_bf16_wvsplitk(", resident)
        self.assertIn('weights.find("lm_head.weight")', resident)
        self.assertIn(
            'decode_workspace.find("native.lm_head.candidate_weights")',
            resident,
        )
        self.assertIn('metrics.decode_sampling = "temperature-top-p"', resident)
        self.assertIn("temperature must be in [0, 2]", http)
        self.assertIn("top_p requires temperature > 0", http)
        self.assertIn("native_request.sampling_seed = parsed.seed", http)
        self.assertIn('native/src/native_sampling.cpp"', build)

    def test_documented_contract_exposes_seeded_text_and_vl_sampling(self) -> None:
        api = (ROOT / "docs/API.md").read_text(encoding="utf-8")
        contract = (
            ROOT / "native/product-contract-v1.5.1-native-vl.2.json"
        ).read_text(encoding="utf-8")
        self.assertIn("Positive-temperature requests", api)
        self.assertIn("raw-weight BF16 LM-head projection", api)
        self.assertIn('"seeded_text_vl_stream_replay": true', contract)

    def test_raw_artifacts_use_the_declared_public_evidence_directory(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('output.stem + "-raw"', source)
        self.assertIn('raw_dir / "native-weight-load.json"', source)
        self.assertIn('raw_dir / "stderr.txt"', source)
        self.assertIn('f"{raw_dir.name}/native-weight-load.json"', source)
        self.assertIn('f"{raw_dir.name}/stderr.txt"', source)
        self.assertIn('"${AIMA_NATIVE_BUILD_DIR}"', source)
        self.assertIn('"${AIMA_MODEL_DIR}"', source)
        self.assertIn("ready = publicize(ready, replacements)", source)
        self.assertIn("atomic_json(load_report, load_payload)", source)

    def test_publicize_rewrites_nested_machine_paths(self) -> None:
        value = {
            "ready": {
                "provider": "/private/build/libprovider.so",
                "nested": ["/private/model/config.json"],
            }
        }
        self.assertEqual(
            sampling.publicize(
                value,
                (
                    ("/private/build", "${AIMA_NATIVE_BUILD_DIR}"),
                    ("/private/model", "${AIMA_MODEL_DIR}"),
                ),
            ),
            {
                "ready": {
                    "provider": "${AIMA_NATIVE_BUILD_DIR}/libprovider.so",
                    "nested": ["${AIMA_MODEL_DIR}/config.json"],
                }
            },
        )

    def test_qualification_is_machine_bound_to_completed_g5(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        provenance = (
            ROOT / "scripts/generate-native-vl-release-provenance.py"
        ).read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--g5-result"', source)
        self.assertIn("require_g5_result(g5_result_path)", source)
        self.assertIn('"post_g5_boundary"', source)
        self.assertIn('get("post_g5_boundary", {}).get("g5_result")', provenance)


if __name__ == "__main__":
    unittest.main()
