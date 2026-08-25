from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate-native-vl-product-qualification.py"
CONTRACT = ROOT / "native/product-contract-v1.5.1-native-vl.2.json"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "native_vl_g5_product_qualification_test", GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeVlG5ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            cls.generator = load_generator()
        finally:
            sys.path.remove(str(ROOT / "scripts"))
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_release_candidate_contract_is_exact_and_fail_closed(self) -> None:
        contract = self.contract
        self.assertEqual(
            contract["schema"],
            "aima-amd395-qwen36/native-vl-product-contract/v1",
        )
        self.assertEqual(contract["release"], "1.5.1-native-vl.2")
        self.assertEqual(contract["release_tag"], "v1.5.1-native-vl.2")
        self.assertEqual(contract["engine_version"], "1.5.1-native")
        self.assertFalse(contract["target"]["model_weights_in_archive"])
        self.assertEqual(
            contract["target"]["maximum_resident_memory_bytes"],
            96 * 1024**3,
        )
        self.assertEqual(contract["model"]["total_tensor_count"], 1026)
        self.assertEqual(contract["capability"]["maximum_total_tokens"], 262144)
        self.assertTrue(contract["candidate"]["single_resident_process"])
        self.assertTrue(
            contract["candidate"]["ready_includes_language_and_vision"]
        )
        self.assertEqual(
            set(contract["promotion_gates"]["g5_required"]),
            {
                "make check",
                "make security-scan",
                "make verify-evidence",
                "isolated portable bundle",
                "second AMD395 host",
                "long resident mixed-workload soak",
                "rollback to the exact v1.5.1 portable baseline",
            },
        )

    def test_package_input_generator_binds_release_tooling(self) -> None:
        inputs = self.generator.DEFAULT_INPUTS
        for name in (
            "product_contract",
            "product_qualification_generator",
            "package_script",
            "bundle_manifest_generator",
            "bundle_qualifier",
            "temperature_sampling_qualifier",
            "resident_soak_qualifier",
            "rollback_qualifier",
            "release_gates_qualifier",
            "g5_qualification_generator",
            "release_provenance_generator",
            "package_input_verifier",
            "bundle_closure",
            "package_qualification",
            "public_hygiene",
            "release_evidence",
            "makefile",
            "systemd_service",
            "systemd_environment",
        ):
            with self.subTest(name=name):
                self.assertTrue(inputs[name].is_file())
        self.assertTrue(
            all(self.generator.source_architecture_checks(inputs).values())
        )
        self.assertTrue(
            self.contract["capability"]["sampling"][
                "seeded_text_vl_stream_replay"
            ]
        )
        self.assertEqual(
            self.contract["post_g5_extension_gates"][
                "temperature_sampling"
            ],
            "benchmarks/results/native-temperature-sampling-v0.1.0.json",
        )
        self.assertIn(
            "after",
            self.contract["post_g5_extension_gates"]["ordering"],
        )

    def test_archive_and_isolated_vl_qualification_are_part_of_check(self) -> None:
        package = (ROOT / "scripts/package-native-foundation.sh").read_text(
            encoding="utf-8"
        )
        qualifier = (
            ROOT / "scripts/qualify-native-portable-bundle.py"
        ).read_text(encoding="utf-8")
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn('"${ROOT}"/benchmarks/results/*.json.sha256', package)
        self.assertIn(
            '"${ROOT}/benchmarks/results/native-vl-envelope-v0.1.0-raw"',
            package,
        )
        self.assertNotIn("AIMA_RELEASE_COMMIT", package)
        self.assertIn("generate-native-vl-product-qualification.py", makefile)
        self.assertIn("generate-native-bundle-manifest.py", makefile)
        self.assertIn("qualify-native-vl-resident-soak.py", makefile)
        self.assertIn("qualify-native-vl-rollback.py", makefile)
        self.assertIn("qualify-native-vl-release-gates.py", makefile)
        self.assertIn("generate-native-vl-g5-qualification.py", makefile)
        self.assertIn("generate-native-vl-release-provenance.py", makefile)
        self.assertIn("--media-root", qualifier)
        self.assertIn('"image_a_restored"', qualifier)
        self.assertIn('"a_b_a_cache_reuse_observed"', qualifier)
        self.assertIn("run_doctor(", qualifier)
        self.assertIn(
            '"fingerprint_sha256": host_fingerprint_sha256()', qualifier
        )
        self.assertIn('"vl_smoke": vl_smoke', qualifier)
        self.assertIn('"public_hygiene": hygiene', qualifier)
        self.assertGreaterEqual(qualifier.count("require_gpu_idle()"), 3)
        self.assertIn(
            '"--context-tokens",\n        "16384",', qualifier
        )
        self.assertIn(
            '"--cache-capacity",\n        "17408",', qualifier
        )
        self.assertIn("except OSError:", qualifier)

        soak = (ROOT / "scripts/qualify-native-vl-resident-soak.py").read_text(
            encoding="utf-8"
        )
        g5 = (ROOT / "scripts/generate-native-vl-g5-qualification.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("post_warm_peak_rss = max", soak)
        self.assertIn("require_gpu_idle()", soak)
        self.assertIn('"gtt_within_96_gib"', soak)
        self.assertIn("mem_info_gtt_used", soak)
        self.assertIn(
            '"--context-tokens",\n            "16384",', soak
        )
        self.assertIn('primary_host.get("fingerprint_sha256")', g5)
        rollback = (
            ROOT / "scripts/qualify-native-vl-rollback.py"
        ).read_text(encoding="utf-8")
        self.assertIn("require_gpu_idle()", rollback)


if __name__ == "__main__":
    unittest.main()
