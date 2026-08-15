from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate-native-vl-g1-coverage-audit.py"
RESULT = ROOT / "benchmarks/results/native-vl-g1-coverage-audit-v0.1.0.json"
RESULT_SIDECAR = RESULT.with_name(RESULT.name + ".sha256")


class NativeVlG1CoverageAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = RESULT.read_bytes()
        cls.result = json.loads(cls.payload)

    def test_audit_is_current_and_reproducible(self) -> None:
        completed = subprocess.run(
            ["python3", str(SCRIPT), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("native VL G1 coverage audit: PASS", completed.stdout)

    def test_evidence_sidecar_and_canonical_seal(self) -> None:
        digest = hashlib.sha256(self.payload).hexdigest()
        self.assertEqual(
            RESULT_SIDECAR.read_text(encoding="utf-8"),
            f"{digest}  {RESULT.name}\n",
        )
        canonical_payload = {
            key: value
            for key, value in self.result.items()
            if key != "integrity"
        }
        canonical_bytes = json.dumps(
            canonical_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(self.result["integrity"]["algorithm"], "sha256")
        self.assertEqual(
            self.result["integrity"]["canonical_payload_sha256"],
            hashlib.sha256(canonical_bytes).hexdigest(),
        )

    def test_all_audit_inputs_are_content_bound(self) -> None:
        for component in self.result["inputs"].values():
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                component["sha256"],
            )

    def test_coverage_counts_are_explicit_and_conservative(self) -> None:
        result = self.result
        self.assertEqual(
            result["schema"],
            "aima-amd395-qwen36/native-vl-g1-coverage-audit/v1",
        )
        self.assertTrue(result["complete"])
        self.assertFalse(result["qualified"])
        coverage = result["coverage"]
        self.assertEqual(coverage["requirements"], 14)
        self.assertEqual(
            coverage["counts"],
            {"covered": 7, "partial": 7, "missing": 0},
        )
        items = coverage["items"]
        self.assertEqual(len(items), len({item["requirement_id"] for item in items}))
        self.assertTrue(
            all(
                (item["status"] == "covered") == (item["gaps"] == [])
                for item in items
            )
        )
        self.assertEqual(len(result["blocking_gaps"]), 7)

    def test_referenced_native_cases_exist_and_are_qualified(self) -> None:
        native = json.loads(
            (
                ROOT / "benchmarks/results/native-vl-capability-v0.1.0.json"
            ).read_text(encoding="utf-8")
        )
        execution = json.loads(
            (
                ROOT / "benchmarks/results/native-vl-envelope-v0.1.0.json"
            ).read_text(encoding="utf-8")
        )
        case_maps = {
            "native_capability": {
                item["case_id"]: item for item in native["matrix"]["cases"]
            },
            "execution_envelope": {
                item["case_id"]: item
                for item in execution["matrix"]["observations"]
            },
            "mixed_conversation_native": {
                item["case_id"]: item
                for item in json.loads(
                    (
                        ROOT
                        / "benchmarks/results/native-vl-g1-extension-v0.1.0.json"
                    ).read_text(encoding="utf-8")
                )["cases"]
            },
        }
        for requirement in self.result["coverage"]["items"]:
            for record in requirement["evidence"]:
                for case_id in record.get("case_ids", []):
                    case = case_maps[record["artifact"]][case_id]
                    self.assertTrue(case["qualified"], case_id)

    def test_next_evidence_names_every_blocking_workstream(self) -> None:
        expected = {
            "g1-transport-cache-extension",
            "g1-error-extension",
            "g1-generation-and-current-head-requalification",
        }
        self.assertEqual(
            {item["evidence_id"] for item in self.result["next_evidence"]},
            expected,
        )
        covered = {
            requirement_id
            for item in self.result["next_evidence"]
            for requirement_id in item["requirement_ids"]
        }
        blocked = {
            item["requirement_id"] for item in self.result["blocking_gaps"]
        }
        self.assertEqual(covered, blocked)

        transport_cache = next(
            item
            for item in self.result["next_evidence"]
            if item["evidence_id"] == "g1-transport-cache-extension"
        )
        self.assertNotIn("cache_http_url_mutation_aba", transport_cache["cases"])
        self.assertNotIn(
            "cache_video_data_local_equivalence", transport_cache["cases"]
        )
        self.assertIn(
            "cache_hit_miss_long_generation_usage", transport_cache["cases"]
        )

    def test_audit_does_not_promote_any_product_gate(self) -> None:
        decision = self.result["decision"]
        self.assertTrue(decision["audit_complete"])
        self.assertTrue(decision["all_referenced_cases_qualified"])
        self.assertFalse(decision["coverage_complete"])
        self.assertTrue(decision["new_evidence_required"])
        for gate in (
            "g1_passed",
            "g2_passed",
            "g3_passed",
            "g4_passed",
            "g5_passed",
        ):
            self.assertFalse(decision[gate])

    def test_audit_contains_no_private_machine_paths(self) -> None:
        serialized = self.payload.decode("utf-8")
        for private_prefix in ("/home/", "/Users/", "/data/", "/tmp/"):
            self.assertNotIn(f'"{private_prefix}', serialized)


if __name__ == "__main__":
    unittest.main()
