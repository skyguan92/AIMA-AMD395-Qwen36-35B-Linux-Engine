from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from aima_engine.release_evidence import (
    DEFAULT_RELEASE,
    NATIVE_VL_ARCHIVE_ONLY_COMPONENTS,
    _verify_recorded_artifacts,
    evidence_tree,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseEvidencePathResolutionTest(unittest.TestCase):
    def test_completed_native_vl_release_is_the_default(self) -> None:
        self.assertEqual(DEFAULT_RELEASE, "1.5.1-native-vl.4")

    def test_inline_http_response_digest_is_not_treated_as_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = root / "summary.json"
            value = {
                "response": {"choices": [{"message": {"content": "ok"}}]},
                "response_sha256": "0" * 64,
            }

            self.assertEqual(
                _verify_recorded_artifacts(root, owner, value), []
            )

    def test_nested_output_path_resolves_from_nearest_evidence_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "benchmarks/runs/native-product-surfaces-final"
            owner = run / "raw/prefix-cache/pair-01/baseline.json"
            target = run / "raw/prefix-cache/pair-01/baseline.load.json"
            owner.parent.mkdir(parents=True)
            target.write_text("{}\n", encoding="utf-8")
            value = {
                "load_report": (
                    "${AIMA_OUTPUT_DIR}/raw/prefix-cache/pair-01/"
                    "baseline.load.json"
                ),
                "load_report_sha256": digest(target),
            }

            self.assertEqual(
                _verify_recorded_artifacts(root, owner, value), []
            )

    def test_unique_sealed_run_suffix_resolves_cross_run_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "benchmarks/runs"
            owner = runs / "native-vl-rollback-final/rollback.json"
            target = runs / "native-vl-resident-soak-final/soak.json"
            owner.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            target.write_text("{}\n", encoding="utf-8")
            value = {
                "path": "benchmarks/runs/native-vl-resident-soak/soak.json",
                "sha256": digest(target),
            }

            self.assertEqual(
                _verify_recorded_artifacts(root, owner, value), []
            )

    def test_real_path_digest_shape_remains_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = root / "summary.json"
            value = {
                "report": ["raw/one.json", "raw/two.json"],
                "report_sha256": "0" * 64,
            }

            errors = _verify_recorded_artifacts(root, owner, value)
            self.assertEqual(len(errors), 1)
            self.assertIn("malformed report/report_sha256 pair", errors[0])

    def test_exact_archive_only_component_can_be_projected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = Path(
                "benchmarks/results/native-vl-envelope-v0.1.0-raw/"
                "processor-probe.stdout.log"
            )
            component_digest, component_bytes = NATIVE_VL_ARCHIVE_ONLY_COMPONENTS[
                relative
            ]
            owner = root / "benchmarks/results/native-vl-envelope-v0.1.0.json"
            value = {
                "bytes": component_bytes,
                "path": relative.relative_to("benchmarks/results").as_posix(),
                "sha256": component_digest,
            }

            self.assertEqual(
                _verify_recorded_artifacts(
                    root,
                    owner,
                    value,
                    archive_only_components=NATIVE_VL_ARCHIVE_ONLY_COMPONENTS,
                ),
                [],
            )
            self.assertIn(
                "missing evidence artifact",
                _verify_recorded_artifacts(root, owner, value)[0],
            )

    def test_archive_only_projection_rejects_changed_identity_or_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = Path(
                "benchmarks/results/native-vl-envelope-v0.1.0-raw/"
                "processor-probe.stdout.log"
            )
            component_digest, component_bytes = NATIVE_VL_ARCHIVE_ONLY_COMPONENTS[
                relative
            ]
            owner = root / "benchmarks/results/native-vl-envelope-v0.1.0.json"
            value = {
                "bytes": component_bytes + 1,
                "path": relative.relative_to("benchmarks/results").as_posix(),
                "sha256": component_digest,
            }
            errors = _verify_recorded_artifacts(
                root,
                owner,
                value,
                archive_only_components=NATIVE_VL_ARCHIVE_ONLY_COMPONENTS,
            )
            self.assertEqual(len(errors), 1)
            self.assertIn("missing evidence artifact", errors[0])

            value["bytes"] = component_bytes
            value["sha256"] = "0" * 64
            errors = _verify_recorded_artifacts(
                root,
                owner,
                value,
                archive_only_components=NATIVE_VL_ARCHIVE_ONLY_COMPONENTS,
            )
            self.assertEqual(len(errors), 1)
            self.assertIn("missing evidence artifact", errors[0])

            value["path"] = "unsealed-raw/processor-probe.stdout.log"
            value["sha256"] = component_digest
            errors = _verify_recorded_artifacts(
                root,
                owner,
                value,
                archive_only_components=NATIVE_VL_ARCHIVE_ONLY_COMPONENTS,
            )
            self.assertEqual(len(errors), 1)
            self.assertIn("missing evidence artifact", errors[0])

            target = root / relative
            target.parent.mkdir(parents=True)
            target.write_text("tampered\n", encoding="utf-8")
            value["path"] = relative.relative_to("benchmarks/results").as_posix()
            errors = _verify_recorded_artifacts(
                root,
                owner,
                value,
                archive_only_components=NATIVE_VL_ARCHIVE_ONLY_COMPONENTS,
            )
            self.assertEqual(len(errors), 1)
            self.assertIn("evidence hash mismatch", errors[0])

    def test_virtual_component_reproduces_materialized_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory)
            (tree / "present.json").write_text("{}\n", encoding="utf-8")
            payload = b"archived component\n"
            virtual = {
                Path("archived.log"): (
                    hashlib.sha256(payload).hexdigest(),
                    len(payload),
                )
            }
            projected = evidence_tree(tree, virtual_components=virtual)
            (tree / "archived.log").write_bytes(payload)

            self.assertEqual(projected, evidence_tree(tree))


if __name__ == "__main__":
    unittest.main()
