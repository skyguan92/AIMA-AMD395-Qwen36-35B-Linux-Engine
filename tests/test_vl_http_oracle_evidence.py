from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from aima_engine.vl_http_oracle import validate_http_oracle_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/results/vl-http-oracle-manifest-v0.1.0.json"
SIDECAR = ROOT / "benchmarks/results/vl-http-oracle-manifest-v0.1.0.json.sha256"
RENDER = ROOT / "benchmarks/results/vl-serving-render-manifest-v0.1.0.json"
PRIVATE = ROOT / "benchmarks/results/vl-oracle-manifest.json"
ORACLE_ROOT = ROOT / "benchmarks/oracles/vl-http-v0.1.0"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VlHttpOracleEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_bytes())
        cls.render = json.loads(RENDER.read_bytes())
        cls.private = json.loads(PRIVATE.read_bytes())

    def test_manifest_and_all_raw_tensors_are_hash_bound(self) -> None:
        manifest_sha256 = sha256_file(MANIFEST)
        self.assertEqual(
            SIDECAR.read_text(encoding="utf-8").split()[0], manifest_sha256
        )
        self.assertEqual(
            validate_http_oracle_manifest(
                self.manifest,
                render_manifest=self.render,
                render_manifest_sha256=sha256_file(RENDER),
                oracle_root=ORACLE_ROOT,
            ),
            [],
        )
        self.assertEqual(
            self.manifest["capture_source"]["commit"],
            "d0c0097504f22e46e1738a5e94166bf23673a671",
        )
        self.assertFalse(self.manifest["capture_source"]["dirty"])
        for component in self.manifest["capture_scripts"].values():
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(sha256_file(path), component["sha256"])
        for component in self.manifest["bindings"].values():
            path = ROOT / component["path"]
            self.assertEqual(path.stat().st_size, component["bytes"])
            self.assertEqual(sha256_file(path), component["sha256"])

    def test_http_prompts_and_private_generations_remain_distinct(self) -> None:
        render_by_id = {case["case_id"]: case for case in self.render["cases"]}
        private_by_id = {
            case["case_id"]: case for case in self.private["cases"]
        }
        self.assertEqual(len(self.manifest["cases"]), 5)
        self.assertEqual(
            [
                len(case["processor"]["prompt_token_ids"])
                for case in self.manifest["cases"]
            ],
            [82, 64, 186, 131, 134],
        )
        self.assertEqual(
            sum(
                len(
                    case["boundaries"]["full_vocabulary_logits"][
                        "selected_rows"
                    ]
                )
                for case in self.manifest["cases"]
            ),
            42,
        )
        for case in self.manifest["cases"]:
            case_id = case["case_id"]
            render_case = render_by_id[case_id]
            private_case = private_by_id[case_id]
            self.assertEqual(
                case["processor"]["prompt_token_ids"],
                render_case["prompt_token_ids"],
            )
            self.assertFalse(render_case["private_prompt_matches_real_http"])
            self.assertEqual(
                case["generation"]["output_token_ids_sha256"],
                private_case["generation"]["output_token_ids_sha256"],
            )
        self.assertTrue(
            self.manifest["decision"]["five_real_http_numerical_oracles_exact"]
        )
        self.assertFalse(self.manifest["decision"]["g1_passed"])
        self.assertFalse(self.manifest["decision"]["g2_passed"])


if __name__ == "__main__":
    unittest.main()
