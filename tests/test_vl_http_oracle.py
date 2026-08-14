from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from aima_engine.vl_http_oracle import http_render_case_checks


ROOT = Path(__file__).resolve().parents[1]


class VlHttpOracleTest(unittest.TestCase):
    def test_http_prompt_binding_is_exact_and_fail_closed(self) -> None:
        prompt_ids = [1, 2, 3, 4]
        prompt_hash = (
            "f6bd10506e9a4daed7c03eda2f2fde54be3bd58eee49dab471c18a888ffbdb6f"
        )
        render = {
            "case_id": "image_local_png",
            "oracle_request_sha256": "1" * 64,
            "render_transport_request_sha256": "2" * 64,
            "prompt_tokens": 4,
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": prompt_hash,
            "private_prompt_token_ids_sha256": "3" * 64,
            "private_prompt_matches_real_http": False,
            "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
        }
        case = {
            "case_id": "image_local_png",
            "request_sha256": "1" * 64,
            "processor": {
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": prompt_hash,
                "placeholders": {
                    "image": [{"offset": 1, "length": 2, "num_embeds": 2}]
                },
            },
            "generation": {
                "prompt_tokens": 4,
                "prompt_token_ids_sha256": prompt_hash,
            },
            "http_render": {
                "prompt_tokens": 4,
                "prompt_token_ids_sha256": prompt_hash,
                "render_transport_request_sha256": "2" * 64,
                "private_prompt_token_ids_sha256": "3" * 64,
                "private_prompt_matches_real_http": False,
            },
        }
        self.assertTrue(all(http_render_case_checks(case, render).values()))
        drifted = deepcopy(case)
        drifted["processor"]["prompt_token_ids"][2] = 9
        checks = http_render_case_checks(drifted, render)
        self.assertFalse(checks["prompt_token_ids_exact"])
        self.assertFalse(checks["prompt_token_ids_sha256_exact"])

    def test_capture_reuses_base_hooks_but_forces_string_layout(self) -> None:
        base = (ROOT / "scripts/capture-vllm-vl-oracles.py").read_text()
        wrapper = (
            ROOT / "scripts/capture-vllm-vl-http-oracles.py"
        ).read_text()
        self.assertIn('getattr(\n        args, "_chat_template_content_format"', base)
        self.assertIn("fixture_root = args.fixture_root.resolve()", base)
        self.assertIn("_llm_kwargs(model_dir, fixture_root)", base)
        self.assertIn("preprocessed prompt differs from the bound render", base)
        self.assertIn('args._chat_template_content_format = "string"', wrapper)
        self.assertIn("VLLM_ALLOW_INSECURE_SERIALIZATION", wrapper)
        self.assertIn("validate_http_oracle_manifest", wrapper)
        self.assertIn("serving_render_manifest", wrapper)


if __name__ == "__main__":
    unittest.main()
