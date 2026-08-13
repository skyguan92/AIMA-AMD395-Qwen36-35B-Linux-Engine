from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from aima_engine.vl_reference import (
    CAPABILITY_SCHEMA,
    LAUNCH_SCHEMA,
    REFERENCE_ATTENTION_BACKEND,
    REFERENCE_MAX_BATCHED_TOKENS,
    REFERENCE_MEDIA_LIMITS,
    canonical_json_sha256,
    model_file_components,
    seal_manifest,
    validate_launch_config,
    verify_manifest_integrity,
)


def valid_launch_config() -> dict:
    limits = dict(REFERENCE_MEDIA_LIMITS)
    media_io = {"video": {"fps": 2.0, "video_backend": "opencv"}}
    processor_kwargs = {}
    return {
        "schema": LAUNCH_SCHEMA,
        "argv": [
            "${AIMA_VLLM_PYTHON}",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            "${AIMA_MODEL_DIR}",
            "--served-model-name",
            "qwen36-vl-reference",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "262144",
            "--max-num-batched-tokens",
            str(REFERENCE_MAX_BATCHED_TOKENS),
            "--max-num-seqs",
            "1",
            "--enable-chunked-prefill",
            "--attention-backend",
            REFERENCE_ATTENTION_BACKEND,
            "--mm-encoder-attn-backend",
            REFERENCE_ATTENTION_BACKEND,
            "--gdn-prefill-backend",
            "triton",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_xml",
            "--enforce-eager",
            "--no-async-scheduling",
            "--no-enable-prefix-caching",
            "--no-language-model-only",
            "--no-skip-mm-profiling",
            "--limit-mm-per-prompt",
            json.dumps(limits, separators=(",", ":"), sort_keys=True),
            "--allowed-local-media-path",
            "${AIMA_ALLOWED_MEDIA_ROOT}",
            "--allowed-media-domains",
            "vl-fixtures.invalid",
            "--media-io-kwargs",
            json.dumps(media_io, separators=(",", ":"), sort_keys=True),
            "--mm-processor-kwargs",
            json.dumps(processor_kwargs, separators=(",", ":"), sort_keys=True),
            "--mm-processor-cache-gb",
            "4",
            "--video-pruning-rate",
            "0",
            "--load-format",
            "safetensors",
            "--tensor-parallel-size",
            "1",
        ],
        "environment_policy": {
            "inherit": False,
            "variables": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONHASHSEED": "0",
                "PYTORCH_ROCM_ARCH": "gfx1151",
                "VLLM_IMAGE_FETCH_TIMEOUT": "10",
                "VLLM_VIDEO_FETCH_TIMEOUT": "30",
            },
        },
        "media_policy": {
            "limits": limits,
            "allowed_local_media_paths": ["${AIMA_ALLOWED_MEDIA_ROOT}"],
            "allowed_media_domains": ["vl-fixtures.invalid"],
            "media_io_kwargs": media_io,
            "mm_processor_kwargs": processor_kwargs,
            "processor_cache_gb": 4,
            "video_pruning_rate": 0,
        },
        "measurement_boundary": {
            "media_fetch": True,
            "media_decode": True,
            "processor": True,
            "vision_encode": True,
            "llm_prefill": True,
            "decode": True,
            "ttft": True,
            "total_latency": True,
        },
        "capability_manifest": {
            "schema": CAPABILITY_SCHEMA,
            "path": "benchmarks/results/vl-capability-manifest.json",
            "sha256": "a" * 64,
        },
    }


class VlReferenceManifestTest(unittest.TestCase):
    def test_strict_multimodal_launch_is_accepted(self) -> None:
        self.assertEqual(validate_launch_config(valid_launch_config()), [])

    def test_text_only_flags_are_rejected(self) -> None:
        config = valid_launch_config()
        config["argv"].extend(["--language-model-only", "--skip-mm-profiling"])
        errors = validate_launch_config(config)
        self.assertTrue(any("not a VL reference" in error for error in errors))
        self.assertGreaterEqual(len(errors), 2)

    def test_single_media_mvp_is_rejected(self) -> None:
        config = valid_launch_config()
        config["media_policy"]["limits"] = {"image": 1, "video": 1}
        errors = validate_launch_config(config)
        self.assertIn("media_policy.limits.image must be at least 2", errors)
        self.assertIn("media_policy.limits.video must be at least 2", errors)

    def test_processor_time_cannot_be_moved_outside_measurement(self) -> None:
        config = valid_launch_config()
        config["measurement_boundary"]["processor"] = False
        self.assertIn(
            "measurement boundary must include processor",
            validate_launch_config(config),
        )

    def test_unsafe_encoder_budget_or_backend_is_rejected(self) -> None:
        config = valid_launch_config()
        index = config["argv"].index("--max-num-batched-tokens") + 1
        config["argv"][index] = "262144"
        index = config["argv"].index("--mm-encoder-attn-backend") + 1
        config["argv"][index] = "TORCH_SDPA"
        errors = validate_launch_config(config)
        self.assertTrue(any("--max-num-batched-tokens" in error for error in errors))
        self.assertTrue(any("--mm-encoder-attn-backend" in error for error in errors))

    def test_inherited_or_credential_environment_is_rejected(self) -> None:
        config = valid_launch_config()
        config["environment_policy"]["inherit"] = True
        config["environment_policy"]["variables"]["API_TOKEN"] = "not-safe"
        errors = validate_launch_config(config)
        self.assertIn(
            "reference launch must use a clean non-inherited environment", errors
        )
        self.assertTrue(any("credential variables" in error for error in errors))

    def test_manifest_integrity_detects_semantic_tampering(self) -> None:
        manifest = seal_manifest({"schema": "test/v1", "complete": True})
        self.assertEqual(verify_manifest_integrity(manifest), [])
        manifest["complete"] = False
        self.assertEqual(
            verify_manifest_integrity(manifest),
            ["manifest canonical payload SHA-256 mismatch"],
        )

    def test_canonical_hash_does_not_depend_on_key_order(self) -> None:
        self.assertEqual(
            canonical_json_sha256({"b": 2, "a": 1}),
            canonical_json_sha256({"a": 1, "b": 2}),
        )

    def test_every_checkpoint_shard_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = {
                "config.json": b"{}\n",
                "tokenizer.json": b"{}\n",
                "tokenizer_config.json": b"{}\n",
                "preprocessor_config.json": b"{}\n",
                "video_preprocessor_config.json": b"{}\n",
            }
            for name, payload in required.items():
                (root / name).write_bytes(payload)
            (root / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")
            (root / "model-00002-of-00002.safetensors").write_bytes(b"shard-two")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 18},
                        "weight_map": {
                            "model.a": "model-00001-of-00002.safetensors",
                            "model.b": "model-00002-of-00002.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )
            components = model_file_components(root)
        self.assertEqual(
            set(components),
            {
                *required,
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            },
        )

    def test_non_string_checkpoint_shard_is_rejected_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
            ):
                (root / name).write_text("{}\n", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.a": 7}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "non-string shards"):
                model_file_components(root)


if __name__ == "__main__":
    unittest.main()
