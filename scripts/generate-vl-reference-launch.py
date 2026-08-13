#!/usr/bin/env python3
"""Generate the sole hash-bound VL reference launch contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_capability import validate_capability_manifest  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    CAPABILITY_SCHEMA,
    LAUNCH_SCHEMA,
    REFERENCE_ATTENTION_BACKEND,
    REFERENCE_MAX_BATCHED_TOKENS,
    REFERENCE_MEDIA_LIMITS,
    atomic_json,
    load_json_object,
    require_valid_launch_config,
    sha256_file,
    verify_manifest_integrity,
)


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def build_launch(capability_path: Path) -> dict:
    capability = load_json_object(capability_path)
    errors = validate_capability_manifest(capability)
    errors.extend(verify_manifest_integrity(capability))
    if errors:
        raise ValueError("invalid capability manifest:\n- " + "\n- ".join(errors))

    media_io = {"video": {"fps": 2.0, "video_backend": "opencv"}}
    processor_kwargs: dict[str, object] = {}
    allowed_root = "${AIMA_ALLOWED_MEDIA_ROOT}"
    allowed_domains = ["localhost", "127.0.0.1"]
    argv = [
        "${AIMA_VLLM_PYTHON}",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "${AIMA_MODEL_DIR}",
        "--served-model-name",
        "qwen36-vl-reference",
        "--host",
        "127.0.0.1",
        "--port",
        "${AIMA_VL_PORT}",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "262144",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(REFERENCE_MAX_BATCHED_TOKENS),
        "--enable-chunked-prefill",
        "--gpu-memory-utilization",
        "0.95",
        "--attention-backend",
        REFERENCE_ATTENTION_BACKEND,
        "--mm-encoder-attn-backend",
        REFERENCE_ATTENTION_BACKEND,
        "--gdn-prefill-backend",
        "triton",
        "--enforce-eager",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--no-language-model-only",
        "--no-skip-mm-profiling",
        "--limit-mm-per-prompt",
        compact(REFERENCE_MEDIA_LIMITS),
        "--allowed-local-media-path",
        allowed_root,
        "--allowed-media-domains",
        *allowed_domains,
        "--media-io-kwargs",
        compact(media_io),
        "--mm-processor-kwargs",
        compact(processor_kwargs),
        "--mm-processor-cache-gb",
        "4",
        "--video-pruning-rate",
        "0",
        "--load-format",
        "safetensors",
        "--tensor-parallel-size",
        "1",
    ]
    return {
        "schema": LAUNCH_SCHEMA,
        "argv": argv,
        "environment_policy": {
            "inherit": False,
            "variables": {
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "LANG": "C.UTF-8",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONHASHSEED": "0",
                "PYTORCH_ROCM_ARCH": "gfx1151",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "ROCM_PATH": "/opt/rocm",
                "HIP_PATH": "/opt/rocm",
                "LD_LIBRARY_PATH": "/opt/rocm/lib:/opt/rocm/lib64",
                "VLLM_IMAGE_FETCH_TIMEOUT": "10",
                "VLLM_VIDEO_FETCH_TIMEOUT": "30",
                "VLLM_VIDEO_LOADER_BACKEND": "opencv",
            },
        },
        "media_policy": {
            "limits": REFERENCE_MEDIA_LIMITS,
            "aggregate_encoder_token_budget": REFERENCE_MAX_BATCHED_TOKENS,
            "max_tokens_per_item": {"image": 16_384, "video": 12_288},
            "allowed_local_media_paths": [allowed_root],
            "allowed_media_domains": allowed_domains,
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
            "bytes": capability_path.stat().st_size,
            "sha256": sha256_file(capability_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch = build_launch(args.capability.resolve())
    require_valid_launch_config(launch)
    print(atomic_json(args.output, launch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
