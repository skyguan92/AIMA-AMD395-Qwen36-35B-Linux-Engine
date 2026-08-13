#!/usr/bin/env python3
"""Probe fixed Qwen3.6 processor and vLLM multimodal budget boundaries."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_capability import (  # noqa: E402
    PROCESSOR_PROBE_SCHEMA,
    validate_processor_probe,
)
from aima_engine.vl_reference import (  # noqa: E402
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PINNED_PACKAGES,
    ReferenceManifestError,
    atomic_json,
    file_component,
    processor_identity,
    seal_manifest,
    sha256_bytes,
)


def outcome(case_id: str, function: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = function()
        return {"case_id": case_id, "outcome": "accepted", "result": value}
    except Exception as exc:  # The exception class/message are capability evidence.
        return {
            "case_id": case_id,
            "outcome": "rejected",
            "error_class": type(exc).__name__,
            "error_message": str(exc),
        }


def tensor_component(value: Any) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    payload = tensor.numpy().tobytes(order="C")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def resize_result(
    resized: tuple[int, int], *, frames: int, patch: int, merge: int, temporal: int
) -> dict[str, Any]:
    height, width = resized
    padded_frames = ((frames + temporal - 1) // temporal) * temporal
    grid = [max(padded_frames // temporal, 1), height // patch, width // patch]
    return {
        "resized_height": height,
        "resized_width": width,
        "grid_thw": grid,
        "vision_tokens": grid[0] * grid[1] * grid[2] // (merge * merge),
    }


def package_versions() -> dict[str, str]:
    names = (*PINNED_PACKAGES, "numpy", "pillow", "opencv-python", "av")
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def build_probe(model_dir: Path, captured_at: str | None) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        smart_resize as image_smart_resize,
    )
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        smart_resize as video_smart_resize,
    )
    from transformers.video_utils import VideoMetadata
    from vllm.engine.arg_utils import EngineArgs
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import MultiModalBudget

    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    image_processor = processor.image_processor
    video_processor = processor.video_processor
    image_size = dict(image_processor.size)
    video_size = dict(video_processor.size)
    patch = int(image_processor.patch_size)
    merge = int(image_processor.merge_size)
    temporal = int(image_processor.temporal_patch_size)
    factor = patch * merge

    engine_args = EngineArgs(
        model=str(model_dir),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=262_144,
        max_num_seqs=1,
        max_num_batched_tokens=262_144,
        enable_chunked_prefill=True,
        limit_mm_per_prompt={"image": 262_144, "video": 262_144},
        language_model_only=False,
        skip_mm_profiling=False,
    )
    engine_config = engine_args.create_engine_config()
    budget = MultiModalBudget(engine_config, MULTIMODAL_REGISTRY)
    info = budget.processor.info

    image_cases = [
        ("minimum_source", 1, 1),
        ("aspect_ratio_200", 1, 200),
        ("aspect_ratio_over_200", 1, 201),
        ("factor_minus_one", factor - 1, factor - 1),
        ("factor_exact", factor, factor),
        ("factor_plus_one", factor + 1, factor + 1),
        ("portrait", 1024, 256),
        ("landscape", 256, 1024),
        ("maximum_pixels", 4096, 4096),
        ("above_maximum_pixels", 8192, 8192),
    ]

    def image_case(case_id: str, height: int, width: int) -> dict[str, Any]:
        record = outcome(
            case_id,
            lambda: resize_result(
                image_smart_resize(
                    height=height,
                    width=width,
                    factor=factor,
                    min_pixels=image_size["shortest_edge"],
                    max_pixels=image_size["longest_edge"],
                ),
                frames=1,
                patch=patch,
                merge=merge,
                temporal=temporal,
            ),
        )
        record["source"] = {"height": height, "width": width}
        return record

    maximum_video_width, maximum_video_height = info.get_image_size_with_most_features(
        max_pixels=video_size["longest_edge"] // temporal
    )
    video_cases = [
        ("below_temporal_factor", 1, factor, factor),
        ("temporal_factor", temporal, factor, factor),
        ("below_spatial_factor", temporal, factor - 1, factor),
        ("spatial_factor", temporal, factor, factor),
        ("aspect_ratio_200", temporal, factor, factor * 200),
        ("aspect_ratio_over_200", temporal, factor, factor * 201),
        ("typical", 4, 256, 256),
        (
            "maximum_feature_shape",
            2,
            maximum_video_height,
            maximum_video_width,
        ),
    ]

    def video_case(
        case_id: str, frames: int, height: int, width: int
    ) -> dict[str, Any]:
        record = outcome(
            case_id,
            lambda: resize_result(
                video_smart_resize(
                    num_frames=frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal,
                    factor=factor,
                    min_pixels=video_size["shortest_edge"],
                    max_pixels=video_size["longest_edge"],
                ),
                frames=frames,
                patch=patch,
                merge=merge,
                temporal=temporal,
            ),
        )
        record["source"] = {
            "frames": frames,
            "height": height,
            "width": width,
        }
        return record

    def sample(
        total_frames: int,
        source_fps: float,
        *,
        fps: float | None = None,
        num_frames: int | None = None,
    ) -> dict[str, Any]:
        metadata = VideoMetadata(
            total_num_frames=total_frames,
            fps=source_fps,
            width=256,
            height=256,
            duration=total_frames / source_fps,
            video_backend="opencv",
        )
        indices = video_processor.sample_frames(
            metadata, fps=fps, num_frames=num_frames
        )
        return {"count": len(indices), "indices": indices.tolist()}

    sampling_specs: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("below_min_frames", lambda: sample(3, 24, fps=2)),
        ("minimum_frames", lambda: sample(48, 24, fps=2)),
        ("typical_fps", lambda: sample(240, 24, fps=2)),
        ("maximum_frames", lambda: sample(9_216, 24, fps=2)),
        ("above_maximum_frames", lambda: sample(18_432, 24, fps=2)),
        ("explicit_num_frames", lambda: sample(240, 24, num_frames=32)),
        (
            "fps_num_frames_conflict",
            lambda: sample(240, 24, fps=2, num_frames=32),
        ),
    ]
    sampling_cases = [outcome(case_id, function) for case_id, function in sampling_specs]

    pattern = (
        np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3) % 256
    ).astype(np.uint8)
    image = Image.fromarray(pattern, "RGB")
    image_outputs = processor(
        text=["<|vision_start|><|image_pad|><|vision_end|>"],
        images=[image],
        return_tensors="pt",
    )
    frames = np.stack([np.roll(pattern, index, axis=1) for index in range(4)])
    metadata = VideoMetadata(
        total_num_frames=4,
        fps=2,
        width=256,
        height=256,
        duration=2,
        video_backend="opencv",
        frames_indices=list(range(4)),
    )
    video_outputs = processor(
        text=["<|vision_start|><|video_pad|><|vision_end|>"],
        videos=[frames],
        video_metadata=[metadata],
        do_sample_frames=False,
        return_tensors="pt",
    )

    def fixture(
        modality: str, outputs: Mapping[str, Any], generator: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "fixture_id": f"deterministic-{modality}-256",
            "modality": modality,
            "generator": generator,
            "outputs": {name: tensor_component(value) for name, value in outputs.items()},
        }

    fixtures = [
        fixture(
            "image",
            image_outputs,
            {
                "type": "uint8-arange-mod-256",
                "shape": [256, 256, 3],
                "source_sha256": sha256_bytes(pattern.tobytes(order="C")),
            },
        ),
        fixture(
            "video",
            video_outputs,
            {
                "type": "four-horizontal-rolls-of-image-fixture",
                "shape": [4, 256, 256, 3],
                "source_sha256": sha256_bytes(frames.tobytes(order="C")),
                "metadata": {
                    "fps": 2,
                    "duration": 2,
                    "frames_indices": [0, 1, 2, 3],
                },
            },
        ),
    ]

    max_tokens = dict(budget.mm_max_toks_per_item)
    max_items_prompt = dict(budget.mm_max_items_per_prompt)
    max_items_batch = dict(budget.mm_max_items_per_batch)
    probe: dict[str, Any] = {
        "schema": PROCESSOR_PROBE_SCHEMA,
        "complete": True,
        "qualified": True,
        "captured_at": captured_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "model_dir": "${AIMA_MODEL_DIR}",
            "config": file_component(model_dir / "config.json", "config.json"),
            "preprocessor_config": file_component(
                model_dir / "preprocessor_config.json", "preprocessor_config.json"
            ),
            "video_preprocessor_config": file_component(
                model_dir / "video_preprocessor_config.json",
                "video_preprocessor_config.json",
            ),
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "packages": package_versions(),
        },
        "processor": processor_identity(model_dir),
        "vllm_budget": {
            "resolved_architecture": engine_config.model_config.architecture,
            "max_model_len": engine_config.model_config.max_model_len,
            "supported_limits": dict(info.supported_mm_limits),
            "probe_limits": {"image": 262_144, "video": 262_144},
            "max_tokens_per_item": max_tokens,
            "max_items_per_prompt": max_items_prompt,
            "max_items_per_batch": max_items_batch,
            "encoder_budget_tokens": budget.get_encoder_budget(),
            "encoder_compute_budget_tokens": budget.encoder_compute_budget,
            "encoder_cache_tokens": budget.encoder_cache_size,
            "derivation": (
                "min(user_limit, max_model_len // worst_case_tokens_per_item); "
                "batch size is one"
            ),
        },
        "image_resize_cases": [
            image_case(case_id, height, width)
            for case_id, height, width in image_cases
        ],
        "video_resize_cases": [
            video_case(case_id, frame_count, height, width)
            for case_id, frame_count, height, width in video_cases
        ],
        "video_sampling_cases": sampling_cases,
        "deterministic_processor_fixtures": fixtures,
    }
    errors = validate_processor_probe(probe)
    if errors:
        probe["qualified"] = False
        probe["validation_errors"] = errors
        raise ReferenceManifestError(
            "processor capability probe failed:\n- " + "\n- ".join(errors)
        )
    return seal_manifest(probe)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--captured-at")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        probe = build_probe(args.model_dir.resolve(), args.captured_at)
        digest = atomic_json(args.output.resolve(), probe)
    except ReferenceManifestError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"VL processor capability probe: PASS ({digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
