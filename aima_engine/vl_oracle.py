"""Contracts and raw-tensor helpers for fixed native-VL reference oracles."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from aima_engine.vl_reference import verify_manifest_integrity


ORACLE_SCHEMA = "aima-amd395-qwen36/vl-oracle-manifest/v1"
TENSOR_SCHEMA = "aima-amd395-qwen36/vl-raw-tensor/v1"

REQUIRED_ORACLE_CASES = {
    "image_local_png",
    "video_local_mp4",
    "multi_image",
    "multi_video",
    "mixed_image_video",
}
REQUIRED_BOUNDARIES = {
    "vision_patch_embed",
    "vision_block_0",
    "vision_block_13",
    "vision_block_26",
    "vision_merger",
    "mrope_positions",
    "injected_embeddings",
    "language_layer_0",
    "language_final_norm",
    "full_vocabulary_logits",
}
_IMAGE_PROCESSOR_TENSORS = {"pixel_values", "image_grid_thw"}
_VIDEO_PROCESSOR_TENSORS = {"pixel_values_videos", "video_grid_thw"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_raw_tensor(
    root: Path,
    logical_name: str,
    tensor: Any,
    *,
    original_shape: list[int] | None = None,
    selected_rows: list[int] | None = None,
) -> dict[str, Any]:
    """Write a tensor without numeric conversion and return its component."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"oracle value is not a tensor: {logical_name}")
    value = tensor.detach().contiguous().cpu()
    raw = value.view(torch.uint8).numpy().tobytes(order="C")
    relative = Path(logical_name + ".bin")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    component: dict[str, Any] = {
        "schema": TENSOR_SCHEMA,
        "path": str(relative),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "element_size": value.element_size(),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }
    if original_shape is not None:
        component["original_shape"] = original_shape
    if selected_rows is not None:
        component["selected_rows"] = selected_rows
    return component


def verify_raw_tensor(component: Mapping[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if component.get("schema") != TENSOR_SCHEMA:
        errors.append("raw tensor schema mismatch")
    relative = component.get("path")
    if not isinstance(relative, str) or not relative:
        return errors + ["raw tensor path is missing"]
    path_value = Path(relative)
    if path_value.is_absolute() or ".." in path_value.parts:
        return errors + [f"unsafe raw tensor path: {relative}"]
    path = (root / path_value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return errors + [f"raw tensor escapes oracle root: {relative}"]
    if not path.is_file():
        return errors + [f"raw tensor is missing: {relative}"]
    raw = path.read_bytes()
    if component.get("bytes") != len(raw):
        errors.append(f"raw tensor size mismatch: {relative}")
    if component.get("sha256") != sha256_bytes(raw):
        errors.append(f"raw tensor SHA-256 mismatch: {relative}")
    shape = component.get("shape")
    element_size = component.get("element_size")
    if (
        not isinstance(shape, list)
        or not all(isinstance(item, int) and item >= 0 for item in shape)
        or not isinstance(element_size, int)
        or element_size <= 0
    ):
        errors.append(f"raw tensor metadata is malformed: {relative}")
    else:
        expected = element_size
        for dimension in shape:
            expected *= dimension
        if expected != len(raw):
            errors.append(f"raw tensor shape/byte count mismatch: {relative}")
    return errors


def validate_oracle_manifest(
    manifest: Mapping[str, Any], *, oracle_root: Path | None = None
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != ORACLE_SCHEMA:
        errors.append(f"oracle schema must be {ORACLE_SCHEMA}")
    if manifest.get("complete") is not True:
        errors.append("oracle manifest is not complete")
    if manifest.get("qualified_for_native_boundary_comparison") is not True:
        errors.append("oracle manifest is not qualified for boundary comparison")
    errors.extend(verify_manifest_integrity(manifest))
    reference = manifest.get("reference_manifest")
    if not isinstance(reference, dict) or not isinstance(
        reference.get("sha256"), str
    ):
        errors.append("oracle has no reference manifest binding")

    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return errors + ["oracle cases must be an array"]
    by_id = {
        item.get("case_id"): item
        for item in cases
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    missing_cases = REQUIRED_ORACLE_CASES - set(by_id)
    if missing_cases:
        errors.append("missing oracle cases: " + ", ".join(sorted(missing_cases)))
    if len(by_id) != len(cases):
        errors.append("oracle case IDs must be unique")

    for case_id in REQUIRED_ORACLE_CASES:
        case = by_id.get(case_id)
        if not isinstance(case, dict):
            continue
        if case.get("passed") is not True:
            errors.append(f"oracle case did not pass: {case_id}")

        processor = case.get("processor")
        if not isinstance(processor, dict):
            errors.append(f"oracle case has no processor record: {case_id}")
        else:
            prompt_token_ids = processor.get("prompt_token_ids")
            if not isinstance(prompt_token_ids, list) or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in prompt_token_ids
            ):
                errors.append(f"processor token IDs are malformed: {case_id}")
            elif processor.get("prompt_token_ids_sha256") != canonical_int_list_sha256(
                prompt_token_ids
            ):
                errors.append(f"processor token ID digest mismatch: {case_id}")
            placeholders = processor.get("placeholders")
            if not isinstance(placeholders, dict) or not placeholders:
                errors.append(f"processor placeholders are missing: {case_id}")
            tensors = processor.get("tensors")
            if not isinstance(tensors, dict):
                errors.append(f"processor tensors are missing: {case_id}")
            else:
                required_processor_tensors: set[str] = set()
                if "image" in placeholders:
                    required_processor_tensors.update(_IMAGE_PROCESSOR_TENSORS)
                if "video" in placeholders:
                    required_processor_tensors.update(_VIDEO_PROCESSOR_TENSORS)
                missing_processor = required_processor_tensors - set(tensors)
                if missing_processor:
                    errors.append(
                        f"oracle case {case_id} is missing processor tensors: "
                        + ", ".join(sorted(missing_processor))
                    )
                if oracle_root is not None:
                    for component in tensors.values():
                        if isinstance(component, dict):
                            errors.extend(verify_raw_tensor(component, oracle_root))
                        else:
                            errors.append(
                                f"malformed processor tensor component: {case_id}"
                            )

        boundaries = case.get("boundaries")
        if not isinstance(boundaries, dict):
            errors.append(f"oracle case has no boundaries: {case_id}")
            continue
        missing = REQUIRED_BOUNDARIES - set(boundaries)
        if missing:
            errors.append(
                f"oracle case {case_id} is missing boundaries: "
                + ", ".join(sorted(missing))
            )
        for name, component in boundaries.items():
            if not isinstance(component, dict):
                errors.append(f"malformed oracle boundary: {case_id}/{name}")
            elif oracle_root is not None:
                errors.extend(verify_raw_tensor(component, oracle_root))
        mrope = boundaries.get("mrope_positions") if isinstance(boundaries, dict) else None
        if isinstance(mrope, dict) and not isinstance(mrope.get("position_delta"), int):
            errors.append(f"M-RoPE position delta is missing: {case_id}")
        logits = (
            boundaries.get("full_vocabulary_logits")
            if isinstance(boundaries, dict)
            else None
        )
        if isinstance(logits, dict):
            rows = logits.get("selected_rows")
            targets = logits.get("teacher_forced_target_token_ids")
            if (
                not isinstance(rows, list)
                or not rows
                or not isinstance(targets, list)
                or len(rows) != len(targets)
            ):
                errors.append(f"teacher-forced logit rows are malformed: {case_id}")
        generation = case.get("generation")
        if not isinstance(generation, dict):
            errors.append(f"oracle generation result is missing: {case_id}")
        else:
            for name in ("prompt_token_ids_sha256", "output_token_ids_sha256"):
                value = generation.get(name)
                if not isinstance(value, str) or len(value) != 64:
                    errors.append(f"oracle generation digest is invalid: {case_id}/{name}")
            output_ids = generation.get("output_token_ids")
            if not isinstance(output_ids, list) or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in output_ids
            ):
                errors.append(f"oracle output token IDs are malformed: {case_id}")
            elif generation.get(
                "output_token_ids_sha256"
            ) != canonical_int_list_sha256(output_ids):
                errors.append(f"oracle output token digest mismatch: {case_id}")
    return errors


def canonical_int_list_sha256(values: list[int]) -> str:
    return sha256_bytes(
        json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
