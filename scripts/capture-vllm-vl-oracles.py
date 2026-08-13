#!/usr/bin/env python3
"""Capture fixed processor, model-boundary, logits and generation VL oracles."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import mimetypes
import os
from pathlib import Path
import socket
import sys
from types import MethodType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_capability import validate_processor_probe  # noqa: E402
from aima_engine.vl_oracle import (  # noqa: E402
    ORACLE_SCHEMA,
    REQUIRED_BOUNDARIES,
    canonical_int_list_sha256,
    validate_oracle_manifest,
    write_raw_tensor,
)
from aima_engine.vl_reference import (  # noqa: E402
    MODEL_REVISION,
    PINNED_PACKAGES,
    REFERENCE_ATTENTION_BACKEND,
    REFERENCE_MAX_BATCHED_TOKENS,
    REFERENCE_MEDIA_LIMITS,
    atomic_json,
    canonical_json_sha256,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_bytes,
    sha256_file,
    validate_launch_config,
    verify_manifest_integrity,
)


CASE_SPECS = (
    {
        "case_id": "image_local_png",
        "content": (
            ("text", "Name one visual property."),
            ("image", "image-rgb-256.png"),
        ),
    },
    {
        "case_id": "video_local_mp4",
        "content": (
            ("text", "Describe the motion briefly."),
            ("video", "video-8f-4fps-128.mp4"),
        ),
    },
    {
        "case_id": "multi_image",
        "content": (
            ("text", "First:"),
            ("image", "image-rgb-256.png"),
            ("text", "Second:"),
            ("image", "image-landscape-512x192.jpg"),
            ("text", "Compare them briefly."),
        ),
    },
    {
        "case_id": "multi_video",
        "content": (
            ("video", "video-8f-4fps-128.mp4"),
            ("text", "and"),
            ("video", "video-12f-6fps-192x128.avi"),
            ("text", "Compare their motion briefly."),
        ),
    },
    {
        "case_id": "mixed_image_video",
        "content": (
            ("image", "image-rgb-256.png"),
            ("text", "then"),
            ("video", "video-8f-4fps-128.mp4"),
            ("text", "Compare the visual patterns briefly."),
        ),
    },
)

_VISION_BOUNDARIES = {
    "vision_patch_embed",
    "vision_block_0",
    "vision_block_13",
    "vision_block_26",
    "vision_merger",
}
_LANGUAGE_BOUNDARIES = {
    "mrope_positions",
    "injected_embeddings",
    "language_layer_0",
    "language_final_norm",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _first_tensor(value: Any) -> Any | None:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _find_model_root(model: Any) -> Any:
    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, "visual") and hasattr(candidate, "language_model"):
            return candidate
        for name in ("model", "module"):
            child = getattr(candidate, name, None)
            if child is not None:
                queue.append(child)
    raise RuntimeError("could not locate the multimodal model root")


def _module_identity(module: Any) -> str:
    return f"{type(module).__module__}.{type(module).__name__}"


class InstallOracleHooks:
    """Serializable worker callable that installs capture hooks."""

    def __init__(
        self,
        *,
        output_root: str,
        case_id: str,
        selected_rows: list[int],
        target_token_ids: list[int],
    ) -> None:
        self.output_root = output_root
        self.case_id = case_id
        self.selected_rows = selected_rows
        self.target_token_ids = target_token_ids

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = _find_model_root(model)
        old_state = getattr(root, "_aima_vl_oracle_state", None)
        if isinstance(old_state, dict):
            for handle in old_state.get("handles", []):
                handle.remove()

        state: dict[str, Any] = {
            "output_root": self.output_root,
            "case_id": self.case_id,
            "selected_rows": list(self.selected_rows),
            "target_token_ids": list(self.target_token_ids),
            "captures": {},
            "call_shapes": {},
            "handles": [],
            "mrope_position_delta": None,
            "logits_original_shape": None,
            "original_visual_forward": None,
        }

        def capture(name: str, value: Any, *, multiple: bool = False) -> None:
            tensor = _first_tensor(value)
            if tensor is None:
                return
            if name in _LANGUAGE_BOUNDARIES and tensor.shape[-2 if tensor.ndim > 1 else 0] <= 1:
                return
            if not multiple and name in state["captures"]:
                return
            cpu = tensor.detach().contiguous().cpu()
            if multiple:
                state["captures"].setdefault(name, []).append(cpu)
                state["call_shapes"].setdefault(name, []).append(list(cpu.shape))
            else:
                state["captures"][name] = cpu

        def output_hook(name: str, *, multiple: bool = False):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                capture(name, output, multiple=multiple)

            return hook

        visual = root.visual
        language = root.language_model
        modules = {
            "vision_patch_embed": visual.patch_embed,
            "vision_block_0": visual.blocks[0],
            "vision_block_13": visual.blocks[13],
            "vision_block_26": visual.blocks[26],
            "vision_merger": visual.merger,
            "language_layer_0": language.model.layers[0],
            "language_final_norm": language.model.norm,
        }
        for name in sorted(_LANGUAGE_BOUNDARIES - {"mrope_positions", "injected_embeddings"}):
            module = modules[name]
            state["handles"].append(
                module.register_forward_hook(output_hook(name))
            )

        state["original_visual_forward"] = visual.forward

        def instrumented_visual_forward(
            visual_self: Any,
            x: Any,
            grid_thw: Any,
            *,
            encoder_metadata: dict[str, Any] | None = None,
        ) -> Any:
            hidden_states = x.to(
                device=visual_self.device,
                dtype=visual_self.dtype,
                non_blocking=True,
            )
            hidden_states = visual_self.patch_embed.forward(hidden_states)
            capture("vision_patch_embed", hidden_states, multiple=True)

            if encoder_metadata is None:
                grid_thw_list = (
                    grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
                )
                encoder_metadata = visual_self.prepare_encoder_metadata(grid_thw_list)

            hidden_states = hidden_states + encoder_metadata["pos_embeds"]
            hidden_states = hidden_states.unsqueeze(1)
            deepstack_features = []
            for layer_num, block in enumerate(visual_self.blocks):
                hidden_states = block.forward(
                    hidden_states,
                    cu_seqlens=encoder_metadata["cu_seqlens"],
                    rotary_pos_emb_cos=encoder_metadata["rotary_pos_emb_cos"],
                    rotary_pos_emb_sin=encoder_metadata["rotary_pos_emb_sin"],
                    max_seqlen=encoder_metadata["max_seqlen"],
                    sequence_lengths=encoder_metadata.get("sequence_lengths"),
                )
                boundary = f"vision_block_{layer_num}"
                if boundary in _VISION_BOUNDARIES:
                    capture(boundary, hidden_states, multiple=True)
                if layer_num in visual_self.deepstack_visual_indexes:
                    merger_index = visual_self.deepstack_visual_indexes.index(layer_num)
                    deepstack_features.append(
                        visual_self.deepstack_merger_list[merger_index].forward(
                            hidden_states
                        )
                    )
            hidden_states = visual_self.merger.forward(hidden_states)
            capture("vision_merger", hidden_states, multiple=True)
            return torch.cat([hidden_states] + deepstack_features, dim=1)

        visual.forward = MethodType(instrumented_visual_forward, visual)

        def root_pre_hook(_module: Any, args: Any, kwargs: Any) -> None:
            positions = kwargs.get("positions")
            inputs_embeds = kwargs.get("inputs_embeds")
            if positions is None and len(args) > 1:
                positions = args[1]
            if inputs_embeds is None and len(args) > 3:
                inputs_embeds = args[3]
            if isinstance(positions, torch.Tensor) and positions.shape[-1] > 1:
                capture("mrope_positions", positions)
                if state["mrope_position_delta"] is None:
                    state["mrope_position_delta"] = (
                        int(positions.max().item()) + 1 - int(positions.shape[-1])
                    )
            if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.shape[0] > 1:
                capture("injected_embeddings", inputs_embeds)

        state["handles"].append(
            root.register_forward_pre_hook(root_pre_hook, with_kwargs=True)
        )

        def logits_hook(_module: Any, _args: Any, output: Any) -> None:
            logits = _first_tensor(output)
            if logits is None or logits.ndim != 2 or logits.shape[0] <= 1:
                return
            if "full_vocabulary_logits" in state["captures"]:
                return
            missing = [row for row in self.selected_rows if row >= logits.shape[0]]
            if missing:
                raise RuntimeError(
                    "teacher-forced logit rows exceed captured prompt logits: "
                    f"shape={list(logits.shape)}, rows={missing}"
                )
            index = torch.tensor(self.selected_rows, device=logits.device)
            state["captures"]["full_vocabulary_logits"] = (
                logits.index_select(0, index).detach().contiguous().cpu()
            )
            state["logits_original_shape"] = list(logits.shape)

        state["handles"].append(
            language.logits_processor.register_forward_hook(logits_hook)
        )
        root._aima_vl_oracle_state = state
        return {
            "root": _module_identity(root),
            "modules": {name: _module_identity(module) for name, module in modules.items()}
            | {"full_vocabulary_logits": _module_identity(language.logits_processor)},
        }


class FinalizeOracleHooks:
    """Serializable worker callable that writes captures and removes hooks."""

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = _find_model_root(model)
        state = getattr(root, "_aima_vl_oracle_state", None)
        if not isinstance(state, dict):
            raise RuntimeError("oracle hooks were not installed")
        for handle in state.get("handles", []):
            handle.remove()
        state["handles"] = []
        original_visual_forward = state.get("original_visual_forward")
        if original_visual_forward is not None:
            root.visual.forward = original_visual_forward

        missing = REQUIRED_BOUNDARIES - set(state["captures"])
        if missing:
            raise RuntimeError("missing model boundaries: " + ", ".join(sorted(missing)))

        output_root = Path(state["output_root"])
        case_id = state["case_id"]
        components: dict[str, dict[str, Any]] = {}
        for name in sorted(REQUIRED_BOUNDARIES):
            value = state["captures"][name]
            call_shapes = state["call_shapes"].get(name)
            if isinstance(value, list):
                if not value:
                    raise RuntimeError(f"empty capture list: {name}")
                try:
                    value = torch.cat(value, dim=0)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"cannot concatenate capture calls for {name}: {call_shapes}"
                    ) from exc
            component = write_raw_tensor(
                output_root,
                f"{case_id}/boundaries/{name}",
                value,
                original_shape=(
                    state["logits_original_shape"]
                    if name == "full_vocabulary_logits"
                    else None
                ),
                selected_rows=(
                    state["selected_rows"]
                    if name == "full_vocabulary_logits"
                    else None
                ),
            )
            if call_shapes:
                component["call_shapes"] = call_shapes
            if name == "mrope_positions":
                component["position_delta"] = state["mrope_position_delta"]
            if name == "full_vocabulary_logits":
                component["teacher_forced_target_token_ids"] = state[
                    "target_token_ids"
                ]
            components[name] = component
        delattr(root, "_aima_vl_oracle_state")
        return {"boundaries": components}


class RemoveOracleHooks:
    def __call__(self, model: Any) -> bool:
        root = _find_model_root(model)
        state = getattr(root, "_aima_vl_oracle_state", None)
        if not isinstance(state, dict):
            return False
        for handle in state.get("handles", []):
            handle.remove()
        original_visual_forward = state.get("original_visual_forward")
        if original_visual_forward is not None:
            root.visual.forward = original_visual_forward
        delattr(root, "_aima_vl_oracle_state")
        return True


def _media_part(modality: str, path: Path) -> dict[str, Any]:
    url = path.resolve().as_uri()
    if modality == "image":
        return {"type": "image_url", "image_url": {"url": url}}
    if modality == "video":
        return {"type": "video_url", "video_url": {"url": url}}
    raise ValueError(f"unsupported modality: {modality}")


def _build_messages(spec: Mapping[str, Any], fixture_root: Path) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for modality, value in spec["content"]:
        if modality == "text":
            content.append({"type": "text", "text": value})
        else:
            content.append(_media_part(modality, fixture_root / value))
    return [{"role": "user", "content": content}]


def _load_fixture_records(fixture_root: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json_object(fixture_root / "fixtures-manifest.json")
    records = manifest.get("fixtures")
    if not isinstance(records, list):
        raise ValueError("fixture manifest has no fixtures array")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("malformed fixture record")
        path = fixture_root / record["path"]
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"fixture failed verification: {record['path']}")
        result[record["path"]] = record
    return result


def _semantic_case(
    spec: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    content = []
    for modality, value in spec["content"]:
        if modality == "text":
            content.append({"type": "text", "text": value})
            continue
        record = records[value]
        content.append(
            {
                "type": modality,
                "fixture": value,
                "transport": "local-file",
                "mime": mimetypes.guess_type(value)[0],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )
    return {"role": "user", "content": content}


def _placeholder_record(value: Any) -> dict[str, Any]:
    record = {
        "offset": int(value.offset),
        "length": int(value.length),
        "num_embeds": int(value.get_num_embeds()),
    }
    mask = getattr(value, "is_embed", None)
    if mask is not None:
        raw = mask.detach().contiguous().cpu().view(-1).tolist()
        record["is_embed"] = [bool(item) for item in raw]
        record["is_embed_sha256"] = canonical_json_sha256(record["is_embed"])
    return record


def _tensor_leaves(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    import torch

    if isinstance(value, torch.Tensor):
        return [(prefix or "tensor", value)]
    if isinstance(value, Mapping):
        result: list[tuple[str, Any]] = []
        for key, item in value.items():
            child = f"{prefix}__{key}" if prefix else str(key)
            result.extend(_tensor_leaves(item, child))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for index, item in enumerate(value):
            child = f"{prefix}__{index}" if prefix else str(index)
            result.extend(_tensor_leaves(item, child))
        return result
    if is_dataclass(value):
        return _tensor_leaves(asdict(value), prefix)
    return []


def _processor_record(
    engine_input: Mapping[str, Any], output_root: Path, case_id: str
) -> dict[str, Any]:
    prompt_token_ids = [int(item) for item in engine_input["prompt_token_ids"]]
    placeholders = {
        str(modality): [_placeholder_record(item) for item in ranges]
        for modality, ranges in engine_input["mm_placeholders"].items()
    }
    mm_kwargs = engine_input["mm_kwargs"]
    data = mm_kwargs.get_data() if hasattr(mm_kwargs, "get_data") else mm_kwargs
    tensors: dict[str, dict[str, Any]] = {}
    for name, tensor in _tensor_leaves(data):
        if name in tensors:
            raise RuntimeError(f"duplicate processor tensor name: {name}")
        tensors[name] = write_raw_tensor(
            output_root, f"{case_id}/processor/{name}", tensor
        )
    return {
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": canonical_int_list_sha256(prompt_token_ids),
        "prompt_text_sha256": (
            sha256_bytes(engine_input["prompt"].encode("utf-8"))
            if isinstance(engine_input.get("prompt"), str)
            else None
        ),
        "placeholders": placeholders,
        "mm_hashes": {
            str(modality): list(values)
            for modality, values in engine_input["mm_hashes"].items()
        },
        "tensors": tensors,
    }


def _selected_teacher_rows(
    prompt_token_ids: list[int], placeholders: Mapping[str, Sequence[Any]]
) -> list[int]:
    maximum = len(prompt_token_ids) - 2
    if maximum < 0:
        raise ValueError("prompt is too short for teacher-forced logits")
    candidates = {0, maximum}
    for ranges in placeholders.values():
        for value in ranges:
            offset = int(value.offset)
            length = int(value.length)
            candidates.update(
                {
                    offset - 1,
                    offset,
                    offset + length - 2,
                    offset + length - 1,
                }
            )
    return sorted(row for row in candidates if 0 <= row <= maximum)


def _runtime_versions() -> dict[str, str]:
    names = (*PINNED_PACKAGES, "numpy", "pillow", "opencv-python-headless")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _require_inputs(
    *,
    reference_path: Path,
    launch_path: Path,
    processor_path: Path,
    fixture_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference = load_json_object(reference_path)
    reference_errors = verify_manifest_integrity(reference)
    if reference.get("complete") is not True or reference.get(
        "qualified_for_oracle_capture"
    ) is not True:
        reference_errors.append("reference is not qualified for oracle capture")
    if reference_errors:
        raise ValueError("invalid reference manifest:\n- " + "\n- ".join(reference_errors))

    launch = load_json_object(launch_path)
    launch_errors = validate_launch_config(launch)
    if launch_errors:
        raise ValueError("invalid launch contract:\n- " + "\n- ".join(launch_errors))

    processor = load_json_object(processor_path)
    processor_errors = validate_processor_probe(processor)
    if processor_errors:
        raise ValueError("invalid processor probe:\n- " + "\n- ".join(processor_errors))

    _load_fixture_records(fixture_root)
    return reference, launch, processor


def _llm_kwargs(model_dir: Path, fixture_root: Path) -> dict[str, Any]:
    return {
        "model": str(model_dir),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": 262_144,
        "max_num_seqs": 1,
        "max_num_batched_tokens": REFERENCE_MAX_BATCHED_TOKENS,
        "enable_chunked_prefill": True,
        "gpu_memory_utilization": 0.95,
        "attention_backend": REFERENCE_ATTENTION_BACKEND,
        "mm_encoder_attn_backend": REFERENCE_ATTENTION_BACKEND,
        "gdn_prefill_backend": "triton",
        "enforce_eager": True,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "language_model_only": False,
        "skip_mm_profiling": False,
        "limit_mm_per_prompt": dict(REFERENCE_MEDIA_LIMITS),
        "allowed_local_media_path": str(fixture_root),
        "allowed_media_domains": ["localhost", "127.0.0.1"],
        "media_io_kwargs": {"video": {"fps": 2.0, "video_backend": "opencv"}},
        "mm_processor_kwargs": {},
        "mm_processor_cache_gb": 4,
        "video_pruning_rate": 0,
        "load_format": "safetensors",
        "tensor_parallel_size": 1,
        "seed": 0,
        "disable_log_stats": True,
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"oracle output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    reference, _launch, _processor = _require_inputs(
        reference_path=args.reference_manifest,
        launch_path=args.launch_config,
        processor_path=args.processor_probe,
        fixture_root=args.fixture_root,
    )
    source = git_identity(ROOT)
    if source["dirty"]:
        raise ValueError("oracle capture source must be a clean commit")
    if socket.gethostname() != reference.get("host", {}).get("hostname"):
        raise ValueError("oracle capture hostname differs from the frozen reference")
    versions = _runtime_versions()
    for name, expected in PINNED_PACKAGES.items():
        actual = versions.get(name)
        if not isinstance(actual, str) or not (
            actual == expected or actual.startswith(expected + ".")
        ):
            raise ValueError(f"runtime pin mismatch for {name}: {actual!r}")

    records = _load_fixture_records(args.fixture_root)
    llm_kwargs = _llm_kwargs(args.model_dir, args.fixture_root)
    llm = LLM(**llm_kwargs)
    print(json.dumps({"event": "engine_ready"}, sort_keys=True), flush=True)
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        prompt_logprobs=1,
        seed=0,
    )
    cases: list[dict[str, Any]] = []

    try:
        for spec in CASE_SPECS:
            case_id = spec["case_id"]
            print(
                json.dumps({"event": "case_start", "case_id": case_id}, sort_keys=True),
                flush=True,
            )
            llm.reset_mm_cache()
            llm.llm_engine.reset_encoder_cache()
            messages = _build_messages(spec, args.fixture_root)
            engine_input = llm._preprocess_chat_one(
                messages, chat_template_content_format="openai"
            )
            if engine_input.get("type") != "multimodal":
                raise RuntimeError(f"case did not produce multimodal input: {case_id}")
            processor = _processor_record(engine_input, output_root, case_id)
            prompt_token_ids = processor["prompt_token_ids"]
            rows = _selected_teacher_rows(
                prompt_token_ids, engine_input["mm_placeholders"]
            )
            targets = [prompt_token_ids[row + 1] for row in rows]

            def install_callable(model: Any) -> dict[str, Any]:
                return InstallOracleHooks(
                    output_root=str(output_root),
                    case_id=case_id,
                    selected_rows=rows,
                    target_token_ids=targets,
                )(model)

            def finalize_callable(model: Any) -> dict[str, Any]:
                return FinalizeOracleHooks()(model)

            def cleanup_callable(model: Any) -> bool:
                return RemoveOracleHooks()(model)

            installation = llm.apply_model(install_callable)
            try:
                outputs = llm._render_and_run_requests(
                    prompts=iter((engine_input,)),
                    params=[sampling],
                    output_type=RequestOutput,
                    use_tqdm=False,
                )
                finalization = llm.apply_model(finalize_callable)
            except BaseException:
                llm.apply_model(cleanup_callable)
                raise

            if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                raise RuntimeError(f"unexpected generation cardinality: {case_id}")
            if len(finalization) != 1 or len(installation) != 1:
                raise RuntimeError("oracle capture requires tensor_parallel_size=1")
            output = outputs[0]
            completion = output.outputs[0]
            output_token_ids = [int(item) for item in completion.token_ids]
            generation = {
                "sampling": {
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "prompt_logprobs": 1,
                    "seed": 0,
                },
                "prompt_token_ids_sha256": canonical_int_list_sha256(prompt_token_ids),
                "output_token_ids": output_token_ids,
                "output_token_ids_sha256": canonical_int_list_sha256(output_token_ids),
                "output_text_sha256": sha256_bytes(completion.text.encode("utf-8")),
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(output_token_ids),
                "num_cached_tokens": int(getattr(output, "num_cached_tokens", 0) or 0),
            }
            cases.append(
                {
                    "case_id": case_id,
                    "passed": True,
                    "request": _semantic_case(spec, records),
                    "request_sha256": canonical_json_sha256(_semantic_case(spec, records)),
                    "cache_state": {
                        "processor_cache": "reset-before-case",
                        "encoder_cache": "reset-before-case",
                        "prefix_cache_enabled": False,
                    },
                    "processor": processor,
                    "model_modules": installation[0],
                    "boundaries": finalization[0]["boundaries"],
                    "generation": generation,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "case_complete",
                        "case_id": case_id,
                        "prompt_tokens": len(prompt_token_ids),
                        "completion_tokens": len(output_token_ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = seal_manifest(
        {
            "schema": ORACLE_SCHEMA,
            "captured_at": utc_now(),
            "complete": True,
            "qualified_for_native_boundary_comparison": True,
            "reference_manifest": file_component(
                args.reference_manifest, "benchmarks/results/vl-reference-manifest.json"
            ),
            "bindings": {
                "launch_config": file_component(
                    args.launch_config, "benchmarks/results/vl-reference-launch.json"
                ),
                "processor_probe": file_component(
                    args.processor_probe,
                    "benchmarks/results/vl-processor-capability-v0.1.0.json",
                ),
                "fixture_manifest": file_component(
                    args.fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
                ),
            },
            "capture_source": source,
            "capture_control_plane": {
                "vllm_allow_insecure_serialization": True,
                "scope": "isolated-offline-oracle-hook-rpc-only",
                "product_runtime_dependency": False,
            },
            "host": {
                "label": args.host_label,
                "hostname": socket.gethostname(),
            },
            "runtime": {
                "python_version": sys.version.split()[0],
                "packages": versions,
            },
            "model": {
                "repository": "Qwen/Qwen3.6-35B-A3B",
                "revision": MODEL_REVISION,
                "dtype": "bfloat16",
            },
            "launch": {
                "kind": "vllm-offline-single-resident-process",
                "kwargs": {
                    key: (
                        "${AIMA_MODEL_DIR}"
                        if key == "model"
                        else "${AIMA_ALLOWED_MEDIA_ROOT}"
                        if key == "allowed_local_media_path"
                        else value
                    )
                    for key, value in llm_kwargs.items()
                },
            },
            "oracle_root": "benchmarks/oracles/vl-v0.1.0",
            "required_boundaries": sorted(REQUIRED_BOUNDARIES),
            "cases": cases,
        }
    )
    errors = validate_oracle_manifest(manifest, oracle_root=output_root)
    if errors:
        raise RuntimeError("oracle manifest validation failed:\n- " + "\n- ".join(errors))
    atomic_json(args.output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--processor-probe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-label", default="amd395")
    parser.add_argument("--max-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
        raise ValueError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required for the isolated "
            "offline apply_model oracle hooks"
        )
    manifest = capture(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(manifest["cases"]),
                "qualified": manifest["qualified_for_native_boundary_comparison"],
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
