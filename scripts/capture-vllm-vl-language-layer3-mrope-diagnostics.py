#!/usr/bin/env python3
"""Capture pinned vLLM layer-3 M-RoPE consumption boundaries."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_oracle import (  # noqa: E402
    canonical_int_list_sha256,
    write_raw_tensor,
)
from aima_engine.vl_reference import (  # noqa: E402
    MODEL_REVISION,
    PINNED_PACKAGES,
    REFERENCE_ATTENTION_BACKEND,
    atomic_json,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
    verify_manifest_integrity,
)


SCHEMA = "aima-amd395-qwen36/vl-language-layer3-mrope-diagnostic-oracle/v1"
VL_ORACLE_SHA256 = "87dcdf76b7251f78da01a2a5f4312a9fb5c7d07a1ca2b2420566e77930f23d44"
CASE_IDS = (
    "image_local_png",
    "video_local_mp4",
    "multi_image",
    "multi_video",
    "mixed_image_video",
)
MROPE_SECTION = (11, 11, 10)
ROTARY_PAIRS = sum(MROPE_SECTION)
SOURCE_HASHES = {
    "vllm.model_executor.models.qwen3_5": (
        "6cbbe29a102a5e6207a1b1828976cbf442eca0fe9f5895b7e1ca74542bb5e8c0"
    ),
    "vllm.model_executor.models.qwen3_next": (
        "0b3a7f577757712b48a09d4ae849d091949be35975de3eb95da81f7ea5670934"
    ),
    "vllm.model_executor.layers.layernorm": (
        "4c4be7e915fa2977dee683a2304a9469719785449ed204dec197c30921fe4d1e"
    ),
    "vllm.model_executor.layers.linear": (
        "715ba882f6029d2cba21314b0e189a1a80947128b7cd4d505f4af9a86c3cc542"
    ),
    "vllm.model_executor.layers.rotary_embedding": (
        "e9bf24a9560187c426028b419e46e96549f2083ae3884da6b8ac65f34e33da17"
    ),
    "vllm.model_executor.layers.rotary_embedding.base": (
        "32e2145231012f98c8c613cc37dfe621996a3508af77ee7e0265baefde5c1139"
    ),
    "vllm.model_executor.layers.rotary_embedding.mrope": (
        "d5fc5ca640bf47fdfc27447bcc6b67b2294c9ac840bbc736fe75cc6098e2a51c"
    ),
}
REQUIRED_COMPONENTS = {
    "positions",
    "attention_input",
    "qkv_projection",
    "q_gate_projection",
    "raw_q",
    "raw_k",
    "raw_v",
    "q_norm_input",
    "k_norm_input",
    "q_norm_weight",
    "k_norm_weight",
    "normalized_q",
    "normalized_k",
    "axis_cos",
    "axis_sin",
    "effective_cos",
    "effective_sin",
    "rotary_q",
    "rotary_k",
    "attention_output",
    "attention_residual",
    "post_attention_norm",
    "layer3_first_tensor",
    "layer_output",
}
ORACLE_LABELS = {
    "layer-003-return-full_attention-inp": "attention_input",
    "layer-003-return-full_attention-q_gate": "q_gate_projection",
    "layer-003-return-full_attention-q": "rotary_q",
    "layer-003-return-full_attention-k": "rotary_k",
    "layer-003-return-full_attention-v": "raw_v",
    "layer-003-return-full_attention-output": "attention_output",
    "layer-003-launch-001-residual_out": "attention_residual",
    "layer-003-launch-001-norm_out": "post_attention_norm",
    "layer-003-return-layer_body-output": "layer_output",
}
STATE_ATTRIBUTE = "_aima_vl_language_layer3_mrope_diagnostic_state"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mrope_axis_for_pair(pair: int) -> int:
    """Return the pinned interleaved T/H/W axis for one rotary pair."""
    if pair < 0 or pair >= ROTARY_PAIRS:
        raise ValueError("M-RoPE pair is outside the rotary dimension")
    if pair % 3 == 1 and pair <= 3 * MROPE_SECTION[1]:
        return 1
    if pair % 3 == 2 and pair <= 3 * MROPE_SECTION[2]:
        return 2
    return 0


def _load_base_capture() -> Any:
    path = ROOT / "scripts/capture-vllm-vl-oracles.py"
    spec = importlib.util.spec_from_file_location(
        "aima_vl_language_layer3_base_capture", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen VL capture helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _remove_hooks(root: Any, state: dict[str, Any]) -> None:
    for handle in state.get("handles", []):
        handle.remove()
    state["handles"] = []
    if hasattr(root, STATE_ATTRIBUTE):
        delattr(root, STATE_ATTRIBUTE)


class InstallLanguageLayer3MropeDiagnosticHooks:
    """Serializable worker callable for the first full-attention layer."""

    def __init__(self, *, output_root: str, case_id: str) -> None:
        self.output_root = output_root
        self.case_id = case_id

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = _find_model_root(model)
        previous = getattr(root, STATE_ATTRIBUTE, None)
        if isinstance(previous, dict):
            _remove_hooks(root, previous)

        language = root.language_model
        if len(language.model.layers) < 5:
            raise RuntimeError("language model has no layer-4 residual boundary")
        layer3 = language.model.layers[3]
        layer4 = language.model.layers[4]
        if getattr(layer3, "layer_type", None) != "full_attention":
            raise RuntimeError("language layer 3 is not full attention")
        attention = layer3.self_attn
        rotary = attention.rotary_emb
        if tuple(getattr(rotary, "mrope_section", ())) != MROPE_SECTION:
            raise RuntimeError("language layer 3 M-RoPE sections differ")
        if not getattr(rotary, "mrope_interleaved", False):
            raise RuntimeError("language layer 3 M-RoPE is not interleaved")

        state: dict[str, Any] = {
            "output_root": self.output_root,
            "case_id": self.case_id,
            "captures": {},
            "handles": [],
        }

        def capture(name: str, value: Any) -> None:
            tensor = _first_tensor(value)
            if tensor is None or name in state["captures"]:
                return
            if name == "positions":
                if tensor.ndim != 2 or tensor.shape[0] != 3 or tensor.shape[1] <= 1:
                    return
            elif name not in {"q_norm_weight", "k_norm_weight"}:
                if tensor.ndim == 0 or tensor.shape[0] <= 1:
                    return
            state["captures"][name] = tensor.detach().contiguous().cpu()

        def output_hook(name: str):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                capture(name, output)

            return hook

        def qkv_projection_hook(
            _module: Any, _args: Any, output: Any
        ) -> None:
            tensor = _first_tensor(output)
            if tensor is None or tensor.ndim != 2 or tensor.shape[1] != 9216:
                raise RuntimeError("layer-3 QKV projection geometry differs")
            q_gate, raw_k, raw_v = tensor.split((8192, 512, 512), dim=-1)
            token_count = tensor.shape[0]
            q_gate_heads = q_gate.view(token_count, 16, 512)
            capture("qkv_projection", tensor)
            capture("q_gate_projection", q_gate)
            capture("raw_q", q_gate_heads[:, :, :256].reshape(token_count, 4096))
            capture("raw_k", raw_k)
            capture("raw_v", raw_v)

        def q_norm_pre_hook(_module: Any, args: Any) -> None:
            tensor = _first_tensor(args)
            if tensor is not None:
                capture("q_norm_input", tensor.reshape(tensor.shape[0], -1))

        def k_norm_pre_hook(_module: Any, args: Any) -> None:
            tensor = _first_tensor(args)
            if tensor is not None:
                capture("k_norm_input", tensor.reshape(tensor.shape[0], -1))

        def q_norm_hook(_module: Any, _args: Any, output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                capture("normalized_q", tensor.reshape(tensor.shape[0], -1))

        def k_norm_hook(_module: Any, _args: Any, output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                capture("normalized_k", tensor.reshape(tensor.shape[0], -1))

        def rotary_pre_hook(
            module: Any, args: Any, kwargs: dict[str, Any]
        ) -> None:
            positions = kwargs.get("positions")
            query = kwargs.get("query")
            key = kwargs.get("key")
            if positions is None and len(args) > 0:
                positions = args[0]
            if query is None and len(args) > 1:
                query = args[1]
            if key is None and len(args) > 2:
                key = args[2]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (positions, query, key)
            ):
                raise RuntimeError("layer-3 rotary inputs are unavailable")
            if positions.ndim != 2 or positions.shape[0] != 3:
                raise RuntimeError("layer-3 rotary positions are not three-axis")
            if query.shape[0] != positions.shape[1] or key.shape[0] != query.shape[0]:
                raise RuntimeError("layer-3 rotary token geometry differs")
            capture("positions", positions)
            capture("normalized_q", query)
            capture("normalized_k", key)

            cache = module._match_cos_sin_cache_dtype(query)
            cos_sin = cache[positions]
            axis_cos, axis_sin = cos_sin.chunk(2, dim=-1)
            if axis_cos.shape != (3, positions.shape[1], ROTARY_PAIRS):
                raise RuntimeError("layer-3 rotary cache geometry differs")
            axes = [mrope_axis_for_pair(pair) for pair in range(ROTARY_PAIRS)]
            effective_cos = torch.stack(
                [axis_cos[axis, :, pair] for pair, axis in enumerate(axes)],
                dim=-1,
            )
            effective_sin = torch.stack(
                [axis_sin[axis, :, pair] for pair, axis in enumerate(axes)],
                dim=-1,
            )
            capture("axis_cos", axis_cos)
            capture("axis_sin", axis_sin)
            capture("effective_cos", effective_cos)
            capture("effective_sin", effective_sin)

        def rotary_hook(_module: Any, _args: Any, output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("layer-3 rotary output geometry differs")
            capture("rotary_q", output[0])
            capture("rotary_k", output[1])

        def attention_hook(
            _module: Any,
            args: Any,
            kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            output = kwargs.get("output")
            if output is None and len(args) > 1:
                output = args[1]
            capture("attention_output", output)

        def post_attention_hook(_module: Any, _args: Any, output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("layer-3 post-attention norm geometry differs")
            capture("post_attention_norm", output[0])
            capture("attention_residual", output[1])

        def layer3_hook(_module: Any, _args: Any, output: Any) -> None:
            capture("layer3_first_tensor", output)

        def layer4_norm_hook(_module: Any, _args: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)) and len(output) == 2:
                capture("layer_output", output[1])

        capture("q_norm_weight", attention.q_norm.weight)
        capture("k_norm_weight", attention.k_norm.weight)
        hooks = state["handles"]
        hooks.append(
            layer3.input_layernorm.register_forward_hook(
                output_hook("attention_input")
            )
        )
        hooks.append(attention.qkv_proj.register_forward_hook(qkv_projection_hook))
        hooks.append(attention.q_norm.register_forward_pre_hook(q_norm_pre_hook))
        hooks.append(attention.q_norm.register_forward_hook(q_norm_hook))
        hooks.append(attention.k_norm.register_forward_pre_hook(k_norm_pre_hook))
        hooks.append(attention.k_norm.register_forward_hook(k_norm_hook))
        hooks.append(
            rotary.register_forward_pre_hook(rotary_pre_hook, with_kwargs=True)
        )
        hooks.append(rotary.register_forward_hook(rotary_hook))
        hooks.append(
            attention.register_forward_hook(attention_hook, with_kwargs=True)
        )
        hooks.append(
            layer3.post_attention_layernorm.register_forward_hook(
                post_attention_hook
            )
        )
        hooks.append(layer3.register_forward_hook(layer3_hook))
        hooks.append(
            layer4.input_layernorm.register_forward_hook(layer4_norm_hook)
        )
        setattr(root, STATE_ATTRIBUTE, state)
        return {
            "layer3": _module_identity(layer3),
            "attention": _module_identity(attention),
            "q_norm": _module_identity(attention.q_norm),
            "k_norm": _module_identity(attention.k_norm),
            "rotary": _module_identity(rotary),
            "post_attention_norm": _module_identity(
                layer3.post_attention_layernorm
            ),
            "layer4_input_norm": _module_identity(layer4.input_layernorm),
            "mrope_section": list(MROPE_SECTION),
            "mrope_interleaved": True,
        }


class FinalizeLanguageLayer3MropeDiagnosticHooks:
    """Write layer-3 tensors and a native-oracle compatibility ledger."""

    def __call__(self, model: Any) -> dict[str, Any]:
        root = _find_model_root(model)
        state = getattr(root, STATE_ATTRIBUTE, None)
        if not isinstance(state, dict):
            raise RuntimeError("language layer-3 M-RoPE hooks were not installed")
        captures = state["captures"]
        missing = REQUIRED_COMPONENTS - set(captures)
        if missing:
            _remove_hooks(root, state)
            raise RuntimeError(
                "missing language layer-3 M-RoPE components: "
                + ", ".join(sorted(missing))
            )
        output_root = Path(state["output_root"])
        case_id = state["case_id"]
        components = {
            name: write_raw_tensor(
                output_root, f"{case_id}/components/{name}", captures[name]
            )
            for name in sorted(REQUIRED_COMPONENTS)
        }
        case_root = output_root / case_id
        oracle_lines = []
        for label, component_name in ORACLE_LABELS.items():
            component_path = Path(components[component_name]["path"])
            relative_path = component_path.relative_to(case_id)
            oracle_lines.append(
                json.dumps(
                    {
                        "event": "native_layer_oracle_tensor",
                        "label": label,
                        "file": relative_path.as_posix(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        (case_root / "oracle.jsonl").write_text(
            "\n".join(oracle_lines) + "\n", encoding="utf-8"
        )
        _remove_hooks(root, state)
        return {
            "components": components,
            "oracle_labels": dict(ORACLE_LABELS),
            "oracle_jsonl_sha256": sha256_file(case_root / "oracle.jsonl"),
        }


class RemoveLanguageLayer3MropeDiagnosticHooks:
    def __call__(self, model: Any) -> bool:
        root = _find_model_root(model)
        state = getattr(root, STATE_ATTRIBUTE, None)
        if not isinstance(state, dict):
            return False
        _remove_hooks(root, state)
        return True


def _runtime_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PINNED_PACKAGES}


def _verify_serving_sources() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for module_name, expected_sha256 in SOURCE_HASHES.items():
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"frozen serving source differs: {module_name}")
        marker = "/site-packages/"
        path_text = path.as_posix()
        relative = (
            path_text.split(marker, 1)[1] if marker in path_text else path.name
        )
        result[module_name] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }
    return result


def _read_component(root: Path, record: Mapping[str, Any]) -> bytes:
    path = root / str(record["path"])
    payload = path.read_bytes()
    if len(payload) != int(record["bytes"]):
        raise RuntimeError(f"diagnostic tensor byte count differs: {path.name}")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise RuntimeError(f"diagnostic tensor hash differs: {path.name}")
    return payload


def _compare_component(
    *,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    actual_root: Path,
    expected_root: Path,
) -> dict[str, Any]:
    actual_payload = _read_component(actual_root, actual)
    expected_payload = _read_component(expected_root, expected)
    shape_exact = actual.get("shape") == expected.get("shape")
    dtype_exact = actual.get("dtype") == expected.get("dtype")
    payload_exact = actual_payload == expected_payload
    return {
        "shape_exact": shape_exact,
        "dtype_exact": dtype_exact,
        "payload_exact": payload_exact,
        "exact": shape_exact and dtype_exact and payload_exact,
        "actual_sha256": actual["sha256"],
        "expected_sha256": expected["sha256"],
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
        raise ValueError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required for the isolated "
            "offline apply_model diagnostic hooks"
        )
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
    import cloudpickle

    base = _load_base_capture()
    output_root = args.output_root.resolve()
    fixture_root = args.fixture_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"diagnostic output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source = git_identity(ROOT)
    if source["dirty"]:
        raise ValueError("language layer-3 diagnostic source must be a clean commit")

    reference, _launch, _processor = base._require_inputs(
        reference_path=args.reference_manifest,
        launch_path=args.launch_config,
        processor_path=args.processor_probe,
        fixture_root=fixture_root,
    )
    if socket.gethostname() != reference.get("host", {}).get("hostname"):
        raise ValueError("diagnostic host differs from the frozen reference")
    vl_manifest_path = args.vl_oracle_manifest.resolve()
    if sha256_file(vl_manifest_path) != VL_ORACLE_SHA256:
        raise ValueError("VL oracle manifest differs from the frozen contract")
    vl_manifest = load_json_object(vl_manifest_path)
    integrity_errors = verify_manifest_integrity(vl_manifest)
    if integrity_errors:
        raise ValueError(
            "VL oracle manifest integrity failed:\n- "
            + "\n- ".join(integrity_errors)
        )
    versions = _runtime_versions()
    for name, expected in PINNED_PACKAGES.items():
        actual = versions.get(name)
        if actual != expected and not actual.startswith(expected + "."):
            raise ValueError(f"runtime pin mismatch for {name}: {actual!r}")
    serving_sources = _verify_serving_sources()

    selected_case_ids = CASE_IDS if args.case_id == "all" else (args.case_id,)
    specs = {item["case_id"]: item for item in base.CASE_SPECS}
    vl_cases = {item["case_id"]: item for item in vl_manifest["cases"]}
    if set(CASE_IDS) != set(specs) or set(CASE_IDS) != set(vl_cases):
        raise ValueError("frozen VL case set differs from the diagnostic contract")

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    llm_kwargs = base._llm_kwargs(args.model_dir.resolve(), fixture_root)
    llm_kwargs["skip_mm_profiling"] = True
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,
        seed=0,
    )
    cases: list[dict[str, Any]] = []
    try:
        for case_id in selected_case_ids:
            print(
                json.dumps(
                    {"event": "layer3_mrope_case_start", "case_id": case_id}
                ),
                flush=True,
            )
            llm.reset_mm_cache()
            llm.llm_engine.reset_encoder_cache()
            messages = base._build_messages(specs[case_id], fixture_root)
            engine_input = llm._preprocess_chat_one(
                messages, chat_template_content_format="openai"
            )
            prompt_token_ids = [
                int(item) for item in engine_input["prompt_token_ids"]
            ]
            prompt_sha256 = canonical_int_list_sha256(prompt_token_ids)
            frozen_prompt_sha256 = vl_cases[case_id]["processor"][
                "prompt_token_ids_sha256"
            ]
            if prompt_sha256 != frozen_prompt_sha256:
                raise RuntimeError(
                    f"prompt tokens differ from frozen case: {case_id}"
                )

            def install_callable(model: Any) -> dict[str, Any]:
                return InstallLanguageLayer3MropeDiagnosticHooks(
                    output_root=str(output_root), case_id=case_id
                )(model)

            def finalize_callable(model: Any) -> dict[str, Any]:
                return FinalizeLanguageLayer3MropeDiagnosticHooks()(model)

            def cleanup_callable(model: Any) -> bool:
                return RemoveLanguageLayer3MropeDiagnosticHooks()(model)

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
            if (
                len(outputs) != 1
                or len(installation) != 1
                or len(finalization) != 1
            ):
                raise RuntimeError("diagnostic capture requires one request and TP=1")
            record = finalization[0]
            positions_comparison = _compare_component(
                actual=record["components"]["positions"],
                expected=vl_cases[case_id]["boundaries"]["mrope_positions"],
                actual_root=output_root,
                expected_root=args.vl_oracle_root.resolve(),
            )
            if not positions_comparison["exact"]:
                raise RuntimeError(
                    f"layer-3 positions differ from frozen M-RoPE: {case_id}"
                )
            cases.append(
                {
                    "case_id": case_id,
                    "prompt_tokens": len(prompt_token_ids),
                    "prompt_token_ids_sha256": prompt_sha256,
                    "model_modules": installation[0],
                    "components": record["components"],
                    "oracle_labels": record["oracle_labels"],
                    "oracle_jsonl_sha256": record["oracle_jsonl_sha256"],
                    "frozen_mrope_positions_comparison": positions_comparison,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "layer3_mrope_case_complete",
                        "case_id": case_id,
                        "prompt_tokens": len(prompt_token_ids),
                    }
                ),
                flush=True,
            )
    finally:
        del llm

    for weight_name in ("q_norm_weight", "k_norm_weight"):
        hashes = {
            case["components"][weight_name]["sha256"] for case in cases
        }
        if len(hashes) != 1:
            raise RuntimeError(f"layer-3 {weight_name} changed between cases")

    return seal_manifest(
        {
            "schema": SCHEMA,
            "captured_at": utc_now(),
            "complete": True,
            "qualified_for_attribution_only": True,
            "acceptance_threshold_unchanged": True,
            "source": source,
            "model": {
                "revision": MODEL_REVISION,
                "reference_manifest": file_component(
                    args.reference_manifest,
                    "benchmarks/results/vl-reference-manifest.json",
                ),
                "vl_oracle_manifest_sha256": VL_ORACLE_SHA256,
            },
            "runtime": versions,
            "reference": {
                "attention_backend": REFERENCE_ATTENTION_BACKEND,
                "enforce_eager": True,
                "skip_mm_profiling": True,
                "vllm_allow_insecure_serialization": True,
                "serving_sources": serving_sources,
            },
            "mrope": {
                "layer_index": 3,
                "rotary_dimension": 64,
                "rotary_pairs": ROTARY_PAIRS,
                "section": list(MROPE_SECTION),
                "interleaved": True,
                "pair_axes": [
                    mrope_axis_for_pair(pair) for pair in range(ROTARY_PAIRS)
                ],
            },
            "case_selector": args.case_id,
            "required_components": sorted(REQUIRED_COMPONENTS),
            "oracle_labels": dict(ORACLE_LABELS),
            "cases": cases,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--processor-probe", type=Path, required=True)
    parser.add_argument("--vl-oracle-manifest", type=Path, required=True)
    parser.add_argument("--vl-oracle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-id", choices=("all", *CASE_IDS), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = capture(args)
    manifest_path = args.output_root.resolve() / "manifest.json"
    atomic_json(manifest_path, result)
    print(
        json.dumps(
            {
                "output": str(manifest_path),
                "cases": len(result["cases"]),
                "sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            f"capture VL language layer-3 M-RoPE diagnostics: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
