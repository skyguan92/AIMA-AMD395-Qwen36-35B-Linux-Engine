#!/usr/bin/env python3
"""Capture Qwen3.6 vision block 0 from the frozen full-model serving path."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_oracle import write_raw_tensor  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    MODEL_REVISION,
    PINNED_PACKAGES,
    REFERENCE_ATTENTION_BACKEND,
    atomic_json,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
    verify_manifest_integrity,
)


SCHEMA = "aima-amd395-qwen36/vision-block-oracle/v1"
VL_ORACLE_SHA256 = "87dcdf76b7251f78da01a2a5f4312a9fb5c7d07a1ca2b2420566e77930f23d44"
POSITION_ORACLE_SHA256 = (
    "9d316fd6904764f88cd5f25726ecaed33d95bb6cfb4bbe21454c909d66c5d9f6"
)
SOURCE_HASHES = {
    "qwen3_vl": "8ba3592a0fb481a959d6952af25a721cfaeab966558ac11214304e5cf7524d1a",
    "qwen2_5_vl": "377f46682f60bc1ad72c5bd2054ce43216be8e8de6efe52be76b5e5e06a22c78",
    "vit_attn_wrappers": "cb72abab31419d83ef05cd46d8170b08cf34c4a8c78c38fbaf600cf5148c2865",
    "rotary_common": "edc44caa8f697fdc7cea2195c525cf45d232cff8b488de5c4c41ba4e494ad837",
    "linear_utils": "67bc1e3c6983387005ee5a4e7ec9e7993650dcca524bfc67eb90a6a01714973a",
}
CASE_IDS = ("image_local_png", "video_local_mp4")
REQUIRED_COMPONENTS = {
    "block_input",
    "norm1",
    "qkv_linear",
    "rotary_cos",
    "rotary_sin",
    "query_rotated",
    "key_rotated",
    "value",
    "attention",
    "attention_projection",
    "attention_residual",
    "norm2",
    "mlp_fc1",
    "mlp_activation",
    "mlp_fc2",
    "block_output",
}


def _load_base_capture() -> Any:
    path = ROOT / "scripts/capture-vllm-vl-oracles.py"
    spec = importlib.util.spec_from_file_location("aima_vl_base_capture", path)
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


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


class InstallVisionBlockHooks:
    """Serializable worker callable that installs block-0 boundary hooks."""

    def __init__(self, *, output_root: str, case_id: str) -> None:
        self.output_root = output_root
        self.case_id = case_id

    def __call__(self, model: Any) -> dict[str, Any]:
        import torch

        root = _find_model_root(model)
        previous = getattr(root, "_aima_vision_block_state", None)
        if isinstance(previous, dict):
            for handle in previous.get("handles", []):
                handle.remove()
        block = root.visual.blocks[0]
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
            state["captures"][name] = tensor.detach().contiguous().cpu()

        def output_hook(name: str):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                capture(name, output)

            return hook

        def block_pre_hook(
            _module: Any, args: Any, kwargs: dict[str, Any]
        ) -> None:
            capture("block_input", args[0])

        def rotary_pre_hook(
            _module: Any, args: Any, _kwargs: dict[str, Any]
        ) -> None:
            capture("rotary_cos", args[1])
            capture("rotary_sin", args[2])

        def attention_pre_hook(
            _module: Any, args: Any, kwargs: dict[str, Any]
        ) -> None:
            query = kwargs.get("query", args[0] if len(args) > 0 else None)
            key = kwargs.get("key", args[1] if len(args) > 1 else None)
            value = kwargs.get("value", args[2] if len(args) > 2 else None)
            capture("query_rotated", query)
            capture("key_rotated", key)
            capture("value", value)
            capture("cu_seqlens", kwargs.get("cu_seqlens"))
            capture("max_seqlen", kwargs.get("max_seqlen"))

        def norm2_pre_hook(_module: Any, args: Any) -> None:
            capture("attention_residual", args[0])

        hooks = state["handles"]
        hooks.append(block.register_forward_pre_hook(block_pre_hook, with_kwargs=True))
        hooks.append(block.norm1.register_forward_hook(output_hook("norm1")))
        hooks.append(block.attn.qkv.register_forward_hook(output_hook("qkv_linear")))
        hooks.append(
            block.attn.apply_rotary_emb.register_forward_pre_hook(
                rotary_pre_hook, with_kwargs=True
            )
        )
        hooks.append(
            block.attn.attn.register_forward_pre_hook(
                attention_pre_hook, with_kwargs=True
            )
        )
        hooks.append(block.attn.attn.register_forward_hook(output_hook("attention")))
        hooks.append(
            block.attn.proj.register_forward_hook(output_hook("attention_projection"))
        )
        hooks.append(block.norm2.register_forward_pre_hook(norm2_pre_hook))
        hooks.append(block.norm2.register_forward_hook(output_hook("norm2")))
        hooks.append(block.mlp.linear_fc1.register_forward_hook(output_hook("mlp_fc1")))
        hooks.append(block.mlp.act_fn.register_forward_hook(output_hook("mlp_activation")))
        hooks.append(block.mlp.linear_fc2.register_forward_hook(output_hook("mlp_fc2")))
        hooks.append(block.register_forward_hook(output_hook("block_output")))
        root._aima_vision_block_state = state
        return {
            "block": _module_identity(block),
            "norm1": _module_identity(block.norm1),
            "qkv": _module_identity(block.attn.qkv),
            "rotary": _module_identity(block.attn.apply_rotary_emb),
            "attention": _module_identity(block.attn.attn),
            "projection": _module_identity(block.attn.proj),
            "norm2": _module_identity(block.norm2),
            "mlp_fc1": _module_identity(block.mlp.linear_fc1),
            "activation": _module_identity(block.mlp.act_fn),
            "mlp_fc2": _module_identity(block.mlp.linear_fc2),
        }


class FinalizeVisionBlockHooks:
    """Serializable worker callable that writes and removes block hooks."""

    def __call__(self, model: Any) -> dict[str, Any]:
        root = _find_model_root(model)
        state = getattr(root, "_aima_vision_block_state", None)
        if not isinstance(state, dict):
            raise RuntimeError("vision block hooks were not installed")
        for handle in state.get("handles", []):
            handle.remove()
        missing = REQUIRED_COMPONENTS - set(state["captures"])
        if missing:
            raise RuntimeError(
                "missing vision block components: " + ", ".join(sorted(missing))
            )
        output_root = Path(state["output_root"])
        case_id = state["case_id"]
        components = {
            name: write_raw_tensor(
                output_root, f"{case_id}/{name}", state["captures"][name]
            )
            for name in sorted(REQUIRED_COMPONENTS)
        }
        metadata = {
            name: write_raw_tensor(
                output_root,
                f"{case_id}/{name}",
                state["captures"][name].reshape(-1),
            )
            for name in ("cu_seqlens", "max_seqlen")
            if name in state["captures"]
        }
        delattr(root, "_aima_vision_block_state")
        return {"components": components, "metadata": metadata}


class RemoveVisionBlockHooks:
    def __call__(self, model: Any) -> bool:
        root = _find_model_root(model)
        state = getattr(root, "_aima_vision_block_state", None)
        if not isinstance(state, dict):
            return False
        for handle in state.get("handles", []):
            handle.remove()
        delattr(root, "_aima_vision_block_state")
        return True


def _compare_component(
    actual: Mapping[str, Any], expected: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    actual_path = root / str(actual["path"])
    expected_path = root / str(expected["path"])
    actual_payload = actual_path.read_bytes()
    expected_payload = expected_path.read_bytes()
    return {
        "actual_sha256": hashlib.sha256(actual_payload).hexdigest(),
        "expected_sha256": hashlib.sha256(expected_payload).hexdigest(),
        "actual_bytes": len(actual_payload),
        "expected_bytes": len(expected_payload),
        "exact": actual_payload == expected_payload,
    }


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PINNED_PACKAGES:
        versions[name] = importlib.metadata.version(name)
    return versions


def _verify_serving_sources() -> dict[str, str]:
    from vllm.model_executor.layers import utils as linear_utils
    from vllm.model_executor.layers.rotary_embedding import common as rotary_common
    from vllm.model_executor.models import qwen2_5_vl, qwen3_vl
    from vllm.v1.attention.ops import vit_attn_wrappers

    modules = {
        "qwen3_vl": qwen3_vl,
        "qwen2_5_vl": qwen2_5_vl,
        "vit_attn_wrappers": vit_attn_wrappers,
        "rotary_common": rotary_common,
        "linear_utils": linear_utils,
    }
    paths = {name: Path(inspect.getfile(module)).resolve() for name, module in modules.items()}
    for name, path in paths.items():
        if sha256_file(path) != SOURCE_HASHES[name]:
            raise RuntimeError(f"frozen serving source differs: {name}")
    return {name: str(path) for name, path in paths.items()}


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
        raise ValueError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required for the isolated "
            "offline apply_model vision block hooks"
        )
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput

    base = _load_base_capture()
    output_root = args.output_root.resolve()
    fixture_root = args.fixture_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"oracle output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source = git_identity(ROOT)
    if source["dirty"]:
        raise ValueError("vision block capture source must be a clean commit")

    reference, _launch, _processor = base._require_inputs(
        reference_path=args.reference_manifest,
        launch_path=args.launch_config,
        processor_path=args.processor_probe,
        fixture_root=fixture_root,
    )
    if socket.gethostname() != reference.get("host", {}).get("hostname"):
        raise ValueError("vision block capture host differs from the frozen reference")
    vl_manifest_path = args.vl_oracle_manifest.resolve()
    if sha256_file(vl_manifest_path) != VL_ORACLE_SHA256:
        raise ValueError("VL oracle manifest differs from the frozen contract")
    vl_manifest = load_json_object(vl_manifest_path)
    oracle_errors = verify_manifest_integrity(vl_manifest)
    if oracle_errors:
        raise ValueError("VL oracle manifest integrity failed")
    position_manifest_path = args.position_oracle_root.resolve() / "manifest.json"
    if sha256_file(position_manifest_path) != POSITION_ORACLE_SHA256:
        raise ValueError("position oracle manifest differs from the serving contract")
    versions = _runtime_versions()
    for name, expected in PINNED_PACKAGES.items():
        actual = versions.get(name)
        if actual != expected and not actual.startswith(expected + "."):
            raise ValueError(f"runtime pin mismatch for {name}: {actual!r}")
    serving_sources = _verify_serving_sources()

    specs = {item["case_id"]: item for item in base.CASE_SPECS}
    vl_cases = {item["case_id"]: item for item in vl_manifest["cases"]}
    llm_kwargs = base._llm_kwargs(args.model_dir.resolve(), fixture_root)
    # The independent full-model oracle already froze maximum-budget encoder
    # profiling. This capture needs only two real small-media forwards, and its
    # final block output must remain byte-identical to that independent oracle.
    llm_kwargs["skip_mm_profiling"] = True
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0, max_tokens=1, seed=0)
    cases: list[dict[str, Any]] = []
    try:
        for case_id in CASE_IDS:
            llm.reset_mm_cache()
            llm.llm_engine.reset_encoder_cache()
            messages = base._build_messages(specs[case_id], fixture_root)
            engine_input = llm._preprocess_chat_one(
                messages, chat_template_content_format="openai"
            )

            def install_callable(model: Any) -> dict[str, Any]:
                return InstallVisionBlockHooks(
                    output_root=str(output_root), case_id=case_id
                )(model)

            def finalize_callable(model: Any) -> dict[str, Any]:
                return FinalizeVisionBlockHooks()(model)

            def cleanup_callable(model: Any) -> bool:
                return RemoveVisionBlockHooks()(model)

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
            if len(outputs) != 1 or len(installation) != 1 or len(finalization) != 1:
                raise RuntimeError("vision block capture requires one request and TP=1")
            record = finalization[0]
            expected = vl_cases[case_id]["boundaries"]["vision_block_0"]
            expected_relative = Path(expected["path"])
            expected_copy = output_root / case_id / "full_model_block_output.bin"
            expected_source = args.vl_oracle_root.resolve() / expected_relative
            payload = expected_source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise RuntimeError(f"full-model block oracle failed verification: {case_id}")
            _write_atomic_bytes(expected_copy, payload)
            expected_component = dict(expected)
            expected_component["path"] = str(expected_copy.relative_to(output_root))
            comparison = _compare_component(
                record["components"]["block_output"], expected_component, output_root
            )
            if not comparison["exact"]:
                raise RuntimeError(
                    f"independent block capture is not byte-identical: {case_id}"
                )
            cases.append(
                {
                    "case_id": case_id,
                    "model_modules": installation[0],
                    "components": record["components"],
                    "metadata": record["metadata"],
                    "full_model_block_output": expected_component,
                    "independent_comparison": comparison,
                }
            )
    finally:
        del llm

    result = {
        "schema": SCHEMA,
        "complete": True,
        "qualified_for_native_boundary_comparison": True,
        "source": source,
        "model": {
            "revision": MODEL_REVISION,
            "reference_manifest_sha256": sha256_file(args.reference_manifest),
            "vl_oracle_manifest_sha256": VL_ORACLE_SHA256,
            "position_oracle_manifest_sha256": POSITION_ORACLE_SHA256,
        },
        "runtime": versions,
        "reference": {
            "attention_backend": REFERENCE_ATTENTION_BACKEND,
            "enforce_eager": True,
            "skip_mm_profiling": True,
            "vllm_allow_insecure_serialization": True,
            "serving_source_sha256": SOURCE_HASHES,
            "serving_source_paths": serving_sources,
        },
        "cases": cases,
    }
    return seal_manifest(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--processor-probe", type=Path, required=True)
    parser.add_argument("--vl-oracle-manifest", type=Path, required=True)
    parser.add_argument("--vl-oracle-root", type=Path, required=True)
    parser.add_argument("--position-oracle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args)
    atomic_json(args.output_root.resolve() / "manifest.json", result)
    print(args.output_root.resolve() / "manifest.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"capture vision block oracle: {error}", file=sys.stderr)
        raise SystemExit(1)
