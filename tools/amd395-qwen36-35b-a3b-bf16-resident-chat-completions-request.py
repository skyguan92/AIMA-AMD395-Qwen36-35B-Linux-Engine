#!/usr/bin/env python3
"""Run one OpenAI-shaped request through the in-process resident engine path."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = "/data/models/Qwen3.6-35B-A3B"
DEFAULT_MODEL_ID = "aima-amd395-qwen36-35b"
DEFAULT_LAYERS = list(range(40))
DEFAULT_LINEAR_ATTENTION_VARIANT = "vllm_fla_auto_prestates_native_refswap_chunk32"
DEFAULT_LINEAR_ATTENTION_POST_CONV_PREP_BLOCK_T = 256
DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_BLOCK_T = None
DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_BLOCK_C = None
DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_NUM_WARPS = None
DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_POST_PREP_FUSION = False
DEFAULT_LINEAR_ATTENTION_PREFILL_VLLM_STATE_HANDOFF = False
DEFAULT_LINEAR_ATTENTION_PREFILL_FUSED_H_O = True
DEFAULT_LINEAR_ATTENTION_PREFILL_FUSED_U_H_O = True
DEFAULT_LINEAR_ATTENTION_CHUNK_GDN_INTERNAL_TIMING = False
DEFAULT_LINEAR_ATTENTION_CONV_STATE_REFSWAP = False
DEFAULT_OVERLAP_SHARED_EXPERT_MOE = True
DEFAULT_OVERLAP_SHARED_EXPERT_ROUTER_MOE = True
DEFAULT_SHARED_EXPERT_OVERLAP_STREAM_PRIORITY = None
DEFAULT_DECODE_LOOP_FAST_HOUSEKEEPING = False
DEFAULT_DEFER_DECODE_TOKEN_CPU_SYNC = False
DEFAULT_DECODE_TOKEN_CPU_SYNC_INTERVAL = 1
DEFAULT_DECODE_LOOP_DIAGNOSTIC = False
DEFAULT_OVERLAP_DECODE_STATE_PROMOTION_LM_HEAD = False
DEFAULT_FULL_ATTENTION_FUSED_GATE_O_PROJ = False
DEFAULT_FULL_ATTENTION_FUSED_NORM_ROPE_KV_WRITE = False
DEFAULT_FULL_ATTENTION_KV_CACHE_LAYOUT = "seq"
DEFAULT_SKIP_LAYER_DISPATCH_METADATA = False
DEFAULT_RESIDENT_NATIVE_DECODE_HOTSET_LAYERS = 13
DEFAULT_EXACT_PREFIX_CACHE = False
DEFAULT_EXACT_PREFIX_CACHE_MAX_ENTRIES = 1
DEFAULT_EXACT_PREFIX_CACHE_MAX_TOKENS = 8192
DEFAULT_ADMITTED_CONTEXT_POLICY = True
MAX_REQUEST_PROMPT_TOKEN_IDS = 262144
STARTUP_MAX2_PREWARM_COUNT = 2
_startup_max2_prewarms_done = False
_startup_max2_prewarms_in_progress = False
_context_policy_module: Any | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "output" / f"resident-chat-completions-request-{stamp}"


def load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context_policy_module() -> Any:
    global _context_policy_module
    if _context_policy_module is None:
        _context_policy_module = load_script_module(
            "amd395_aotriton_context_policy",
            repo_root() / "tools" / "amd395-qwen36-35b-a3b-bf16-aotriton-context-policy.py",
        )
    return _context_policy_module


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_contract(
    *,
    engine: Any,
    manifest: dict[str, Any],
    model_dir: Path,
    prompt_token_ids: list[int],
    args: argparse.Namespace,
    decode_stop_token_ids: list[int],
    reuse_tensor_cache: bool,
) -> dict[str, Any]:
    contract = engine.planned_contract(
        manifest,
        model_dir,
        DEFAULT_LAYERS,
        "prefill",
        args.seq_len,
        len(prompt_token_ids),
        "linear_and_full_attention",
        args.moe_variant,
        None,
        getattr(args, "moe_override_config_by_layer", None),
        args.overlap_shared_expert_moe,
        args.overlap_shared_expert_router_moe,
        "triton_topk_softmax",
        DEFAULT_LINEAR_ATTENTION_VARIANT,
        "prefill_fused_t_decode_fused_t_conv_triton",
        "triton_matvec",
        "decode_direct_triton",
        args.linear_attention_conv_state_refswap,
        "triton",
        args.linear_attention_post_conv_prep_block_t,
        args.linear_attention_prefill_conv_block_t,
        args.linear_attention_prefill_conv_block_c,
        args.linear_attention_prefill_conv_num_warps,
        args.linear_attention_prefill_conv_post_prep_fusion,
        args.linear_attention_prefill_vllm_state_handoff,
        args.linear_attention_prefill_fused_h_o,
        args.linear_attention_prefill_fused_u_h_o,
        args.linear_attention_chunk_gdn_internal_timing,
        "triton",
        "decode_grouped_bmm_bf16",
        "triton_fused_qkv_matvec",
        "triton",
        args.full_attention_kv_cache_layout,
        args.full_attention_fused_gate_o_proj,
        "int8_certified_global_tie",
        "triton_fused_in_matvec",
        False,
        args.skip_layer_dispatch_metadata,
        True,
        "token_ids",
        True,
        full_attention_fused_norm_rope_kv_write=args.full_attention_fused_norm_rope_kv_write,
        shared_expert_overlap_stream_priority=getattr(args, "shared_expert_overlap_stream_priority", None),
    )
    contract["include_shared_expert"] = True
    contract["decode_output_token"] = True
    contract["logit_topk"] = args.logit_topk
    contract["decode_loop_steps"] = max(0, args.max_tokens - 1)
    contract["prefill_seed_output"] = True
    contract["decode_loop_fast_housekeeping"] = args.decode_loop_fast_housekeeping
    contract["decode_loop_defer_token_cpu_sync"] = args.defer_decode_token_cpu_sync
    contract["decode_loop_token_cpu_sync_interval"] = args.effective_decode_token_cpu_sync_interval
    contract["decode_loop_diagnostic"] = args.decode_loop_diagnostic
    contract["overlap_decode_state_promotion_lm_head"] = args.overlap_decode_state_promotion_lm_head
    contract["shared_expert_overlap_stream_priority"] = getattr(
        args,
        "shared_expert_overlap_stream_priority",
        None,
    )
    contract["skip_layer_dispatch_metadata"] = args.skip_layer_dispatch_metadata
    contract["decode_sampling"] = "argmax"
    contract["sampling_temperature"] = args.sampling_temperature
    contract["sampling_top_k"] = args.sampling_top_k
    contract["decode_stop_token_ids"] = decode_stop_token_ids
    contract["attention_substage_timing"] = False
    contract["moe_substage_timing"] = False
    contract["collect_resident_stage_timeline"] = args.collect_resident_stage_timeline
    contract["measurement_mode"] = "resident_only"
    contract["reuse_tensor_cache"] = reuse_tensor_cache
    contract["resident_native_decode_hotset_layers"] = (
        args.resident_native_decode_hotset_layers
    )
    contract["exact_prefix_cache"] = args.exact_prefix_cache
    contract["exact_prefix_cache_max_entries"] = args.exact_prefix_cache_max_entries
    contract["exact_prefix_cache_max_tokens"] = args.exact_prefix_cache_max_tokens
    contract["exact_prefix_cache_matching_rule"] = "longest_exact_token_prefix_and_runtime_contract"
    contract["admitted_context_policy"] = args.admitted_context_policy
    contract["admitted_context_policy_selection"] = getattr(
        args, "admitted_context_policy_selection", None
    )
    contract["resident_request_repeats"] = args.repeat_same_request
    contract["input_token_ids_sha256"] = engine.token_ids_digest(prompt_token_ids)
    contract["input_token_count"] = len(prompt_token_ids)
    return contract


def run_engine_once(
    *,
    engine: Any,
    manifest: dict[str, Any],
    model_dir: Path,
    prompt_token_ids: list[int],
    user_prompt: str,
    decode_stop_token_ids: list[int],
    args: argparse.Namespace,
    reuse_tensor_cache: bool,
    load_only: bool = False,
) -> dict[str, Any]:
    global _startup_max2_prewarms_done, _startup_max2_prewarms_in_progress
    startup_max2_prewarms: list[dict[str, Any]] = []
    if (
        not load_only
        and not _startup_max2_prewarms_done
        and not _startup_max2_prewarms_in_progress
    ):
        _startup_max2_prewarms_in_progress = True
        try:
            prewarm_args = argparse.Namespace(**vars(args))
            prewarm_args.max_tokens = 2
            for index in range(STARTUP_MAX2_PREWARM_COUNT):
                prewarm_result = run_engine_once(
                    engine=engine,
                    manifest=manifest,
                    model_dir=model_dir,
                    prompt_token_ids=prompt_token_ids,
                    user_prompt=user_prompt,
                    decode_stop_token_ids=decode_stop_token_ids,
                    args=prewarm_args,
                    reuse_tensor_cache=reuse_tensor_cache,
                )
                prewarm_measurement = prewarm_result["measurement"]
                prewarm_cache = prewarm_measurement["engine_tensor_cache"]
                prewarm_owner = prewarm_cache["resident_native_prefill"]
                startup_max2_prewarms.append(
                    {
                        "index": index,
                        "max_tokens": 2,
                        "pipeline_ms": prewarm_measurement["pipeline"]["resident_ms_per_iter"],
                        "decode_loop_ms": prewarm_measurement["decode_loop"]["elapsed_ms"],
                        "prefill_state_source": prewarm_measurement["decode_loop"]["prefill_state_source"],
                        "completion_token_ids": [
                            int(prewarm_measurement["text_smoke"]["generated_token_id"]),
                            *[
                                int(item)
                                for item in prewarm_measurement["decode_loop"]["visible_generated_token_ids"]
                            ],
                        ],
                        "completion_token_ids_sha256": engine.token_ids_digest(
                            [
                                int(prewarm_measurement["text_smoke"]["generated_token_id"]),
                                *[
                                    int(item)
                                    for item in prewarm_measurement["decode_loop"]["visible_generated_token_ids"]
                                ],
                            ]
                        ),
                        "decode_suffix_token_ids_sha256": prewarm_measurement["decode_loop"]["visible_generated_token_ids_sha256"],
                        "raw_weight_misses": prewarm_cache["misses_by_scope"]["raw_weights"],
                        "layout_entries_at_load": prewarm_owner["layout_entries_at_load"],
                        "hotset_entries_after_run": prewarm_owner["native_decode_hotset_cache_entries_after_run"],
                        "peak_memory_bytes": prewarm_measurement["peak_memory_bytes"],
                    }
                )
        finally:
            _startup_max2_prewarms_in_progress = False
        if len(startup_max2_prewarms) != STARTUP_MAX2_PREWARM_COUNT:
            raise RuntimeError("startup max2 prewarm sequence incomplete")
        _startup_max2_prewarms_done = True

    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        measurement = engine.run_with_torch(
            manifest=manifest,
            model_dir=model_dir,
            layers=DEFAULT_LAYERS,
            mode="prefill",
            seq_len=args.seq_len,
            tokens=len(prompt_token_ids),
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            moe_chunk_size=args.moe_chunk_size,
            attention_mode="linear_and_full_attention",
            moe_variant=args.moe_variant,
            moe_override_config=None,
            moe_override_config_by_layer=getattr(args, "moe_override_config_by_layer", None),
            overlap_shared_expert_moe=args.overlap_shared_expert_moe,
            overlap_shared_expert_router_moe=args.overlap_shared_expert_router_moe,
            shared_expert_overlap_stream_priority=getattr(args, "shared_expert_overlap_stream_priority", None),
            router_variant="triton_topk_softmax",
            linear_attention_variant=DEFAULT_LINEAR_ATTENTION_VARIANT,
            linear_attention_input_proj_variant="prefill_fused_t_decode_fused_t_conv_triton",
            linear_attention_output_proj_variant="triton_matvec",
            linear_attention_conv_variant="decode_direct_triton",
            linear_attention_conv_state_refswap=args.linear_attention_conv_state_refswap,
            linear_attention_gated_norm_variant="triton",
            linear_attention_post_conv_prep_block_t=args.linear_attention_post_conv_prep_block_t,
            linear_attention_prefill_conv_block_t=args.linear_attention_prefill_conv_block_t,
            linear_attention_prefill_conv_block_c=args.linear_attention_prefill_conv_block_c,
            linear_attention_prefill_conv_num_warps=args.linear_attention_prefill_conv_num_warps,
            linear_attention_prefill_conv_post_prep_fusion=args.linear_attention_prefill_conv_post_prep_fusion,
            linear_attention_prefill_vllm_state_handoff=args.linear_attention_prefill_vllm_state_handoff,
            linear_attention_prefill_fused_h_o=args.linear_attention_prefill_fused_h_o,
            linear_attention_prefill_fused_u_h_o=args.linear_attention_prefill_fused_u_h_o,
            linear_attention_chunk_gdn_internal_timing=args.linear_attention_chunk_gdn_internal_timing,
            rmsnorm_variant="triton",
            full_attention_variant="decode_grouped_bmm_bf16",
            full_attention_proj_variant="triton_fused_qkv_matvec",
            full_attention_norm_rope_variant="triton",
            full_attention_kv_cache_layout=args.full_attention_kv_cache_layout,
            full_attention_fused_gate_o_proj=args.full_attention_fused_gate_o_proj,
            full_attention_fused_norm_rope_kv_write=args.full_attention_fused_norm_rope_kv_write,
            lm_head_variant="int8_certified_global_tie",
            shared_expert_proj_variant="triton_fused_in_matvec",
            include_shared_expert=True,
            input_token_ids=prompt_token_ids,
            input_text=user_prompt,
            include_lm_head=True,
            decode_output_token=True,
            logit_topk=args.logit_topk,
            attention_substage_timing=False,
            moe_substage_timing=False,
            collect_resident_stage_timeline=args.collect_resident_stage_timeline,
            measurement_mode="resident_only",
            decode_loop_steps=max(0, args.max_tokens - 1),
            decode_sampling="argmax",
            sampling_temperature=args.sampling_temperature,
            sampling_top_k=args.sampling_top_k,
            decode_stop_token_ids=set(decode_stop_token_ids),
            prefill_seed_output=True,
            decode_loop_fast_housekeeping=args.decode_loop_fast_housekeeping,
            decode_loop_defer_token_cpu_sync=args.defer_decode_token_cpu_sync,
            decode_loop_token_cpu_sync_interval=args.effective_decode_token_cpu_sync_interval,
            decode_loop_diagnostic=args.decode_loop_diagnostic,
            overlap_decode_state_promotion_lm_head=args.overlap_decode_state_promotion_lm_head,
            skip_layer_dispatch_metadata=args.skip_layer_dispatch_metadata,
            reuse_tensor_cache=reuse_tensor_cache,
            resident_native_decode_hotset_layers=args.resident_native_decode_hotset_layers,
            exact_prefix_cache=args.exact_prefix_cache,
            exact_prefix_cache_max_entries=args.exact_prefix_cache_max_entries,
            exact_prefix_cache_max_tokens=args.exact_prefix_cache_max_tokens,
            load_only=load_only,
        )
    return {
        "measurement": measurement,
        "captured_stdout": captured_stdout.getvalue(),
        "startup_max2_prewarms": startup_max2_prewarms,
        "startup_max2_prewarms_executed": len(startup_max2_prewarms),
        "startup_max2_prewarms_done": _startup_max2_prewarms_done,
    }


def prepare_resident_engine(
    *,
    engine: Any,
    manifest: dict[str, Any],
    model_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    global _startup_max2_prewarms_done
    prepare_args = argparse.Namespace(**vars(args))
    prepare_args.max_tokens = 2
    prepared = run_engine_once(
        engine=engine,
        manifest=manifest,
        model_dir=model_dir,
        prompt_token_ids=[0],
        user_prompt="",
        decode_stop_token_ids=[],
        args=prepare_args,
        reuse_tensor_cache=True,
        load_only=True,
    )
    measurement = prepared["measurement"]
    if measurement.get("load_only") is not True or measurement.get("model_loaded") is not True:
        raise RuntimeError("resident engine load-only preparation did not report model_loaded")
    _startup_max2_prewarms_done = True
    measurement["service_request_prewarm_policy"] = {
        "same_prompt_max2_prewarms_suppressed": True,
        "suppressed_prewarms": STARTUP_MAX2_PREWARM_COUNT,
        "reason": "user requests must not own hidden compile or exact-prefix warmup work",
    }
    return measurement


def build_generation(
    *,
    skeleton: Any,
    engine_result: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = skeleton.metrics_for_result(engine_result)
    case = {
        "name": "prefill",
        "artifact": "resident-direct",
        "stdout": None,
        "stderr": None,
        "command": None,
        "result": engine_result,
        "metrics": metrics,
    }
    generation = skeleton.generation_report(metadata, [case])
    if not isinstance(generation, dict):
        raise RuntimeError("resident request did not produce generation metrics")
    validate_generation_contract(generation)
    return metrics, generation


def validate_generation_contract(generation: dict[str, Any]) -> None:
    requested = generation.get("requested_completion_tokens")
    completion = generation.get("completion_tokens")
    visible = generation.get("visible_completion_tokens")
    stop_reason = generation.get("stop_reason")
    if not isinstance(requested, int) or requested <= 0:
        raise RuntimeError("generation report is missing a positive requested completion count")
    if not isinstance(completion, int) or completion <= 0 or completion > requested:
        raise RuntimeError(
            f"generation report returned invalid completion count {completion!r} for request {requested}"
        )
    if not isinstance(visible, int) or visible < 0 or visible > completion:
        raise RuntimeError(
            f"generation report returned invalid visible completion count {visible!r}"
        )
    if stop_reason == "length":
        if completion != requested:
            raise RuntimeError("length-finished generation did not fill the requested completion budget")
    elif stop_reason == "stop_token":
        stop_ids = generation.get("stop_token_ids")
        stopped_on = generation.get("stopped_on_token_id")
        if not isinstance(stop_ids, list) or stopped_on not in stop_ids:
            raise RuntimeError("stop-token generation is missing its matched stop token")
    else:
        raise RuntimeError(f"generation report has unsupported stop reason {stop_reason!r}")
    if generation.get("token_count_matches_request") is not True:
        raise RuntimeError("generation report failed its completion-token contract")


def build_response(
    *,
    adapter: Any,
    request: dict[str, Any],
    created: int,
    output_dir: Path,
    metadata: dict[str, Any],
    generation: dict[str, Any],
    resident_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    model_id = str(request.get("model") or DEFAULT_MODEL_ID)
    content = generation.get("completion_text")
    if not isinstance(content, str):
        content = ""
    prompt_tokens = generation.get("prompt_tokens")
    # OpenAI/vLLM usage counts every sampled token, including a terminal stop
    # token that is intentionally omitted from the assistant-visible content.
    completion_tokens = generation.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        completion_tokens = generation.get("visible_completion_tokens")
    if not isinstance(prompt_tokens, int):
        prompt_tokens = 0
    if not isinstance(completion_tokens, int):
        completion_tokens = 0
    return {
        "id": adapter.request_id(request, created),
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "system_fingerprint": "aima-amd395-qwen36-v1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": adapter.finish_reason(generation.get("stop_reason")),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "aima_amd395": {
            "artifact_dir": str(output_dir),
            "summary_json": str(output_dir / "summary.json"),
            "source_commit": metadata.get("source_commit"),
            "source_diff_scope": metadata.get("source_diff_scope"),
            "route": "in-process resident tensor-cache engine",
            "ttft_ms": generation.get("ttft_ms"),
            "tpot_ms": generation.get("tpot_ms"),
            "decode_tok_s": generation.get("decode_tok_s"),
            "total_latency_ms": generation.get("total_latency_ms"),
            "prefill_tok_s": generation.get("prefill_tok_s"),
            "prefill_state_reused": generation.get("prefill_state_reused"),
            "stop_reason": generation.get("stop_reason"),
            "admitted_context_policy": metadata.get("admitted_context_policy_selection"),
            "resident_runs": resident_runs,
        },
    }


def response_markdown(response: dict[str, Any]) -> str:
    choice = response["choices"][0]
    metrics = response["aima_amd395"]
    final_run = metrics["resident_runs"][-1] if metrics.get("resident_runs") else {}
    return "\n".join(
        [
            "# Resident Chat Completions Request",
            "",
            f"- id: `{response['id']}`",
            f"- model: `{response['model']}`",
            f"- finish reason: `{choice['finish_reason']}`",
            f"- source commit: `{metrics.get('source_commit')}`",
            f"- artifact dir: `{metrics.get('artifact_dir')}`",
            f"- TTFT ms: `{metrics.get('ttft_ms')}`",
            f"- TPOT ms: `{metrics.get('tpot_ms')}`",
            f"- decode tok/s: `{metrics.get('decode_tok_s')}`",
            f"- total latency ms: `{metrics.get('total_latency_ms')}`",
            f"- final engine wall time ms: `{final_run.get('engine_wall_time_ms')}`",
            f"- final cache hits: `{final_run.get('cache_hits')}`",
            "",
            "## Assistant",
            "",
            str(choice["message"]["content"]),
            "",
        ]
    )


def compact_resident_run(index: int, measurement: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    stage = measurement.get("engine_stage_wall_time_ms")
    stage = stage if isinstance(stage, dict) else {}
    cache = measurement.get("engine_tensor_cache")
    cache = cache if isinstance(cache, dict) else {}
    native_prefill = cache.get("resident_native_prefill")
    native_prefill = native_prefill if isinstance(native_prefill, dict) else {}
    prefix_cache = measurement.get("exact_prefix_cache")
    prefix_cache = prefix_cache if isinstance(prefix_cache, dict) else {}
    prefill_seed_token_id = metrics.get("decode_loop_prefill_seed_token_id")
    prefill_seed_token_text = metrics.get("decode_loop_prefill_seed_token_text")
    if not isinstance(prefill_seed_token_id, int):
        prefill_seed_token_id = metrics.get("generated_token_id")
        prefill_seed_token_text = metrics.get("generated_token_text")
    return {
        "run_index": index,
        "engine_wall_time_ms": measurement.get("engine_wall_time_ms"),
        "engine_stage_wall_time_ms": stage,
        "cache_hits": cache.get("hits"),
        "cache_misses": cache.get("misses"),
        "cache_entries_after_run": cache.get("entries_after_run"),
        "native_decode_hotset_layers_requested": native_prefill.get(
            "native_decode_hotset_layers_requested"
        ),
        "native_decode_hotset_layer_indices": native_prefill.get(
            "native_decode_hotset_layer_indices"
        ),
        "native_decode_hotset_layers_available": native_prefill.get(
            "native_decode_hotset_layers_available"
        ),
        "native_decode_hotset_cache_entries_after_run": native_prefill.get(
            "native_decode_hotset_cache_entries_after_run"
        ),
        "exact_prefix_cache_enabled": prefix_cache.get("enabled"),
        "exact_prefix_cache_hit": prefix_cache.get("hit"),
        "exact_prefix_cache_lookup": prefix_cache.get("lookup"),
        "exact_prefix_cache_match_kind": prefix_cache.get("match_kind"),
        "exact_prefix_cache_matched_tokens": prefix_cache.get("matched_tokens"),
        "exact_prefix_cache_suffix_tokens": prefix_cache.get("suffix_tokens"),
        "exact_prefix_cache_request_tokens": prefix_cache.get("request_tokens"),
        "exact_prefix_cache_restore_ms": (
            prefix_cache.get("restore", {}).get("wall_time_ms")
            if isinstance(prefix_cache.get("restore"), dict)
            else None
        ),
        "exact_prefix_cache_retained_bytes": prefix_cache.get("retained_bytes"),
        "exact_prefix_cache_suffix_pipeline_ms": prefix_cache.get("suffix_pipeline_ms"),
        "exact_prefix_cache_restore_plus_suffix_prefill_ms": prefix_cache.get(
            "prefix_restore_plus_suffix_prefill_ms"
        ),
        "resident_ms_per_iter": metrics.get("pipeline_ms"),
        "ttft_ms": metrics.get("text_path_ms"),
        "decode_loop_elapsed_ms": metrics.get("decode_loop_elapsed_ms"),
        "decode_loop_tokens_per_s": metrics.get("decode_loop_tokens_per_s"),
        "decode_loop_prefill_state_source": metrics.get("decode_loop_prefill_state_source"),
        "decode_loop_fast_housekeeping": metrics.get("decode_loop_fast_housekeeping"),
        "decode_loop_defer_token_cpu_sync": metrics.get("decode_loop_defer_token_cpu_sync"),
        "decode_loop_token_cpu_sync_interval": metrics.get("decode_loop_token_cpu_sync_interval"),
        "decode_loop_diagnostic": measurement.get("decode_loop_diagnostic"),
        "overlap_decode_state_promotion_lm_head": metrics.get("overlap_decode_state_promotion_lm_head"),
        "prefill_seed_token_id": prefill_seed_token_id,
        "prefill_seed_token_text": prefill_seed_token_text,
        "admitted_context_policy": measurement.get("admitted_context_policy"),
    }


def prompt_token_ids_from_request(
    *,
    request: dict[str, Any],
    skeleton: Any,
    model_dir: Path,
    user_prompt: str,
    system_prompt: str,
    chat_disable_thinking: bool,
) -> tuple[list[int], str]:
    supplied = request.get("prompt_token_ids")
    if supplied is None:
        return (
            skeleton.tokenize_text(
                model_dir,
                user_prompt,
                "chat-completions",
                system_prompt,
                chat_disable_thinking,
            ),
            "messages_chat_template",
        )
    if not isinstance(supplied, list) or not supplied:
        raise SystemExit("prompt_token_ids must be a non-empty list when supplied")
    if len(supplied) > MAX_REQUEST_PROMPT_TOKEN_IDS:
        raise SystemExit(
            f"prompt_token_ids exceeds the {MAX_REQUEST_PROMPT_TOKEN_IDS}-token limit"
        )
    for index, token_id in enumerate(supplied):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise SystemExit(f"prompt_token_ids[{index}] must be an integer token id")
    tokenizer = skeleton.cached_tokenizer(model_dir, "request.prompt_token_ids")
    tokenizer_size = len(tokenizer)
    for index, token_id in enumerate(supplied):
        if token_id < 0 or token_id >= tokenizer_size:
            raise SystemExit(
                f"prompt_token_ids[{index}]={token_id} is outside tokenizer range "
                f"[0, {tokenizer_size})"
            )
    return [int(token_id) for token_id in supplied], "request_prompt_token_ids"


def execute_resident_request(
    *,
    adapter: Any,
    skeleton: Any,
    engine: Any,
    request: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    model_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    source_commit: str,
    source_diff: str,
    request_json_path: Path | None,
) -> dict[str, Any]:
    adapter.validate_request(request)
    system_prompt, user_prompt = adapter.parse_messages(request, "")
    max_tokens = adapter.max_tokens_from_request(request)
    request_args = argparse.Namespace(**vars(args))
    request_args.max_tokens = max_tokens
    chat_disable_thinking = bool(getattr(request_args, "chat_disable_thinking", False))

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "request.json", request)

    prompt_token_ids, prompt_token_ids_source = prompt_token_ids_from_request(
        request=request,
        skeleton=skeleton,
        model_dir=model_dir,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        chat_disable_thinking=chat_disable_thinking,
    )
    decode_stop_token_ids = skeleton.chat_stop_token_ids(model_dir)
    context_policy = context_policy_module()
    context_selection = context_policy.select_policy(
        prompt_tokens=len(prompt_token_ids),
        enabled=bool(request_args.admitted_context_policy),
        exact_prefix_cache=bool(request_args.exact_prefix_cache),
        exact_prefix_cache_max_tokens=int(request_args.exact_prefix_cache_max_tokens),
        fallback_layout=str(request_args.full_attention_kv_cache_layout),
    )
    request_args.full_attention_kv_cache_layout = context_selection["kv_layout"]
    request_args.admitted_context_policy_selection = context_selection
    prompt_info = {
        "system_prompt_sha256": adapter.sha256_text(system_prompt),
        "system_prompt_chars": len(system_prompt),
        "user_prompt_sha256": adapter.sha256_text(user_prompt),
        "user_prompt_chars": len(user_prompt),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_source": prompt_token_ids_source,
        "prompt_token_ids_sha256": engine.token_ids_digest(prompt_token_ids),
        "prompt_tokens": len(prompt_token_ids),
        "decode_stop_token_ids": decode_stop_token_ids,
        "max_tokens": max_tokens,
    }
    write_json(output_dir / "prompt.json", prompt_info)

    reuse_tensor_cache = args.reuse_tensor_cache or args.repeat_same_request > 1 or args.stdin_loop
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "model_dir": str(model_dir),
        "source_commit": source_commit,
        "source_diff_scope": source_diff,
        "manifest": str(manifest_path),
        "manifest_sha256": skeleton.sha256_file(manifest_path),
        "request_json": str(request_json_path.resolve()) if request_json_path else None,
        "request_sha256": adapter.sha256_text(json.dumps(request, sort_keys=True, separators=(",", ":"))),
        "execute": args.execute,
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "layers": ",".join(str(item) for item in DEFAULT_LAYERS),
        "prompt_format": "chat-completions",
        "chat_disable_thinking": chat_disable_thinking,
        "moe_variant": args.moe_variant,
        "chat_stop_tokens": True,
        "system_prompt": system_prompt,
        "system_prompt_sha256": prompt_info["system_prompt_sha256"],
        "system_prompt_chars": len(system_prompt),
        "prompt_text": user_prompt,
        "prompt_text_sha256": prompt_info["user_prompt_sha256"],
        "prompt_text_chars": len(user_prompt),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_source": prompt_token_ids_source,
        "prompt_token_ids_sha256": prompt_info["prompt_token_ids_sha256"],
        "seq_len": args.seq_len,
        "decode_input_token_id": prompt_token_ids[-1],
        "case_mode": "generation",
        "generation_token_count": max_tokens,
        "decode_stop_token_ids": decode_stop_token_ids,
        "measurement_mode": "resident_only",
        "collect_resident_stage_timeline": args.collect_resident_stage_timeline,
        "linear_attention_post_conv_prep_block_t": args.linear_attention_post_conv_prep_block_t,
        "linear_attention_prefill_conv_block_t": args.linear_attention_prefill_conv_block_t,
        "linear_attention_prefill_conv_block_c": args.linear_attention_prefill_conv_block_c,
        "linear_attention_prefill_conv_num_warps": args.linear_attention_prefill_conv_num_warps,
        "linear_attention_prefill_conv_effective_block_t": args.linear_attention_prefill_conv_block_t or 16,
        "linear_attention_prefill_conv_effective_block_c": args.linear_attention_prefill_conv_block_c or 32,
        "linear_attention_prefill_conv_effective_num_warps": args.linear_attention_prefill_conv_num_warps or 4,
        "linear_attention_prefill_conv_post_prep_fusion": args.linear_attention_prefill_conv_post_prep_fusion,
        "linear_attention_prefill_vllm_state_handoff": args.linear_attention_prefill_vllm_state_handoff,
        "linear_attention_prefill_fused_h_o": args.linear_attention_prefill_fused_h_o,
        "linear_attention_prefill_fused_u_h_o": args.linear_attention_prefill_fused_u_h_o,
        "linear_attention_chunk_gdn_internal_timing": args.linear_attention_chunk_gdn_internal_timing,
        "linear_attention_conv_state_refswap": args.linear_attention_conv_state_refswap,
        "moe_override_config_by_layer": getattr(args, "moe_override_config_by_layer", None),
        "overlap_shared_expert_moe": args.overlap_shared_expert_moe,
        "overlap_shared_expert_router_moe": args.overlap_shared_expert_router_moe,
        "shared_expert_overlap_stream_priority": getattr(args, "shared_expert_overlap_stream_priority", None),
        "decode_loop_fast_housekeeping": args.decode_loop_fast_housekeeping,
        "decode_loop_defer_token_cpu_sync": args.defer_decode_token_cpu_sync,
        "decode_loop_token_cpu_sync_interval": args.effective_decode_token_cpu_sync_interval,
        "decode_loop_diagnostic": args.decode_loop_diagnostic,
        "overlap_decode_state_promotion_lm_head": args.overlap_decode_state_promotion_lm_head,
        "full_attention_fused_gate_o_proj": args.full_attention_fused_gate_o_proj,
        "full_attention_kv_cache_layout": request_args.full_attention_kv_cache_layout,
        "admitted_context_policy": request_args.admitted_context_policy,
        "admitted_context_policy_selection": context_selection,
        "full_attention_fused_norm_rope_kv_write": args.full_attention_fused_norm_rope_kv_write,
        "skip_layer_dispatch_metadata": args.skip_layer_dispatch_metadata,
        "decode_sampling": "argmax",
        "reuse_tensor_cache": reuse_tensor_cache,
        "exact_prefix_cache": args.exact_prefix_cache,
        "exact_prefix_cache_max_entries": args.exact_prefix_cache_max_entries,
        "exact_prefix_cache_max_tokens": args.exact_prefix_cache_max_tokens,
        "resident_native_decode_hotset_layers": args.resident_native_decode_hotset_layers,
        "repeat_same_request": args.repeat_same_request,
        "stdin_loop": args.stdin_loop,
        "env": {
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": os.environ.get(
                "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
            ),
        },
    }

    resident_runs: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    final_generation: dict[str, Any] | None = None
    final_metrics: dict[str, Any] | None = None
    start_wall = time.perf_counter()
    for run_index in range(args.repeat_same_request):
        cache_eviction_pre_run = engine.evict_engine_tensor_cache(
            native_moe=not reuse_tensor_cache,
            reason=f"resident_request_run_{run_index}_preflight",
        )
        metadata["cache_eviction_pre_run"] = cache_eviction_pre_run
        with context_policy.bind_for_request(context_selection) as context_binding:
            run_result = run_engine_once(
                engine=engine,
                manifest=manifest,
                model_dir=model_dir,
                prompt_token_ids=prompt_token_ids,
                user_prompt=user_prompt,
                decode_stop_token_ids=decode_stop_token_ids,
                args=request_args,
                reuse_tensor_cache=reuse_tensor_cache,
            )
        measurement = run_result["measurement"]
        measurement["admitted_context_policy"] = {
            "selection": context_selection,
            "binding": context_binding,
        }
        contract = build_contract(
            engine=engine,
            manifest=manifest,
            model_dir=model_dir,
            prompt_token_ids=prompt_token_ids,
            args=request_args,
            decode_stop_token_ids=decode_stop_token_ids,
            reuse_tensor_cache=reuse_tensor_cache,
        )
        engine_result = {
            "schema_version": 1,
            "mode": "four-layer-mini-engine",
            "contract": contract,
            "executed": True,
            "measurement": measurement,
            "source": {"commit": source_commit, "diff_scope": source_diff},
        }
        metrics, generation = build_generation(
            skeleton=skeleton,
            engine_result=engine_result,
            metadata=metadata,
        )
        final_metrics = metrics
        final_generation = generation
        compact = compact_resident_run(run_index, measurement, metrics)
        resident_runs.append(compact)
        run_summaries.append(
            {
                "run_index": run_index,
                "cache_eviction_pre_run": cache_eviction_pre_run,
                "engine_result": engine_result,
                "metrics": metrics,
                "generation": generation,
                "captured_stdout": run_result.get("captured_stdout"),
            }
        )

    if final_generation is None or final_metrics is None:
        raise RuntimeError("resident request produced no runs")
    metadata["request_wall_time_ms"] = (time.perf_counter() - start_wall) * 1000.0
    summary = {
        "metadata": metadata,
        "resident_runs": run_summaries,
        "final_generation": final_generation,
    }
    write_json(output_dir / "summary.json", summary)
    created = int(time.time())
    response = build_response(
        adapter=adapter,
        request=request,
        created=created,
        output_dir=output_dir,
        metadata=metadata,
        generation=final_generation,
        resident_runs=resident_runs,
    )
    write_json(output_dir / "response.json", response)
    (output_dir / "response.md").write_text(response_markdown(response), encoding="utf-8")
    return response


def main() -> None:
    root = repo_root()
    shape_lab = root / "benchmarks" / "shape-lab"
    sys.path.insert(0, str(shape_lab))
    adapter = load_script_module(
        "aima_chat_contract",
        root / "tools" / "aima_chat_contract.py",
    )
    skeleton = load_script_module("amd395_run_full_model_skeleton", shape_lab / "run_full_model_skeleton.py")
    engine = load_script_module("amd395_four_layer_mini_engine", shape_lab / "four_layer_mini_engine.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--manifest", default=str(root / "doc/reference/amd395-qwen36-35b-a3b-bf16/qwen36-shape-manifest.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--moe-chunk-size", type=int, default=64)
    parser.add_argument(
        "--moe-variant",
        choices=sorted(engine.moe_variants()),
        default="vllm_fused_prefill_m32_n32_decode_m32_n16_k512",
    )
    parser.add_argument("--logit-topk", type=int, default=5)
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--sampling-top-k", type=int, default=50)
    parser.add_argument(
        "--chat-disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable thinking in the Qwen chat template for target-contract performance runs.",
    )
    parser.add_argument(
        "--moe-override-config-by-layer-json",
        help=(
            "Optional JSON object mapping model layer id to a vLLM fused_moe.override_config "
            "object for one-token decode."
        ),
    )
    parser.add_argument(
        "--linear-attention-post-conv-prep-block-t",
        type=int,
        choices=[8, 16, 32, 64, 128, 256],
        default=DEFAULT_LINEAR_ATTENTION_POST_CONV_PREP_BLOCK_T,
        help="Retained vLLM fused_post_conv_prep BLOCK_T for resident prefill.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-block-t",
        type=int,
        choices=[8, 16, 32, 64],
        default=DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_BLOCK_T,
        help="Retained Triton prefill causal-conv BLOCK_T override for resident prefill.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-block-c",
        type=int,
        choices=[16, 32, 64],
        default=DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_BLOCK_C,
        help="Retained Triton prefill causal-conv BLOCK_C override for resident prefill.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-num-warps",
        type=int,
        choices=[4, 8],
        default=DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_NUM_WARPS,
        help="Retained Triton prefill causal-conv num_warps override for resident prefill.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-post-prep-fusion",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_POST_PREP_FUSION,
        help="Default-off prefill candidate that fuses Triton causal-conv with post-conv prep.",
    )
    parser.add_argument(
        "--linear-attention-prefill-vllm-state-handoff",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LINEAR_ATTENTION_PREFILL_VLLM_STATE_HANDOFF,
        help=(
            "Keep prefill chunk-GDN final state in vLLM layout for native-vLLM "
            "decode state handoff."
        ),
    )
    parser.add_argument(
        "--linear-attention-prefill-fused-h-o",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LINEAR_ATTENTION_PREFILL_FUSED_H_O,
        help="Use the retained fused prefill chunk-GDN h/o boundary for chunk16 Qwen shapes.",
    )
    parser.add_argument(
        "--linear-attention-prefill-fused-u-h-o",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LINEAR_ATTENTION_PREFILL_FUSED_U_H_O,
        help="Use the retained W-only plus fused U+h/o prefill chunk-GDN boundary.",
    )
    parser.add_argument(
        "--linear-attention-chunk-gdn-internal-timing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LINEAR_ATTENTION_CHUNK_GDN_INTERNAL_TIMING,
        help="Collect a separate diagnostic timeline for retained prefill chunk-GDN internal vLLM FLA kernels.",
    )
    parser.add_argument(
        "--linear-attention-conv-state-refswap",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LINEAR_ATTENTION_CONV_STATE_REFSWAP,
        help="Probe one-token decode linear-attention causal-conv state promotion by tensor reference.",
    )
    parser.add_argument(
        "--collect-resident-stage-timeline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect the extra per-layer resident stage timeline diagnostic pass. Disabled by default for serving.",
    )
    parser.add_argument(
        "--decode-loop-fast-housekeeping",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DECODE_LOOP_FAST_HOUSEKEEPING,
        help="Precompute decode-loop RoPE rows and reuse the next-token embedding buffer.",
    )
    parser.add_argument(
        "--defer-decode-token-cpu-sync",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DEFER_DECODE_TOKEN_CPU_SYNC,
        help="Materialize generated token ids on CPU once after the decode loop.",
    )
    parser.add_argument(
        "--decode-token-cpu-sync-interval",
        type=int,
        default=DEFAULT_DECODE_TOKEN_CPU_SYNC_INTERVAL,
        help=(
            "Materialize generated token ids on CPU every N tokens; 1 preserves "
            "per-token sync and 0 defers until loop end."
        ),
    )
    parser.add_argument(
        "--decode-loop-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DECODE_LOOP_DIAGNOSTIC,
        help=(
            "Record per-step decode-loop top-k logits and linear-attention state "
            "statistics. Timings include diagnostic sync overhead."
        ),
    )
    parser.add_argument(
        "--overlap-decode-state-promotion-lm-head",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERLAP_DECODE_STATE_PROMOTION_LM_HEAD,
        help="Overlap decode state promotion with final RMSNorm/LM-head during decode loops.",
    )
    parser.add_argument(
        "--full-attention-fused-gate-o-proj",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FULL_ATTENTION_FUSED_GATE_O_PROJ,
        help="Probe fused full-attention output gate plus o_proj in one-token decode.",
    )
    parser.add_argument(
        "--full-attention-kv-cache-layout",
        choices=["seq", "grouped"],
        default=DEFAULT_FULL_ATTENTION_KV_CACHE_LAYOUT,
        help="Probe full-attention KV cache layout for grouped-BMM decode.",
    )
    parser.add_argument(
        "--admitted-context-policy",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ADMITTED_CONTEXT_POLICY,
        help="Apply the v1.0.0 exact cold-context layout/schedule policy; use --no-admitted-context-policy to opt out.",
    )
    parser.add_argument(
        "--full-attention-fused-norm-rope-kv-write",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FULL_ATTENTION_FUSED_NORM_ROPE_KV_WRITE,
        help="Probe fused full-attention norm/RoPE plus KV-cache write in one-token decode.",
    )
    parser.add_argument(
        "--skip-layer-dispatch-metadata",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SKIP_LAYER_DISPATCH_METADATA,
        help="Skip per-layer dispatch metadata dict construction in the served resident pipeline.",
    )
    parser.add_argument(
        "--overlap-shared-expert-moe",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERLAP_SHARED_EXPERT_MOE,
        help="Enable the retained one-token decode shared-expert/MoE CUDA-stream overlap path.",
    )
    parser.add_argument(
        "--overlap-shared-expert-router-moe",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERLAP_SHARED_EXPERT_ROUTER_MOE,
        help="Start shared-expert overlap before router/top-k for served decode tokens.",
    )
    parser.add_argument(
        "--shared-expert-overlap-stream-priority",
        type=int,
        default=DEFAULT_SHARED_EXPERT_OVERLAP_STREAM_PRIORITY,
        help=(
            "Optional torch.cuda.Stream priority for the shared-expert overlap "
            "side stream; unset preserves the default stream priority."
        ),
    )
    parser.add_argument("--repeat-same-request", type=int, default=1)
    parser.add_argument("--reuse-tensor-cache", action="store_true")
    parser.add_argument(
        "--exact-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXACT_PREFIX_CACHE,
        help="Reuse process-live prefill state only for an exact token-tuple and runtime-contract hit.",
    )
    parser.add_argument(
        "--exact-prefix-cache-max-entries",
        type=int,
        default=DEFAULT_EXACT_PREFIX_CACHE_MAX_ENTRIES,
        help="Bound the process-live exact-token prefix-state cache by entry count.",
    )
    parser.add_argument(
        "--exact-prefix-cache-max-tokens",
        type=int,
        default=DEFAULT_EXACT_PREFIX_CACHE_MAX_TOKENS,
        help="Do not store prompt state above this exact token count.",
    )
    parser.add_argument(
        "--resident-native-decode-hotset-layers",
        type=int,
        default=DEFAULT_RESIDENT_NATIVE_DECODE_HOTSET_LAYERS,
        help="Retain native selected-expert decode layouts beside raw weights for this many leading layers.",
    )
    parser.add_argument(
        "--stdin-loop",
        action="store_true",
        help="Read one OpenAI-shaped request JSON object per stdin line and keep the resident tensor cache alive.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repeat_same_request <= 0:
        raise SystemExit("--repeat-same-request must be positive")
    if args.exact_prefix_cache_max_entries < 0:
        raise SystemExit("--exact-prefix-cache-max-entries must be non-negative")
    if args.exact_prefix_cache_max_tokens < 0:
        raise SystemExit("--exact-prefix-cache-max-tokens must be non-negative")
    if args.exact_prefix_cache and args.exact_prefix_cache_max_entries == 0:
        raise SystemExit("--exact-prefix-cache requires --exact-prefix-cache-max-entries >= 1")
    if (
        args.resident_native_decode_hotset_layers < 0
        or args.resident_native_decode_hotset_layers > len(DEFAULT_LAYERS)
    ):
        raise SystemExit("--resident-native-decode-hotset-layers must be between 0 and 40")
    if args.execute and args.dry_run:
        raise SystemExit("--execute and --dry-run are mutually exclusive")
    if not args.execute and not args.dry_run:
        raise SystemExit("use --execute to run the resident path, or --dry-run to write inputs only")
    if args.stdin_loop:
        if args.request_json is not None:
            raise SystemExit("--stdin-loop reads requests from stdin; do not pass --request-json")
        if args.dry_run:
            raise SystemExit("--stdin-loop requires --execute")
        if args.output_dir is None:
            raise SystemExit("--stdin-loop requires --output-dir")
    elif args.request_json is None:
        raise SystemExit("--request-json is required unless --stdin-loop is used")
    if args.decode_token_cpu_sync_interval < 0:
        raise SystemExit("--decode-token-cpu-sync-interval must be non-negative")
    args.effective_decode_token_cpu_sync_interval = (
        0 if args.defer_decode_token_cpu_sync else args.decode_token_cpu_sync_interval
    )
    try:
        args.moe_override_config_by_layer = engine.normalize_moe_override_config_by_layer(
            args.moe_override_config_by_layer_json
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = (args.output_dir or default_output_dir(root)).resolve()
    source_commit = skeleton.git_value(root, "rev-parse", "HEAD") or "unknown"
    source_diff = skeleton.source_diff_scope(root)

    if args.stdin_loop:
        output_dir.mkdir(parents=True, exist_ok=False)
        model_dir = Path(args.model_dir).resolve()
        manifest_path = Path(args.manifest).resolve()
        manifest = engine.load_json(manifest_path)
        served = 0
        for line_index, line in enumerate(sys.stdin):
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"stdin line {line_index + 1} is not valid JSON") from exc
            if not isinstance(request, dict):
                raise SystemExit(f"stdin line {line_index + 1} must contain a JSON object")
            request_output_dir = output_dir / f"request-{served:03d}"
            response = execute_resident_request(
                adapter=adapter,
                skeleton=skeleton,
                engine=engine,
                request=request,
                output_dir=request_output_dir,
                args=args,
                model_dir=model_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                source_commit=source_commit,
                source_diff=source_diff,
                request_json_path=None,
            )
            final_run = response["aima_amd395"]["resident_runs"][-1]
            print(
                json.dumps(
                    {
                        "request_index": served,
                        "output_dir": str(request_output_dir),
                        "response_json": str(request_output_dir / "response.json"),
                        "finish_reason": response["choices"][0]["finish_reason"],
                        "cache_hits": final_run.get("cache_hits"),
                        "cache_misses": final_run.get("cache_misses"),
                        "engine_wall_time_ms": final_run.get("engine_wall_time_ms"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            served += 1
        return

    request_json = args.request_json
    assert request_json is not None
    request = adapter.json_load(request_json)
    adapter.validate_request(request)
    system_prompt, user_prompt = adapter.parse_messages(request, "")
    max_tokens = adapter.max_tokens_from_request(request)

    reuse_tensor_cache = args.reuse_tensor_cache or args.repeat_same_request > 1
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(output_dir / "request.json", request)
        write_json(
            output_dir / "dry-run.json",
            {
                "request_json": str(request_json.resolve()),
                "output_dir": str(output_dir),
                "model_dir": str(Path(args.model_dir).resolve()),
                "manifest": str(Path(args.manifest).resolve()),
                "system_prompt_sha256": adapter.sha256_text(system_prompt),
                "user_prompt_sha256": adapter.sha256_text(user_prompt),
                "max_tokens": max_tokens,
                "repeat_same_request": args.repeat_same_request,
                "reuse_tensor_cache": reuse_tensor_cache,
                "source_commit": source_commit,
                "source_diff_scope": source_diff,
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "env": {
                    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": os.environ.get(
                        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
                    ),
                },
            },
        )
        print(json.dumps({"output_dir": str(output_dir), "dry_run": True}, sort_keys=True))
        return

    model_dir = Path(args.model_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = engine.load_json(manifest_path)
    response = execute_resident_request(
        adapter=adapter,
        skeleton=skeleton,
        engine=engine,
        request=request,
        output_dir=output_dir,
        args=args,
        model_dir=model_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        source_commit=source_commit,
        source_diff=source_diff,
        request_json_path=request_json,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "response_json": str(output_dir / "response.json"),
                "finish_reason": response["choices"][0]["finish_reason"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
