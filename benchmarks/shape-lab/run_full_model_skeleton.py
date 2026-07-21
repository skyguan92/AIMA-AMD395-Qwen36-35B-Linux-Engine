#!/usr/bin/env python3
"""Run the retained full-model skeleton smoke and save artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_LAYERS = ",".join(str(item) for item in range(40))
DEFAULT_PROMPT_TEXT = "The capital of France is"
DEFAULT_PROMPT_TOKEN_IDS = "760,6511,314,9338,369"
D275_SYSTEM_PROMPT = (
    "You are serving a performance benchmark. Answer the user request directly and in detail. "
    "Prefer complete structured answers and do not stop early unless the answer is complete."
)

_TOKENIZER_CACHE: dict[str, Any] = {}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def token_ids_digest(token_ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(item) for item in token_ids).encode("utf-8")).hexdigest()


def cached_tokenizer(model_dir: Path, feature: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"transformers is required for {feature}") from exc
    key = str(model_dir.resolve())
    tokenizer = _TOKENIZER_CACHE.get(key)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
        _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer


def tokenize_text(
    model_dir: Path,
    text: str,
    prompt_format: str = "raw",
    system_prompt: str | None = None,
    chat_disable_thinking: bool = False,
) -> list[int]:
    tokenizer = cached_tokenizer(model_dir, "--prompt-jsonl")
    if prompt_format == "raw":
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    elif prompt_format == "chat-completions":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=not chat_disable_thinking,
        )
    else:
        raise SystemExit(f"unsupported prompt format: {prompt_format}")
    if not token_ids:
        raise SystemExit("prompt text produced zero tokens")
    return [int(item) for item in token_ids]


def chat_stop_token_ids(model_dir: Path) -> list[int]:
    tokenizer = cached_tokenizer(model_dir, "--chat-stop-tokens")
    candidates: list[Any] = [
        getattr(tokenizer, "eos_token_id", None),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    ]
    unk = getattr(tokenizer, "unk_token_id", None)
    stop_ids = {
        int(item)
        for item in candidates
        if isinstance(item, int) and item >= 0 and (unk is None or int(item) != int(unk))
    }
    if not stop_ids:
        raise SystemExit("could not resolve chat stop token ids")
    return sorted(stop_ids)


def load_prompt_jsonl_record(path: Path, record_index: int) -> dict[str, Any]:
    if record_index < 0:
        raise SystemExit("--prompt-record-index must be non-negative")
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index != record_index:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{index + 1} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"{path}:{index + 1} must contain a JSON object")
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise SystemExit(f"{path}:{index + 1} must contain a non-empty prompt string")
            return record
    raise SystemExit(f"{path} does not contain record index {record_index}")


def prompt_source_from_record(path: Path, record_index: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "jsonl",
        "path": str(path),
        "sha256": sha256_file(path),
        "record_index": record_index,
        "record": {key: value for key, value in record.items() if key != "prompt"},
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_diff_scope(root: Path) -> str:
    diff_paths = [
        line
        for line in (git_value(root, "diff", "--name-only") or "").splitlines()
        if line and not line.startswith("output/")
    ]
    diff_stat = git_value(root, "diff", "--stat", "--", *diff_paths) if diff_paths else ""
    diff_stat = diff_stat or ""
    untracked = git_value(root, "ls-files", "--others", "--exclude-standard") or ""
    untracked = "\n".join(line for line in untracked.splitlines() if not line.startswith("output/"))
    parts: list[str] = []
    if diff_stat:
        parts.append(diff_stat)
    if untracked:
        parts.append("untracked:\n" + untracked)
    return "\n".join(parts) if parts else "clean"


def default_output_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "output" / f"full-model-skeleton-retained-{stamp}"


def parse_json_stdout(stdout: str, command: list[str]) -> dict[str, Any]:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        json_start = stdout.find("{")
        if json_start < 0:
            raise RuntimeError(f"case did not produce JSON: {' '.join(command)}\n{stdout}") from exc
        try:
            return json.loads(stdout[json_start:])
        except json.JSONDecodeError as nested_exc:
            raise RuntimeError(f"case did not produce JSON: {' '.join(command)}\n{stdout}") from nested_exc


def build_case_command(
    python: str,
    harness: Path,
    manifest: Path,
    model_dir: Path,
    mode: str,
    seq_len: int,
    input_token_ids: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python,
        str(harness),
        "--manifest",
        str(manifest),
        "--model-dir",
        str(model_dir),
        "--layers",
        args.layers,
        "--mode",
        mode,
        "--seq-len",
        str(seq_len),
        "--input-token-ids",
        input_token_ids,
        "--attention-mode",
        args.attention_mode,
        "--full-attention-variant",
        args.full_attention_variant,
        "--full-attention-proj-variant",
        args.full_attention_proj_variant,
        "--full-attention-norm-rope-variant",
        args.full_attention_norm_rope_variant,
        "--full-attention-kv-cache-layout",
        args.full_attention_kv_cache_layout,
        "--linear-attention-variant",
        args.linear_attention_variant,
        "--linear-attention-input-proj-variant",
        args.linear_attention_input_proj_variant,
        "--linear-attention-output-proj-variant",
        args.linear_attention_output_proj_variant,
        "--linear-attention-conv-variant",
        args.linear_attention_conv_variant,
        "--linear-attention-gated-norm-variant",
        args.linear_attention_gated_norm_variant,
        "--rmsnorm-variant",
        args.rmsnorm_variant,
        "--lm-head-variant",
        args.lm_head_variant,
        "--shared-expert-proj-variant",
        args.shared_expert_proj_variant,
        "--moe-variant",
        args.moe_variant,
        "--router-variant",
        args.router_variant,
        "--measurement-mode",
        args.measurement_mode,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
        "--moe-chunk-size",
        str(args.moe_chunk_size),
        "--include-lm-head",
        "--decode-output-token",
        "--logit-topk",
        str(args.logit_topk),
        "--decode-sampling",
        args.decode_sampling,
        "--sampling-temperature",
        str(args.sampling_temperature),
        "--sampling-top-k",
        str(args.sampling_top_k),
    ]
    if args.moe_override_config_by_layer_json:
        command.extend(
            [
                "--moe-override-config-by-layer-json",
                args.moe_override_config_by_layer_json,
            ]
        )
    if args.linear_attention_post_conv_prep_block_t is not None:
        command.extend(
            [
                "--linear-attention-post-conv-prep-block-t",
                str(args.linear_attention_post_conv_prep_block_t),
            ]
        )
    if args.linear_attention_prefill_conv_block_t is not None:
        command.extend(
            [
                "--linear-attention-prefill-conv-block-t",
                str(args.linear_attention_prefill_conv_block_t),
            ]
        )
    if args.linear_attention_prefill_conv_block_c is not None:
        command.extend(
            [
                "--linear-attention-prefill-conv-block-c",
                str(args.linear_attention_prefill_conv_block_c),
            ]
        )
    if args.linear_attention_prefill_conv_num_warps is not None:
        command.extend(
            [
                "--linear-attention-prefill-conv-num-warps",
                str(args.linear_attention_prefill_conv_num_warps),
            ]
        )
    if args.linear_attention_prefill_conv_post_prep_fusion:
        command.append("--linear-attention-prefill-conv-post-prep-fusion")
    if args.linear_attention_prefill_vllm_state_handoff:
        command.append("--linear-attention-prefill-vllm-state-handoff")
    if args.linear_attention_prefill_fused_h_o:
        command.append("--linear-attention-prefill-fused-h-o")
    if args.linear_attention_prefill_fused_u_h_o:
        command.append("--linear-attention-prefill-fused-u-h-o")
    if args.linear_attention_chunk_gdn_internal_timing:
        command.append("--linear-attention-chunk-gdn-internal-timing")
    if args.overlap_shared_expert_moe:
        command.append("--overlap-shared-expert-moe")
    if args.overlap_shared_expert_router_moe:
        command.append("--overlap-shared-expert-router-moe")
    if args.shared_expert_overlap_stream_priority is not None:
        command.extend(
            [
                "--shared-expert-overlap-stream-priority",
                str(args.shared_expert_overlap_stream_priority),
            ]
        )
    if args.linear_attention_conv_state_refswap:
        command.append("--linear-attention-conv-state-refswap")
    if args.attention_substage_timing:
        command.append("--attention-substage-timing")
    if args.moe_substage_timing:
        command.append("--moe-substage-timing")
    if not args.collect_resident_stage_timeline:
        command.append("--no-collect-resident-stage-timeline")
    if args.decode_stop_token_ids:
        command.extend(["--decode-stop-token-ids", args.decode_stop_token_ids])
    loop_steps = decode_loop_steps_for_case(args, mode)
    if loop_steps:
        command.extend(["--decode-loop-steps", str(loop_steps)])
    if args.decode_loop_fast_housekeeping:
        command.append("--decode-loop-fast-housekeeping")
    if args.defer_decode_token_cpu_sync:
        command.append("--defer-decode-token-cpu-sync")
    if args.decode_token_cpu_sync_interval != 1:
        command.extend(["--decode-token-cpu-sync-interval", str(args.decode_token_cpu_sync_interval)])
    if args.decode_loop_diagnostic:
        command.append("--decode-loop-diagnostic")
    if args.overlap_decode_state_promotion_lm_head:
        command.append("--overlap-decode-state-promotion-lm-head")
    if args.skip_layer_dispatch_metadata:
        command.append("--skip-layer-dispatch-metadata")
    if args.cuda_graph_replay_timing:
        command.append("--cuda-graph-replay-timing")
    command.append("--include-shared-expert" if args.include_shared_expert else "--no-include-shared-expert")
    if args.execute:
        command.extend(["--execute", "--device", args.device])
    return command


def active_decode_loop_mode(args: argparse.Namespace) -> str:
    if args.case_mode == "generation":
        return "prefill"
    return str(args.decode_loop_mode)


def decode_loop_steps_for_case(args: argparse.Namespace, mode: str) -> int:
    if args.case_mode == "generation":
        if mode != "prefill":
            return 0
        if args.generation_token_count is None:
            raise RuntimeError("--generation-token-count is required for --case-mode generation")
        return max(0, int(args.generation_token_count) - 1)
    if mode == args.decode_loop_mode and args.decode_loop_steps:
        return int(args.decode_loop_steps)
    return 0


def run_case(command: list[str], output_dir: Path, stem: str) -> dict[str, Any]:
    wall_start = time.perf_counter()
    proc = subprocess.run(command, text=True, capture_output=True)
    subprocess_wall_time_ms = (time.perf_counter() - wall_start) * 1000.0
    stdout_path = output_dir / f"{stem}.stdout"
    stderr_path = output_dir / f"{stem}.stderr"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        failure = {
            "command": command,
            "returncode": proc.returncode,
            "stdout": str(stdout_path.name),
            "stderr": str(stderr_path.name),
        }
        (output_dir / f"{stem}.failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"case failed: {' '.join(command)}")
    data = parse_json_stdout(proc.stdout, command)
    artifact_path = output_dir / f"{stem}.json"
    artifact_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "artifact": artifact_path.name,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
        "command": command,
        "subprocess_wall_time_ms": subprocess_wall_time_ms,
        "result": data,
        "metrics": metrics_for_result(data),
    }


def stage_sum_if(result: dict[str, Any], predicate: Callable[[str], bool]) -> float | None:
    measurement = result.get("measurement")
    if not isinstance(measurement, dict):
        return None
    timeline = measurement.get("resident_stage_timeline")
    if not isinstance(timeline, list):
        return None
    return float(sum(item["ms_per_iter"] for item in timeline if predicate(item.get("name", ""))))


def stage_sum_ms(result: dict[str, Any], needle: str) -> float | None:
    return stage_sum_if(result, lambda name: needle in name)


def moe_stage_sum_ms(result: dict[str, Any]) -> float | None:
    return stage_sum_if(
        result,
        lambda name: "vllm_fused_routed_moe" in name or name.endswith("_routed_moe"),
    )


def metrics_for_result(result: dict[str, Any]) -> dict[str, Any]:
    contract = result.get("contract", {})
    measurement = result.get("measurement", {})
    text_smoke = measurement.get("text_smoke", {}) if isinstance(measurement, dict) else {}
    pipeline = measurement.get("pipeline", {}) if isinstance(measurement, dict) else {}
    comparison = measurement.get("comparison", {}) if isinstance(measurement, dict) else {}
    engine_stage_wall = measurement.get("engine_stage_wall_time_ms", {}) if isinstance(measurement, dict) else {}
    engine_stage_wall = engine_stage_wall if isinstance(engine_stage_wall, dict) else {}
    decode_loop = measurement.get("decode_loop") if isinstance(measurement, dict) else None
    decode_loop = decode_loop if isinstance(decode_loop, dict) else {}
    logits_comparison = text_smoke.get("logits_comparison", {}) if isinstance(text_smoke, dict) else {}
    logits_comparison = logits_comparison or {}
    lm_head = text_smoke.get("resident_final_norm_lm_head", {}) if isinstance(text_smoke, dict) else {}
    pipeline_ms = pipeline.get("resident_ms_per_iter")
    graph_pipeline_ms = pipeline.get("resident_cuda_graph_replay_ms_per_iter")
    lm_head_ms = lm_head.get("ms_per_iter")
    text_path_ms = None
    if isinstance(pipeline_ms, (int, float)) and isinstance(lm_head_ms, (int, float)):
        text_path_ms = float(pipeline_ms) + float(lm_head_ms)
    token_count = contract.get("tokens")
    tokens_per_s = None
    if isinstance(token_count, int) and text_path_ms and text_path_ms > 0:
        tokens_per_s = token_count * 1000.0 / text_path_ms
    peak_bytes = measurement.get("peak_memory_bytes") if isinstance(measurement, dict) else None
    return {
        "mode": contract.get("mode"),
        "logical_seq_len": contract.get("logical_seq_len"),
        "tokens": token_count,
        "layers": len(contract.get("layers", [])) if isinstance(contract.get("layers"), list) else None,
        "full_attention_variant": contract.get("full_attention_variant"),
        "full_attention_proj_variant": contract.get("full_attention_proj_variant"),
        "linear_attention_input_proj_variant": contract.get("linear_attention_input_proj_variant"),
        "linear_attention_output_proj_variant": contract.get("linear_attention_output_proj_variant"),
        "linear_attention_conv_variant": contract.get("linear_attention_conv_variant"),
        "linear_attention_conv_state_refswap": contract.get("linear_attention_conv_state_refswap"),
        "linear_attention_gated_norm_variant": contract.get("linear_attention_gated_norm_variant"),
        "linear_attention_post_conv_prep_block_t": contract.get("linear_attention_post_conv_prep_block_t"),
        "linear_attention_prefill_conv_block_t": contract.get("linear_attention_prefill_conv_block_t"),
        "linear_attention_prefill_conv_block_c": contract.get("linear_attention_prefill_conv_block_c"),
        "linear_attention_prefill_conv_num_warps": contract.get("linear_attention_prefill_conv_num_warps"),
        "linear_attention_prefill_conv_effective_block_t": contract.get(
            "linear_attention_prefill_conv_effective_block_t"
        ),
        "linear_attention_prefill_conv_effective_block_c": contract.get(
            "linear_attention_prefill_conv_effective_block_c"
        ),
        "linear_attention_prefill_conv_effective_num_warps": contract.get(
            "linear_attention_prefill_conv_effective_num_warps"
        ),
        "linear_attention_prefill_conv_post_prep_fusion": contract.get(
            "linear_attention_prefill_conv_post_prep_fusion"
        ),
        "linear_attention_prefill_vllm_state_handoff": contract.get(
            "linear_attention_prefill_vllm_state_handoff"
        ),
        "linear_attention_prefill_fused_h_o": contract.get("linear_attention_prefill_fused_h_o"),
        "linear_attention_prefill_fused_u_h_o": contract.get("linear_attention_prefill_fused_u_h_o"),
        "linear_attention_chunk_gdn_internal_timing": contract.get("linear_attention_chunk_gdn_internal_timing"),
        "router_variant": contract.get("router_variant"),
        "rmsnorm_variant": contract.get("rmsnorm_variant"),
        "lm_head_variant": contract.get("lm_head_variant"),
        "shared_expert_proj_variant": contract.get("shared_expert_proj_variant"),
        "skip_layer_dispatch_metadata": contract.get("skip_layer_dispatch_metadata"),
        "pipeline_ms": pipeline_ms,
        "cuda_graph_pipeline_ms": graph_pipeline_ms,
        "cuda_graph_vs_eager_speedup": pipeline.get("resident_cuda_graph_replay_vs_eager_speedup"),
        "cuda_graph_max_abs_diff": pipeline.get("resident_cuda_graph_replay_max_abs_diff"),
        "lm_head_ms": lm_head_ms,
        "text_path_ms": text_path_ms,
        "tokens_per_s": tokens_per_s,
        "engine_wall_time_ms": measurement.get("engine_wall_time_ms") if isinstance(measurement, dict) else None,
        "engine_python_import_wall_time_ms": engine_stage_wall.get("python_imports"),
        "engine_runtime_setup_wall_time_ms": engine_stage_wall.get("runtime_setup"),
        "engine_layer_tensor_load_derive_wall_time_ms": engine_stage_wall.get("layer_tensor_load_derive"),
        "engine_global_tensor_load_derive_wall_time_ms": engine_stage_wall.get("global_tensor_load_derive"),
        "engine_workspace_alloc_init_wall_time_ms": engine_stage_wall.get("workspace_alloc_init"),
        "engine_measurement_and_reporting_wall_time_ms": engine_stage_wall.get("measurement_and_reporting"),
        "linear_attention_sum_ms": stage_sum_ms(result, "linear_attention"),
        "full_attention_sum_ms": stage_sum_ms(result, "full_attention"),
        "moe_sum_ms": moe_stage_sum_ms(result),
        "hidden_max_relative_diff": comparison.get("max_relative_to_reference_max"),
        "logits_max_relative_diff": logits_comparison.get("max_relative_to_reference_max"),
        "peak_mb": None if peak_bytes is None else float(peak_bytes) / 1e6,
        "checksum_finite": measurement.get("checksum_finite") if isinstance(measurement, dict) else None,
        "logits_checksum_finite": text_smoke.get("logits_checksum_finite") if isinstance(text_smoke, dict) else None,
        "linear_attention_state_validated": measurement.get("workspace", {}).get(
            "linear_attention_state_validated"
        )
        if isinstance(measurement.get("workspace"), dict)
        else None,
        "full_attention_kv_cache_validated": measurement.get("workspace", {}).get(
            "full_attention_kv_cache_validated"
        )
        if isinstance(measurement.get("workspace"), dict)
        else None,
        "generated_token_id": text_smoke.get("generated_token_id") if isinstance(text_smoke, dict) else None,
        "generated_token_text": text_smoke.get("generated_token_text") if isinstance(text_smoke, dict) else None,
        "topk_token_ids": text_smoke.get("topk_token_ids") if isinstance(text_smoke, dict) else None,
        "topk_token_text": text_smoke.get("topk_token_text") if isinstance(text_smoke, dict) else None,
        "decode_loop_steps": decode_loop.get("steps"),
        "decode_loop_model_steps": decode_loop.get("model_steps"),
        "decode_loop_elapsed_ms": decode_loop.get("elapsed_ms"),
        "decode_loop_ms_per_token": decode_loop.get("ms_per_token"),
        "decode_loop_tokens_per_s": decode_loop.get("tokens_per_s"),
        "decode_loop_prefill_state_reused": decode_loop.get("prefill_state_reused"),
        "decode_loop_fast_housekeeping": decode_loop.get("fast_housekeeping"),
        "decode_loop_defer_token_cpu_sync": decode_loop.get("defer_token_cpu_sync"),
        "decode_loop_token_cpu_sync_interval": decode_loop.get("token_cpu_sync_interval"),
        "decode_loop_diagnostic": (
            decode_loop.get("diagnostic", {}).get("enabled")
            if isinstance(decode_loop.get("diagnostic"), dict)
            else False
        ),
        "overlap_decode_state_promotion_lm_head": decode_loop.get("overlap_state_promotion_lm_head"),
        "decode_loop_seed_source": decode_loop.get("seed_source"),
        "decode_loop_prefill_state_source": decode_loop.get("prefill_state_source"),
        "decode_loop_prefill_seed_token_id": decode_loop.get("prefill_seed_token_id"),
        "decode_loop_prefill_seed_token_text": decode_loop.get("prefill_seed_token_text"),
        "decode_loop_generated_token_ids_sha256": decode_loop.get("generated_token_ids_sha256"),
        "decode_loop_decode_sampling": decode_loop.get("decode_sampling"),
        "decode_loop_sampling_temperature": decode_loop.get("sampling_temperature"),
        "decode_loop_sampling_top_k": decode_loop.get("sampling_top_k"),
        "decode_loop_generated_text": decode_loop.get("generated_text"),
    }


def find_case(cases: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for case in cases:
        if case.get("name") == name:
            return case
    return None


def generation_report(metadata: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    if metadata.get("case_mode") != "generation":
        return None
    prefill = find_case(cases, "prefill")
    if prefill is None:
        return None
    metrics = prefill.get("metrics")
    result = prefill.get("result")
    if not isinstance(metrics, dict) or not isinstance(result, dict):
        return None
    measurement = result.get("measurement")
    decode_loop_value = measurement.get("decode_loop") if isinstance(measurement, dict) else None
    decode_loop_present = isinstance(decode_loop_value, dict)
    decode_loop = decode_loop_value if decode_loop_present else {}
    completion_tokens = metadata.get("generation_token_count")
    if not isinstance(completion_tokens, int):
        return None
    prefill_seed_only = completion_tokens == 1 and not decode_loop_present
    loop_steps = decode_loop.get("steps")
    loop_steps = int(loop_steps) if isinstance(loop_steps, int) else 0
    expected_loop_steps = max(0, completion_tokens - 1)
    stop_token_ids_value = decode_loop.get("stop_token_ids")
    if not isinstance(stop_token_ids_value, list):
        stop_token_ids_value = metadata.get("decode_stop_token_ids")
    stop_token_ids = sorted(
        {
            int(item)
            for item in (stop_token_ids_value if isinstance(stop_token_ids_value, list) else [])
            if isinstance(item, int)
        }
    )
    stop_token_id_set = set(stop_token_ids)
    seed_token_id = decode_loop.get("prefill_seed_token_id")
    seed_text = decode_loop.get("prefill_seed_token_text")
    stop_reason = decode_loop.get("stop_reason")
    stopped_on_token_id = decode_loop.get("stopped_on_token_id")
    seed_source = decode_loop.get("seed_source")
    prefill_state_reused = decode_loop.get("prefill_state_reused")
    decode_sampling = decode_loop.get("decode_sampling")
    sampling_temperature = decode_loop.get("sampling_temperature")
    sampling_top_k = decode_loop.get("sampling_top_k")
    initial_context_len = decode_loop.get("initial_context_len")
    final_context_len = decode_loop.get("final_context_len")
    generation_timing_scope = decode_loop.get("timing_scope")
    if prefill_seed_only:
        seed_token_id = metrics.get("generated_token_id")
        seed_text = metrics.get("generated_token_text")
        seed_source = "prefill_top1"
        prefill_state_reused = True
        decode_sampling = metadata.get("decode_sampling", "argmax")
        sampling_temperature = metadata.get("sampling_temperature")
        sampling_top_k = metadata.get("sampling_top_k")
        prompt_token_ids = metadata.get("prompt_token_ids")
        initial_context_len = len(prompt_token_ids) if isinstance(prompt_token_ids, list) else None
        final_context_len = initial_context_len
        generation_timing_scope = "no post-first decode tokens"
        if isinstance(seed_token_id, int):
            if seed_token_id in stop_token_id_set:
                stop_reason = "stop_token"
                stopped_on_token_id = seed_token_id
            else:
                stop_reason = "length"
                stopped_on_token_id = None
    loop_token_ids = decode_loop.get("generated_token_ids")
    loop_token_ids = loop_token_ids if isinstance(loop_token_ids, list) else []
    visible_loop_token_ids = decode_loop.get("visible_generated_token_ids")
    visible_loop_token_ids = visible_loop_token_ids if isinstance(visible_loop_token_ids, list) else loop_token_ids
    raw_completion_token_ids: list[int] = []
    visible_completion_token_ids: list[int] = []
    if isinstance(seed_token_id, int):
        raw_completion_token_ids.append(seed_token_id)
        if seed_token_id not in stop_token_id_set:
            visible_completion_token_ids.append(seed_token_id)
    raw_completion_token_ids.extend(
        int(item) for item in loop_token_ids[: max(0, completion_tokens - len(raw_completion_token_ids))]
    )
    visible_completion_token_ids.extend(
        int(item)
        for item in visible_loop_token_ids[: max(0, completion_tokens - len(visible_completion_token_ids))]
    )
    completion_text = None
    raw_completion_text = None
    generated_text = decode_loop.get("visible_generated_text")
    if not isinstance(generated_text, str):
        generated_text = decode_loop.get("generated_text")
    raw_generated_text = decode_loop.get("generated_text")
    if prefill_seed_only:
        generated_text = ""
        raw_generated_text = ""
    visible_seed_text = "" if seed_token_id in stop_token_id_set else seed_text
    if isinstance(visible_seed_text, str) and isinstance(generated_text, str):
        completion_text = visible_seed_text + generated_text
    if isinstance(seed_text, str) and isinstance(raw_generated_text, str):
        raw_completion_text = seed_text + raw_generated_text
    ttft_ms = metrics.get("text_path_ms")
    generation_ms = decode_loop.get("elapsed_ms") if loop_steps else 0.0
    total_latency_ms = None
    if isinstance(ttft_ms, (int, float)) and isinstance(generation_ms, (int, float)):
        total_latency_ms = float(ttft_ms) + float(generation_ms)
    case_subprocess_wall_time_ms = prefill.get("subprocess_wall_time_ms")
    case_unaccounted_wall_overhead_ms = None
    if isinstance(case_subprocess_wall_time_ms, (int, float)) and isinstance(total_latency_ms, (int, float)):
        case_unaccounted_wall_overhead_ms = float(case_subprocess_wall_time_ms) - float(total_latency_ms)
    tpot_ms = None
    decode_tok_s = None
    if isinstance(generation_ms, (int, float)) and loop_steps > 0 and generation_ms > 0:
        tpot_ms = float(generation_ms) / loop_steps
        decode_tok_s = loop_steps * 1000.0 / float(generation_ms)
    stopped_by_token = (
        stop_reason == "stop_token"
        and isinstance(stopped_on_token_id, int)
        and stopped_on_token_id in stop_token_id_set
        and bool(raw_completion_token_ids)
        and raw_completion_token_ids[-1] == stopped_on_token_id
    )
    stopped_by_length = (
        stop_reason == "length"
        and len(raw_completion_token_ids) == completion_tokens
        and loop_steps == expected_loop_steps
    )
    token_count_matches_request = (
        0 < len(raw_completion_token_ids) <= completion_tokens
        and len(visible_completion_token_ids) <= len(raw_completion_token_ids)
        and loop_steps <= expected_loop_steps
        and (stopped_by_token or stopped_by_length)
    )
    return {
        "comparison_scope": "single-batch serving-style accounting; first token from prefill logits, post-first tokens from prefill-seeded decode loop",
        "prompt_tokens": len(metadata.get("prompt_token_ids", []))
        if isinstance(metadata.get("prompt_token_ids"), list)
        else None,
        "requested_completion_tokens": completion_tokens,
        "completion_tokens": len(raw_completion_token_ids),
        "visible_completion_tokens": len(visible_completion_token_ids),
        "first_token_count": 1 if isinstance(seed_token_id, int) else 0,
        "prefill_seed_only": prefill_seed_only,
        "post_first_decode_tokens": loop_steps,
        "expected_post_first_decode_tokens": expected_loop_steps,
        "token_count_matches_request": token_count_matches_request,
        "ttft_ms": ttft_ms,
        "prefill_tok_s": metrics.get("tokens_per_s"),
        "generation_ms": generation_ms,
        "tpot_ms": tpot_ms,
        "decode_tok_s": decode_tok_s,
        "total_latency_ms": total_latency_ms,
        "case_subprocess_wall_time_ms": case_subprocess_wall_time_ms,
        "case_unaccounted_wall_overhead_ms": case_unaccounted_wall_overhead_ms,
        "engine_wall_time_ms": metrics.get("engine_wall_time_ms"),
        "engine_layer_tensor_load_derive_wall_time_ms": metrics.get(
            "engine_layer_tensor_load_derive_wall_time_ms"
        ),
        "engine_global_tensor_load_derive_wall_time_ms": metrics.get(
            "engine_global_tensor_load_derive_wall_time_ms"
        ),
        "engine_workspace_alloc_init_wall_time_ms": metrics.get("engine_workspace_alloc_init_wall_time_ms"),
        "engine_measurement_and_reporting_wall_time_ms": metrics.get(
            "engine_measurement_and_reporting_wall_time_ms"
        ),
        "peak_mb": metrics.get("peak_mb"),
        "prefill_state_reused": prefill_state_reused,
        "seed_source": seed_source,
        "stop_reason": stop_reason,
        "stop_token_ids": stop_token_ids,
        "stopped_on_token_id": stopped_on_token_id,
        "prefill_seed_token_id": seed_token_id,
        "prefill_seed_token_text": seed_text,
        "completion_token_ids_preview": raw_completion_token_ids[:16],
        "completion_token_ids_sha256": token_ids_digest(raw_completion_token_ids),
        "visible_completion_token_ids_preview": visible_completion_token_ids[:16],
        "visible_completion_token_ids_sha256": token_ids_digest(visible_completion_token_ids),
        "completion_text": completion_text,
        "raw_completion_text": raw_completion_text,
        "completion_text_preview": completion_text[:512] if isinstance(completion_text, str) else None,
        "loop_generated_token_ids_sha256": decode_loop.get("generated_token_ids_sha256"),
        "decode_sampling": decode_sampling,
        "sampling_temperature": sampling_temperature,
        "sampling_top_k": sampling_top_k,
        "initial_context_len": initial_context_len,
        "final_context_len": final_context_len,
        "timing_scope": {
            "ttft_ms": "prefill resident pipeline plus final norm and LM-head",
            "generation_ms": generation_timing_scope,
            "tpot_ms": "generation_ms divided by post-first decode tokens, matching SGLang request tpot_s convention",
        },
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def prompt_ids_summary(metadata: dict[str, Any]) -> str:
    token_ids = metadata["prompt_token_ids"]
    if len(token_ids) <= 32:
        return str(token_ids)
    preview = ", ".join(str(item) for item in token_ids[:16])
    digest = metadata.get("prompt_token_ids_sha256", "unknown")
    return f"{len(token_ids)} ids; preview [{preview}, ...]; sha256 {digest}"


def prompt_text_summary(metadata: dict[str, Any]) -> str:
    text = metadata.get("prompt_text")
    if not isinstance(text, str) or not text:
        return "n/a"
    if len(text) <= 240:
        return text
    preview = text[:200].replace("\n", "\\n")
    digest = metadata.get("prompt_text_sha256", "unknown")
    return f"{len(text)} chars; sha256 {digest}; preview {preview}..."


def markdown_summary(
    metadata: dict[str, Any],
    cases: list[dict[str, Any]],
    generation: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Full-Model Skeleton Retained-Route Smoke",
        "",
        "## Metadata",
        "",
        f"- created utc: `{metadata['created_utc']}`",
        f"- host: `{metadata['host']}`",
        f"- platform: `{metadata['platform']}`",
        f"- python: `{metadata['python']}`",
        f"- model dir: `{metadata['model_dir']}`",
        f"- source commit: `{metadata['source_commit']}`",
        f"- source diff scope: `{metadata['source_diff_scope']}`",
        f"- manifest sha256: `{metadata['manifest_sha256']}`",
        f"- execute: `{metadata['execute']}`",
        f"- device: `{metadata['device']}`",
        f"- layers: `{metadata['layers']}`",
        f"- prompt source: `{metadata['prompt_source']}`",
        f"- prompt format: `{metadata['prompt_format']}`",
        f"- chat disable thinking: `{metadata.get('chat_disable_thinking')}`",
        f"- chat stop tokens: `{metadata.get('chat_stop_tokens')}`",
        f"- system prompt chars: `{metadata.get('system_prompt_chars')}`",
        f"- system prompt sha256: `{metadata.get('system_prompt_sha256')}`",
        f"- prompt text: `{prompt_text_summary(metadata)}`",
        f"- prompt token ids: `{prompt_ids_summary(metadata)}`",
        f"- decode input token id: `{metadata['decode_input_token_id']}`",
        f"- attention mode: `{metadata['attention_mode']}`",
        f"- full attention variant: `{metadata['full_attention_variant']}`",
        f"- full attention projection variant: `{metadata['full_attention_proj_variant']}`",
        f"- full attention norm+RoPE variant: `{metadata['full_attention_norm_rope_variant']}`",
        f"- full attention KV cache layout: `{metadata['full_attention_kv_cache_layout']}`",
        f"- linear attention variant: `{metadata['linear_attention_variant']}`",
        f"- linear attention input projection variant: `{metadata['linear_attention_input_proj_variant']}`",
        f"- linear attention output projection variant: `{metadata['linear_attention_output_proj_variant']}`",
        f"- linear attention conv variant: `{metadata['linear_attention_conv_variant']}`",
        f"- linear attention conv-state refswap: `{metadata.get('linear_attention_conv_state_refswap')}`",
        f"- linear attention gated norm variant: `{metadata['linear_attention_gated_norm_variant']}`",
        f"- linear attention post-conv prep BLOCK_T: `{metadata.get('linear_attention_post_conv_prep_block_t')}`",
        f"- linear attention prefill conv BLOCK_T: `{metadata.get('linear_attention_prefill_conv_block_t')}`",
        f"- linear attention prefill conv BLOCK_C: `{metadata.get('linear_attention_prefill_conv_block_c')}`",
        f"- linear attention prefill conv num_warps: `{metadata.get('linear_attention_prefill_conv_num_warps')}`",
        f"- linear attention prefill conv effective tile: "
        f"`{metadata.get('linear_attention_prefill_conv_effective_block_t')}/"
        f"{metadata.get('linear_attention_prefill_conv_effective_block_c')}/"
        f"{metadata.get('linear_attention_prefill_conv_effective_num_warps')}`",
        f"- linear attention prefill conv+post-prep fusion: `{metadata.get('linear_attention_prefill_conv_post_prep_fusion')}`",
        f"- linear attention prefill vLLM state handoff: `{metadata.get('linear_attention_prefill_vllm_state_handoff')}`",
        f"- linear attention prefill fused h/o: `{metadata.get('linear_attention_prefill_fused_h_o')}`",
        f"- linear attention prefill fused U+h/o: `{metadata.get('linear_attention_prefill_fused_u_h_o')}`",
        f"- linear attention chunk-GDN internal timing: `{metadata.get('linear_attention_chunk_gdn_internal_timing')}`",
        f"- RMSNorm variant: `{metadata['rmsnorm_variant']}`",
        f"- LM-head variant: `{metadata['lm_head_variant']}`",
        f"- shared expert projection variant: `{metadata['shared_expert_proj_variant']}`",
        f"- skip layer dispatch metadata: `{metadata.get('skip_layer_dispatch_metadata')}`",
        f"- MoE variant: `{metadata['moe_variant']}`",
        f"- shared/MoE overlap: `{metadata.get('overlap_shared_expert_moe')}`",
        f"- router-early shared/MoE overlap: `{metadata.get('overlap_shared_expert_router_moe')}`",
        f"- shared-expert overlap stream priority: `{metadata.get('shared_expert_overlap_stream_priority')}`",
        f"- router variant: `{metadata['router_variant']}`",
        f"- measurement mode: `{metadata['measurement_mode']}`",
        f"- case mode: `{metadata['case_mode']}`",
        f"- generation token count: `{metadata.get('generation_token_count')}`",
        f"- decode stop token ids: `{metadata.get('decode_stop_token_ids')}`",
        f"- effective decode loop steps: `{metadata.get('effective_decode_loop_steps')}`",
        f"- decode token CPU sync interval: `{metadata.get('decode_loop_token_cpu_sync_interval')}`",
        f"- MoE chunk size: `{metadata['moe_chunk_size']}`",
        f"- include shared expert: `{metadata['include_shared_expert']}`",
        f"- attention substage timing: `{metadata.get('attention_substage_timing')}`",
        f"- MoE substage timing: `{metadata.get('moe_substage_timing')}`",
        "- TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: "
        f"`{metadata['env']['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL']}`",
        f"- warmup/iters: `{metadata['warmup']}` / `{metadata['iters']}`",
        f"- wrapper wall time ms: `{fmt(metadata.get('wrapper_wall_time_ms'))}`",
        "",
        "## Results",
        "",
        "| case | mode | seq | tokens | generated token | pipeline ms | graph ms | graph/eager | lm-head ms | text path ms | tokens/s | linear ms | full ms | MoE ms | loop steps | loop ms/token | loop tok/s | hidden rel | logits rel | peak MB | finite | artifact |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for case in cases:
        metrics = case["metrics"]
        finite = (
            metrics.get("checksum_finite"),
            metrics.get("logits_checksum_finite"),
            metrics.get("linear_attention_state_validated"),
            metrics.get("full_attention_kv_cache_validated"),
        )
        generated = f"{metrics.get('generated_token_id')} / {metrics.get('generated_token_text')!r}"
        lines.append(
            "| "
            f"`{case['name']}` | "
            f"`{metrics.get('mode')}` | "
            f"{metrics.get('logical_seq_len')} | "
            f"{metrics.get('tokens')} | "
            f"`{generated}` | "
            f"{fmt(metrics.get('pipeline_ms'))} | "
            f"{fmt(metrics.get('cuda_graph_pipeline_ms'))} | "
            f"{fmt(metrics.get('cuda_graph_vs_eager_speedup'))} | "
            f"{fmt(metrics.get('lm_head_ms'))} | "
            f"{fmt(metrics.get('text_path_ms'))} | "
            f"{fmt(metrics.get('tokens_per_s'))} | "
            f"{fmt(metrics.get('linear_attention_sum_ms'))} | "
            f"{fmt(metrics.get('full_attention_sum_ms'))} | "
            f"{fmt(metrics.get('moe_sum_ms'))} | "
            f"{fmt(metrics.get('decode_loop_steps'))} | "
            f"{fmt(metrics.get('decode_loop_ms_per_token'))} | "
            f"{fmt(metrics.get('decode_loop_tokens_per_s'))} | "
            f"{fmt(metrics.get('hidden_max_relative_diff'))} | "
            f"{fmt(metrics.get('logits_max_relative_diff'))} | "
            f"{fmt(metrics.get('peak_mb'), digits=1)} | "
            f"`{finite}` | "
            f"`{case['artifact']}` |"
        )
    if generation:
        lines.extend(
            [
                "",
                "## Generation Accounting",
                "",
                f"- comparison scope: `{generation['comparison_scope']}`",
                f"- prompt tokens: `{generation['prompt_tokens']}`",
                f"- requested completion tokens: `{generation.get('requested_completion_tokens')}`",
                f"- completion tokens: `{generation['completion_tokens']}`",
                f"- visible completion tokens: `{generation.get('visible_completion_tokens')}`",
                f"- first token count: `{generation['first_token_count']}`",
                f"- post-first decode tokens: `{generation['post_first_decode_tokens']}`",
                f"- expected post-first decode tokens: `{generation['expected_post_first_decode_tokens']}`",
                f"- token count matches request: `{generation['token_count_matches_request']}`",
                f"- TTFT ms: `{fmt(generation['ttft_ms'])}`",
                f"- prefill tok/s: `{fmt(generation['prefill_tok_s'])}`",
                f"- generation ms: `{fmt(generation['generation_ms'])}`",
                f"- TPOT ms: `{fmt(generation['tpot_ms'])}`",
                f"- decode tok/s: `{fmt(generation['decode_tok_s'])}`",
                f"- total latency ms: `{fmt(generation['total_latency_ms'])}`",
                f"- case subprocess wall time ms: `{fmt(generation.get('case_subprocess_wall_time_ms'))}`",
                f"- case unaccounted wall overhead ms: `{fmt(generation.get('case_unaccounted_wall_overhead_ms'))}`",
                f"- peak MB: `{fmt(generation['peak_mb'], digits=3)}`",
                f"- prefill state reused: `{generation['prefill_state_reused']}`",
                f"- stop reason: `{generation.get('stop_reason')}`",
                f"- stopped on token id: `{generation.get('stopped_on_token_id')}`",
                f"- decode sampling: `{generation.get('decode_sampling')}`",
                f"- sampling temperature: `{generation.get('sampling_temperature')}`",
                f"- sampling top-k: `{generation.get('sampling_top_k')}`",
                f"- seed: `{generation['prefill_seed_token_id']} / {generation['prefill_seed_token_text']!r}`",
                f"- completion token ids sha256: `{generation['completion_token_ids_sha256']}`",
                f"- completion text preview: `{generation.get('completion_text_preview')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    wrapper_wall_start = time.perf_counter()
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(root / "doc/reference/amd395-qwen36-35b-a3b-bf16/qwen36-shape-manifest.json"),
    )
    parser.add_argument("--model-dir", default="/data/models/Qwen3.6-35B-A3B")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--prompt-token-ids", default=DEFAULT_PROMPT_TOKEN_IDS)
    parser.add_argument("--prompt-jsonl")
    parser.add_argument("--prompt-record-index", type=int, default=0)
    parser.add_argument("--prompt-format", choices=["raw", "chat-completions"], default="raw")
    parser.add_argument("--chat-disable-thinking", action="store_true")
    parser.add_argument("--system-prompt", default=D275_SYSTEM_PROMPT)
    parser.add_argument("--decode-input-token-id", type=int)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--decode-seq-len", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--moe-chunk-size", type=int, default=64)
    parser.add_argument(
        "--moe-variant",
        choices=[
            "resident_dispatch",
            "padded_batched",
            "count_batched",
            "vllm_fused",
            "vllm_fused_inplace",
            "vllm_fused_m32_n16_k512",
            "vllm_fused_prefill_m32_n32_decode_m32_n16_k512",
        ],
        default="vllm_fused",
    )
    parser.add_argument(
        "--router-variant",
        choices=["torch", "torch_out", "triton_topk", "triton_topk_softmax"],
        default="torch",
    )
    parser.add_argument(
        "--moe-override-config-by-layer-json",
        help=(
            "Forward a JSON object mapping model layer id to a vLLM fused_moe.override_config "
            "object for one-token decode."
        ),
    )
    parser.add_argument(
        "--overlap-shared-expert-moe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward the one-token decode shared-expert/MoE CUDA-stream overlap probe to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--overlap-shared-expert-router-moe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward router-early shared-expert/MoE overlap to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--shared-expert-overlap-stream-priority",
        type=int,
        help=(
            "Forward an optional torch.cuda.Stream priority for the "
            "shared-expert overlap side stream."
        ),
    )
    parser.add_argument(
        "--measurement-mode",
        choices=["correctness", "resident_only"],
        default="correctness",
    )
    parser.add_argument(
        "--attention-mode",
        choices=["stub", "full_for_full_attention", "linear_and_full_attention"],
        default="linear_and_full_attention",
    )
    parser.add_argument(
        "--full-attention-variant",
        choices=["sdpa", "decode_grouped_bmm", "decode_grouped_bmm_bf16"],
        default="sdpa",
    )
    parser.add_argument(
        "--full-attention-proj-variant",
        choices=["torch", "triton_matvec", "triton_fused_qkv_matvec"],
        default="torch",
    )
    parser.add_argument(
        "--full-attention-norm-rope-variant",
        choices=["torch", "triton"],
        default="torch",
    )
    parser.add_argument(
        "--full-attention-kv-cache-layout",
        choices=["seq", "grouped"],
        default="seq",
        help="Forward full-attention KV cache layout to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--linear-attention-variant",
        choices=[
            "torch_ref",
            "vllm_fla_chunk",
            "vllm_fla_auto",
            "vllm_fla_auto_prestates",
            "vllm_fla_auto_prestates_chunk16",
            "vllm_fla_auto_prestates_chunk32",
            "vllm_fla_auto_prestates_native_chunk16",
            "vllm_fla_auto_prestates_native_refswap_chunk16",
            "vllm_fla_packed_decode",
            "vllm_fla_packed_refswap_decode",
            "vllm_fla_packed_refswap_decode_chunk16",
        ],
        default="vllm_fla_auto",
    )
    parser.add_argument(
        "--linear-attention-input-proj-variant",
        choices=[
            "separate",
            "decode_fused",
            "decode_fused_t",
            "decode_fused_t_triton",
            "decode_fused_t_conv_triton",
            "decode_fused_t_conv_qkv_triton",
            "prefill_fused_t_decode_fused_t_conv_triton",
            "prefill_fused_t_decode_fused_t_conv_qkv_triton",
        ],
        default="separate",
    )
    parser.add_argument(
        "--linear-attention-output-proj-variant",
        choices=["torch", "triton_matvec"],
        default="torch",
    )
    parser.add_argument(
        "--linear-attention-conv-variant",
        choices=["conv1d", "decode_direct", "decode_direct_triton"],
        default="conv1d",
    )
    parser.add_argument(
        "--linear-attention-conv-state-refswap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward one-token decode linear-attention causal-conv state refswap to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--linear-attention-gated-norm-variant",
        choices=["torch", "triton"],
        default="torch",
    )
    parser.add_argument(
        "--linear-attention-post-conv-prep-block-t",
        type=int,
        choices=[8, 16, 32, 64, 128, 256],
        help="Forward a vLLM fused_post_conv_prep BLOCK_T override to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-block-t",
        type=int,
        choices=[8, 16, 32, 64],
        help="Forward a Triton prefill causal-conv BLOCK_T override to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-block-c",
        type=int,
        choices=[16, 32, 64],
        help="Forward a Triton prefill causal-conv BLOCK_C override to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-num-warps",
        type=int,
        choices=[4, 8],
        help="Forward a Triton prefill causal-conv num_warps override to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-post-prep-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward the default-off fused prefill causal-conv plus post-conv-prep candidate.",
    )
    parser.add_argument(
        "--linear-attention-prefill-vllm-state-handoff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward prefill vLLM final-state handoff to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--linear-attention-chunk-gdn-internal-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward prefill chunk-GDN internal timing to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--linear-attention-prefill-fused-h-o",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward experimental prefill fused chunk-GDN h/o boundary to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--linear-attention-prefill-fused-u-h-o",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward experimental prefill W-only plus fused U+h/o boundary to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--rmsnorm-variant",
        choices=["torch", "triton"],
        default="torch",
    )
    parser.add_argument(
        "--lm-head-variant",
        choices=["view", "pretransposed", "pretransposed_out"],
        default="view",
    )
    parser.add_argument(
        "--shared-expert-proj-variant",
        choices=["torch", "triton_matvec", "triton_fused_in_matvec"],
        default="torch",
    )
    parser.add_argument("--include-shared-expert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logit-topk", type=int, default=5)
    parser.add_argument(
        "--attention-substage-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask four_layer_mini_engine.py to run the extra attention-substage diagnostic pass.",
    )
    parser.add_argument(
        "--moe-substage-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask four_layer_mini_engine.py to run the extra MoE-substage diagnostic pass.",
    )
    parser.add_argument(
        "--collect-resident-stage-timeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward resident stage timeline collection to four_layer_mini_engine.py.",
    )
    parser.add_argument("--decode-sampling", choices=["argmax", "top_k"], default="argmax")
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--sampling-top-k", type=int, default=50)
    parser.add_argument("--decode-stop-token-ids", default="")
    parser.add_argument("--chat-stop-tokens", action="store_true")
    parser.add_argument("--decode-loop-steps", type=int, default=0)
    parser.add_argument("--decode-loop-mode", choices=["prefill", "decode"], default="decode")
    parser.add_argument(
        "--decode-loop-fast-housekeeping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward decode-loop RoPE precompute and embedding-buffer reuse to "
            "four_layer_mini_engine.py."
        ),
    )
    parser.add_argument(
        "--defer-decode-token-cpu-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward deferred generated-token CPU materialization to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--decode-token-cpu-sync-interval",
        type=int,
        default=1,
        help=(
            "Forward generated-token CPU materialization interval to four_layer_mini_engine.py "
            "(1=current per-token sync, 0=loop-end materialization)."
        ),
    )
    parser.add_argument(
        "--decode-loop-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward decode-loop logits/state diagnostics to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--overlap-decode-state-promotion-lm-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward decode state-promotion/LM-head overlap to four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--skip-layer-dispatch-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip per-layer dispatch metadata dict construction in four_layer_mini_engine.py.",
    )
    parser.add_argument(
        "--cuda-graph-replay-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ask four_layer_mini_engine.py to capture/replay the resident pipeline "
            "with torch.cuda.CUDAGraph for launch-boundary diagnostics."
        ),
    )
    parser.add_argument("--case-mode", choices=["paired", "generation"], default="paired")
    parser.add_argument("--generation-token-count", type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-diff-scope")
    args = parser.parse_args()
    if args.decode_token_cpu_sync_interval < 0:
        raise SystemExit("--decode-token-cpu-sync-interval must be non-negative")
    if args.case_mode == "generation":
        if args.generation_token_count is None:
            raise SystemExit("--generation-token-count is required for --case-mode generation")
        if args.generation_token_count < 2:
            raise SystemExit("--generation-token-count must be at least 2 for serving-style TPOT")
        if args.decode_loop_steps:
            raise SystemExit("--decode-loop-steps is derived from --generation-token-count in generation mode")

    manifest = Path(args.manifest).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    prompt_text = args.prompt_text
    prompt_source: dict[str, Any] = {"kind": "cli_token_ids"}
    if args.prompt_jsonl:
        prompt_jsonl = Path(args.prompt_jsonl).resolve()
        prompt_record = load_prompt_jsonl_record(prompt_jsonl, args.prompt_record_index)
        prompt_text = str(prompt_record["prompt"])
        prompt_token_ids = tokenize_text(
            model_dir,
            prompt_text,
            args.prompt_format,
            args.system_prompt if args.prompt_format == "chat-completions" else None,
            args.chat_disable_thinking,
        )
        prompt_source = prompt_source_from_record(prompt_jsonl, args.prompt_record_index, prompt_record)
    else:
        prompt_token_ids = csv_ints(args.prompt_token_ids)
    if not prompt_token_ids:
        raise SystemExit("--prompt-token-ids must contain at least one token id")
    decode_stop_token_ids = csv_ints(args.decode_stop_token_ids) if args.decode_stop_token_ids else []
    if args.chat_stop_tokens:
        decode_stop_token_ids = sorted(set(decode_stop_token_ids).union(chat_stop_token_ids(model_dir)))
    args.decode_stop_token_ids = ",".join(str(item) for item in decode_stop_token_ids)
    decode_token_id = args.decode_input_token_id if args.decode_input_token_id is not None else prompt_token_ids[-1]
    decode_seq_len = args.decode_seq_len if args.decode_seq_len is not None else args.seq_len
    harness = Path(__file__).resolve().with_name("four_layer_mini_engine.py")

    effective_decode_loop_steps = {
        "prefill": decode_loop_steps_for_case(args, "prefill"),
        "decode": decode_loop_steps_for_case(args, "decode"),
    }

    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "model_dir": str(model_dir),
        "source_commit": args.source_commit or git_value(root, "rev-parse", "HEAD") or "unknown",
        "source_diff_scope": args.source_diff_scope or source_diff_scope(root),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "execute": args.execute,
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "layers": args.layers,
        "prompt_source": prompt_source,
        "prompt_format": args.prompt_format,
        "chat_disable_thinking": args.chat_disable_thinking,
        "system_prompt": args.system_prompt if args.prompt_format == "chat-completions" else None,
        "system_prompt_sha256": (
            hashlib.sha256(args.system_prompt.encode("utf-8")).hexdigest()
            if args.prompt_format == "chat-completions"
            else None
        ),
        "system_prompt_chars": len(args.system_prompt) if args.prompt_format == "chat-completions" else 0,
        "prompt_text": prompt_text,
        "prompt_text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest() if prompt_text else None,
        "prompt_text_chars": len(prompt_text) if prompt_text else 0,
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": token_ids_digest(prompt_token_ids),
        "seq_len": args.seq_len,
        "decode_seq_len": decode_seq_len,
        "decode_input_token_id": decode_token_id,
        "moe_chunk_size": args.moe_chunk_size,
        "moe_variant": args.moe_variant,
        "overlap_shared_expert_moe": args.overlap_shared_expert_moe,
        "overlap_shared_expert_router_moe": args.overlap_shared_expert_router_moe,
        "shared_expert_overlap_stream_priority": args.shared_expert_overlap_stream_priority,
        "router_variant": args.router_variant,
        "measurement_mode": args.measurement_mode,
        "attention_mode": args.attention_mode,
        "full_attention_variant": args.full_attention_variant,
        "full_attention_proj_variant": args.full_attention_proj_variant,
        "full_attention_norm_rope_variant": args.full_attention_norm_rope_variant,
        "full_attention_kv_cache_layout": args.full_attention_kv_cache_layout,
        "linear_attention_variant": args.linear_attention_variant,
        "linear_attention_input_proj_variant": args.linear_attention_input_proj_variant,
        "linear_attention_output_proj_variant": args.linear_attention_output_proj_variant,
        "linear_attention_conv_variant": args.linear_attention_conv_variant,
        "linear_attention_conv_state_refswap": args.linear_attention_conv_state_refswap,
        "linear_attention_gated_norm_variant": args.linear_attention_gated_norm_variant,
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
        "rmsnorm_variant": args.rmsnorm_variant,
        "lm_head_variant": args.lm_head_variant,
        "shared_expert_proj_variant": args.shared_expert_proj_variant,
        "skip_layer_dispatch_metadata": args.skip_layer_dispatch_metadata,
        "include_shared_expert": args.include_shared_expert,
        "logit_topk": args.logit_topk,
        "attention_substage_timing": args.attention_substage_timing,
        "moe_substage_timing": args.moe_substage_timing,
        "collect_resident_stage_timeline": args.collect_resident_stage_timeline,
        "decode_sampling": args.decode_sampling,
        "sampling_temperature": args.sampling_temperature,
        "sampling_top_k": args.sampling_top_k,
        "decode_stop_token_ids": decode_stop_token_ids,
        "chat_stop_tokens": args.chat_stop_tokens,
        "case_mode": args.case_mode,
        "generation_token_count": args.generation_token_count,
        "decode_loop_steps": args.decode_loop_steps,
        "decode_loop_mode": active_decode_loop_mode(args),
        "decode_loop_fast_housekeeping": args.decode_loop_fast_housekeeping,
        "decode_loop_defer_token_cpu_sync": args.defer_decode_token_cpu_sync,
        "decode_loop_token_cpu_sync_interval": (
            0 if args.defer_decode_token_cpu_sync else args.decode_token_cpu_sync_interval
        ),
        "decode_loop_diagnostic": args.decode_loop_diagnostic,
        "overlap_decode_state_promotion_lm_head": args.overlap_decode_state_promotion_lm_head,
        "cuda_graph_replay_timing": args.cuda_graph_replay_timing,
        "effective_decode_loop_steps": effective_decode_loop_steps,
        "env": {
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": os.environ.get(
                "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
            ),
        },
    }

    cases: list[dict[str, Any]] = []
    prefill_command = build_case_command(
        sys.executable,
        harness,
        manifest,
        model_dir,
        "prefill",
        args.seq_len,
        ",".join(str(item) for item in prompt_token_ids),
        args,
    )
    prefill = run_case(prefill_command, output_dir, "prefill-full-model-skeleton")
    prefill["name"] = "prefill"
    cases.append(prefill)

    if args.case_mode == "paired":
        decode_command = build_case_command(
            sys.executable,
            harness,
            manifest,
            model_dir,
            "decode",
            decode_seq_len,
            str(decode_token_id),
            args,
        )
        decode = run_case(decode_command, output_dir, "decode-full-model-skeleton")
        decode["name"] = "decode"
        cases.append(decode)

    generation = generation_report(metadata, cases)
    metadata["wrapper_wall_time_ms"] = (time.perf_counter() - wrapper_wall_start) * 1000.0
    summary = {"metadata": metadata, "cases": cases}
    if generation:
        summary["generation"] = generation
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(markdown_summary(metadata, cases, generation), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
