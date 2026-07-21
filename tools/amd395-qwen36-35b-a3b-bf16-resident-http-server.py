#!/usr/bin/env python3
"""Serve the resident single-batch path behind a minimal HTTP API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18080
DEFAULT_MODEL_DIR = "/data/models/Qwen3.6-35B-A3B"
DEFAULT_MODEL_ID = "aima-amd395-qwen36-35b"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "output" / f"resident-chat-completions-http-server-{stamp}"


def load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def error_payload(message: str, *, code: str = "bad_request", error_type: str = "invalid_request_error") -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


class ServerState:
    def __init__(
        self,
        *,
        adapter: Any,
        skeleton: Any,
        engine: Any,
        model_dir: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        output_dir: Path,
        args: argparse.Namespace,
        source_commit: str,
        source_diff: str,
        started_wall: float,
        startup_load: dict[str, Any],
    ) -> None:
        self.adapter = adapter
        self.skeleton = skeleton
        self.engine = engine
        self.model_dir = model_dir
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.output_dir = output_dir
        self.args = args
        self.source_commit = source_commit
        self.source_diff = source_diff
        self.started_wall = started_wall
        self.startup_load = startup_load
        self.served = 0
        self.lock = threading.Lock()


def next_request_dir(state: ServerState) -> tuple[int, Path]:
    with state.lock:
        request_index = state.served
        state.served += 1
    return request_index, state.output_dir / f"request-{request_index:03d}"


def build_handler(state: ServerState) -> type[BaseHTTPRequestHandler]:
    class ResidentHandler(BaseHTTPRequestHandler):
        server_version = "aima-amd395-engine/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            message = fmt % args
            print(
                json.dumps(
                    {
                        "event": "access",
                        "remote": self.client_address[0],
                        "message": message,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                json_response(
                    self,
                    200,
                    {
                        "status": "ok",
                        "model": DEFAULT_MODEL_ID,
                        "route": "in-process resident tensor-cache engine",
                        "model_loaded": state.startup_load.get("model_loaded") is True,
                        "load_only_engine_wall_time_ms": state.startup_load.get(
                            "engine_wall_time_ms"
                        ),
                        "served": state.served,
                        "source_commit": state.source_commit,
                        "admitted_context_policy": state.args.admitted_context_policy,
                        "uptime_s": time.perf_counter() - state.started_wall,
                    },
                )
                return
            if self.path == "/v1/models":
                created = int(time.time())
                json_response(
                    self,
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": DEFAULT_MODEL_ID,
                                "object": "model",
                                "created": created,
                                "owned_by": "approaching-ai",
                            }
                        ],
                    },
                )
                return
            json_response(self, 404, error_payload(f"unsupported path: {self.path}", code="not_found"))

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/shutdown":
                json_response(self, 200, {"status": "shutting_down"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != "/v1/chat/completions":
                json_response(self, 404, error_payload(f"unsupported path: {self.path}", code="not_found"))
                return
            content_length = self.headers.get("Content-Length")
            try:
                length = int(content_length or "0")
            except ValueError:
                json_response(self, 400, error_payload("invalid Content-Length"))
                return
            if length <= 0 or length > state.args.max_request_bytes:
                json_response(self, 400, error_payload("request body is empty or too large"))
                return
            raw_body = self.rfile.read(length)
            try:
                request = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                json_response(self, 400, error_payload("request body must be a JSON object"))
                return
            if not isinstance(request, dict):
                json_response(self, 400, error_payload("request body must be a JSON object"))
                return

            request_index, request_dir = next_request_dir(state)
            started = time.perf_counter()
            try:
                response = state.resident.execute_resident_request(
                    adapter=state.adapter,
                    skeleton=state.skeleton,
                    engine=state.engine,
                    request=request,
                    output_dir=request_dir,
                    args=state.args,
                    model_dir=state.model_dir,
                    manifest_path=state.manifest_path,
                    manifest=state.manifest,
                    source_commit=state.source_commit,
                    source_diff=state.source_diff,
                    request_json_path=None,
                )
            except SystemExit as exc:
                json_response(self, 400, error_payload(str(exc)))
                return
            except Exception as exc:  # pragma: no cover - preserves remote error artifacts.
                error = error_payload(str(exc), code="resident_engine_error", error_type="server_error")
                write_json(
                    request_dir / "error.json",
                    {
                        "request_index": request_index,
                        "request": request,
                        "error": error,
                    },
                )
                json_response(self, 500, error)
                return

            final_run = response.get("aima_amd395", {}).get("resident_runs", [{}])[-1]
            print(
                json.dumps(
                    {
                        "event": "served",
                        "request_index": request_index,
                        "output_dir": str(request_dir),
                        "finish_reason": response["choices"][0]["finish_reason"],
                        "cache_hits": final_run.get("cache_hits"),
                        "cache_misses": final_run.get("cache_misses"),
                        "engine_wall_time_ms": final_run.get("engine_wall_time_ms"),
                        "http_request_wall_time_ms": (time.perf_counter() - started) * 1000.0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            json_response(self, 200, response)
            if state.args.max_requests and state.served >= state.args.max_requests:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

    ResidentHandler.resident_state = state  # type: ignore[attr-defined]
    return ResidentHandler


def main() -> None:
    root = repo_root()
    shape_lab = root / "benchmarks" / "shape-lab"
    sys.path.insert(0, str(shape_lab))
    resident = load_script_module(
        "amd395_resident_chat_completions_request",
        root / "tools" / "amd395-qwen36-35b-a3b-bf16-resident-chat-completions-request.py",
    )
    adapter = resident.load_script_module(
        "aima_chat_contract",
        root / "tools" / "aima_chat_contract.py",
    )
    skeleton = resident.load_script_module("amd395_run_full_model_skeleton", shape_lab / "run_full_model_skeleton.py")
    engine = resident.load_script_module("amd395_four_layer_mini_engine", shape_lab / "four_layer_mini_engine.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
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
        default="native_selected_expert_consumer",
    )
    parser.add_argument("--logit-topk", type=int, default=5)
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--sampling-top-k", type=int, default=50)
    parser.add_argument(
        "--moe-override-config-by-layer-json",
        help=(
            "Optional JSON object mapping model layer id to a vLLM fused_moe.override_config "
            "object for served one-token decode."
        ),
    )
    parser.add_argument(
        "--linear-attention-post-conv-prep-block-t",
        type=int,
        choices=[8, 16, 32, 64, 128, 256],
        default=resident.DEFAULT_LINEAR_ATTENTION_POST_CONV_PREP_BLOCK_T,
        help="Retained vLLM fused_post_conv_prep BLOCK_T for resident prefill.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-block-t",
        type=int,
        choices=[8, 16, 32, 64],
        default=resident.DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_BLOCK_T,
        help="Retained Triton prefill causal-conv BLOCK_T override for resident prefill.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-block-c",
        type=int,
        choices=[16, 32, 64],
        default=resident.DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_BLOCK_C,
        help="Retained Triton prefill causal-conv BLOCK_C override for resident prefill.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-num-warps",
        type=int,
        choices=[4, 8],
        default=resident.DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_NUM_WARPS,
        help="Retained Triton prefill causal-conv num_warps override for resident prefill.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-post-prep-fusion",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_LINEAR_ATTENTION_PREFILL_CONV_POST_PREP_FUSION,
        help="Default-off prefill candidate that fuses Triton causal-conv with post-conv prep.",
    )
    parser.add_argument(
        "--linear-attention-prefill-vllm-state-handoff",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_LINEAR_ATTENTION_PREFILL_VLLM_STATE_HANDOFF,
        help=(
            "Keep prefill chunk-GDN final state in vLLM layout for native-vLLM "
            "decode state handoff."
        ),
    )
    parser.add_argument(
        "--linear-attention-prefill-fused-h-o",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_LINEAR_ATTENTION_PREFILL_FUSED_H_O,
        help="Use the retained fused prefill chunk-GDN h/o boundary for chunk16 Qwen shapes.",
    )
    parser.add_argument(
        "--linear-attention-prefill-fused-u-h-o",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_LINEAR_ATTENTION_PREFILL_FUSED_U_H_O,
        help="Use the retained W-only plus fused U+h/o prefill chunk-GDN boundary.",
    )
    parser.add_argument(
        "--linear-attention-chunk-gdn-internal-timing",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_LINEAR_ATTENTION_CHUNK_GDN_INTERNAL_TIMING,
        help="Collect a separate diagnostic timeline for retained prefill chunk-GDN internal vLLM FLA kernels.",
    )
    parser.add_argument(
        "--linear-attention-conv-state-refswap",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_LINEAR_ATTENTION_CONV_STATE_REFSWAP,
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
        default=resident.DEFAULT_DECODE_LOOP_FAST_HOUSEKEEPING,
        help="Precompute decode-loop RoPE rows and reuse the next-token embedding buffer.",
    )
    parser.add_argument(
        "--defer-decode-token-cpu-sync",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_DEFER_DECODE_TOKEN_CPU_SYNC,
        help="Materialize generated token ids on CPU once after the decode loop.",
    )
    parser.add_argument(
        "--decode-token-cpu-sync-interval",
        type=int,
        default=resident.DEFAULT_DECODE_TOKEN_CPU_SYNC_INTERVAL,
        help=(
            "Materialize generated token ids on CPU every N tokens; 1 preserves "
            "per-token sync and 0 defers until loop end."
        ),
    )
    parser.add_argument(
        "--decode-loop-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_DECODE_LOOP_DIAGNOSTIC,
        help=(
            "Record per-step decode-loop top-k logits and linear-attention state "
            "statistics. Timings include diagnostic sync overhead."
        ),
    )
    parser.add_argument(
        "--overlap-decode-state-promotion-lm-head",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_OVERLAP_DECODE_STATE_PROMOTION_LM_HEAD,
        help="Overlap decode state promotion with final RMSNorm/LM-head during decode loops.",
    )
    parser.add_argument(
        "--full-attention-fused-gate-o-proj",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_FULL_ATTENTION_FUSED_GATE_O_PROJ,
        help="Probe fused full-attention output gate plus o_proj in served one-token decode.",
    )
    parser.add_argument(
        "--full-attention-kv-cache-layout",
        choices=["seq", "grouped"],
        default=resident.DEFAULT_FULL_ATTENTION_KV_CACHE_LAYOUT,
        help="Probe full-attention KV cache layout for grouped-BMM decode.",
    )
    parser.add_argument(
        "--admitted-context-policy",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_ADMITTED_CONTEXT_POLICY,
        help="Apply the v1.0.0 exact cold-context layout/schedule policy; use --no-admitted-context-policy to opt out.",
    )
    parser.add_argument(
        "--full-attention-fused-norm-rope-kv-write",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_FULL_ATTENTION_FUSED_NORM_ROPE_KV_WRITE,
        help="Probe fused full-attention norm/RoPE plus KV-cache write in served one-token decode.",
    )
    parser.add_argument(
        "--skip-layer-dispatch-metadata",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_SKIP_LAYER_DISPATCH_METADATA,
        help="Skip per-layer dispatch metadata dict construction in served resident requests.",
    )
    parser.add_argument(
        "--overlap-shared-expert-moe",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_OVERLAP_SHARED_EXPERT_MOE,
        help="Enable the retained one-token decode shared-expert/MoE CUDA-stream overlap path for served requests.",
    )
    parser.add_argument(
        "--overlap-shared-expert-router-moe",
        action=argparse.BooleanOptionalAction,
        default=resident.DEFAULT_OVERLAP_SHARED_EXPERT_ROUTER_MOE,
        help="Start shared-expert overlap before router/top-k for served decode tokens.",
    )
    parser.add_argument(
        "--shared-expert-overlap-stream-priority",
        type=int,
        default=resident.DEFAULT_SHARED_EXPERT_OVERLAP_STREAM_PRIORITY,
        help=(
            "Optional torch.cuda.Stream priority for the shared-expert overlap "
            "side stream in served decode tokens."
        ),
    )
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--max-request-bytes", type=int, default=1024 * 1024)
    parser.add_argument(
        "--exact-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the longest exact token-identical cached prefix under the runtime contract.",
    )
    parser.add_argument("--exact-prefix-cache-max-entries", type=int, default=2)
    parser.add_argument("--exact-prefix-cache-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--resident-native-decode-hotset-layers",
        type=int,
        default=12,
        help="Bind the strict-admitted R4 native selected-expert hotset depth.",
    )
    parser.add_argument(
        "--startup-warmup-request-json",
        type=Path,
        help="Run one OpenAI-shaped request before binding the HTTP server so the first served request hits the tensor cache.",
    )
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        raise SystemExit("use --execute to run the resident HTTP server")
    if args.max_requests < 0:
        raise SystemExit("--max-requests must be non-negative")
    if args.max_request_bytes <= 0:
        raise SystemExit("--max-request-bytes must be positive")
    if args.exact_prefix_cache and args.exact_prefix_cache_max_entries <= 0:
        raise SystemExit("--exact-prefix-cache requires --exact-prefix-cache-max-entries >= 1")
    if args.exact_prefix_cache_max_tokens < 0:
        raise SystemExit("--exact-prefix-cache-max-tokens must be non-negative")
    if not 0 <= args.resident_native_decode_hotset_layers <= 40:
        raise SystemExit("--resident-native-decode-hotset-layers must be between 0 and 40")
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
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = Path(args.model_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = engine.load_json(manifest_path)
    source_commit = skeleton.git_value(root, "rev-parse", "HEAD") or "unknown"
    source_diff = skeleton.source_diff_scope(root)

    request_args = argparse.Namespace(**vars(args))
    request_args.repeat_same_request = 1
    request_args.reuse_tensor_cache = True
    request_args.stdin_loop = False
    request_args.dry_run = False

    started_wall = time.perf_counter()
    startup_load = resident.prepare_resident_engine(
        engine=engine,
        manifest=manifest,
        model_dir=model_dir,
        args=request_args,
    )
    startup_warmup: dict[str, Any] | None = None
    state = ServerState(
        adapter=adapter,
        skeleton=skeleton,
        engine=engine,
        model_dir=model_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        output_dir=output_dir,
        args=request_args,
        source_commit=source_commit,
        source_diff=source_diff,
        started_wall=started_wall,
        startup_load=startup_load,
    )
    state.resident = resident
    if args.startup_warmup_request_json:
        warmup_path = args.startup_warmup_request_json.resolve()
        warmup_request = adapter.json_load(warmup_path)
        warmup_started = time.perf_counter()
        response = resident.execute_resident_request(
            adapter=adapter,
            skeleton=skeleton,
            engine=engine,
            request=warmup_request,
            output_dir=output_dir / "startup-warmup",
            args=request_args,
            model_dir=model_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            source_commit=source_commit,
            source_diff=source_diff,
            request_json_path=warmup_path,
        )
        final_run = response.get("aima_amd395", {}).get("resident_runs", [{}])[-1]
        startup_warmup = {
            "request_json": str(warmup_path),
            "output_dir": str(output_dir / "startup-warmup"),
            "wall_time_ms": (time.perf_counter() - warmup_started) * 1000.0,
            "finish_reason": response["choices"][0]["finish_reason"],
            "cache_hits": final_run.get("cache_hits"),
            "cache_misses": final_run.get("cache_misses"),
            "engine_wall_time_ms": final_run.get("engine_wall_time_ms"),
        }
        print(json.dumps({"event": "startup_warmup", **startup_warmup}, sort_keys=True), flush=True)
    server = HTTPServer((args.host, args.port), build_handler(state))
    actual_host, actual_port = server.server_address
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "pid": os.getpid(),
        "listen_host": actual_host,
        "listen_port": actual_port,
        "model_dir": str(model_dir),
        "manifest": str(manifest_path),
        "source_commit": source_commit,
        "source_diff_scope": source_diff,
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "seq_len": args.seq_len,
        "linear_attention_variant": resident.DEFAULT_LINEAR_ATTENTION_VARIANT,
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
        "moe_override_config_by_layer": args.moe_override_config_by_layer,
        "overlap_shared_expert_moe": args.overlap_shared_expert_moe,
        "overlap_shared_expert_router_moe": args.overlap_shared_expert_router_moe,
        "shared_expert_overlap_stream_priority": args.shared_expert_overlap_stream_priority,
        "decode_loop_fast_housekeeping": args.decode_loop_fast_housekeeping,
        "decode_loop_defer_token_cpu_sync": args.defer_decode_token_cpu_sync,
        "decode_loop_token_cpu_sync_interval": args.effective_decode_token_cpu_sync_interval,
        "decode_loop_diagnostic": args.decode_loop_diagnostic,
        "overlap_decode_state_promotion_lm_head": args.overlap_decode_state_promotion_lm_head,
        "full_attention_fused_gate_o_proj": args.full_attention_fused_gate_o_proj,
        "full_attention_kv_cache_layout": args.full_attention_kv_cache_layout,
        "admitted_context_policy": args.admitted_context_policy,
        "full_attention_fused_norm_rope_kv_write": args.full_attention_fused_norm_rope_kv_write,
        "skip_layer_dispatch_metadata": args.skip_layer_dispatch_metadata,
        "max_requests": args.max_requests,
        "startup_warmup": startup_warmup,
        "startup_load": startup_load,
        "model_loaded": startup_load.get("model_loaded") is True,
        "model_load_to_api_ready_ms": (time.perf_counter() - started_wall) * 1000.0,
        "output_dir": str(output_dir),
        "route": "in-process resident tensor-cache engine",
        "api": "POST /v1/chat/completions",
        "env": {
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
        },
    }
    write_json(output_dir / "server.json", metadata)
    if args.ready_file:
        write_json(args.ready_file.resolve(), metadata)
    print(json.dumps({"event": "ready", **metadata}, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        write_json(
            output_dir / "server-exit.json",
            {
                **metadata,
                "served": state.served,
                "uptime_s": time.perf_counter() - started_wall,
            },
        )
        print(json.dumps({"event": "exit", "served": state.served, "output_dir": str(output_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
