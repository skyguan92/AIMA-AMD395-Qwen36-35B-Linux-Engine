#!/usr/bin/env python3
"""Benchmark-only vLLM middleware for per-request VL stage evidence.

Load this module with vLLM's ``--middleware`` option and set
``AIMA_VLLM_VL_BENCHMARK_LOG`` to a JSONL path.  Only requests carrying a
valid ``x-aima-vl-benchmark-id`` header are instrumented.  The middleware
does not inspect or persist request/response content.
"""

from __future__ import annotations

import asyncio
from functools import wraps
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence
from urllib.parse import urlsplit


BENCHMARK_HEADER = b"x-aima-vl-benchmark-id"
LOG_ENV = "AIMA_VLLM_VL_BENCHMARK_LOG"
_RUN_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

_MEDIA_LOCK = threading.Lock()
_ACTIVE_MEDIA_RUN: str | None = None
_MEDIA_INTERVALS: dict[str, list[dict[str, Any]]] = {}
_PATCH_LOCK = threading.Lock()
_MEDIA_PATCHED = False
_LOG_LOCK = threading.Lock()


def _media_kind(media_io: object) -> str:
    name = type(media_io).__name__
    return name.removesuffix("MediaIO").lower() or "unknown"


def _media_scheme(url: object) -> str:
    try:
        return urlsplit(str(url)).scheme or "unknown"
    except Exception:
        return "unknown"


def _record_media_interval(
    start_ns: int,
    end_ns: int,
    url: object,
    media_io: object,
) -> None:
    with _MEDIA_LOCK:
        run_id = _ACTIVE_MEDIA_RUN
        if run_id is None:
            return
        _MEDIA_INTERVALS.setdefault(run_id, []).append(
            {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "modality": _media_kind(media_io),
                "scheme": _media_scheme(url),
            }
        )


def _install_media_patch() -> None:
    """Patch the frozen vLLM connector at its common sync/async boundary."""

    global _MEDIA_PATCHED
    with _PATCH_LOCK:
        if _MEDIA_PATCHED:
            return

        from vllm.multimodal.media.connector import MediaConnector

        original_sync = MediaConnector.load_from_url
        original_async = MediaConnector.load_from_url_async

        @wraps(original_sync)
        def timed_sync(self, url, media_io, *args, **kwargs):
            start_ns = time.perf_counter_ns()
            try:
                return original_sync(self, url, media_io, *args, **kwargs)
            finally:
                _record_media_interval(
                    start_ns, time.perf_counter_ns(), url, media_io
                )

        @wraps(original_async)
        async def timed_async(self, url, media_io, *args, **kwargs):
            start_ns = time.perf_counter_ns()
            try:
                return await original_async(self, url, media_io, *args, **kwargs)
            finally:
                _record_media_interval(
                    start_ns, time.perf_counter_ns(), url, media_io
                )

        MediaConnector.load_from_url = timed_sync
        MediaConnector.load_from_url_async = timed_async
        _MEDIA_PATCHED = True


def _begin_media_capture(run_id: str) -> None:
    global _ACTIVE_MEDIA_RUN
    with _MEDIA_LOCK:
        if _ACTIVE_MEDIA_RUN is not None:
            raise RuntimeError("overlapping instrumented VL requests are unsupported")
        _ACTIVE_MEDIA_RUN = run_id
        _MEDIA_INTERVALS[run_id] = []


def _merged_interval_ns(intervals: Sequence[Mapping[str, Any]]) -> int:
    spans = sorted(
        (int(item["start_ns"]), int(item["end_ns"])) for item in intervals
    )
    if not spans:
        return 0

    total = 0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += max(0, current_end - current_start)
            current_start, current_end = start, end
    return total + max(0, current_end - current_start)


def _end_media_capture(run_id: str) -> dict[str, Any]:
    global _ACTIVE_MEDIA_RUN
    with _MEDIA_LOCK:
        if _ACTIVE_MEDIA_RUN != run_id:
            raise RuntimeError("VL media capture ownership drifted")
        _ACTIVE_MEDIA_RUN = None
        intervals = _MEDIA_INTERVALS.pop(run_id, [])

    public_intervals = [
        {
            "duration_secs": max(
                0.0, (int(item["end_ns"]) - int(item["start_ns"])) / 1e9
            ),
            "modality": item["modality"],
            "scheme": item["scheme"],
        }
        for item in intervals
    ]
    return {
        "items": public_intervals,
        "item_count": len(public_intervals),
        "media_load_decode_secs": _merged_interval_ns(intervals) / 1e9,
        "summed_item_secs": sum(
            item["duration_secs"] for item in public_intervals
        ),
    }


def _merge_mm_stats(
    processor_stats: Mapping[str, Mapping[str, float]],
    worker_stats: Sequence[Mapping[str, Mapping[str, float | int]] | None],
) -> dict[str, dict[str, float | int]]:
    """Mirror the merge semantics in vLLM's official mm-processor bench."""

    encoder_stats: dict[str, dict[str, float | int]] = {}
    for worker in worker_stats:
        if not worker:
            continue
        for request_id, stats in worker.items():
            if request_id not in encoder_stats:
                encoder_stats[request_id] = dict(stats)
                continue
            current = encoder_stats[request_id]
            current["encoder_forward_secs"] = max(
                float(current.get("encoder_forward_secs", 0.0)),
                float(stats.get("encoder_forward_secs", 0.0)),
            )
            current["num_encoder_calls"] = max(
                int(current.get("num_encoder_calls", 0)),
                int(stats.get("num_encoder_calls", 0)),
            )

    merged: dict[str, dict[str, float | int]] = {
        request_id: dict(stats)
        for request_id, stats in processor_stats.items()
    }
    for request_id, stats in encoder_stats.items():
        target = request_id
        if target not in merged:
            possible_original = request_id.rpartition("-")[0]
            if possible_original in merged:
                target = possible_original
        merged.setdefault(target, {}).update(stats)
    return merged


async def _drain_mm_stats(engine_client: Any) -> dict[str, Any]:
    registry = getattr(engine_client.renderer, "_mm_timing_registry", None)
    processor_stats = registry.stat() if registry is not None else {}
    worker_stats = await engine_client.collective_rpc(
        "get_encoder_timing_stats"
    )
    return {
        "processor": processor_stats,
        "workers": worker_stats,
        "merged": _merge_mm_stats(processor_stats, worker_stats),
    }


def _header_value(scope: Mapping[str, Any], name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if bytes(key).lower() == name:
            try:
                return bytes(value).decode("ascii")
            except UnicodeDecodeError:
                return None
    return None


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class VlBenchmarkMetricsMiddleware:
    """Collect frozen vLLM VL stages after a marked response completes."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app
        self._request_lock = asyncio.Lock()
        _install_media_patch()

    async def __call__(self, scope, receive, send) -> None:
        run_id = _header_value(scope, BENCHMARK_HEADER)
        log_path = os.environ.get(LOG_ENV)
        if (
            scope.get("type") != "http"
            or scope.get("path") != "/v1/chat/completions"
            or run_id is None
            or _RUN_ID.fullmatch(run_id) is None
            or not log_path
        ):
            await self.app(scope, receive, send)
            return

        async with self._request_lock:
            engine_client = scope["app"].state.engine_client
            await _drain_mm_stats(engine_client)
            _begin_media_capture(run_id)

            started_ns = time.perf_counter_ns()
            response_start_ns: int | None = None
            first_body_ns: int | None = None
            response_end_ns: int | None = None
            status_code: int | None = None
            response_bytes = 0
            error: str | None = None

            async def send_wrapper(message: Mapping[str, Any]) -> None:
                nonlocal response_start_ns, first_body_ns, response_end_ns
                nonlocal status_code, response_bytes
                now_ns = time.perf_counter_ns()
                if message.get("type") == "http.response.start":
                    response_start_ns = now_ns
                    status_code = int(message.get("status", 0))
                elif message.get("type") == "http.response.body":
                    body = message.get("body", b"")
                    response_bytes += len(body)
                    if body and first_body_ns is None:
                        first_body_ns = now_ns
                    if not message.get("more_body", False):
                        response_end_ns = now_ns
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                finished_ns = time.perf_counter_ns()
                media = _end_media_capture(run_id)
                try:
                    mm_stats = await _drain_mm_stats(engine_client)
                    stats_error = None
                except BaseException as exc:
                    mm_stats = {"processor": {}, "workers": [], "merged": {}}
                    stats_error = type(exc).__name__

                def elapsed(end_ns: int | None) -> float | None:
                    if end_ns is None:
                        return None
                    return max(0.0, (end_ns - started_ns) / 1e9)

                _append_jsonl(
                    Path(log_path),
                    {
                        "schema": "aima-amd395-qwen36/vllm-vl-stage-sample/v1",
                        "benchmark_id": run_id,
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "status_code": status_code,
                        "response_bytes": response_bytes,
                        "timings": {
                            "asgi_total_secs": (finished_ns - started_ns) / 1e9,
                            "response_start_secs": elapsed(response_start_ns),
                            "first_body_secs": elapsed(first_body_ns),
                            "response_end_secs": elapsed(response_end_ns),
                        },
                        "media": media,
                        "multimodal": mm_stats,
                        "request_error": error,
                        "stats_error": stats_error,
                    },
                )


__all__ = ["VlBenchmarkMetricsMiddleware"]
