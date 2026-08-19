#!/usr/bin/env python3
"""Capture one marked streaming request for paired native/vLLM VL timing."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
from typing import Any, Mapping
import urllib.error
import urllib.parse
import urllib.request


SCHEMA = "aima-amd395-qwen36/vl-performance-request-sample/v1"
BENCHMARK_HEADER = "x-aima-vl-benchmark-id"
MEDIA_ROOT_PLACEHOLDER = "${AIMA_VL_MEDIA_ROOT}"
PROMPT_NONCE_PLACEHOLDER = "${AIMA_VL_PROMPT_NONCE}"
TEXT_PADDING_UNIT = " x"
TEXT_PADDING_TOKEN_ID = 830
_RUN_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
PROMETHEUS_METRICS = (
    "vllm:prompt_tokens",
    "vllm:prompt_tokens_cached",
    "vllm:generation_tokens",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def loopback_endpoint(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint must be an explicit loopback HTTP port")
    return value.rstrip("/")


def metric_totals(text: str) -> dict[str, float]:
    totals = {name: 0.0 for name in PROMETHEUS_METRICS}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            continue
        sample, raw_value = fields
        name = sample.split("{", 1)[0]
        if name not in totals:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            totals[name] += value
    return totals


def metric_delta(
    before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, float]:
    return {
        name: float(after.get(name, 0.0)) - float(before.get(name, 0.0))
        for name in PROMETHEUS_METRICS
    }


def semantic_delta(event: Mapping[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            return reasoning
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return canonical_bytes(tool_calls).decode("utf-8")
    return ""


def normalized_stream_error(event: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_error = event.get("error")
    if not isinstance(raw_error, Mapping):
        return None
    result: dict[str, Any] = {}
    for key in ("message", "type", "code", "param"):
        if key not in raw_error:
            continue
        value = raw_error.get(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
    if not result:
        result["payload_sha256"] = sha256_bytes(canonical_bytes(raw_error))
    return result


def normalized_capture_error(kind: str, error: BaseException) -> dict[str, str]:
    """Describe a transport/capture failure without copying response content."""

    message = str(error)
    return {
        "kind": kind,
        "exception_type": type(error).__name__,
        "message_sha256": sha256_bytes(message.encode("utf-8")),
    }


def request_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    images = 0
    videos = 0
    text_characters = 0
    message_count = 0
    messages = payload.get("messages")
    if isinstance(messages, list):
        message_count = len(messages)
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if isinstance(content, str):
                text_characters += len(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") in {"image_url", "input_image"}:
                    images += 1
                elif part.get("type") == "video_url":
                    videos += 1
                text = part.get("text")
                if isinstance(text, str):
                    text_characters += len(text)
    return {
        "message_count": message_count,
        "image_count": images,
        "video_count": videos,
        "text_characters": text_characters,
        "max_tokens": payload.get(
            "max_completion_tokens", payload.get("max_tokens")
        ),
        "stream": payload.get("stream"),
        "temperature": payload.get("temperature"),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def media_components(
    payload: Mapping[str, Any], media_root: Path
) -> list[dict[str, Any]]:
    root = media_root.resolve()
    result: list[dict[str, Any]] = []
    hashed: dict[Path, tuple[int, str]] = {}
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return result
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind not in {"image_url", "input_image", "video_url"}:
                continue
            field = "image_url" if kind in {"image_url", "input_image"} else "video_url"
            source = part.get(field)
            url = source.get("url") if isinstance(source, Mapping) else source
            if not isinstance(url, str):
                raise ValueError("performance media URL must be a string")
            parsed = urllib.parse.urlparse(url)
            if (
                parsed.scheme != "file"
                or parsed.netloc not in {"", "localhost"}
                or parsed.params
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "performance evidence currently requires a local file URL"
                )
            path = Path(urllib.parse.unquote(parsed.path)).resolve()
            try:
                relative = path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "performance media escaped the allowed media root"
                ) from error
            if not path.is_file():
                raise ValueError("performance media file is missing")
            if path not in hashed:
                hashed[path] = (path.stat().st_size, file_sha256(path))
            byte_count, digest = hashed[path]
            result.append(
                {
                    "index": len(result),
                    "modality": "image" if field == "image_url" else "video",
                    "path": f"${{AIMA_VL_MEDIA_ROOT}}/{relative.as_posix()}",
                    "bytes": byte_count,
                    "sha256": digest,
                }
            )
    return result


def substitute_media_root(value: Any, media_root: Path) -> Any:
    if isinstance(value, str):
        return value.replace(MEDIA_ROOT_PLACEHOLDER, str(media_root))
    if isinstance(value, list):
        return [substitute_media_root(item, media_root) for item in value]
    if isinstance(value, dict):
        return {
            key: substitute_media_root(item, media_root)
            for key, item in value.items()
        }
    return value


def count_string(value: Any, needle: str) -> int:
    if isinstance(value, str):
        return value.count(needle)
    if isinstance(value, list):
        return sum(count_string(item, needle) for item in value)
    if isinstance(value, dict):
        return sum(count_string(item, needle) for item in value.values())
    return 0


def substitute_prompt_nonce(value: Any, nonce: str) -> Any:
    if _RUN_ID.fullmatch(nonce) is None:
        raise ValueError("prompt nonce contains unsupported characters")
    if count_string(value, PROMPT_NONCE_PLACEHOLDER) != 1:
        raise ValueError("request must contain exactly one prompt nonce placeholder")
    if isinstance(value, str):
        return value.replace(PROMPT_NONCE_PLACEHOLDER, nonce)
    if isinstance(value, list):
        return [substitute_prompt_nonce_value(item, nonce) for item in value]
    if isinstance(value, dict):
        return {
            key: substitute_prompt_nonce_value(item, nonce)
            for key, item in value.items()
        }
    return value


def substitute_prompt_nonce_value(value: Any, nonce: str) -> Any:
    if isinstance(value, str):
        return value.replace(PROMPT_NONCE_PLACEHOLDER, nonce)
    if isinstance(value, list):
        return [substitute_prompt_nonce_value(item, nonce) for item in value]
    if isinstance(value, dict):
        return {
            key: substitute_prompt_nonce_value(item, nonce)
            for key, item in value.items()
        }
    return value


def append_text_padding(payload: dict[str, Any], token_count: int) -> None:
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or token_count < 0
        or token_count > 262_144
    ):
        raise ValueError("text padding token count is outside the model window")
    if token_count == 0:
        return
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("request has no messages for text padding")
    padding = TEXT_PADDING_UNIT * token_count
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content + padding
            return
        if not isinstance(content, list):
            continue
        for part in reversed(content):
            if (
                isinstance(part, dict)
                and part.get("type") in {"text", "input_text"}
                and isinstance(part.get("text"), str)
            ):
                part["text"] += padding
                return
    raise ValueError("request has no text field for deterministic padding")


def process_tree(root_pid: int) -> set[int]:
    found: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        try:
            children = Path(
                f"/proc/{pid}/task/{pid}/children"
            ).read_text(encoding="ascii")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for value in children.split():
            try:
                pending.append(int(value))
            except ValueError:
                continue
    return found


def pid_is_live(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def tree_rss_bytes(root_pid: int) -> int | None:
    total_kib = 0
    observed = False
    for pid in process_tree(root_pid):
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) >= 2:
                    total_kib += int(fields[1])
                    observed = True
                break
    return total_kib * 1024 if observed else None


def amd_memory_bytes() -> dict[str, int | None]:
    cards = sorted(Path("/sys/class/drm").glob("card[0-9]*"))
    for card in cards:
        try:
            if (card / "device/vendor").read_text().strip() != "0x1002":
                continue
        except (FileNotFoundError, PermissionError):
            continue
        result: dict[str, int | None] = {}
        for label, filename in (
            ("gtt_used_bytes", "mem_info_gtt_used"),
            ("vram_used_bytes", "mem_info_vram_used"),
        ):
            try:
                result[label] = int(
                    (card / "device" / filename).read_text().strip()
                )
            except (FileNotFoundError, PermissionError, ValueError):
                result[label] = None
        result["card"] = card.name  # type: ignore[assignment]
        return result
    return {"gtt_used_bytes": None, "vram_used_bytes": None, "card": None}


class MemorySampler:
    def __init__(self, pid: int, interval_seconds: float = 0.1) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples = 0
        self.peak_rss: int | None = None
        self.peak_gtt: int | None = None
        self.peak_vram: int | None = None
        self.card: str | None = None

    def _sample(self) -> None:
        rss = tree_rss_bytes(self.pid)
        gpu = amd_memory_bytes()
        gtt = gpu.get("gtt_used_bytes")
        vram = gpu.get("vram_used_bytes")
        card = gpu.get("card")
        if isinstance(rss, int):
            self.peak_rss = max(self.peak_rss or 0, rss)
        if isinstance(gtt, int):
            self.peak_gtt = max(self.peak_gtt or 0, gtt)
        if isinstance(vram, int):
            self.peak_vram = max(self.peak_vram or 0, vram)
        if isinstance(card, str):
            self.card = card
        self.samples += 1

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval_seconds)
        self._sample()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def result(self) -> dict[str, Any]:
        return {
            "process_tree_root_pid": self.pid,
            "sample_interval_seconds": self.interval_seconds,
            "samples": self.samples,
            "peak_host_rss_bytes": self.peak_rss,
            "peak_gtt_used_bytes": self.peak_gtt,
            "peak_vram_used_bytes": self.peak_vram,
            "drm_card": self.card,
        }


def fetch_metrics(opener, endpoint: str, timeout: float) -> dict[str, float]:
    request = urllib.request.Request(endpoint + "/metrics", method="GET")
    with opener.open(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"metrics endpoint returned HTTP {response.status}")
        return metric_totals(response.read().decode("utf-8", errors="strict"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--prompt-nonce")
    parser.add_argument("--text-padding-tokens", type=int, default=0)
    parser.add_argument("--expected-completion-tokens", type=int)
    parser.add_argument(
        "--engine-role",
        choices=("reference", "candidate"),
        required=True,
    )
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--prometheus", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if not args.request.is_file():
        raise SystemExit(f"request file is missing: {args.request}")
    if not args.media_root.is_dir() or not args.media_root.is_absolute():
        raise SystemExit("media root must be an existing absolute directory")
    if _RUN_ID.fullmatch(args.benchmark_id) is None:
        raise SystemExit("benchmark ID contains unsupported characters")
    if not pid_is_live(args.server_pid):
        raise SystemExit("server PID is not live")
    if (
        not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
        or args.timeout_seconds > 3600
    ):
        raise SystemExit("timeout must be positive and at most 3600 seconds")
    try:
        endpoint = loopback_endpoint(args.endpoint)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    request_template = args.request.read_bytes()
    payload = json.loads(request_template)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("messages"), list
    ):
        raise SystemExit("performance request must contain a messages array")
    payload = substitute_media_root(payload, args.media_root)
    prompt_nonce_sha256: str | None = None
    if args.prompt_nonce is not None:
        try:
            payload = substitute_prompt_nonce(payload, args.prompt_nonce)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        prompt_nonce_sha256 = sha256_bytes(args.prompt_nonce.encode("ascii"))
    elif count_string(payload, PROMPT_NONCE_PLACEHOLDER) != 0:
        raise SystemExit("request contains an unresolved prompt nonce placeholder")
    try:
        append_text_padding(payload, args.text_padding_tokens)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if float(payload.get("temperature", 0)) != 0.0:
        raise SystemExit("G4 request must use temperature=0")
    payload["temperature"] = 0
    payload["model"] = args.model
    payload["stream"] = True
    stream_options = payload.setdefault("stream_options", {})
    if not isinstance(stream_options, dict):
        raise SystemExit("stream_options must be an object")
    stream_options["include_usage"] = True
    expected_completion_tokens = args.expected_completion_tokens
    if expected_completion_tokens is None:
        expected_completion_tokens = payload.get(
            "max_completion_tokens", payload.get("max_tokens")
        )
    if (
        not isinstance(expected_completion_tokens, int)
        or isinstance(expected_completion_tokens, bool)
        or expected_completion_tokens <= 0
        or expected_completion_tokens > 1024
    ):
        raise SystemExit(
            "expected completion tokens must be an integer from 1 through 1024"
        )
    if payload.get(
        "max_completion_tokens", payload.get("max_tokens")
    ) != expected_completion_tokens:
        raise SystemExit("request output limit differs from expected completion tokens")
    try:
        media = media_components(payload, args.media_root)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not media:
        raise SystemExit("VL performance request must contain local media")
    request_body = canonical_bytes(payload)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    metrics_before: dict[str, float] | None = None
    if args.prometheus:
        metrics_before = fetch_metrics(opener, endpoint, args.timeout_seconds)

    request = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        data=request_body,
        headers={
            "Content-Type": "application/json",
            BENCHMARK_HEADER: args.benchmark_id,
        },
        method="POST",
    )
    event_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    output_bytes = 0
    semantic_chunks = 0
    event_count = 0
    first_semantic_ns: int | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    native_metrics: dict[str, Any] | None = None
    stream_error: dict[str, Any] | None = None
    done = False
    started_ns = time.perf_counter_ns()
    finished_ns = started_ns
    status_code: int | None = None
    capture_error: dict[str, Any] | None = None
    with MemorySampler(args.server_pid) as memory:
        try:
            response = opener.open(request, timeout=args.timeout_seconds)
        except urllib.error.HTTPError as error:
            detail = error.read()
            status_code = error.code
            capture_error = {
                **normalized_capture_error("http_error", error),
                "body_bytes": len(detail),
                "body_sha256": sha256_bytes(detail),
            }
        except Exception as error:
            capture_error = normalized_capture_error("transport_error", error)
        else:
            try:
                with response:
                    status_code = response.status
                    if status_code != 200:
                        raise RuntimeError(
                            f"performance endpoint returned HTTP {status_code}"
                        )
                    for raw_line in response:
                        line = raw_line.strip()
                        if not line or not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            done = True
                            continue
                        event = json.loads(data)
                        if not isinstance(event, dict):
                            raise RuntimeError("SSE event is not an object")
                        encoded_event = canonical_bytes(event)
                        event_digest.update(encoded_event)
                        event_digest.update(b"\n")
                        event_count += 1
                        semantic = semantic_delta(event)
                        if semantic:
                            if first_semantic_ns is None:
                                first_semantic_ns = time.perf_counter_ns()
                            encoded = semantic.encode("utf-8")
                            output_digest.update(encoded)
                            output_bytes += len(encoded)
                            semantic_chunks += 1
                        if isinstance(event.get("usage"), dict):
                            usage = dict(event["usage"])
                        if isinstance(event.get("aima_amd395"), dict):
                            native_metrics = dict(event["aima_amd395"])
                        normalized_error = normalized_stream_error(event)
                        if normalized_error is not None and stream_error is None:
                            stream_error = normalized_error
                        choices = event.get("choices")
                        if isinstance(choices, list):
                            for choice in choices:
                                if not isinstance(choice, Mapping):
                                    continue
                                value = choice.get("finish_reason")
                                if isinstance(value, str):
                                    finish_reason = value
            except Exception as error:
                capture_error = normalized_capture_error("stream_error", error)
        finally:
            finished_ns = time.perf_counter_ns()

    prometheus: dict[str, Any] | None = None
    if metrics_before is not None:
        metrics_after = metrics_before
        try:
            for _ in range(40):
                metrics_after = fetch_metrics(
                    opener, endpoint, args.timeout_seconds
                )
                delta = metric_delta(metrics_before, metrics_after)
                if delta["vllm:request_prefill_time_seconds_count"] >= 1:
                    break
                time.sleep(0.05)
        except Exception as error:
            prometheus = {
                "before": metrics_before,
                "after": None,
                "delta": None,
                "fetch_error": normalized_capture_error(
                    "metrics_after_error", error
                ),
            }
        else:
            prometheus = {
                "before": metrics_before,
                "after": metrics_after,
                "delta": metric_delta(metrics_before, metrics_after),
                "fetch_error": None,
            }

    total_seconds = (finished_ns - started_ns) / 1e9
    ttft_seconds = (
        (first_semantic_ns - started_ns) / 1e9
        if first_semantic_ns is not None
        else None
    )
    completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage, dict) else None
    )
    decode_tokens_per_second: float | None = None
    if (
        isinstance(completion_tokens, int)
        and completion_tokens > 1
        and isinstance(ttft_seconds, float)
        and total_seconds > ttft_seconds
    ):
        decode_tokens_per_second = (completion_tokens - 1) / (
            total_seconds - ttft_seconds
        )

    complete = (
        done
        and capture_error is None
        and stream_error is None
        and first_semantic_ns is not None
        and finish_reason in {"stop", "length", "tool_calls"}
        and isinstance(usage, dict)
        and isinstance(completion_tokens, int)
        and completion_tokens == expected_completion_tokens
    )
    result = {
        "schema": SCHEMA,
        "captured_at": utc_now(),
        "complete": complete,
        "benchmark_id": args.benchmark_id,
        "engine_role": args.engine_role,
        "endpoint": "${AIMA_LOOPBACK_ENDPOINT}",
        "server_pid": args.server_pid,
        "host": {"hostname": socket.gethostname()},
        "request": {
            "path": str(args.request),
            "template_sha256": sha256_bytes(request_template),
            "sha256": sha256_bytes(request_body),
            "bytes": len(request_body),
            "summary": request_summary(payload),
            "media": media,
            "prompt_nonce_sha256": prompt_nonce_sha256,
            "text_padding": {
                "tokens": args.text_padding_tokens,
                "unit_sha256": sha256_bytes(TEXT_PADDING_UNIT.encode("ascii")),
                "frozen_single_token_id": TEXT_PADDING_TOKEN_ID,
            },
            "expected_completion_tokens": expected_completion_tokens,
        },
        "response": {
            "status_code": status_code,
            "sse_done": done,
            "event_count": event_count,
            "event_stream_sha256": event_digest.hexdigest(),
            "semantic_chunks": semantic_chunks,
            "content_bytes": output_bytes,
            "content_sha256": output_digest.hexdigest(),
            "finish_reason": finish_reason,
            "usage": usage,
            "error": stream_error,
            "capture_error": capture_error,
        },
        "timings": {
            "ttft_seconds": ttft_seconds,
            "total_seconds": total_seconds,
            "decode_tokens_per_second": decode_tokens_per_second,
        },
        "memory": memory.result(),
        "native_metrics": native_metrics,
        "prometheus": prometheus,
    }
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "complete": complete}))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
