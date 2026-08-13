#!/usr/bin/env python3
"""Probe the fixed vLLM OpenAI VL surface and emit hash-bound evidence."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_capability import (  # noqa: E402
    REQUIRED_API_CASES,
    validate_capability_manifest,
)
from aima_engine.vl_reference import (  # noqa: E402
    CAPABILITY_SCHEMA,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    REFERENCE_ATTENTION_BACKEND,
    REFERENCE_MAX_BATCHED_TOKENS,
    REFERENCE_MEDIA_LIMITS,
    atomic_json,
    file_component,
    seal_manifest,
    sha256_file,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def recursive_replace(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: recursive_replace(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_replace(item, replacements) for item in value]
    if isinstance(value, str) and value in replacements:
        return replacements[value]
    return value


def recursive_redact(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: recursive_redact(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_redact(item, replacements) for item in value]
    if isinstance(value, str):
        for private, public in replacements.items():
            value = value.replace(private, public)
    return value


class Fixtures:
    def __init__(self, root: Path, http_base: str):
        self.root = root.resolve()
        self.http_base = http_base.rstrip("/")
        manifest_path = self.root / "fixtures-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("fixtures")
        if not isinstance(records, list):
            raise ValueError("fixture manifest has no fixtures array")
        self.records: dict[str, dict[str, Any]] = {}
        for item in records:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("fixture manifest contains a malformed record")
            path = self.root / item["path"]
            if not path.is_file() or sha256_file(path) != item.get("sha256"):
                raise ValueError(f"fixture failed hash verification: {item['path']}")
            self.records[item["path"]] = item

    def part(
        self,
        modality: str,
        fixture_id: str,
        transport: str,
        replacements: dict[str, Any],
    ) -> dict[str, Any]:
        record = self.records[fixture_id]
        path = self.root / fixture_id
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if transport == "local":
            url = path.as_uri()
        elif transport == "http":
            url = f"{self.http_base}/{fixture_id}"
        elif transport == "data":
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            url = f"data:{mime};base64,{encoded}"
        else:
            raise ValueError(f"unsupported fixture transport: {transport}")
        replacements[url] = {
            "fixture": fixture_id,
            "transport": transport,
            "mime": mime,
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        if modality == "image":
            return {"type": "image_url", "image_url": {"url": url}}
        if modality == "video":
            return {"type": "video_url", "video_url": {"url": url}}
        raise ValueError(f"unsupported fixture modality: {modality}")


def text(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


def tool_contract() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "inspect_visual",
                "description": "Record a short label for visual content.",
                "parameters": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                    "required": ["label"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def request_payload(
    model: str,
    messages: list[dict[str, Any]],
    *,
    stream: bool = False,
    max_tokens: int = 1,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["parallel_tool_calls"] = False
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def normalize_json_response(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    result.pop("id", None)
    result.pop("created", None)
    result.pop("system_fingerprint", None)
    return result


def parse_stream(raw: bytes) -> dict[str, Any]:
    events: list[Any] = []
    done = False
    content = ""
    tool_call_seen = False
    for line in raw.splitlines():
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            done = True
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            events.append({"malformed_sha256": digest(payload)})
            continue
        event = normalize_json_response(event)
        events.append(event)
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content += delta["content"]
            if delta.get("tool_calls"):
                tool_call_seen = True
    return {
        "done": done,
        "event_count": len(events),
        "events": events,
        "aggregate_content": content,
        "tool_call_seen": tool_call_seen,
    }


def perform_request(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> tuple[int, bytes, Any, dict[str, float | None]]:
    body = canonical_bytes(payload)
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    ttft: float | None = None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        elapsed = time.monotonic() - start
        try:
            normalized = normalize_json_response(json.loads(raw))
        except json.JSONDecodeError:
            normalized = {"unparsed_body_sha256": digest(raw)}
        return exc.code, raw, normalized, {
            "ttft_seconds": None,
            "total_seconds": elapsed,
        }

    with response:
        status = response.status
        if payload.get("stream"):
            chunks: list[bytes] = []
            while True:
                line = response.readline()
                if not line:
                    break
                chunks.append(line)
                if ttft is None and line.startswith(b"data: ") and line[6:].strip() not in {
                    b"",
                    b"[DONE]",
                }:
                    ttft = time.monotonic() - start
            raw = b"".join(chunks)
            normalized = parse_stream(raw)
        else:
            raw = response.read()
            try:
                normalized = normalize_json_response(json.loads(raw))
            except json.JSONDecodeError:
                normalized = {"unparsed_body_sha256": digest(raw)}
        elapsed = time.monotonic() - start
    return status, raw, normalized, {
        "ttft_seconds": ttft,
        "total_seconds": elapsed,
    }


def response_has_tool_call(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    for choice in response.get("choices", []):
        message = choice.get("message") or {}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") == "inspect_visual":
                return True
    return False


def execute_case(
    endpoint: str,
    case_id: str,
    surfaces: list[str],
    expected_accept: bool,
    payload: dict[str, Any],
    replacements: dict[str, Any],
    *,
    timeout: float,
    response_redactions: dict[str, str],
    require_tool_call: bool = False,
) -> dict[str, Any]:
    request_body = canonical_bytes(payload)
    status, raw, normalized, timings = perform_request(
        endpoint, payload, timeout=timeout
    )
    normalized = recursive_redact(normalized, response_redactions)
    accepted = 200 <= status < 300
    passed = accepted == expected_accept
    if accepted and payload.get("stream"):
        passed = passed and bool(normalized.get("done")) and bool(
            normalized.get("event_count")
        )
    elif accepted:
        passed = passed and isinstance(normalized, dict) and bool(
            normalized.get("choices")
        )
    if require_tool_call:
        passed = passed and response_has_tool_call(normalized)
    return {
        "case_id": case_id,
        "surfaces": surfaces,
        "expected_accept": expected_accept,
        "accepted": accepted,
        "passed": passed,
        "status_code": status,
        "request_sha256": digest(request_body),
        "request": recursive_replace(payload, replacements),
        "response_sha256": digest(raw),
        "response_bytes": len(raw),
        "response": normalized,
        "timings": timings,
    }


def build_cases(fixtures: Fixtures, model: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def add(
        case_id: str,
        surfaces: list[str],
        messages: list[dict[str, Any]],
        replacements: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        max_tokens: int = 1,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        require_tool_call: bool = False,
        payload_override: dict[str, Any] | None = None,
    ) -> None:
        payload = payload_override or request_payload(
            model,
            messages,
            stream=stream,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )
        specs.append(
            {
                "case_id": case_id,
                "surfaces": surfaces,
                "expected_accept": REQUIRED_API_CASES[case_id],
                "payload": payload,
                "replacements": replacements or {},
                "require_tool_call": require_tool_call,
            }
        )

    add(
        "residency_text_before",
        ["api", "generation", "residency"],
        [{"role": "user", "content": "Reply with one token."}],
    )

    r: dict[str, Any] = {}
    add(
        "image_local_png",
        ["image", "transport", "generation"],
        [{"role": "user", "content": [text("Name one visual property."), fixtures.part("image", "image-rgb-256.png", "local", r)]}],
        r,
    )
    r = {}
    add(
        "image_data_jpeg",
        ["image", "transport"],
        [{"role": "user", "content": [fixtures.part("image", "image-landscape-512x192.jpg", "data", r), text("Describe briefly.")]}],
        r,
    )
    r = {}
    add(
        "image_http_webp",
        ["image", "transport"],
        [{"role": "user", "content": [text("Describe briefly."), fixtures.part("image", "image-portrait-192x512.webp", "http", r)]}],
        r,
    )
    r = {}
    add(
        "image_transparent_png",
        ["image"],
        [{"role": "user", "content": [fixtures.part("image", "image-transparent-160x320.png", "local", r), text("Describe briefly.")]}],
        r,
    )
    r = {}
    add(
        "multi_image_interleaved",
        ["image", "conversation"],
        [{"role": "user", "content": [text("First:"), fixtures.part("image", "image-rgb-256.png", "local", r), text("Second:"), fixtures.part("image", "image-landscape-512x192.jpg", "local", r), text("Compare.")]}],
        r,
    )

    r = {}
    add(
        "video_local_mp4",
        ["video", "transport", "generation"],
        [{"role": "user", "content": [text("Describe the motion."), fixtures.part("video", "video-8f-4fps-128.mp4", "local", r)]}],
        r,
    )
    r = {}
    add(
        "video_data_mp4",
        ["video", "transport"],
        [{"role": "user", "content": [fixtures.part("video", "video-8f-4fps-128.mp4", "data", r), text("Describe briefly.")]}],
        r,
    )
    r = {}
    add(
        "video_http_avi",
        ["video", "transport"],
        [{"role": "user", "content": [text("Describe briefly."), fixtures.part("video", "video-12f-6fps-192x128.avi", "http", r)]}],
        r,
    )
    r = {}
    add(
        "multi_video",
        ["video"],
        [{"role": "user", "content": [fixtures.part("video", "video-8f-4fps-128.mp4", "local", r), text("and"), fixtures.part("video", "video-12f-6fps-192x128.avi", "local", r), text("Compare.")]}],
        r,
    )

    r = {}
    add(
        "mixed_image_then_video",
        ["mixed", "conversation"],
        [{"role": "user", "content": [fixtures.part("image", "image-rgb-256.png", "local", r), text("then"), fixtures.part("video", "video-8f-4fps-128.mp4", "local", r), text("Compare.")]}],
        r,
    )
    r = {}
    add(
        "mixed_video_then_image",
        ["mixed", "conversation"],
        [{"role": "user", "content": [fixtures.part("video", "video-8f-4fps-128.mp4", "local", r), text("then"), fixtures.part("image", "image-landscape-512x192.jpg", "local", r), text("Compare.")]}],
        r,
    )

    r = {}
    prior_image = fixtures.part("image", "image-rgb-256.png", "local", r)
    add(
        "conversation_prior_image",
        ["conversation", "image"],
        [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": [prior_image, text("Remember this image.")]},
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "user", "content": "What did you see?"},
        ],
        r,
    )
    r = {}
    add(
        "conversation_media_replace",
        ["conversation", "image"],
        [
            {"role": "user", "content": [fixtures.part("image", "image-rgb-256.png", "local", r), text("First image.")]},
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "user", "content": [fixtures.part("image", "image-transparent-160x320.png", "local", r), text("Now use only this image.")]},
        ],
        r,
    )

    tools = tool_contract()
    r = {}
    add(
        "tool_history_with_image",
        ["tool", "conversation", "image"],
        [
            {"role": "user", "content": "Record prior state."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_fixture_prior", "type": "function", "function": {"name": "inspect_visual", "arguments": "{\"label\":\"prior\"}"}}]},
            {"role": "tool", "tool_call_id": "call_fixture_prior", "content": "recorded"},
            {"role": "user", "content": [fixtures.part("image", "image-rgb-256.png", "local", r), text("Confirm briefly.")]},
        ],
        r,
        tools=tools,
        tool_choice="none",
    )
    r = {}
    forced_choice = {"type": "function", "function": {"name": "inspect_visual"}}
    add(
        "tool_forced_image",
        ["tool", "image"],
        [{"role": "user", "content": [text("Call inspect_visual for this image."), fixtures.part("image", "image-rgb-256.png", "local", r)]}],
        r,
        max_tokens=64,
        tools=tools,
        tool_choice=forced_choice,
        require_tool_call=True,
    )
    r = {}
    add(
        "tool_auto_image",
        ["tool", "image"],
        [{"role": "user", "content": [text("You must call inspect_visual with a short label."), fixtures.part("image", "image-rgb-256.png", "local", r)]}],
        r,
        max_tokens=192,
        tools=tools,
        tool_choice="auto",
        require_tool_call=True,
    )

    r = {}
    add(
        "stream_image",
        ["stream", "api", "image"],
        [{"role": "user", "content": [fixtures.part("image", "image-rgb-256.png", "local", r), text("Describe briefly.")]}],
        r,
        stream=True,
        max_tokens=4,
    )
    r = {}
    add(
        "stream_video",
        ["stream", "api", "video"],
        [{"role": "user", "content": [fixtures.part("video", "video-8f-4fps-128.mp4", "local", r), text("Describe briefly.")]}],
        r,
        stream=True,
        max_tokens=4,
    )
    add(
        "residency_text_after",
        ["api", "generation", "residency"],
        [{"role": "user", "content": "Reply with one token after VL requests."}],
    )

    r = {}
    add(
        "error_corrupt_image",
        ["error", "image"],
        [{"role": "user", "content": [fixtures.part("image", "corrupt-image.png", "local", r), text("Describe.")]}],
        r,
    )
    r = {}
    add(
        "error_corrupt_video",
        ["error", "video"],
        [{"role": "user", "content": [fixtures.part("video", "corrupt-video.mp4", "local", r), text("Describe.")]}],
        r,
    )
    r = {}
    add(
        "error_aspect_ratio",
        ["error", "image"],
        [{"role": "user", "content": [fixtures.part("image", "image-invalid-aspect-201x1.png", "local", r), text("Describe.")]}],
        r,
    )
    add(
        "error_outside_local_root",
        ["error", "transport"],
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///etc/hosts"}}, text("Describe.")]}],
    )
    add(
        "error_disallowed_domain",
        ["error", "transport"],
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://vl-disallowed.invalid/image.png"}}, text("Describe.")]}],
    )
    r = {}
    image_parts = [fixtures.part("image", "image-rgb-256.png", "local", r) for _ in range(REFERENCE_MEDIA_LIMITS["image"] + 1)]
    add(
        "error_image_count_over_limit",
        ["error", "image"],
        [{"role": "user", "content": [*image_parts, text("Too many images.")]}],
        r,
    )
    r = {}
    video_parts = [fixtures.part("video", "video-8f-4fps-128.mp4", "local", r) for _ in range(REFERENCE_MEDIA_LIMITS["video"] + 1)]
    add(
        "error_video_count_over_limit",
        ["error", "video"],
        [{"role": "user", "content": [*video_parts, text("Too many videos.")]}],
        r,
    )
    add(
        "error_malformed_data_uri",
        ["error", "transport"],
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%invalid%%%"}}, text("Describe.")]}],
    )
    r = {}
    mismatched = fixtures.part("video", "video-8f-4fps-128.mp4", "local", r)
    mismatched = {"type": "image_url", "image_url": mismatched["video_url"]}
    add(
        "error_type_mismatch",
        ["error", "transport", "image", "video"],
        [{"role": "user", "content": [mismatched, text("Describe.")]}],
        r,
    )
    add(
        "error_audio_out_of_scope",
        ["error", "api"],
        [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}, text("Transcribe.")]}],
    )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen36-vl-reference")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--fixture-http-base", required=True)
    parser.add_argument("--processor-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--captured-at")
    args = parser.parse_args()

    fixtures = Fixtures(args.fixtures, args.fixture_http_base)
    specs = build_cases(fixtures, args.model)
    cases = []
    for spec in specs:
        print(f"CASE {spec['case_id']}", flush=True)
        cases.append(
            execute_case(
                args.endpoint,
                timeout=args.timeout,
                response_redactions={
                    str(args.fixtures.resolve()): "${AIMA_ALLOWED_MEDIA_ROOT}"
                },
                **spec,
            )
        )
        print(
            f"RESULT {cases[-1]['case_id']} status={cases[-1]['status_code']} "
            f"passed={cases[-1]['passed']}",
            flush=True,
        )

    captured_at = args.captured_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    payload: dict[str, Any] = {
        "schema": CAPABILITY_SCHEMA,
        "complete": {item["case_id"] for item in cases} == set(REQUIRED_API_CASES),
        "qualified": all(item["passed"] for item in cases),
        "captured_at": captured_at,
        "reference": {
            "endpoint": args.endpoint,
            "model": args.model,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "max_model_len": 262_144,
            "max_num_batched_tokens": REFERENCE_MAX_BATCHED_TOKENS,
            "max_num_seqs": 1,
            "media_limits": REFERENCE_MEDIA_LIMITS,
            "attention_backend": REFERENCE_ATTENTION_BACKEND,
            "mm_encoder_attention_backend": REFERENCE_ATTENTION_BACKEND,
            "tool_call_parser": "qwen3_xml",
        },
        "bindings": {
            "processor_probe": file_component(
                args.processor_probe.resolve(),
                "benchmarks/results/vl-processor-capability-v0.1.0.json",
            ),
            "fixture_manifest": file_component(
                (args.fixtures / "fixtures-manifest.json").resolve(),
                "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
            ),
        },
        "cases": cases,
    }
    manifest = seal_manifest(payload)
    atomic_json(args.output, manifest)
    errors = validate_capability_manifest(manifest)
    if errors:
        print("CAPABILITY_PROBE_FAILED", flush=True)
        for error in errors:
            print(f"- {error}", flush=True)
        return 1
    print("CAPABILITY_PROBE_QUALIFIED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
