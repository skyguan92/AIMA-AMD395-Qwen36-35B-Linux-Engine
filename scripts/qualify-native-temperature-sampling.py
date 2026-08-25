#!/usr/bin/env python3
"""Qualify seeded full-vocabulary temperature/top-p sampling over native HTTP."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import selectors
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
    verify_manifest_integrity,
)


MODEL_ID = "aima-amd395-qwen36-35b"
SCHEMA = "aima-amd395-qwen36/native-temperature-sampling/v1"
G5_SCHEMA = "aima-amd395-qwen36/native-vl-g5-release-qualification/v1"
RELEASE = "1.5.1-native-vl.4"
VOCABULARY_SIZE = 248_320
BF16_LOGIT_BYTES = VOCABULARY_SIZE * 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_object_from_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("command returned a non-object JSON value")
    return value


def publicize(
    value: Any, replacements: tuple[tuple[str, str], ...]
) -> Any:
    if isinstance(value, str):
        result = value
        for private, logical in replacements:
            result = result.replace(private, logical)
        return result
    if isinstance(value, list):
        return [publicize(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: publicize(item, replacements) for key, item in value.items()
        }
    return value


def require_g5_result(path: Path) -> dict[str, Any]:
    payload = load_object(path)
    if (
        payload.get("schema") != G5_SCHEMA
        or payload.get("release") != RELEASE
        or payload.get("complete") is not True
        or payload.get("qualified") is not True
        or payload.get("decision", {}).get("g5_native_release_product")
        is not True
        or verify_manifest_integrity(payload)
    ):
        raise RuntimeError("sealed G5 result is incomplete or invalid")
    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{sha256(path)}  {path.name}\n"
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8") != expected_sidecar
    ):
        raise RuntimeError("sealed G5 result checksum sidecar differs")
    return payload


def require_gpu_idle() -> None:
    occupied = subprocess.run(
        ["fuser", "/dev/kfd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if occupied.returncode != 0:
        return
    subprocess.run(["fuser", "-v", "/dev/kfd"], check=False)
    raise SystemExit(75)


def read_lifecycle(
    process: subprocess.Popen[str], timeout_seconds: float
) -> dict[str, Any]:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    events = selector.select(timeout_seconds)
    selector.close()
    if not events:
        raise RuntimeError("native sampling lifecycle event timed out")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"native sampling server exited: {process.poll()}")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError("native sampling lifecycle event is not an object")
    return value


def request_json(
    port: int,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        if payload is not None
        else None
    )
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    connection.request(
        method,
        path,
        body=body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    response = connection.getresponse()
    raw = response.read()
    status = response.status
    connection.close()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {path} returned non-object JSON")
    return status, value


def request_stream(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
    )
    response = connection.getresponse()
    content: list[str] = []
    usage: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    finish_reason: str | None = None
    done = False
    while True:
        line = response.readline()
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            done = True
            break
        event = json.loads(data)
        if event.get("usage") is not None:
            usage = event["usage"]
        if event.get("aima_amd395") is not None:
            metrics = event["aima_amd395"]
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    status = response.status
    content_type = response.getheader("Content-Type")
    connection.close()
    return {
        "status": status,
        "content_type": content_type,
        "content": "".join(content),
        "usage": usage,
        "metrics": metrics,
        "finish_reason": finish_reason,
        "done": done,
    }


def content_sha256(response: dict[str, Any]) -> str:
    content = response["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise RuntimeError("sampling response content is not text")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sampling_metrics_pass(
    response: dict[str, Any], *, temperature: float, top_p: float, seed: int
) -> bool:
    usage = response.get("usage") or {}
    metrics = response.get("aima_amd395") or {}
    sampling = metrics.get("sampling") or {}
    selections = sampling.get("token_selections")
    return bool(
        sampling.get("mode") == "temperature-top-p"
        and sampling.get("logits_source") == "exact-bf16-full-vocabulary"
        and sampling.get("temperature") == temperature
        and sampling.get("top_p") == top_p
        and sampling.get("seed_provided") is True
        and sampling.get("seed") == seed
        and isinstance(selections, int)
        and selections == usage.get("completion_tokens")
        and sampling.get("logits_device_to_host_bytes")
        == selections * BF16_LOGIT_BYTES
        and float(sampling.get("wall_ms", 0)) > 0.0
    )


def compact_response(response: dict[str, Any]) -> dict[str, Any]:
    metrics = response["aima_amd395"]
    return {
        "content_sha256": content_sha256(response),
        "output_token_ids_sha256": metrics["output_token_ids_sha256"],
        "finish_reason": response["choices"][0]["finish_reason"],
        "usage": response["usage"],
        "sampling": metrics["sampling"],
        "vl": {
            key: metrics["vl"][key]
            for key in (
                "enabled",
                "media_count",
                "image_count",
                "video_count",
                "vision_patches",
                "visual_tokens",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--g5-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fmha-provider", type=Path)
    parser.add_argument("--vision-attention-image", type=Path)
    parser.add_argument("--port", type=int, default=18129)
    args = parser.parse_args()

    engine = args.engine.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    media_root = args.media_root.expanduser().resolve()
    g5_result_path = args.g5_result.expanduser().resolve()
    output = args.output.expanduser().resolve()
    provider = (
        args.fmha_provider.expanduser().resolve()
        if args.fmha_provider is not None
        else engine.parent / "libaima-fmha-ck.so"
    )
    vision_image = (
        args.vision_attention_image.expanduser().resolve()
        if args.vision_attention_image is not None
        else engine.parent / "aima-vision-attention.hsaco"
    )
    image = media_root / "vl-capability-v0.1.0/image-rgb-256.png"
    for path in (engine, provider, vision_image, image, g5_result_path):
        if not path.is_file():
            raise SystemExit(f"sampling qualification input is missing: {path}")
    g5_result = require_g5_result(g5_result_path)
    build_info = load_object_from_command([str(engine), "--build-info"])
    engine_digest = sha256(engine)
    g5_source = g5_result.get("source", {})
    if (
        build_info.get("source_commit")
        != g5_source.get("native_source_commit")
        or engine_digest
        != g5_result.get("candidate", {}).get("native_engine_sha256")
    ):
        raise SystemExit("temperature candidate differs from sealed G5 result")
    raw_dir = output.with_name(output.stem + "-raw")
    if output.exists() or raw_dir.exists():
        raise SystemExit(
            "refusing to reuse temperature-sampling qualification output"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir()
    load_report = raw_dir / "native-weight-load.json"
    stderr_path = raw_dir / "stderr.txt"
    command = [
        str(engine),
        "serve",
        "--model-dir",
        str(model_dir),
        "--context-tokens",
        "1024",
        "--cache-capacity",
        "2048",
        "--fmha-provider",
        str(provider),
        "--vision-attention-image",
        str(vision_image),
        "--allowed-local-media-path",
        str(media_root),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--request-timeout-ms",
        "120000",
        "--report",
        str(load_report),
    ]
    require_gpu_idle()
    with stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
        try:
            ready = read_lifecycle(process, 180)
            base = {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Continue with one imaginative sentence: The sky"
                        ),
                    }
                ],
                "max_tokens": 16,
            }
            greedy_status, greedy = request_json(
                args.port,
                "POST",
                "/v1/chat/completions",
                {**base, "temperature": 0, "top_p": 1},
            )
            stochastic = {
                **base,
                "temperature": 0.8,
                "top_p": 0.9,
                "seed": 424242,
            }
            first_status, first = request_json(
                args.port, "POST", "/v1/chat/completions", stochastic
            )
            replay_status, replay = request_json(
                args.port, "POST", "/v1/chat/completions", stochastic
            )
            different_status, different = request_json(
                args.port,
                "POST",
                "/v1/chat/completions",
                {**stochastic, "seed": 424243},
            )
            stream = request_stream(
                args.port,
                {
                    **stochastic,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            vl_payload = {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image creatively in one "
                                    "sentence."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "file://" + str(image)},
                            },
                        ],
                    }
                ],
                "temperature": 0.7,
                "top_p": 0.85,
                "seed": 777,
                "max_tokens": 8,
            }
            vl_first_status, vl_first = request_json(
                args.port, "POST", "/v1/chat/completions", vl_payload
            )
            vl_replay_status, vl_replay = request_json(
                args.port, "POST", "/v1/chat/completions", vl_payload
            )

            invalid_cases: list[dict[str, Any]] = []
            for case_id, patch in (
                ("negative_temperature", {"temperature": -0.1}),
                ("high_temperature", {"temperature": 2.1}),
                ("zero_top_p", {"temperature": 1, "top_p": 0}),
                ("greedy_top_p", {"temperature": 0, "top_p": 0.9}),
                ("negative_seed", {"temperature": 1, "seed": -1}),
            ):
                status, response = request_json(
                    args.port,
                    "POST",
                    "/v1/chat/completions",
                    {**base, "max_tokens": 1, **patch},
                )
                invalid_cases.append(
                    {
                        "case_id": case_id,
                        "status": status,
                        "error": response.get("error"),
                    }
                )
            shutdown_status, shutdown = request_json(
                args.port, "POST", "/shutdown"
            )
            stopped = read_lifecycle(process, 30)
            returncode = process.wait(timeout=30)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

    replacements = (
        (str(provider), "${AIMA_FMHA_PROVIDER}"),
        (str(vision_image), "${AIMA_VISION_ATTENTION_IMAGE}"),
        (str(engine.parent), "${AIMA_NATIVE_BUILD_DIR}"),
        (str(model_dir), "${AIMA_MODEL_DIR}"),
        (str(media_root), "${AIMA_VL_MEDIA_ROOT}"),
        (str(raw_dir), "${AIMA_TEMPERATURE_RAW_DIR}"),
        (str(ROOT), "${AIMA_REPO_ROOT}"),
    )
    ready = publicize(ready, replacements)
    stopped = publicize(stopped, replacements)
    load_payload = publicize(load_object(load_report), replacements)
    atomic_json(load_report, load_payload)
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    for private, logical in replacements:
        stderr_text = stderr_text.replace(private, logical)
    stderr_path.write_text(stderr_text, encoding="utf-8")

    greedy_sampling = greedy.get("aima_amd395", {}).get("sampling", {})
    greedy_pass = bool(
        greedy_status == 200
        and greedy_sampling.get("mode") == "argmax"
        and greedy_sampling.get("logits_source") == "certified-top1"
        and greedy_sampling.get("temperature") == 0
        and greedy_sampling.get("top_p") == 1
        and greedy_sampling.get("seed") is None
        and greedy_sampling.get("token_selections") == 0
        and greedy_sampling.get("logits_device_to_host_bytes") == 0
    )
    same_seed_pass = bool(
        first_status == replay_status == 200
        and sampling_metrics_pass(
            first, temperature=0.8, top_p=0.9, seed=424242
        )
        and sampling_metrics_pass(
            replay, temperature=0.8, top_p=0.9, seed=424242
        )
        and content_sha256(first) == content_sha256(replay)
        and first["aima_amd395"]["output_token_ids_sha256"]
        == replay["aima_amd395"]["output_token_ids_sha256"]
    )
    different_seed_pass = bool(
        different_status == 200
        and sampling_metrics_pass(
            different, temperature=0.8, top_p=0.9, seed=424243
        )
        and different["aima_amd395"]["output_token_ids_sha256"]
        != first["aima_amd395"]["output_token_ids_sha256"]
    )
    stream_metrics = stream.get("metrics") or {}
    stream_sampling = stream_metrics.get("sampling") or {}
    stream_pass = bool(
        stream.get("status") == 200
        and str(stream.get("content_type", "")).startswith(
            "text/event-stream"
        )
        and stream.get("done") is True
        and stream.get("content")
        == first["choices"][0]["message"]["content"]
        and stream.get("usage") == first.get("usage")
        and stream_metrics.get("output_token_ids_sha256")
        == first["aima_amd395"]["output_token_ids_sha256"]
        and stream_sampling.get("mode") == "temperature-top-p"
        and stream_sampling.get("seed") == 424242
        and stream_sampling.get("logits_source")
        == "exact-bf16-full-vocabulary"
    )
    vl_pass = bool(
        vl_first_status == vl_replay_status == 200
        and sampling_metrics_pass(
            vl_first, temperature=0.7, top_p=0.85, seed=777
        )
        and sampling_metrics_pass(
            vl_replay, temperature=0.7, top_p=0.85, seed=777
        )
        and vl_first["aima_amd395"]["vl"]["enabled"] is True
        and vl_first["aima_amd395"]["vl"]["image_count"] == 1
        and vl_replay["aima_amd395"]["vl"]["enabled"] is True
        and content_sha256(vl_first) == content_sha256(vl_replay)
        and vl_first["aima_amd395"]["output_token_ids_sha256"]
        == vl_replay["aima_amd395"]["output_token_ids_sha256"]
    )
    validation_pass = all(case["status"] == 400 for case in invalid_cases)
    lifecycle_pass = bool(
        ready.get("event") == "ready"
        and ready.get("runtime_python") is False
        and ready.get("runtime_torch") is False
        and ready.get("runtime_vllm") is False
        and shutdown_status == 200
        and shutdown.get("status") == "shutting_down"
        and stopped.get("event") == "stopped"
        and stopped.get("model_loads") == 1
        and stopped.get("served") == 7
        and returncode == 0
        and stderr_path.stat().st_size == 0
    )
    checks = {
        "greedy_fast_path_preserved": greedy_pass,
        "same_seed_text_exact_replay": same_seed_pass,
        "different_seed_text_diverges": different_seed_pass,
        "seeded_stream_nonstream_exact": stream_pass,
        "same_seed_vl_exact_replay": vl_pass,
        "invalid_sampling_boundaries_fail_closed": validation_pass,
        "single_load_clean_lifecycle": lifecycle_pass,
    }
    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "recorded_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "complete": True,
            "qualified": all(checks.values()),
            "candidate": {
                "source_commit": build_info["source_commit"],
                "binary_sha256": engine_digest,
            },
            "post_g5_boundary": {
                "g5_result": file_component(
                    g5_result_path,
                    "benchmarks/results/"
                    "native-vl-g5-release-v1.5.1-native-vl.4.json",
                ),
                "g5_recorded_at": g5_result["recorded_at"],
                "ordering_proof": (
                    "the sealed complete G5 result was validated and identity-"
                    "matched before the temperature engine launch"
                ),
            },
            "sampling_contract": {
                "temperature": "0 retains greedy; positive range is (0,2]",
                "top_p": "(0,1] on the positive-temperature path",
                "seed": "optional non-negative uint64; effective seed reported",
                "prng": "splitmix64-upper53",
                "logits": (
                    "exact full-vocabulary BF16 wvSplitK projection from raw "
                    "LM-head weights"
                ),
                "vocabulary_size": VOCABULARY_SIZE,
                "bf16_logits_per_selection_bytes": BF16_LOGIT_BYTES,
            },
            "checks": checks,
            "cases": {
                "greedy": compact_response(greedy),
                "same_seed_first": compact_response(first),
                "same_seed_replay": compact_response(replay),
                "different_seed": compact_response(different),
                "stream": {
                    "content_sha256": hashlib.sha256(
                        stream["content"].encode("utf-8")
                    ).hexdigest(),
                    "output_token_ids_sha256": stream_metrics[
                        "output_token_ids_sha256"
                    ],
                    "finish_reason": stream["finish_reason"],
                    "usage": stream["usage"],
                    "sampling": stream_sampling,
                },
                "vl_first": compact_response(vl_first),
                "vl_replay": compact_response(vl_replay),
                "invalid": invalid_cases,
            },
            "lifecycle": {"ready": ready, "stopped": stopped},
            "artifacts": {
                "load_report": file_component(
                    load_report,
                    f"{raw_dir.name}/native-weight-load.json",
                ),
                "stderr": file_component(
                    stderr_path, f"{raw_dir.name}/stderr.txt"
                ),
            },
            "decision": {
                "nonzero_temperature_supported": all(checks.values()),
                "greedy_contract_unchanged": greedy_pass,
                "text_and_vl_seeded_sampling_qualified": (
                    same_seed_pass
                    and different_seed_pass
                    and stream_pass
                    and vl_pass
                ),
            },
        }
    )
    digest = atomic_json(output, payload)
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": payload["qualified"],
                "output": str(output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
