#!/usr/bin/env python3
"""Resident AMD395 A/B qualification for safe chat-prefix checkpoints."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import array
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import sys
import threading
import time
import zlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aima_engine.vl_reference import canonical_json_sha256, seal_manifest

spec = importlib.util.spec_from_file_location(
    "prefix_protocol", ROOT / "scripts/qualify-native-chat-protocol.py"
)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def write_json(path: Path, value: dict) -> None:
    value = protocol.sanitize_host_paths(value, [(ROOT, "${AIMA_ROOT}")])
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol.seal_file(path)


def make_media_b(path: Path) -> None:
    """A different 256x256 RGB object with the same visual token geometry."""
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload)))

    pixels = (b"\0" + bytes((0, 255, 0)) * 256) * 256
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def text_request(user: str, system: str = "You are a helpful assistant.",
                 output_tokens: int = 64) -> dict:
    return {"model": protocol.MODEL_ID, "temperature": 0, "top_p": 1,
            "max_tokens": output_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}


def prompt_tokens(engine: Path, model: Path, request: dict) -> list[int]:
    if "prompt_token_ids" in request:
        return request["prompt_token_ids"]
    value = subprocess.run(
        [str(engine), "chat-template-probe", "--model-dir", str(model),
         "--disable-thinking", "--request-json", json.dumps(request, ensure_ascii=False)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(value.stdout)["token_ids"]


def build_cases(engine: Path, model: Path, output: Path, performance: bool,
                cache_capacity: int = 32768) -> list[dict]:
    short = text_request("你好")
    extended = text_request("你好，你是谁")
    tokens = prompt_tokens(engine, model, short)
    extended_tokens = prompt_tokens(engine, model, extended)
    common = next((index for index, pair in enumerate(zip(tokens, extended_tokens))
                   if pair[0] != pair[1]), min(len(tokens), len(extended_tokens)))
    boundaries = [index for index, token in enumerate(tokens) if index and token == 248046]
    matched = max(boundary for boundary in boundaries if boundary <= common)
    cases = []

    def add(name: str, payload: dict, lookup: str | None = None,
            matched_tokens: int | None = None, stream: bool = False) -> None:
        if "prompt_token_ids" in payload:
            payload.setdefault("messages", [{"role": "user", "content": "qualification"}])
        cases.append({"id": name, "payload": payload, "lookup": lookup,
                      "matched_tokens": matched_tokens, "stream": stream})

    add("short_cold", short, "miss", 0)
    add("short_exact", short, "exact", len(tokens))
    add("divergent_chat", extended, "prefix", matched)
    add("divergent_exact_sse", extended, "exact", len(extended_tokens), True)
    add("shared_system", text_request("What is two plus two?"), "prefix", boundaries[0])
    history = text_request("继续说一句中文")
    history["messages"][1:1] = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    add("multi_turn", history, "prefix")
    add("append_seed", short)
    append_tokens = json.loads(subprocess.run(
        [str(engine), "tokenizer-probe", "--model-dir", str(model), "--text", "Hello"],
        check=True, capture_output=True, text=True).stdout)["token_ids"]
    add("whole_prompt_append", {"model": protocol.MODEL_ID, "temperature": 0,
        "max_tokens": 64, "prompt_token_ids": tokens + append_tokens}, "prefix", len(tokens))
    add("checkpoint_as_complete_prompt", {"model": protocol.MODEL_ID, "temperature": 0,
        "max_tokens": 64, "prompt_token_ids": tokens[:matched]}, "exact", matched)
    add("full_owner_after_short_checkpoint", short, "exact", len(tokens))
    # Five unrelated owners exceed the four-entry promoted q8192 LRU.
    for index in range(5):
        add(f"eviction_fill_{index}", {"model": protocol.MODEL_ID, "temperature": 0,
            "max_tokens": 8, "prompt_token_ids": [1000 + index, 2000 + index, 3000 + index]},
            "miss", 0)
    add("evicted_short", short, "miss", 0)
    media_b = output / "media-b.png"
    make_media_b(media_b)
    image = ROOT / "benchmarks/fixtures/vl-capability-v0.1.0/image-rgb-256.png"

    def media_request(path: Path) -> dict:
        return {"model": protocol.MODEL_ID, "temperature": 0, "max_tokens": 64,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": path.as_uri()}},
                    {"type": "text", "text": "Describe the colors."}]}]}

    add("media_a", media_request(image), "miss", 0)
    add("media_b_same_geometry", media_request(media_b), "miss", 0)
    media_hit = cache_capacity <= 131072
    add("media_a_restored", media_request(image), "exact" if media_hit else "miss")
    text_hit = cache_capacity <= 32768
    add("text_after_media", short, "exact" if text_hit else "miss", len(tokens) if text_hit else 0)
    if performance:
        system = "Use this reference glossary when answering. " + "alpha beta gamma delta. " * 512
        add("performance_seed", text_request(
            "Count from 1 to 200, separated by commas. Output only the numbers.", system, 128))
        for index in range(5):
            add(f"performance_partial_{index}", text_request(
                f"Count from {index + 2} to 200, separated by commas. Output only the numbers.",
                system, 128), "prefix")
    return cases


def monitor_memory(pid: int, stop: threading.Event, maxima: dict) -> None:
    gtt_files = list(Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used"))
    while not stop.is_set():
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    maxima["rss_bytes"] = max(maxima["rss_bytes"], int(line.split()[1]) * 1024)
            used = sum(int(path.read_text()) for path in gtt_files)
            maxima["gtt_bytes"] = max(maxima["gtt_bytes"], used)
        except (OSError, ValueError):
            pass
        stop.wait(0.05)


def run_cases(cli, engine: Path, mode: str, cases: list[dict]) -> dict:
    protocol.require_gpu_idle()
    directory = cli.output / mode
    directory.mkdir()
    command = [str(engine), "serve", "--model-dir", str(cli.model_dir),
               "--context-tokens", str(cli.context_tokens), "--cache-capacity", str(cli.cache_capacity),
               "--host", "127.0.0.1", "--port", str(cli.port),
               "--allowed-local-media-path", str(ROOT / "benchmarks/fixtures"),
               "--allowed-local-media-path", str(cli.output),
               "--report", str(directory / "weight-load.json")]
    if mode != "cached":
        command.append("--disable-prefix-cache")
    maxima = {"rss_bytes": 0, "gtt_bytes": 0}
    stop = threading.Event()
    observations = []
    with (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, text=True)
        monitor = threading.Thread(target=monitor_memory, args=(process.pid, stop, maxima), daemon=True)
        monitor.start()
        try:
            ready = protocol.read_lifecycle(process, 900)
            write_json(directory / "ready.json", ready)
            protocol.require(ready.get("event") == "ready", f"server did not reach ready: {ready}")
            print(json.dumps({"mode": mode, "event": "ready", "pid": process.pid}), flush=True)
            for case in cases:
                started = time.perf_counter()
                if case["stream"]:
                    response = protocol.request_stream(cli.port, case["payload"])
                    metrics = response["metrics"]
                else:
                    status, response = protocol.request_json(
                        cli.port, "POST", "/v1/chat/completions", case["payload"], timeout=600)
                    protocol.require(status == 200, f"{case['id']}: HTTP {status}: {response}")
                    metrics = response["aima_amd395"]
                observation = {"case_id": case["id"], "metrics": metrics,
                               "http_wall_ms": (time.perf_counter() - started) * 1000,
                               "response": response}
                observations.append(observation)
                write_json(directory / f"{case['id']}.json", observation)
                print(json.dumps({"mode": mode, "case": case["id"], "prefix": metrics["prefix_cache"],
                                  "output_sha256": metrics["output_token_ids_sha256"]}), flush=True)
        finally:
            if process.poll() is None:
                try:
                    protocol.request_json(cli.port, "POST", "/shutdown")
                except Exception:
                    process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            stop.set()
            monitor.join(timeout=2)
            if process.stdout is not None:
                (directory / "stopped.log").write_text(process.stdout.read(), encoding="utf-8")
    return {"ready": ready, "exit_code": process.returncode,
            "peak_memory": maxima, "observations": observations}


def compare_runs(cases: list[dict], cold: dict, cached: dict) -> list[dict]:
    identifiers = [case["id"] for case in cases]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("qualification cases must be nonempty and unique")
    for run in (cold, cached):
        observed = [item["case_id"] for item in run["observations"]]
        if len(observed) != len(identifiers) or set(observed) != set(identifiers):
            raise ValueError("qualification observations must cover each case exactly once")
    references = {item["case_id"]: item["metrics"] for item in cold["observations"]}
    observations = {item["case_id"]: item["metrics"] for item in cached["observations"]}
    results = []
    for case in cases:
        baseline = references[case["id"]]
        candidate = observations[case["id"]]
        prefix = candidate["prefix_cache"]
        checks = {
            "token_identity": candidate["output_token_ids_sha256"] == baseline["output_token_ids_sha256"],
            "completion_count": candidate["completion_tokens"] == baseline["completion_tokens"],
            "prompt_count": candidate["prompt_tokens"] == baseline["prompt_tokens"],
            "cold_disabled": baseline["prefix_cache"]["lookup"] == "disabled"
                             and baseline["prefix_cache"]["matched_tokens"] == 0,
            "lookup": case["lookup"] is None or prefix["lookup"] == case["lookup"],
            "matched_boundary": case["matched_tokens"] is None
                                or prefix["matched_tokens"] == case["matched_tokens"],
            "suffix_only": candidate["aot_prefill_tokens"] == prefix["suffix_tokens"]
                           == candidate["prompt_tokens"] - prefix["matched_tokens"],
            "no_serial_prefill": candidate["cold_prompt_decode_tokens"] == 0
                                 and prefix["suffix_decode_tokens"] == 0,
            "restore_measured": prefix["lookup"] == "miss"
                                or (prefix["restore_bytes"] > 0 and prefix["restore_wall_ms"] > 0),
        }
        results.append({"case_id": case["id"], "checks": checks, "pass": all(checks.values()),
                        "cold_ttft_ms": baseline["ttft_ms"], "cached_ttft_ms": candidate["ttft_ms"],
                        "ttft_speedup": baseline["ttft_ms"] / candidate["ttft_ms"],
                        "cold_decode_tps": baseline["decode_tokens_per_second"],
                        "cached_decode_tps": candidate["decode_tokens_per_second"],
                        "prefix_cache": prefix,
                        "suffix_aot_tokens": candidate["aot_prefill_tokens"],
                        "suffix_bucket_tokens": candidate["aot_prefill_bucket_tokens"]})
    return results


def compare_logits(reference: list[float], actual: list[float]) -> dict:
    if len(reference) != 248320 or len(actual) != len(reference):
        raise ValueError("paired logits must cover all 248320 vocabulary entries")
    if not all(math.isfinite(value) for row in (reference, actual) for value in row):
        raise ValueError("paired logits must be finite")
    reference_top1 = max(range(len(reference)), key=reference.__getitem__)
    actual_top1 = max(range(len(actual)), key=actual.__getitem__)
    reference_max, actual_max = max(reference), max(actual)
    reference_exp = [math.exp(value - reference_max) for value in reference]
    actual_exp = [math.exp(value - actual_max) for value in actual]
    reference_sum, actual_sum = math.fsum(reference_exp), math.fsum(actual_exp)
    reference_log_z = reference_max + math.log(reference_sum)
    actual_log_z = actual_max + math.log(actual_sum)
    kld = math.fsum(probability / reference_sum *
                    (left - reference_log_z - right + actual_log_z)
                    for probability, left, right in zip(reference_exp, reference, actual))
    return {"elements": len(reference), "reference_top1": reference_top1,
            "actual_top1": actual_top1, "top1_match": reference_top1 == actual_top1,
            "kl_divergence": max(0.0, kld),
            "maximum_absolute_error": max(abs(left - right) for left, right in zip(reference, actual)),
            "exact_elements": sum(left == right for left, right in zip(reference, actual)),
            "pass": reference_top1 == actual_top1 and kld < 0.005}


def qualify_logits(cli, cases: list[dict]) -> dict:
    """Inspect one-token prefill distributions, without changing HTTP serving."""
    selected = [case for case in cases if not case["id"].startswith(
        ("eviction_", "evicted_", "media_", "text_after_media"))]
    sequences = [prompt_tokens(cli.engine, cli.model_dir, case["payload"]) for case in selected]
    names = [case["id"] for case in selected]
    # Exercise both sides of FLA chunks and composed AOT segment boundaries.
    short = sequences[0]
    system_end = short.index(248046)
    for boundary in (31, 32, 33, 1023, 1024, 1025, 8192, 8193):
        seed = short[:3] + [1] * (boundary - 3) + short[system_end:]
        divergent = seed[:boundary + 5] + [1] + seed[boundary + 5:]
        sequences.extend((seed, divergent))
        names.extend((f"boundary_{boundary}_seed", f"boundary_{boundary}_partial"))
    # A free-form prompt has no full-generation identity promise across BF16
    # partitions; its prefill distribution still has the same strict KLD gate.
    system = "Use this reference glossary when answering. " + "alpha beta gamma delta. " * 512
    for index in (1, 2):
        sequences.append(prompt_tokens(cli.engine, cli.model_dir, text_request(
            f"Write a short story about a robot visiting city {index}.", system, 1)))
        names.append(f"freeform_{index}")
    sequence_path = cli.output / "logits-prompts.json"
    sequence_path.write_text(json.dumps(sequences) + "\n", encoding="utf-8")
    protocol.seal_file(sequence_path)
    reports = {}
    for mode in ("cold", "cached"):
        protocol.require_gpu_idle()
        command = [str(cli.engine), "resident-session-probe", "--model-dir", str(cli.model_dir),
                   "--context-tokens", str(cli.context_tokens), "--cache-capacity", str(cli.cache_capacity),
                   "--input-token-ids-sequence-file", str(sequence_path), "--max-new-tokens", "1",
                   "--output-logits-dir", str(cli.output / f"logits-{mode}"),
                   "--report", str(cli.output / f"logits-{mode}-weights.json")]
        if mode == "cold":
            command.append("--disable-prefix-cache")
        with (cli.output / f"logits-{mode}.json").open("w") as stdout, \
                (cli.output / f"logits-{mode}.stderr").open("w") as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
        reports[mode] = json.loads((cli.output / f"logits-{mode}.json").read_text())
        write_json(cli.output / f"logits-{mode}.json", reports[mode])
        print(json.dumps({"event": "logits_complete", "mode": mode, "cases": len(names)}), flush=True)
    comparisons = []
    for index, name in enumerate(names):
        values = []
        for mode in ("cold", "cached"):
            row = array.array("f")
            with (cli.output / f"logits-{mode}/request-{index}.f32").open("rb") as stream:
                row.fromfile(stream, 248320)
                protocol.require(not stream.read(1), "extra logits bytes")
            if sys.byteorder != "little":
                row.byteswap()
            values.append(row)
        comparison = compare_logits(*values)
        comparison["case_id"] = name
        matched = reports["cached"]["requests"][index]["prefix_cache_matched_tokens"]
        comparison["matched_tokens"] = matched
        if name.startswith("boundary_") and name.endswith("_partial"):
            comparison["boundary_restored"] = matched == int(name.split("_")[1])
            comparison["pass"] &= comparison["boundary_restored"]
        comparisons.append(comparison)
    # Unlike HTTP this native probe deliberately continues after EOS. Decode
    # from a shorter checkpoint must invalidate the later part of live KV,
    # even though both requests refer to the same persistent snapshot owner.
    replay_sequence = cli.output / "checkpoint-replay-prompts.json"
    boundary = short.index(248046, system_end + 1)
    replay_sequence.write_text(json.dumps([short, short[:boundary], short, short]) + "\n")
    protocol.seal_file(replay_sequence)
    replays = {}
    for mode in ("cold", "cached"):
        protocol.require_gpu_idle()
        command = [str(cli.engine), "resident-session-probe", "--model-dir", str(cli.model_dir),
                   "--context-tokens", str(cli.context_tokens), "--cache-capacity", str(cli.cache_capacity),
                   "--input-token-ids-sequence-file", str(replay_sequence), "--max-new-tokens", "8",
                   "--report", str(cli.output / f"checkpoint-replay-{mode}-weights.json")]
        if mode == "cold":
            command.append("--disable-prefix-cache")
        with (cli.output / f"checkpoint-replay-{mode}.json").open("w") as stdout, \
                (cli.output / f"checkpoint-replay-{mode}.stderr").open("w") as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
        replays[mode] = json.loads((cli.output / f"checkpoint-replay-{mode}.json").read_text())
        write_json(cli.output / f"checkpoint-replay-{mode}.json", replays[mode])
    replay_checks = {
        "outputs_identical": all(left["output_token_ids"] == right["output_token_ids"]
                                 for left, right in zip(replays["cold"]["requests"], replays["cached"]["requests"])),
        "short_checkpoint_restored": replays["cached"]["requests"][1]["prefix_cache_matched_tokens"] == boundary,
        "later_owner_kv_restored": not replays["cached"]["requests"][2]["prefix_cache_active_kv_reused"],
        "full_owner_consecutive_reuse": replays["cached"]["requests"][3]["prefix_cache_active_kv_reused"],
    }
    return {"gate": {"vocabulary": 248320, "kld_strictly_less_than": 0.005,
                     "top1_match": True}, "cases": comparisons,
            "short_checkpoint_replay": replay_checks,
            "qualified": all(item["pass"] for item in comparisons) and all(replay_checks.values())}


def source_binding(engine_commit: str) -> dict:
    checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True).strip()
    runtime_delta = subprocess.run(
        ["git", "diff", "--quiet", engine_commit, "HEAD", "--", "native/src", "native/include",
         "native/aot", "native/generated", "scripts/build-native-runtime.sh"],
        cwd=ROOT, capture_output=True, check=False)
    return {"checkout_commit": checkout, "checkout_clean": not status,
            "native_source_commit": engine_commit,
            "engine_runtime_matches_checkout": runtime_delta.returncode == 0,
            "generator_sha256": protocol.sha256_file(Path(__file__)),
            "protocol_helper_sha256": protocol.sha256_file(ROOT / "scripts/qualify-native-chat-protocol.py")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--cache-capacity", type=int, default=32768)
    parser.add_argument("--port", type=int, default=18132)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--logits", action="store_true")
    parser.add_argument("--development", action="store_true")
    cli = parser.parse_args()
    cli.engine = cli.engine.resolve()
    cli.model_dir = cli.model_dir.resolve()
    cli.output = cli.output.resolve()
    cli.output.mkdir(parents=True, exist_ok=False)
    build_info = json.loads(subprocess.run(
        [str(cli.engine), "--build-info"], check=True, text=True, capture_output=True).stdout)
    clean_source = re.fullmatch(r"[0-9a-f]{40}", build_info["source_commit"]) is not None
    source = source_binding(build_info["source_commit"])
    protocol.require(clean_source or cli.development, "qualification requires a clean source build")
    cases = build_cases(cli.engine, cli.model_dir, cli.output, cli.performance, cli.cache_capacity)
    cached = run_cases(cli, cli.engine, "cached", cases)
    cold = run_cases(cli, cli.engine, "cold", cases)
    comparisons = compare_runs(cases, cold, cached)
    logits = qualify_logits(cli, cases) if cli.logits else None
    qualified = (all(item["pass"] for item in comparisons)
                 and cold["exit_code"] == cached["exit_code"] == 0
                 and 0 < cached["peak_memory"]["gtt_bytes"] <= 96 * 1024**3
                 and (logits is None or logits["qualified"]))
    performance = [item for item in comparisons if item["case_id"].startswith("performance_partial_")]
    result = {"schema": "aima-amd395-qwen36/native-safe-prefix-cache/v1",
              "complete": True,
              "created_at": datetime.now(timezone.utc).isoformat(), "qualified": qualified,
              "release_eligible": (qualified and clean_source and not cli.development
                                   and source["checkout_clean"] and source["engine_runtime_matches_checkout"]
                                   and cli.performance and cli.logits),
              "source": source, "case_input_sha256": canonical_json_sha256(cases),
              "build_info": build_info, "engine_sha256": protocol.sha256_file(cli.engine),
              "configuration": {"context_tokens": cli.context_tokens,
                                "cache_capacity": cli.cache_capacity, "checkpoint_limit_per_entry": 2},
              "cold_peak_memory": cold["peak_memory"], "cached_peak_memory": cached["peak_memory"],
              "host": {"system": platform.system(), "kernel": platform.release(),
                       "architecture": platform.machine()},
              "cases": comparisons, "logits": logits}
    if performance:
        result["performance"] = {
            "median_partial_ttft_speedup": statistics.median(item["ttft_speedup"] for item in performance),
            "median_decode_retention": statistics.median(item["cached_decode_tps"] for item in performance)
                / statistics.median(item["cold_decode_tps"] for item in performance),
        }
        result["performance"]["pass"] = (
            result["performance"]["median_partial_ttft_speedup"] > 1
            and result["performance"]["median_decode_retention"] >= 0.97)
        result["qualified"] &= result["performance"]["pass"]
        result["release_eligible"] &= result["performance"]["pass"]
    result["artifacts"] = [protocol.file_component(path, str(path.relative_to(cli.output)))
                           for path in sorted(cli.output.rglob("*"))
                           if path.is_file() and not path.name.endswith(".sha256")]
    result = seal_manifest(result)
    write_json(cli.output / "qualification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
