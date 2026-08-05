#!/usr/bin/env python3
"""Run a frozen answer-only eval through the exact resident native binary.

The input dataset and pretokenized requests stay outside the public repository.
The generated scorecard contains only hashes, answers, aggregate scores and
runtime metrics; it never copies questions or prompt token IDs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SCHEMA = "aima-amd395-qwen36/native-answer-eval/v1"
MODEL = "aima-amd395-qwen36-35b"
ANSWER = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")
ANSWER_TOKEN_IDS = {32: "A", 33: "B", 34: "C", 35: "D"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(token_ids: list[int]) -> str:
    return sha256_bytes(",".join(str(value) for value in token_ids).encode())


def parse_answer(content: str) -> str | None:
    match = ANSWER.search(content)
    return None if match is None else match.group(1)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_items(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}:{line_number}: expected object")
        result.append(value)
    identifiers = [item.get("item_id") for item in result]
    if not result or any(not isinstance(value, str) for value in identifiers):
        raise RuntimeError("eval items require non-empty string item_id values")
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("eval items contain duplicate item_id values")
    return result


def request_for(
    item: dict[str, Any], requests_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = item.get("request_path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{item.get('item_id')}: request_path is missing")
    root = requests_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"{item.get('item_id')}: request_path is unsafe or missing")
    frozen = read_json(path)
    values = frozen.get("prompt_token_ids")
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, int) and 0 <= value < 248_320 for value in values)
    ):
        raise RuntimeError(f"{item.get('item_id')}: invalid prompt_token_ids")
    digest = token_ids_sha256(values)
    if digest != item.get("prompt_token_ids_sha256"):
        raise RuntimeError(f"{item.get('item_id')}: prompt token hash changed")
    if len(values) != item.get("prompt_tokens"):
        raise RuntimeError(f"{item.get('item_id')}: prompt token count changed")
    max_tokens = item.get("requested_output_tokens", frozen.get("max_tokens"))
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise RuntimeError(f"{item.get('item_id')}: invalid output token count")
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Frozen eval prompt supplied by prompt_token_ids.",
            }
        ],
        "prompt_token_ids": values,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }
    source = {
        "prompt_tokens": len(values),
        "prompt_token_ids_sha256": digest,
        "request_file_sha256": sha256_file(path),
    }
    return payload, source


def http_json(
    url: str,
    *,
    payload: dict[str, Any] | None,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
        method = "POST"
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"request failed: {error}") from error
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError("HTTP response is not an object")
    return value


def engine_record(binary: Path) -> dict[str, Any]:
    resolved = binary.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"native engine is missing: {resolved}")
    result = subprocess.run(
        [str(resolved), "--build-info"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"native --build-info failed: {result.stderr.strip()}")
    build = json.loads(result.stdout)
    if not isinstance(build, dict):
        raise RuntimeError("native --build-info did not return an object")
    return {
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "build_info": build,
    }


def scorecard(
    *,
    items_path: Path,
    item_count: int,
    records: list[dict[str, Any]],
    engine: dict[str, Any],
    health: dict[str, Any],
    minimum_correct: int | None,
    started_at: str,
    complete: bool,
    reference_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    correct = sum(record["correct"] for record in records)
    invalid = sum(record["parsed_answer"] is None for record in records)
    domain_counts: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        domain_counts[record["subject"]].append(record["correct"])
    domains = {
        domain: {
            "correct": sum(values),
            "items": len(values),
            "accuracy": sum(values) / len(values),
        }
        for domain, values in sorted(domain_counts.items())
    }
    gate = minimum_correct is None or correct >= minimum_correct
    result = {
        "schema": SCHEMA,
        "complete": complete,
        "qualified": complete and invalid == 0 and gate,
        "started_at_utc": started_at,
        "updated_at_utc": utc_now(),
        "claim_boundary": (
            "Frozen deterministic answer-only regression subset; not an official "
            "leaderboard score. Prompt text and token IDs are intentionally excluded."
        ),
        "source": {
            "items_file_name": items_path.name,
            "items_file_sha256": sha256_file(items_path),
            "selected_items": item_count,
            "prompt_token_ids_in_scorecard": False,
            "prompt_text_in_scorecard": False,
        },
        "engine": engine,
        "service": {
            "transport": urlsplit("http://127.0.0.1").scheme,
            "model": health.get("model"),
            "runtime": health.get("runtime"),
            "resident": health.get("resident"),
            "context_capacity": health.get("context_capacity"),
            "resident_prefill_buckets": health.get("resident_prefill_buckets"),
            "prefix_cache_entries": health.get("prefix_cache_entries"),
            "prompt_token_ids_extension": health.get(
                "prompt_token_ids_extension"
            ),
        },
        "protocol": {
            "batch_size": 1,
            "temperature": 0,
            "top_p": 1,
            "parser": "first standalone A/B/C/D in decoded completion",
            "same_frozen_prompt_token_ids": True,
        },
        "progress": {"completed": len(records), "items": item_count},
        "score": {
            "correct": correct,
            "items": len(records),
            "accuracy": correct / len(records) if records else 0.0,
            "invalid_answers": invalid,
            "finish_reasons": dict(
                sorted(Counter(record["finish_reason"] for record in records).items())
            ),
        },
        "gate": {
            "minimum_correct": minimum_correct,
            "score_pass": gate if complete else None,
            "all_answers_valid": invalid == 0 if complete else None,
        },
        "domains": domains,
        "records": records,
    }
    if reference_comparison is not None:
        result["reference_comparison"] = reference_comparison
    return result


def compare_reference(
    path: Path,
    items: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = read_json(path.resolve())
    reference_records = reference.get("records")
    if not isinstance(reference_records, list) or len(reference_records) != len(
        items
    ):
        raise RuntimeError("reference result does not cover the selected items")
    if len(records) != len(items):
        raise RuntimeError("reference comparison requires a complete current run")

    prompt_hash_matches = 0
    completion_hash_matches = 0
    answer_matches = 0
    reference_correct = 0
    changed_answers: list[dict[str, Any]] = []
    for item, current, frozen in zip(
        items, records, reference_records, strict=True
    ):
        if (
            frozen.get("item_id") != item["item_id"]
            or current.get("item_id") != item["item_id"]
            or frozen.get("correct_answer") != item.get("correct_answer")
        ):
            raise RuntimeError("reference result item order or answer changed")
        prompt_equal = (
            frozen.get("prompt_token_ids_sha256")
            == current.get("prompt_token_ids_sha256")
        )
        prompt_hash_matches += int(prompt_equal)
        if not prompt_equal:
            raise RuntimeError("reference result prompt token hash changed")
        reference_answer = ANSWER_TOKEN_IDS.get(frozen.get("first_token_id"))
        if reference_answer is None:
            raise RuntimeError("reference result has a non-answer first token")
        current_answer = current.get("parsed_answer")
        if current_answer == reference_answer:
            answer_matches += 1
        else:
            changed_answers.append(
                {
                    "item_id": item["item_id"],
                    "reference_answer": reference_answer,
                    "current_answer": current_answer,
                    "correct_answer": item["correct_answer"],
                }
            )
        reference_correct += int(reference_answer == item["correct_answer"])
        completion_hash_matches += int(
            current.get("output_token_ids_sha256")
            == frozen.get("completion_token_ids_sha256")
        )

    current_correct = sum(record["correct"] for record in records)
    return {
        "claim_boundary": (
            "Paired against a frozen GB10 vLLM result with identical prompt-token "
            "hashes; the reference records and prompt tokens are not redistributed."
        ),
        "reference": {
            "file_name": path.name,
            "file_sha256": sha256_file(path),
            "schema": reference.get("schema"),
            "served_model": reference.get("served_model"),
            "speculative_mode": reference.get("speculative_mode"),
        },
        "items": len(items),
        "prompt_token_hash_matches": prompt_hash_matches,
        "completion_token_hash_matches": completion_hash_matches,
        "answer_matches": answer_matches,
        "changed_answers": changed_answers,
        "reference_correct": reference_correct,
        "current_correct": current_correct,
        "correct_delta": current_correct - reference_correct,
        "score_nonregression_pass": current_correct >= reference_correct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--requests-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-binary", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--minimum-correct", type=int)
    parser.add_argument(
        "--reference-result",
        type=Path,
        help="optional frozen answer-only result for hash/score pairing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items_path = args.items.resolve()
    all_items = read_items(items_path)
    if args.limit is not None:
        if args.limit <= 0 or args.limit > len(all_items):
            raise RuntimeError("--limit is outside the item set")
        items = all_items[: args.limit]
    else:
        items = all_items
    if args.minimum_correct is not None and not (
        0 <= args.minimum_correct <= len(items)
    ):
        raise RuntimeError("--minimum-correct is outside the selected item set")
    requests_root = (
        args.requests_root.resolve()
        if args.requests_root is not None
        else items_path.parent
    )
    engine = engine_record(args.engine_binary)
    parsed_endpoint = urlsplit(args.endpoint)
    health_url = f"{parsed_endpoint.scheme}://{parsed_endpoint.netloc}/health"
    api_key = None
    if args.api_key_file is not None:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
        if not api_key:
            raise RuntimeError("--api-key-file is empty")
    health = http_json(
        health_url, payload=None, timeout=args.timeout, api_key=api_key
    )
    if (
        health.get("status") != "ok"
        or health.get("model_loaded") is not True
        or health.get("prompt_token_ids_extension") is not True
    ):
        raise RuntimeError("native service is not ready for frozen-token eval")

    started_at = utc_now()
    records: list[dict[str, Any]] = []
    output = args.output.resolve()
    if output.exists():
        previous = read_json(output)
        if previous.get("schema") != SCHEMA:
            raise RuntimeError("existing output uses another schema")
        if previous.get("source", {}).get("items_file_sha256") != sha256_file(
            items_path
        ):
            raise RuntimeError("existing output belongs to another item set")
        if previous.get("engine") != engine:
            raise RuntimeError("existing output belongs to another native binary")
        previous_records = previous.get("records")
        if not isinstance(previous_records, list):
            raise RuntimeError("existing output has invalid records")
        records = previous_records
        started_at = str(previous.get("started_at_utc", started_at))
    completed_ids = {record.get("item_id") for record in records}
    expected_prefix = [item["item_id"] for item in items[: len(records)]]
    if [record.get("item_id") for record in records] != expected_prefix:
        raise RuntimeError("existing output is not a valid ordered prefix")

    for item in items:
        if item["item_id"] in completed_ids:
            continue
        payload, source = request_for(item, requests_root)
        request_started = time.monotonic()
        response = http_json(
            args.endpoint,
            payload=payload,
            timeout=args.timeout,
            api_key=api_key,
        )
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
            usage = response["usage"]
            metrics = response["aima_amd395"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"{item['item_id']}: malformed response") from error
        if not isinstance(content, str):
            raise RuntimeError(f"{item['item_id']}: completion content is not text")
        expected = item.get("correct_answer")
        if expected not in {"A", "B", "C", "D"}:
            raise RuntimeError(f"{item['item_id']}: invalid correct_answer")
        parsed = parse_answer(content)
        record = {
            "ordinal": item.get("ordinal"),
            "item_id": item["item_id"],
            "subject": item.get("subject", "unknown"),
            **source,
            "expected_answer": expected,
            "parsed_answer": parsed,
            "correct": parsed == expected,
            "content_sha256": sha256_bytes(content.encode("utf-8")),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "output_token_ids_sha256": metrics.get("output_token_ids_sha256"),
            "prompt_source": metrics.get("prompt_source"),
            "prompt_execution": metrics.get("prompt_execution"),
            "aot_prefill_tokens": metrics.get("aot_prefill_tokens"),
            "request_wall_ms": metrics.get("request_wall_ms"),
            "http_round_trip_ms": (time.monotonic() - request_started) * 1000,
        }
        if record["prompt_source"] != "token_ids":
            raise RuntimeError(f"{item['item_id']}: raw prompt source was not used")
        records.append(record)
        write_json(
            output,
            scorecard(
                items_path=items_path,
                item_count=len(items),
                records=records,
                engine=engine,
                health=health,
                minimum_correct=args.minimum_correct,
                started_at=started_at,
                complete=False,
            ),
        )

    reference_comparison = (
        compare_reference(args.reference_result, items, records)
        if args.reference_result is not None
        else None
    )
    result = scorecard(
        items_path=items_path,
        item_count=len(items),
        records=records,
        engine=engine,
        health=health,
        minimum_correct=args.minimum_correct,
        started_at=started_at,
        complete=True,
        reference_comparison=reference_comparison,
    )
    if reference_comparison is not None:
        reference_pass = reference_comparison["score_nonregression_pass"]
        result["gate"]["reference_score_nonregression"] = reference_pass
        result["qualified"] = bool(result["qualified"] and reference_pass)
    write_json(output, result)
    print(json.dumps({"output": str(output), "score": result["score"], "qualified": result["qualified"]}))
    return 0 if result["qualified"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
