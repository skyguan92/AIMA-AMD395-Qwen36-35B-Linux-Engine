#!/usr/bin/env python3
"""Rebase selected AOT trace launches onto newly captured kernel variants."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HASH_CHUNK_BYTES = 1024 * 1024
POINTER_FIELDS = ("data_ptr", "storage_data_ptr", "storage_offset")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid trace JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"trace record is not an object at {path}:{line_number}"
                )
            records.append(value)
    return records


def parse_mapping(value: str) -> tuple[str, str]:
    old, separator, new = value.partition("=")
    if not separator or not old or not new:
        raise argparse.ArgumentTypeError("replacement must be OLD_SYMBOL=NEW_SYMBOL")
    return old, new


def parse_scalar_requirement(value: str) -> tuple[str, Any]:
    name, separator, raw = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("caller scalar must be NAME=JSON_VALUE")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"caller scalar value must be JSON: {raw}"
        ) from exc
    return name, parsed


def caller_scalar(record: dict[str, Any], name: str) -> Any:
    for frame in record.get("caller_context", []):
        scalars = frame.get("scalars", {})
        if name in scalars:
            return scalars[name]
    return None


def matches_requirements(
    record: dict[str, Any], requirements: list[tuple[str, Any]]
) -> bool:
    return all(caller_scalar(record, name) == value for name, value in requirements)


def selected_replacements(
    records: list[dict[str, Any]], symbols: set[str]
) -> dict[str, dict[str, Any]]:
    hard_errors = [
        record
        for record in records
        if record.get("event") in {"trace_launch_error", "trace_autotune_error"}
    ]
    if hard_errors:
        raise RuntimeError(
            f"replacement trace contains {len(hard_errors)} launch/autotune errors"
        )
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("event") != "triton_launch":
            continue
        symbol = str(record.get("python_qualname", ""))
        if symbol not in symbols:
            continue
        previous = selected.get(symbol)
        if previous is None or int(record["sequence"]) > int(previous["sequence"]):
            selected[symbol] = record
    missing = sorted(symbols - selected.keys())
    if missing:
        raise RuntimeError(f"replacement trace is missing selected launches: {missing}")
    for symbol, selected_record in selected.items():
        artifact_records = [
            record
            for record in records
            if record.get("event") == "triton_launch"
            and record.get("python_qualname") == symbol
            and record.get("kernel_hash") == selected_record.get("kernel_hash")
            and record.get("cache_files")
        ]
        if not artifact_records:
            raise RuntimeError(
                f"selected replacement launch has no captured cache artifacts: {symbol}"
            )
        artifact_record = max(
            artifact_records, key=lambda record: int(record["sequence"])
        )
        if (
            artifact_record.get("metadata") != selected_record.get("metadata")
            or artifact_record.get("signature") != selected_record.get("signature")
        ):
            raise RuntimeError(
                f"selected replacement artifact ABI does not match final launch: {symbol}"
            )
        selected_record = dict(selected_record)
        selected_record["cache_files"] = artifact_record["cache_files"]
        selected[symbol] = selected_record
    return selected


def rebase_launch(
    base: dict[str, Any], replacement: dict[str, Any]
) -> dict[str, Any]:
    base_arguments = {
        str(argument["name"]): argument for argument in base.get("arguments", [])
    }
    if len(base_arguments) != len(base.get("arguments", [])):
        raise RuntimeError("base launch contains duplicate argument names")

    arguments: list[dict[str, Any]] = []
    for replacement_argument in replacement.get("arguments", []):
        argument = dict(replacement_argument)
        if argument.get("kind") == "tensor":
            name = str(argument["name"])
            base_argument = base_arguments.get(name)
            if base_argument is None or base_argument.get("kind") != "tensor":
                raise RuntimeError(
                    f"replacement tensor argument has no base pointer identity: {name}"
                )
            for field in POINTER_FIELDS:
                if field in base_argument:
                    argument[field] = base_argument[field]
                else:
                    argument.pop(field, None)
        arguments.append(argument)

    result = dict(replacement)
    result["arguments"] = arguments
    for field in (
        "sequence",
        "caller_context",
        "captured_monotonic_ns",
        "captured_unix_ns",
        "first_structural_observation",
    ):
        if field in base:
            result[field] = base[field]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-trace", type=Path, required=True)
    parser.add_argument("--replacement-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="append",
        type=parse_mapping,
        required=True,
        metavar="OLD_SYMBOL=NEW_SYMBOL",
    )
    parser.add_argument(
        "--caller-scalar",
        action="append",
        type=parse_scalar_requirement,
        default=[],
        metavar="NAME=JSON_VALUE",
    )
    parser.add_argument("--expect-replacements", type=int)
    args = parser.parse_args()

    base_path = args.base_trace.resolve()
    replacement_path = args.replacement_trace.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite rebased trace: {output_path}")

    mappings = dict(args.replace)
    if len(mappings) != len(args.replace):
        raise RuntimeError("duplicate base symbols in --replace mappings")
    base_records = load_jsonl(base_path)
    replacements = selected_replacements(
        load_jsonl(replacement_path), set(mappings.values())
    )

    counts = {symbol: 0 for symbol in mappings}
    output_records: list[dict[str, Any]] = []
    for record in base_records:
        symbol = str(record.get("python_qualname", ""))
        if (
            record.get("event") == "triton_launch"
            and symbol in mappings
            and matches_requirements(record, args.caller_scalar)
        ):
            record = rebase_launch(record, replacements[mappings[symbol]])
            counts[symbol] += 1
        output_records.append(record)

    missing = sorted(symbol for symbol, count in counts.items() if count == 0)
    if missing:
        raise RuntimeError(f"no qualified base launches matched symbols: {missing}")
    replacement_count = sum(counts.values())
    if (
        args.expect_replacements is not None
        and replacement_count != args.expect_replacements
    ):
        raise RuntimeError(
            f"expected {args.expect_replacements} replacements, got {replacement_count}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        for record in output_records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    print(
        json.dumps(
            {
                "base_trace_sha256": sha256_file(base_path),
                "replacement_trace_sha256": sha256_file(replacement_path),
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "records": len(output_records),
                "replacement_count": replacement_count,
                "replacements_by_symbol": counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
