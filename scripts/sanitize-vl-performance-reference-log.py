#!/usr/bin/env python3
"""Publish a path-safe copy of a raw G4 reference failure log."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile


SCHEMA = "aima-amd395-qwen36/vl-performance-redacted-log/v1"
PREFIX = "AIMA_REDACTED_LOG_METADATA "
RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "private_home",
        re.compile(r"/(?:home|data/home)/[A-Za-z0-9._-]+"),
        "${AIMA_REMOTE_HOME}",
    ),
    (
        "private_ipv4",
        re.compile(
            r"\b(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}"
            r"|192\.168\.[0-9]{1,3}\.[0-9]{1,3}"
            r"|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})\b"
        ),
        "${AIMA_PRIVATE_IPV4}",
    ),
    (
        "benchmark_root",
        re.compile(r"/tmp/aima-native-vl-final-[A-Za-z0-9._-]+"),
        "${AIMA_BENCH_ROOT}",
    ),
    (
        "model_root",
        re.compile(r"/data/models/[^'\",\s)]+"),
        "${AIMA_MODEL_DIR}",
    ),
)


def sanitize(payload: bytes) -> tuple[str, dict[str, object]]:
    text = payload.decode("utf-8", errors="replace")
    counts: dict[str, int] = {}
    for name, pattern, replacement in RULES:
        text, count = pattern.subn(lambda _: replacement, text)
        counts[name] = count
    carriage_returns = text.count("\r")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    trailing_whitespace_lines = sum(line != line.rstrip(" \t") for line in lines)
    text = "\n".join(line.rstrip(" \t") for line in lines) + "\n"
    metadata: dict[str, object] = {
        "schema": SCHEMA,
        "source_bytes": len(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "replacement_counts": counts,
        "normalization_counts": {
            "carriage_returns": carriage_returns,
            "trailing_whitespace_lines": trailing_whitespace_lines,
        },
    }
    return PREFIX + json.dumps(metadata, sort_keys=True) + "\n" + text, metadata


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        parser.error(f"input log is missing: {source}")
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    sanitized, metadata = sanitize(source.read_bytes())
    atomic_write(output, sanitized)
    print(json.dumps({"output": str(output), **metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
