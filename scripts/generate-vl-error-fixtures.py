#!/usr/bin/env python3
"""Derive deterministic media fixtures for VL error/limit qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
)


SCHEMA = "aima-amd395-qwen36/vl-error-fixtures/v1"
SOURCE_RELATIVE = Path(
    "benchmarks/fixtures/vl-capability-v0.1.0/"
    "video-12f-6fps-192x128.avi"
)
SOURCE_MANIFEST_RELATIVE = Path(
    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json"
)
OUTPUT_RELATIVE = Path(
    "benchmarks/fixtures/vl-error-v0.1.0/"
    "video-12f-0.002fps-192x128.avi"
)
OUTPUT_NAME = OUTPUT_RELATIVE.name

SOURCE_MICROSECONDS_PER_FRAME = 166_666
SOURCE_SCALE = 1
SOURCE_RATE = 6
TARGET_MICROSECONDS_PER_FRAME = 500_000_000
TARGET_SCALE = 500
TARGET_RATE = 1
FRAME_COUNT = 12


def _chunks(
    payload: bytes | bytearray, start: int, end: int
) -> Iterator[tuple[bytes, int, int]]:
    """Yield RIFF chunks recursively as (fourcc, payload offset, size)."""
    cursor = start
    while cursor < end:
        if cursor + 8 > end:
            raise ValueError("truncated RIFF chunk header")
        fourcc = bytes(payload[cursor : cursor + 4])
        size = struct.unpack_from("<I", payload, cursor + 4)[0]
        data_offset = cursor + 8
        data_end = data_offset + size
        if data_end > end:
            raise ValueError("RIFF chunk extends beyond its parent")
        yield fourcc, data_offset, size
        if fourcc == b"LIST":
            if size < 4:
                raise ValueError("LIST chunk has no list type")
            yield from _chunks(payload, data_offset + 4, data_end)
        cursor = data_end + (size & 1)
    if cursor != end:
        raise ValueError("RIFF padding extends beyond its parent")


def _riff_chunks(
    payload: bytes | bytearray,
) -> Iterator[tuple[bytes, int, int]]:
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"AVI ":
        raise ValueError("source fixture is not a RIFF AVI file")
    riff_size = struct.unpack_from("<I", payload, 4)[0]
    riff_end = 8 + riff_size
    if riff_end != len(payload):
        raise ValueError("RIFF size does not bind the complete source fixture")
    return _chunks(payload, 12, riff_end)


def derive_long_duration_avi(source: bytes) -> bytes:
    derived = bytearray(source)
    avih_offsets: list[int] = []
    video_strh_offsets: list[int] = []
    for fourcc, offset, size in _riff_chunks(derived):
        if fourcc == b"avih":
            if size < 56:
                raise ValueError("AVI main header is truncated")
            avih_offsets.append(offset)
        elif fourcc == b"strh":
            if size < 56:
                raise ValueError("AVI stream header is truncated")
            if derived[offset : offset + 4] == b"vids":
                video_strh_offsets.append(offset)
    if len(avih_offsets) != 1 or len(video_strh_offsets) != 1:
        raise ValueError("expected exactly one AVI main header and video stream")

    avih = avih_offsets[0]
    strh = video_strh_offsets[0]
    source_headers = (
        struct.unpack_from("<I", derived, avih)[0],
        struct.unpack_from("<I", derived, strh + 20)[0],
        struct.unpack_from("<I", derived, strh + 24)[0],
        struct.unpack_from("<I", derived, avih + 16)[0],
        struct.unpack_from("<I", derived, strh + 32)[0],
    )
    expected_headers = (
        SOURCE_MICROSECONDS_PER_FRAME,
        SOURCE_SCALE,
        SOURCE_RATE,
        FRAME_COUNT,
        FRAME_COUNT,
    )
    if source_headers != expected_headers:
        raise ValueError(
            f"source AVI header drifted: {source_headers} != {expected_headers}"
        )

    struct.pack_into("<I", derived, avih, TARGET_MICROSECONDS_PER_FRAME)
    struct.pack_into("<I", derived, strh + 20, TARGET_SCALE)
    struct.pack_into("<I", derived, strh + 24, TARGET_RATE)
    return bytes(derived)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / SOURCE_RELATIVE)
    parser.add_argument(
        "--output", type=Path, default=ROOT / OUTPUT_RELATIVE.parent
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / OUTPUT_NAME
    output.write_bytes(derive_long_duration_avi(source.read_bytes()))

    manifest = seal_manifest(
        {
            "schema": SCHEMA,
            "complete": True,
            "generator": file_component(
                Path(__file__).resolve(),
                "scripts/generate-vl-error-fixtures.py",
            ),
            "source": {
                "fixture": file_component(source, SOURCE_RELATIVE.as_posix()),
                "manifest": file_component(
                    ROOT / SOURCE_MANIFEST_RELATIVE,
                    SOURCE_MANIFEST_RELATIVE.as_posix(),
                ),
                "header": {
                    "microseconds_per_frame": SOURCE_MICROSECONDS_PER_FRAME,
                    "scale": SOURCE_SCALE,
                    "rate": SOURCE_RATE,
                    "fps": SOURCE_RATE / SOURCE_SCALE,
                    "frame_count": FRAME_COUNT,
                },
            },
            "fixtures": [
                {
                    "fixture_id": "video_long_duration_low_fps",
                    **file_component(output, OUTPUT_RELATIVE.as_posix()),
                    "modality": "video",
                    "format": "avi",
                    "width": 192,
                    "height": 128,
                    "frame_count": FRAME_COUNT,
                    "fps": TARGET_RATE / TARGET_SCALE,
                    "duration_seconds": FRAME_COUNT * TARGET_SCALE / TARGET_RATE,
                    "boundary": "duration_beyond_removed_native_768_second_cap",
                    "derivation": {
                        "method": "riff_header_patch_no_frame_reencode",
                        "microseconds_per_frame": TARGET_MICROSECONDS_PER_FRAME,
                        "scale": TARGET_SCALE,
                        "rate": TARGET_RATE,
                    },
                }
            ],
        }
    )
    digest = atomic_json(output_root / "fixtures-manifest.json", manifest)
    print(
        json.dumps(
            {
                "fixture": str(output),
                "manifest_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
