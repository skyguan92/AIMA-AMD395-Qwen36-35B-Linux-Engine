#!/usr/bin/env python3
"""Capture exact frozen-vLLM RGBA image loader byte oracles."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import socket
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    PINNED_PACKAGES,
    atomic_json,
    file_component,
    git_identity,
    seal_manifest,
    sha256_bytes,
)
from vllm.multimodal.media import ImageMediaIO  # noqa: E402
import vllm  # noqa: E402


SCHEMA = "aima-amd395-qwen36/vl-media-io-reference/v1"
FIXTURE = (
    ROOT
    / "benchmarks/fixtures/vl-capability-v0.1.0"
    / "image-transparent-160x320.png"
)
EXPECTED_RGB_SHA256 = {
    "default_white": "c779b79d2b3dc97c964b1f931bb9056602fba3b40eee297131e247680e36104e",
    "red": "debb77f47c8594a633976b272b192ac42db5f396de52c4ee8789a57854f176ef",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture = args.fixture.resolve()
    output = args.output.resolve()
    if not fixture.is_file():
        raise SystemExit(f"RGBA fixture is missing: {fixture}")
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise SystemExit("media-IO oracle output and sidecar must not exist")
    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("media-IO oracle capture requires clean source")
    expected_vllm = PINNED_PACKAGES["vllm"]
    if not (
        vllm.__version__ == expected_vllm
        or vllm.__version__.startswith(expected_vllm + ".")
    ):
        raise SystemExit(f"reference vLLM version differs: {vllm.__version__!r}")

    payload_bytes = fixture.read_bytes()
    cases = []
    for case_id, background in (
        ("default_white", [255, 255, 255]),
        ("red", [255, 0, 0]),
    ):
        loaded = ImageMediaIO(
            rgba_background_color=background
        ).load_bytes(payload_bytes)
        image = loaded.media
        rgb_bytes = image.tobytes()
        cases.append(
            {
                "case_id": case_id,
                "rgba_background_color": background,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "rgb_bytes": len(rgb_bytes),
                "rgb_sha256": sha256_bytes(rgb_bytes),
                "expected_rgb_sha256": EXPECTED_RGB_SHA256[case_id],
                "qualified": image.mode == "RGB"
                and (image.width, image.height) == (160, 320)
                and len(rgb_bytes) == 160 * 320 * 3
                and sha256_bytes(rgb_bytes) == EXPECTED_RGB_SHA256[case_id],
            }
        )

    media_io_source = Path(inspect.getsourcefile(ImageMediaIO) or "").resolve()
    complete = all(case["qualified"] for case in cases) and (
        cases[0]["rgb_sha256"] != cases[1]["rgb_sha256"]
    )
    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "captured_at": utc_now(),
            "complete": complete,
            "qualified": complete,
            "scope": "frozen-vllm-image-media-io-rgba-byte-oracle",
            "host": {"label": "amd395", "hostname": socket.gethostname()},
            "source": {
                **source,
                "files": [
                    file_component(
                        Path(__file__).resolve(),
                        "scripts/capture-vllm-vl-media-io-oracles.py",
                    )
                ],
            },
            "runtime": {
                "vllm": vllm.__version__,
                "pillow": importlib.metadata.version("pillow"),
                "image_media_io_source": file_component(
                    media_io_source, "vllm/multimodal/media/image.py"
                ),
            },
            "fixture": file_component(
                fixture,
                "benchmarks/fixtures/vl-capability-v0.1.0/"
                "image-transparent-160x320.png",
            ),
            "cases": cases,
            "decision": {
                "default_white_exact": cases[0]["qualified"],
                "request_red_exact": cases[1]["qualified"],
                "background_changes_rgb_bytes": cases[0]["rgb_sha256"]
                != cases[1]["rgb_sha256"],
                "g1_passed": False,
                "g2_passed": False,
                "g3_passed": False,
                "g4_passed": False,
                "g5_passed": False,
            },
        }
    )
    digest = atomic_json(output, payload)
    print(
        json.dumps(
            {
                "complete": complete,
                "output": str(output),
                "qualified": complete,
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
