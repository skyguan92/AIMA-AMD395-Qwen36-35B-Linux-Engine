#!/usr/bin/env python3
"""Capture frozen block-0 components for unequal multi-media shapes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE_CAPTURE = ROOT / "scripts/capture-vllm-vision-block-oracles.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "aima_vl_multimedia_block_capture", BASE_CAPTURE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen vision block capture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SCHEMA = (
        "aima-amd395-qwen36/vision-multimedia-block-oracle/v1"
    )
    module.CASE_IDS = ("multi_image", "multi_video")
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
