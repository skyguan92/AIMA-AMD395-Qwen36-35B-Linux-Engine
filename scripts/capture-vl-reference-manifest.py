#!/usr/bin/env python3
"""Capture or verify the hash-bound native-VL reference manifest."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    ReferenceManifestError,
    atomic_json,
    build_reference_manifest,
    load_json_object,
    verify_reference_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--model-dir", type=Path, required=True)
    capture.add_argument("--launch-config", type=Path, required=True)
    capture.add_argument("--capability-manifest", type=Path, required=True)
    capture.add_argument(
        "--product-contract",
        type=Path,
        default=ROOT / "native/product-contract-v1.5.1.json",
    )
    capture.add_argument(
        "--goal-document",
        type=Path,
        default=ROOT / "docs/NATIVE_VL_GOAL.md",
    )
    capture.add_argument("--host-label", default="amd395")
    capture.add_argument("--captured-at")
    capture.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--model-dir", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "capture":
            manifest = build_reference_manifest(
                root=ROOT,
                model_dir=args.model_dir.resolve(),
                launch_config_path=args.launch_config.resolve(),
                capability_manifest_path=args.capability_manifest.resolve(),
                product_contract_path=args.product_contract.resolve(),
                goal_document_path=args.goal_document.resolve(),
                host_label=args.host_label,
                captured_at=args.captured_at,
            )
            digest = atomic_json(args.output.resolve(), manifest)
            print(f"VL reference manifest: PASS ({digest})")
            return 0

        manifest = load_json_object(args.manifest.resolve())
        errors = verify_reference_manifest(
            manifest,
            model_dir=args.model_dir.resolve(),
        )
        if errors:
            for error in errors:
                print(error)
            print(f"VL reference manifest: FAIL ({len(errors)} error(s))")
            return 1
        print("VL reference manifest: PASS")
        return 0
    except ReferenceManifestError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
