#!/usr/bin/env python3
"""Verify release metadata and every binary input against a qualification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.package_qualification import verify_package_qualification


def component(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("component must use NAME=PATH")
    return name, Path(path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--component", action="append", type=component, required=True)
    args = parser.parse_args()
    components = dict(args.component)
    if len(components) != len(args.component):
        raise SystemExit("duplicate --component name")
    errors = verify_package_qualification(
        args.qualification.resolve(),
        release=args.release,
        release_tag=args.release_tag,
        source_commit=args.source_commit,
        components=components,
    )
    if errors:
        for error in errors:
            print(error)
        print(f"package qualification: FAIL ({len(errors)} error(s))")
        return 1
    print("package qualification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
